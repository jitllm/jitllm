package tier

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// A prefill chunk's attention over pages: the rows of one sequence in position
// order, which is what the paged prefill kernels take (kernels/pagedprefill.go)
// and what lets one K or V load serve a whole query tile -- FlashDecodeKV reads
// every row's keys on its own, which runs a prefill at about half the
// contiguous rate. A ragged batch keeps FlashDecodeKV: its rows are different
// sequences.
//
// Per device, as the contiguous path chooses: the one-kernel flash form where a
// device has one (FlashPrefill70 on sm_70's m8n8k4, FlashPrefillTile on
// Metal), else the staged scores, softmax and accumulate, the scores on
// m16n8k16 where it lowers. The staged kernels are built per span of keys
// (a power of two from 64): a chunk attending 300 keys runs kernels of 512,
// and its planes are that span's, never the context's. Past prefillPassW the
// history goes in passes of that width, folded into a running partial.

// prefillPassW is the most keys one staged prefill launch attends: a longer
// span runs in passes of it (kernels.FlashAttentionMergeRun), which bounds the
// score planes by it rather than by the context.
const prefillPassW = 4096

// prefillPassChunk is a pass's key split: prefillPassW/prefillPassChunk splits
// a launch, each its own partial.
const prefillPassChunk = 256

// prefillPass is the pass width and its key split: Config.PrefillPass or
// prefillPassW, a split of prefillPassChunk keys (or the whole pass, below it).
func (g *devTier) prefillPass() (width, chunk int) {
	w := g.PrefillPass
	if w <= 0 {
		w = prefillPassW
	}
	w = (w + 63) &^ 63
	if w > prefillPassChunk {
		w = (w + prefillPassChunk - 1) / prefillPassChunk * prefillPassChunk
	}
	return w, min(w, prefillPassChunk)
}

// pagedPrefill is a batched scratch's prefill attention set.
type pagedPrefill struct {
	// flash is the one-kernel form, and fmerge its merge where the plan has
	// sinks (the kernel's one split writes partials; the merge adds the sink).
	flash, fmerge backend.Kernel
	// fs is the flash forms' shape; ragBuild builds the flash form's run
	// table (raggedprefill.go) at a shape, ragGroups and ragThreads are its
	// launch and ragWG the rows a workgroup takes, nil where the form has
	// none; ragK is it compiled, by slot count.
	fs         kernels.FlashPrefill70Shape
	ragBuild   func(kernels.FlashPrefill70Shape) (*ir.Kernel, error)
	ragGroups  func(kernels.FlashPrefill70Shape) int
	ragThreads int
	ragWG      int
	ragK       map[int]backend.Kernel
	// ragMerge is the mapped wide merge by slot count, with sinks.
	ragMerge map[int]backend.Kernel
	// ragVars is the staged form's run-table variants by chunk, splits and
	// width; ragFin is the mapped merge that finishes its passes, by width.
	ragVars map[[3]int]*prefillVariant
	ragFin  map[int]backend.Kernel
	// flash80 says flash is kernels.FlashPrefill80, the m16n8k16 form, or
	// its m16n8k8 twin on sm_75 (FlashPrefill80K8).
	flash80         bool
	fGroups, fWidth int
	fRows           int // the merge's threads: rows x heads x value
	// The staged form: a variant per span, the pass folds, the planes.
	lanes, qt, sq int // softmax lanes, the accumulate's query tile, the scores'
	mode          prefillMode
	kt            int
	vars          map[int]*prefillVariant
	passV         *prefillVariant
	mFirst, mFold backend.Kernel
	mFin          backend.Kernel
	sc, wt        backend.Buf // the score and weight planes
	planeFloats   int
	part          backend.Buf // a launch's split partials, and the flash form's
	partFloats    int
	run           [2]backend.Buf // the passes' running partial, ping-ponged
	// The attention descriptors, full [0] and windowed [1]: the writers'
	// with each padded row empty at the last real row's keyEnd, and per pass
	// the same clipped to the pass's keys.
	buf  [2]backend.Buf
	pass [2][]backend.Buf
	plan [2]prefillPlan
}

// prefillMode is the staged form's matrix instruction.
type prefillMode int

const (
	prefillTiled  prefillMode = iota // FMA scores and accumulate
	prefillMMA                       // m16n8k16 scores, FMA accumulate
	prefillMMAAcc                    // m16n8k16 scores, m16n8k8 accumulate
	prefillMMA70                     // sm_70's m8n8k4, both products
)

// accMMANT is PagedAttnAccMMA's query tiles a warp: 16 queries, the scores
// kernel's query tile, which the softmax's (and so the accumulate's) must
// divide.
const accMMANT = 2

// prefillVariant is the staged kernels at one span.
type prefillVariant struct {
	s                   kernels.FlashShape
	scores, soft, acc   backend.Kernel
	scoreT, softG, accT int
	// softW is the softmax's workgroup width, which its lanes decide
	// (kernels.PagedPrefillSoftmaxWidth): a 32-lane softmax is one item a
	// 32-wide group, and Metal, which runs a launch at the width it is given,
	// put two of its subgroups on one item at 64.
	softW int
}

// prefillPlan is one descriptor set's call: its variant, how many passes (0 for
// one direct launch) and its staging.
type prefillPlan struct {
	v      *prefillVariant
	passes int
	desc   []uint32 // one launch's descriptors
	passD  []uint32 // passes*len(desc): pass j's clipped copy
}

// initPagedPrefill builds bs's prefill set, or leaves it nil (and prefill on
// FlashDecodeKV) where a device refuses every form. Callers hold g.mu.
func (g *devTier) initPagedPrefill(bs *blockScratch) error {
	defer g.scratchWin().close() // the scratch's own buffers (scratch.go)
	p, pg := &bs.p, bs.pkv
	// The attention's width: a head, or MLA's whole latent row.
	rows, hd := bs.rows, pg.shape.Dim
	gqa := p.NHead / p.NKVHead
	pf := &pagedPrefill{vars: map[int]*prefillVariant{}}
	comp := func(k *ir.Kernel, err error) backend.Kernel {
		if err != nil {
			return nil
		}
		c, err := g.compileShared(k)
		if err != nil {
			return nil
		}
		return c
	}
	fs := kernels.FlashPrefill70Shape{Heads: p.NHead, KVHeads: p.NHead / gqa, Dim: hd, Rows: rows,
		Scale: pg.shape.Scale, Softcap: p.AttnSoftcap, Page: pg.shape.Page, F16: g.KVF16}
	if p.AttnSinks {
		fs.Splits = 1
	}
	pf.fs = fs
	// The flash forms and the m16n8k16 scores read no latent row (MLA): the
	// FMA tiles and sm_70's pair do.
	if !g.NoFlashPrefill && !p.MLA() {
		tile := func() {
			if t, ok := kernels.FlashTileFor(hd); ok && t.SG >= 4 && rows%t.Rows() == 0 && 32%t.TK == 0 {
				if pf.flash = comp(kernels.FlashPrefillTile(fs, t)); pf.flash != nil {
					pf.fGroups, pf.fWidth = kernels.FlashPrefillTileGroups(fs, t), t.Threads()
					pf.ragBuild = func(s kernels.FlashPrefill70Shape) (*ir.Kernel, error) { return kernels.FlashPrefillTile(s, t) }
					pf.ragGroups = func(s kernels.FlashPrefill70Shape) int { return kernels.FlashPrefillTileGroups(s, t) }
					pf.ragThreads, pf.ragWG = t.Threads(), t.Rows()
				}
			}
		}
		switch {
		case g.PreferTiles && g.tileF32():
			tile()
		case !g.kb.noAttnMMA && g.f16GemmK() == 16 && kernels.FlashPrefill80WhyNot(fs) == "":
			// sm_80 on: the whole attention in one kernel on m16n8k16.
			if pf.flash = comp(kernels.FlashPrefill80(fs)); pf.flash != nil {
				pf.fGroups, pf.fWidth = kernels.FlashPrefill80Groups(fs), kernels.FlashPrefill80Threads
				pf.flash80 = true
				pf.ragBuild, pf.ragGroups = kernels.FlashPrefill80, kernels.FlashPrefill80Groups
				pf.ragThreads, pf.ragWG = kernels.FlashPrefill80Threads, kernels.FlashPrefill80Rows(fs)
			}
		case !g.kb.noAttnMMA && g.f16GemmK() == 8 && kernels.FlashPrefill80WhyNot(fs) == "":
			// sm_75: the same kernel on m16n8k8, two products for each of
			// the k16 form's. A width it does not divide keeps sm_70's pair.
			if pf.flash = comp(kernels.FlashPrefill80K8(fs)); pf.flash != nil {
				pf.fGroups, pf.fWidth = kernels.FlashPrefill80Groups(fs), kernels.FlashPrefill80Threads
				pf.flash80 = true
				pf.ragBuild, pf.ragGroups = kernels.FlashPrefill80K8, kernels.FlashPrefill80Groups
				pf.ragThreads, pf.ragWG = kernels.FlashPrefill80Threads, kernels.FlashPrefill80Rows(fs)
			}
		case g.tileF32():
			// The collective tiles where they lower and FlashPrefill80 does
			// not run: Metal, and a CUDA card it refuses.
			tile()
		case !g.NoVolta && (hd == 64 || hd == 128) && rows%kernels.FlashPrefill70Rows == 0 && g.sm70():
			if pf.flash = comp(kernels.FlashPrefill70(fs)); pf.flash != nil {
				pf.fGroups, pf.fWidth = kernels.FlashPrefill70Groups(fs), kernels.FlashPrefill70Threads
				pf.ragBuild, pf.ragGroups = kernels.FlashPrefill70, kernels.FlashPrefill70Groups
				pf.ragThreads, pf.ragWG = kernels.FlashPrefill70Threads, kernels.FlashPrefill70Rows
			}
		}
	}
	if pf.ragBuild != nil || pf.flash == nil {
		g.RagForms++
	}
	if pf.flash != nil && p.AttnSinks {
		ms := kernels.FlashShape{Heads: p.NHead, KVHeads: p.NKVHead, Dim: hd, Rows: rows, Splits: 1, Sink: true}
		if pf.fmerge = comp(kernels.FlashAttentionMergeWide(ms)); pf.fmerge == nil {
			closeKern(pf.flash)
			pf.flash = nil
		} else if err := g.prefillBuf(&pf.part, &pf.partFloats, kernels.FlashPartialFloats(ms)); err != nil {
			pf.free()
			return err
		}
		pf.fRows = rows * p.NHead * hd
	}
	if pf.flash == nil {
		pf.lanes, pf.kt = g.pickLanes(), g.tiles.ktile
		if pf.lanes != ir.SubgroupLanes {
			pf.lanes = 1
		}
		// The matrix instruction where it lowers, as the contiguous path
		// chooses: m16n8k16, else sm_70's m8n8k4, else the FMA tiles. The
		// pass form is built here for each in turn -- every depth past the pass
		// width takes it, and a refusal is better found at placement.
		mat := !g.NoMMA && !g.kb.noAttnMMA
		var modes []prefillMode
		// The accumulate on the matrix unit too where the m16n8 binary16
		// instruction lowers (f16GemmK): the FMA accumulate was the larger
		// half of a 512-row prompt's attention on an sm_86 card.
		if mat && hd%16 == 0 && rows%16 == 0 && !p.MLA() && g.f16GemmK() > 0 && rows%(8*accMMANT) == 0 {
			modes = append(modes, prefillMMAAcc)
		}
		if mat && hd%16 == 0 && rows%16 == 0 && !p.MLA() {
			modes = append(modes, prefillMMA)
		}
		if mat && !g.NoVolta && hd%32 == 0 && rows%(8*volta70AttnNT) == 0 && g.sm70() {
			modes = append(modes, prefillMMA70)
		}
		modes = append(modes, prefillTiled)
		pw, pc := g.prefillPass()
		// A history that cannot pass the pass width never takes the pass
		// form, and the form's planes are rows x heads x the pass width: 537
		// MB at 512 rows, where a 716-position context's
		// widest span needs a quarter of it. So a short capacity builds the
		// widest single pass it can reach (prefillPrep's own choice at that
		// span) and no pass form; a capacity grown past the pass width is a
		// new scratch, which builds it.
		short := 0
		if p.MaxSeq > 0 && p.MaxSeq <= pw {
			short = 64
			for short < p.MaxSeq {
				short *= 2
			}
		}
		var v *prefillVariant
		for _, m := range modes {
			pf.mode, pf.qt, pf.sq = m, 8, g.tiles.qtile
			switch m {
			case prefillMMA:
				pf.sq = 16
			case prefillMMAAcc:
				pf.qt, pf.sq = 8*accMMANT, 16
			case prefillMMA70:
				pf.qt, pf.sq = 8*volta70AttnNT, 8*volta70AttnNT
			}
			if rows%pf.qt != 0 || rows%pf.sq != 0 {
				continue
			}
			var err error
			if short > 0 {
				v, err = g.prefillVariantAt(bs, pf, short, 0)
			} else {
				v, err = g.prefillVariantAt(bs, pf, pc, pw/pc)
			}
			if err == nil {
				break
			}
		}
		if v == nil {
			pf.free()
			return nil // no staged form on this device: FlashDecodeKV serves
		}
		ms := mergeShape(kernels.FlashShape{Heads: p.NHead, KVHeads: pg.shape.KVHeads, Dim: hd, MLA: pg.shape.MLA,
			Rows: rows, Splits: max(1, v.s.Splits)})
		if short > 0 {
			// The softmax binds the partials in every form, so they exist at
			// one pass's width.
			if err := g.prefillBuf(&pf.part, &pf.partFloats, kernels.FlashPartialFloats(ms)); err != nil {
				pf.free()
				return err
			}
		} else {
			pf.passV = v
			one := ms
			one.Splits, one.Sink = 1, p.AttnSinks
			pf.mFirst = comp(kernels.FlashAttentionMergeRun(ms, true))
			pf.mFold = comp(kernels.FlashAttentionMergeRun(ms, false))
			pf.mFin = comp(kernels.FlashAttentionMergeWide(one))
			if pf.mFirst == nil || pf.mFold == nil || pf.mFin == nil {
				pf.free()
				return nil
			}
			if err := g.prefillBuf(&pf.part, &pf.partFloats, kernels.FlashPartialFloats(ms)); err != nil {
				pf.free()
				return err
			}
			for i := range pf.run {
				var have int
				if err := g.prefillBuf(&pf.run[i], &have, kernels.FlashPartialFloats(one)); err != nil {
					pf.free()
					return err
				}
			}
		}
	}
	for i := range pf.buf {
		if i == 1 && p.SWAWindow <= 0 {
			break
		}
		b, err := g.dev.Alloc(rows * kernels.PRowWords * 4)
		if err != nil {
			pf.free()
			return err
		}
		pf.buf[i] = b
	}
	pg.pf = pf
	return nil
}

// prefillBuf makes *dst hold at least n floats, poisoned when the tier
// poisons its scratch. A buffer it replaces is freed and the recording
// dropped, since one may name it.
func (g *devTier) prefillBuf(dst *backend.Buf, have *int, n int) error {
	defer g.scratchWin().close() // the scratch's own buffers (scratch.go)
	if *dst != nil && *have >= n {
		return nil
	}
	b, err := g.dev.Alloc(n * 4)
	if err != nil {
		return err
	}
	if g.kb.poisonScratch {
		b.Write(nanFill(n * 4))
	}
	if *dst != nil {
		// The buffer is this lane's: only its recordings name it.
		g.dropLaneGraph()
		(*dst).Free()
	}
	*dst, *have = b, n
	return nil
}

// prefillVariantAt compiles the staged kernels at chunk keys a split and
// splits splits (0: the direct form), grows the planes to hold them, and
// caches the variant under its span.
func (g *devTier) prefillVariantAt(bs *blockScratch, pf *pagedPrefill, chunk, splits int) (*prefillVariant, error) {
	key := chunk
	if splits > 0 {
		key = -chunk * splits
	}
	if v, ok := pf.vars[key]; ok {
		return v, nil
	}
	p, pg := &bs.p, bs.pkv
	s := pg.shape
	s.Rows, s.Chunk, s.Splits = bs.rows, chunk, splits
	s.Group, s.Warps, s.KImm = 0, 0, false
	s.Sink = p.AttnSinks && splits == 0
	v, err := g.buildVariant(pf, s)
	if err != nil {
		return nil, fmt.Errorf("paged prefill at %d keys: %w", chunk*max(1, splits), err)
	}
	pf.vars[key] = v
	return v, nil
}

// buildVariant compiles the staged kernels at s on the set's matrix
// instruction (pf.mode) and grows the planes to hold them.
func (g *devTier) buildVariant(pf *pagedPrefill, s kernels.FlashShape) (*prefillVariant, error) {
	v := &prefillVariant{s: s}
	var ks, kf, ka *ir.Kernel
	var err error
	switch pf.mode {
	case prefillMMA, prefillMMAAcc:
		ks, err = kernels.PagedAttnScoresMMA(s, g.tiles.attnNT)
		v.scoreT = kernels.PagedAttnScoresMMAWarps(s, g.tiles.attnNT) * 32
	case prefillMMA70:
		ks, err = kernels.PagedAttnScoresMMA70(s, volta70AttnMT, volta70AttnNT)
		v.scoreT = kernels.PagedAttnScoresMMA70Warps(s, volta70AttnMT, volta70AttnNT) * 32
	default:
		ks, err = kernels.PagedAttnScoresTiled(s, pf.sq, pf.kt)
		v.scoreT = kernels.PagedAttnScoresTiledThreads(s, pf.sq, pf.kt)
	}
	if err == nil {
		kf, err = kernels.PagedPrefillSoftmax(s, pf.lanes, pf.qt)
	}
	if err == nil {
		switch pf.mode {
		case prefillMMA70:
			ka, err = kernels.PagedAttnAccMMA70(s, volta70AccMT, volta70AttnNT)
			v.accT = kernels.PagedAttnAccMMA70Warps(s, volta70AccMT, volta70AttnNT) * 32
		case prefillMMAAcc:
			mt := 4
			for mt > 1 && s.Dim%(16*mt) != 0 {
				mt /= 2
			}
			ka, err = kernels.PagedAttnAccMMA(s, mt, accMMANT)
			v.accT = kernels.PagedAttnAccMMAWarps(s, mt, accMMANT) * 32
		default:
			ka, err = kernels.PagedAttnAccTiled(s, pf.qt)
			v.accT = kernels.PagedAttnAccTiledThreads(s, pf.qt)
		}
	}
	if err != nil {
		return nil, err
	}
	v.softG, v.softW = kernels.PagedPrefillSoftmaxGroups(s, pf.lanes), kernels.PagedPrefillSoftmaxWidth(pf.lanes)
	cs := []*backend.Kernel{&v.scores, &v.soft, &v.acc}
	for i, k := range []*ir.Kernel{ks, kf, ka} {
		c, err := g.compileShared(k)
		if err != nil {
			v.close()
			return nil, err
		}
		*cs[i] = c
	}
	n := kernels.PagedPrefillPlane(s)
	have := pf.planeFloats
	if err := g.prefillBuf(&pf.sc, &have, n); err != nil {
		v.close()
		return nil, err
	}
	have = pf.planeFloats
	if err := g.prefillBuf(&pf.wt, &have, n); err != nil {
		v.close()
		return nil, err
	}
	pf.planeFloats = max(pf.planeFloats, n)
	return v, nil
}

func (v *prefillVariant) close() {
	for _, k := range []backend.Kernel{v.scores, v.soft, v.acc} {
		if k != nil {
			closeKern(k)
		}
	}
}

// free releases a prefill set.
func (pf *pagedPrefill) free() {
	if pf == nil {
		return
	}
	for _, v := range pf.vars {
		v.close()
	}
	for _, k := range []backend.Kernel{pf.flash, pf.fmerge, pf.mFirst, pf.mFold, pf.mFin} {
		if k != nil {
			closeKern(k)
		}
	}
	for _, k := range pf.ragK {
		closeKern(k)
	}
	for _, k := range pf.ragMerge {
		closeKern(k)
	}
	for _, v := range pf.ragVars {
		v.close()
	}
	for _, k := range pf.ragFin {
		closeKern(k)
	}
	bufs := []backend.Buf{pf.sc, pf.wt, pf.part, pf.run[0], pf.run[1], pf.buf[0], pf.buf[1]}
	bufs = append(append(bufs, pf.pass[0]...), pf.pass[1]...)
	for _, b := range bufs {
		if b != nil {
			b.Free()
		}
	}
}

// prefillPrep stages a chunk's attention: nrow real rows of one sequence (the
// writers' descriptors in pg.desc/descW), the rest padding. Each descriptor set
// gets its variant and its passes, compiled and allocated here, before the
// submission. Callers hold g.mu and have run pagedPrep and pagedStage.
func (g *devTier) prefillPrep(bs *blockScratch, nrow int) error {
	defer g.scratchWin().close() // the scratch's own buffers (scratch.go)
	pg := bs.pkv
	pf := pg.pf
	for i, src := range [2][]uint32{pg.desc, pg.descW} {
		pl := &pf.plan[i]
		pl.desc, pl.passes = append(pl.desc[:0], src...), 0
		if len(src) == 0 {
			continue
		}
		// A padded row is empty at the last real row's keyEnd, in its table.
		last := (nrow - 1) * kernels.PRowWords
		end := src[last+kernels.PRowEnd]
		for r := nrow; r < len(src)/kernels.PRowWords; r++ {
			o := r * kernels.PRowWords
			pl.desc[o+kernels.PRowTab], pl.desc[o+kernels.PRowStart], pl.desc[o+kernels.PRowEnd] =
				src[kernels.PRowTab], end, end
		}
		if pf.flash != nil {
			continue
		}
		base := int(src[kernels.PRowStart]) &^ 63
		span := int(end) - base
		pw, _ := g.prefillPass()
		if span <= pw {
			c := 64
			for c < span {
				c *= 2
			}
			v, err := g.prefillVariantAt(bs, pf, c, 0)
			if err != nil {
				return err
			}
			pl.v = v
			continue
		}
		if pf.passV == nil {
			// The scratch was built for a capacity under the pass width
			// (initPagedPrefill), and a span past it means the capacity grew
			// without a rebuild: refused, so the chunk takes the decode path.
			return fmt.Errorf("a %d-key span past the %d-key pass width on a scratch built without the pass form", span, pw)
		}
		pl.v, pl.passes = pf.passV, (span+pw-1)/pw
		n := len(pl.desc)
		pl.passD = pl.passD[:0]
		for j := 0; j < pl.passes; j++ {
			lo, hi := uint32(base+j*pw), uint32(base+(j+1)*pw)
			at := len(pl.passD)
			pl.passD = append(pl.passD, pl.desc...)
			for o := at; o < at+n && !pagedFaulted("passclip"); o += kernels.PRowWords {
				for _, f := range []int{kernels.PRowStart, kernels.PRowEnd} {
					pl.passD[o+f] = min(max(pl.passD[o+f], lo), hi)
				}
			}
		}
		for len(pf.pass[i]) < pl.passes {
			b, err := g.dev.Alloc(n * 4)
			if err != nil {
				return err
			}
			pf.pass[i] = append(pf.pass[i], b)
		}
	}
	return nil
}

// prefillWrites uploads the staged descriptors: w is the call's staging write.
func (pf *pagedPrefill) prefillWrites(w func(backend.Buf, []byte)) {
	for i := range pf.plan {
		pl := &pf.plan[i]
		if len(pl.desc) == 0 {
			continue
		}
		w(pf.buf[i], u32view(pl.desc))
		n := len(pl.desc)
		for j := 0; j < pl.passes; j++ {
			w(pf.pass[i][j], u32view(pl.passD[j*n:(j+1)*n]))
		}
	}
}

// on70 reports that the staged form runs on sm_70's m8n8k4 pair.
func (pf *pagedPrefill) on70() bool { return pf.flash == nil && pf.mode == prefillMMA70 }

// on80 reports that the prompt's attention is kernels.FlashPrefill80.
func (pf *pagedPrefill) on80() bool { return pf.flash != nil && pf.flash80 }

// accOnMMA reports that the staged form's accumulate is PagedAttnAccMMA.
func (pf *pagedPrefill) accOnMMA() bool { return pf.flash == nil && pf.mode == prefillMMAAcc }

// passes is how many passes the current call's descriptor set win runs, 0
// for one launch.
func (pf *pagedPrefill) passes(win bool) int {
	if win {
		return pf.plan[1].passes
	}
	return pf.plan[0].passes
}

// prefillAttn launches one layer's chunk attention: q, the layer's K and V
// pools and table, into out; win selects the windowed descriptors and sink is
// the layer's sinks (nil without). la launches at a thread count in groups of
// 128, s the raw launch.
func (pf *pagedPrefill) prefillAttn(s backend.Session, lc *launcher,
	q, k, v, n, out, tab backend.Buf, win bool, sink backend.Buf) error {
	i := 0
	if win {
		i = 1
	}
	pl := &pf.plan[i]
	if pf.flash != nil {
		if pf.fmerge == nil {
			return lc.launch(pf.flash, pf.fGroups, pf.fWidth, q, k, v, n, out, tab, pf.buf[i])
		}
		if err := lc.launch(pf.flash, pf.fGroups, pf.fWidth, q, k, v, n, pf.part, tab, pf.buf[i]); err != nil {
			return err
		}
		lc.la(pf.fmerge, pf.fRows, pf.part, out, sink)
		return nil
	}
	vr := pl.v
	sh := vr.s
	one := func(desc, dst backend.Buf) error {
		lc.la(vr.scores, vr.scoreT, q, k, n, pf.sc, tab, desc)
		args := []backend.Buf{pf.sc, n, pf.wt, pf.part, tab, desc}
		if sh.Sink {
			args = append(args, sink)
		}
		w := vr.softW
		if pagedFaulted("softwidth") {
			w = 64
		}
		if err := lc.launch(vr.soft, vr.softG, w, args...); err != nil {
			return err
		}
		lc.la(vr.acc, vr.accT, pf.wt, v, n, dst, tab, desc)
		return nil
	}
	if pl.passes == 0 {
		return one(pf.buf[i], out)
	}
	vd := mergeShape(sh).Dim
	rg := sh.Rows * sh.Heads * ((vd + 31) / 32 * 32)
	for j := 0; j < pl.passes; j++ {
		if err := one(pf.pass[i][j], pf.part); err != nil {
			return err
		}
		if j == 0 {
			lc.la(pf.mFirst, rg, pf.part, pf.run[0])
		} else {
			lc.la(pf.mFold, rg, pf.part, pf.run[(j+1)%2], pf.run[j%2])
		}
	}
	fin := []backend.Buf{pf.run[(pl.passes+1)%2], out}
	if sink != nil {
		fin = append(fin, sink)
	}
	lc.la(pf.mFin, sh.Rows*sh.Heads*vd, fin...)
	return nil
}
