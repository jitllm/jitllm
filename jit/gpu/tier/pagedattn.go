package tier

import (
	"fmt"
	"math"
	"slices"
	"unsafe"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// Paged attention history on the device (docs/design/device-kv-paging.md).
//
// A device holds one kvPool: per attention layer its own pages (K and V
// buffers, ids, free list) and a persistent table arena, a sequence's run at
// the same offset in every layer's arena. A sequence is a session and the
// base of its rows (seqID), so a LayersRows batch's sequences and a single
// State are the same thing here. A session holds pages only in the layers it
// runs here, so leaving a layer frees that layer's pages. Nothing a kernel is
// built from depends on how long a history is: the rows' descriptors are data
// written before each submission (the only per-token upload), the tables take
// a new id at a page boundary, and growth appends pages. That is what lets the
// capacity machinery (sesscap.go) go: every session shares one scratch.
//
// Decode and a ragged batch read through the decode plan the device and the
// history choose (kernels.ChooseDecodePlan: FlashDecodeKV or the staged
// kernels), one row per descriptor; a prefill chunk through the prefill
// kernels (pagedprefill.go). A sliding window is only the descriptor's
// keyStart, so a windowed layer reads a second descriptor set rather than a
// second kernel.

// maxKVPage is the largest page the pool picks by itself: the size the paged
// kernels were tuned against the contiguous ones.
const maxKVPage = 256

// pagedScratch is one scratch's paged attention: the writers, the attention
// variants by decode plan, and the per-call descriptor buffers.
type pagedScratch struct {
	// copyK is set where K reaches the pool unrotated or partly rotated (the
	// rope writes only the dimensions it rotates); ropeK where it rotates.
	copyK, ropeK, copyV backend.Kernel
	// ropeQKV is the three decode writers as one launch
	// (kernels.RoPEQKVPaged): q's rotation, k's into the pool, v's copy.
	// nil where the block's rotary is not the plain one it fuses (partial,
	// split, MSA's extra key head) or Config.NoFuse is set.
	ropeQKV backend.Kernel
	// copyLat and copyPE are MLA's writers: the normed latent at the row's
	// head and the rotated key after it, the rotation done into kpeRot at
	// rotOff's per-row offsets first (the rotary kernels write no pages).
	copyLat, copyPE backend.Kernel
	kpeRot, rotOff  backend.Buf
	shape           kernels.FlashShape // the plan's input: Splits, Chunk unset
	vars            map[kernels.DecodePlan]*pagedVariant
	// part holds a split variant's partials, sc and wt the staged path's
	// score and weight planes, each sized for the widest variant built.
	part, sc, wt            backend.Buf
	partFloats, planeFloats int
	// pRow/pRowW are the rows' descriptors, the second with each row's
	// window start; every layer reads them, each with its own table arena.
	pRow, pRowW backend.Buf
	// rows, desc and descW are the per-call staging, reused so a token
	// allocates nothing.
	rows        []pagedRow
	desc, descW []uint32
	// cur is the variant the current call runs, chosen before the submission
	// (pagedPrep) and read by layersOnce; nil when this session holds no
	// attention block here.
	cur *pagedVariant
	// pf is a batched scratch's prefill attention (pagedprefill.go); nil
	// runs a chunk through FlashDecodeKV.
	pf *pagedPrefill
	// prefill says the current call is a prefill chunk that pf serves.
	prefill bool
	// The ragged call's run table (raggedprefill.go): ragRows slots, 0 when
	// the decode plan takes every row; ragMap and ragOrder its host staging,
	// pMap its device copy, and pRowDec/pRowDecW the decode descriptors
	// with the table's rows emptied (descDec/descDecW staged).
	ragRows int
	// ragAlone says the table holds every row of the call (ragTable), so the
	// decode plan does not run.
	ragAlone bool
	// ragStaged says the table runs the staged form (raggedprefill.go), and
	// ragSt is its full and windowed descriptor sets' bases and variants.
	ragStaged                              bool
	ragSt                                  [2]ragStaged
	ragMap                                 []uint32
	ragOrder                               []int
	pMap, pRowDec, pRowDecW                backend.Buf
	pMapBytes, pRowDecBytes, pRowDecWBytes int
	descDec, descDecW                      []uint32
	// st is the current call's pass over evicted history (pagedstream.go), nil
	// when every page it reads is resident.
	st *pagedStream
	// idx is DeepSeek V3.2's lightning indexer (indexer.go), nil without one.
	idx *idxScratch
	// msa is MiniMax-M3's block selection (msa.go), nil without one.
	msa *msaScratch
	// qsa is a QSA block's selection (qsa.go), nil elsewhere.
	qsa *qsaScratch
	// stream is what st points at when set, and streamD each descriptor set's
	// words for every pass, written with the call's staged writes
	// (pagedStream.writes): both reused call to call, so a decode token that
	// streams allocates nothing once the widest call has been made.
	stream  pagedStream
	streamD [2][]uint32
	// streams are the pass sets built so far, by pass width in pages.
	streams map[int]*streamSet
	// streamDesc are the passes' descriptor buffers, full [0] and windowed
	// [1], grown to the most passes a call has run.
	streamDesc [2][]backend.Buf
}

// pagedVariant is one decode plan's attention (kernels.ChooseDecodePlan):
// FlashDecodeKV, or the staged scores, softmax and accumulate, and the wide
// merge of their partials where there are any.
type pagedVariant struct {
	plan                        kernels.DecodePlan
	k, scores, soft, acc, merge backend.Kernel
	groups, width               int // FlashDecodeKV's launch
	scoreT, softG, accT, mergeT int // the staged launches, and the merge's threads
	mergeItems                  int // > 0: the group-per-item merge, 32 lanes an item (DecodePlanOff)
}

func (v *pagedVariant) close() {
	for _, k := range []backend.Kernel{v.k, v.scores, v.soft, v.acc, v.merge} {
		if k != nil {
			closeKern(k)
		}
	}
}

// kvPage is the pool's page size, decided when the pool is made and fixed for
// its life: Config.KVPage, or the smallest power of two from 64 that covers
// the context, up to maxKVPage -- a page longer than the context would hold
// history no sequence can reach.
func (g *devTier) kvPage() int {
	if g.kvp != nil {
		return g.kvp.p
	}
	if g.KVPage > 0 {
		return g.KVPage
	}
	p := 64
	for p < maxKVPage && p < g.kvCap {
		p *= 2
	}
	return p
}

// pagedPlan reports that a plan's attention history is paged on this device:
// every text attention, MLA's latent included, but not a tower's, whose keys
// live for one encode and are not history.
func (g *devTier) pagedPlan(p *nn.LayerPlan) bool {
	return g.paged && !p.NonCausal
}

// kvGeomOf is a plan's page geometry. MLA's is one row-major latent row a
// position, float32 (the paged kernels read no packed latent), and no V
// buffer: the value is the row's prefix.
func (g *devTier) kvGeomOf(p *nn.LayerPlan) kvGeom {
	// DeepSeek V4's row is one latent head that is also the value, as MLA's.
	latent := p.MLA() || p.DS4 != nil
	return kvGeom{kvRow: p.KVRow(), f16: g.KVF16 && !latent, mla: latent}
}

// kvPages is the device's KV pool, made on first use. Callers hold g.mu.
func (g *devTier) kvPages() *kvPool {
	if g.kvp == nil {
		g.kvp = newKVPool(g.kvPage(), g.maxBuffer())
	}
	return g.kvp
}

// initPagedScratch builds bs's paged set. Callers hold g.mu.
func (g *devTier) initPagedScratch(bs *blockScratch) error {
	defer g.scratchWin().close() // the scratch's own buffers (scratch.go)
	p := &bs.p
	P, rows, kvDim := g.kvPage(), bs.rows, p.KVRow()
	pg := &pagedScratch{vars: map[kernels.DecodePlan]*pagedVariant{}}
	bs.pkv = pg
	comp := func(dst *backend.Kernel, k *ir.Kernel, err error) error {
		if err != nil {
			return err
		}
		c, err := g.compileShared(k)
		if err != nil {
			return err
		}
		*dst = c
		return nil
	}
	if p.MLA() {
		return g.initPagedMLA(bs, comp)
	}
	if p.DS4 != nil {
		return g.initPagedDS4(bs)
	}
	// A QSA row is copied whole, its raw key included, before k's heads turn.
	if partialRotary(p) || p.NoPEGlobal || p.NoPosEnc || p.NRot == 0 || p.RopeSplit || p.QSA() {
		k, err := kernels.PagedCopyRowsT(kvDim, rows, P)
		if err := comp(&pg.copyK, k, err); err != nil {
			return err
		}
	}
	// A selecting block's row is k's heads and the indexer's key, written
	// together and turned together (msa.go); its v is k's width, the row's
	// last head unwritten.
	kHeads, vw := p.NKVHead, p.NKVHead*p.HeadDim
	if p.MSA() {
		kHeads++
	}
	// XD-RoPE's k reaches the pages through copyK, turned and normed first.
	if p.NRot > 0 && !p.NoPosEnc && !p.RopeSplit {
		k, err := kernels.PagedRoPERowsT(kHeads, p.HeadDim, p.NRot, p.RopeNeox, rows, P)
		if p.QSA() {
			// k's heads turn in a row that carries the raw key after them.
			k, err = kernels.PagedRoPERowsTIn(kHeads, p.HeadDim, kvDim, p.NRot, p.RopeNeox, rows, P)
		}
		if err := comp(&pg.ropeK, k, err); err != nil {
			return err
		}
	}
	k, err := kernels.PagedCopyRows(vw, rows, P, kvDim, 0, g.KVF16)
	if err := comp(&pg.copyV, k, err); err != nil {
		return err
	}
	if pg.ropeK != nil && pg.copyK == nil && !p.MSA() && !p.QSA() && vw == kvDim && !p.QKNormPost && !g.NoFuse {
		// Built beside the three it fuses, which stay: the per-op path is
		// the reference and the A/B arm, and a layer the fused form does not
		// fit (an unrotated one, a KV-sharing one) still takes them.
		k, err := kernels.RoPEQKVPaged(p.NHead, p.NKVHead, p.HeadDim, p.NRot, p.RopeNeox, rows, P, g.KVF16)
		if err := comp(&pg.ropeQKV, k, err); err != nil {
			return err
		}
	}
	s := kernels.FlashShape{Heads: p.NHead, KVHeads: p.NKVHead, Dim: p.HeadDim, Rows: rows,
		Scale: g.scoreScale(p), Softcap: p.AttnSoftcap, F16: g.KVF16, Sink: p.AttnSinks,
		Page: P, KImm: g.dev.API() == "msl", Warps: g.FlashWarps}
	if gqa := p.NHead / p.NKVHead; g.FlashGroup > 0 && gqa%g.FlashGroup == 0 {
		s.Group = g.FlashGroup
	}
	// Vector V loads lower on PTX only (kernels.FlashShape.VecV).
	s.VecV = g.dev.API() == "ptx"
	pg.shape = s
	al := func(dst *backend.Buf, n int) error {
		b, err := g.dev.Alloc(n)
		if err != nil {
			return err
		}
		*dst = b
		return nil
	}
	if err := al(&pg.pRow, rows*kernels.PRowWords*4); err != nil {
		return err
	}
	if p.SWAWindow > 0 {
		if err := al(&pg.pRowW, rows*kernels.PRowWords*4); err != nil {
			return err
		}
	}
	// A selecting block attends over gathered pages, whose descriptors the
	// prefill kernels do not read: a chunk takes the decode plan row by row.
	if p.MSA() {
		if err := g.initMSA(bs, rows, P, comp); err != nil {
			return err
		}
	}
	if p.QSA() {
		if err := g.initQSA(bs, rows, P, comp); err != nil {
			return err
		}
	}
	if _, err := g.pagedPlanFor(bs, scoreGrain); err != nil {
		return err
	}
	if rows > 1 && !p.MSA() && !p.QSA() {
		return g.initPagedPrefill(bs)
	}
	return nil
}

// initPagedMLA builds bs's paged set for a latent block: the latent row's two
// writers, the rotation's staging, and the attention's shape -- one KV head
// whose row is the whole latent row and whose value is its first KVLoraRank
// floats, which kernels.ChooseDecodePlan sends to the staged kernels.
// Callers hold g.mu.
func (g *devTier) initPagedMLA(bs *blockScratch, comp func(*backend.Kernel, *ir.Kernel, error) error) error {
	defer g.scratchWin().close() // the scratch's own buffers (scratch.go)
	p, pg := &bs.p, bs.pkv
	P, rows, row, lat, rot := g.kvPage(), bs.rows, p.KVRow(), p.KVLoraRank, p.NRot
	k, err := kernels.PagedCopyRows(lat, rows, P, row, 0, false)
	if err := comp(&pg.copyLat, k, err); err != nil {
		return err
	}
	k, err = kernels.PagedCopyRows(rot, rows, P, row, lat, false)
	if err := comp(&pg.copyPE, k, err); err != nil {
		return err
	}
	al := func(dst *backend.Buf, n int) {
		if err == nil {
			*dst, err = g.dev.Alloc(n)
		}
	}
	al(&pg.kpeRot, rows*rot*4)
	al(&pg.rotOff, rows*4)
	al(&pg.pRow, rows*kernels.PRowWords*4)
	if err != nil {
		return err
	}
	off := make([]uint32, rows)
	for r := range off {
		off[r] = uint32(r * rot)
	}
	if err := pg.rotOff.Write(u32view(off)); err != nil {
		return err
	}
	// The attention's row is the latent and its key. With the lightning
	// indexer the pool's row also carries the indexer's key, and the
	// attention reads the gathered rows (IdxGather), which are exactly that
	// wide; there is no prefill kernel over them, so a chunk takes the
	// decode plan row by row.
	w := lat + rot
	if p.IdxHeads != 0 {
		if err := g.initIdx(bs, rows, P, row, w, comp); err != nil {
			return err
		}
	}
	pg.shape = kernels.FlashShape{Heads: p.NHead, KVHeads: 1, Dim: w, MLA: lat, Rows: rows,
		Scale: g.scoreScale(p), Page: P, MLAValueTail: g.kb.mlaFault == MLAFaultValueHeadMajor}
	if _, err := g.pagedPlanFor(bs, scoreGrain); err != nil {
		return err
	}
	if rows > 1 && p.IdxHeads == 0 {
		return g.initPagedPrefill(bs)
	}
	return nil
}

// initPagedDS4 builds bs's paged set for DeepSeek V4: the rows' descriptors
// and the attention's shape over the gathered keys (ds4.go) -- one head
// hd wide that is also the value, with the heads' sinks, which
// kernels.ChooseDecodePlan sends to the staged kernels. There is no prefill
// kernel over gathered keys, so a chunk takes the decode plan row by row in
// one launch. Callers hold g.mu.
func (g *devTier) initPagedDS4(bs *blockScratch) error {
	p, pg := &bs.p, bs.pkv
	b, err := g.dev.Alloc(bs.rows * kernels.PRowWords * 4)
	if err != nil {
		return err
	}
	pg.pRow = b
	// The windowed descriptors are staged for every windowed plan
	// (pagedStage); the gathers read the window from the full ones.
	if p.SWAWindow > 0 {
		if pg.pRowW, err = g.dev.Alloc(bs.rows * kernels.PRowWords * 4); err != nil {
			return err
		}
	}
	pg.shape = kernels.FlashShape{Heads: p.NHead, KVHeads: 1, Dim: p.HeadDim, MLA: p.HeadDim, Rows: bs.rows,
		Scale: g.scoreScale(p), Page: g.kvPage(), Sink: true}
	_, err = g.pagedPlanFor(bs, scoreGrain)
	return err
}

// freePaged releases a scratch's paged set.
func freePaged(pg *pagedScratch) {
	if pg == nil {
		return
	}
	for _, k := range []backend.Kernel{pg.copyK, pg.ropeK, pg.copyV, pg.copyLat, pg.copyPE, pg.ropeQKV} {
		if k != nil {
			closeKern(k)
		}
	}
	for _, v := range pg.vars {
		v.close()
	}
	pg.pf.free()
	pg.msa.free()
	pg.qsa.free()
	pg.idx.free()
	for _, ss := range pg.streams {
		ss.close()
	}
	for _, bufs := range pg.streamDesc {
		for _, b := range bufs {
			b.Free()
		}
	}
	pg.ragFree()
	for _, b := range []backend.Buf{pg.part, pg.sc, pg.wt, pg.pRow, pg.pRowW, pg.kpeRot, pg.rotOff} {
		if b != nil {
			b.Free()
		}
	}
}

// pagedPlanFor makes sure bs has the variant for a call whose rows attend up
// to keys keys, and returns it: kernels.ChooseDecodePlan's path, splits and
// width for this device, or FlashDecodeKV at Config.FlashSplit when that is
// forced (the order-only controls pin it), or the staged path under
// Config.StagedDecode. The kernels are generated for the plan and kept, so a
// new plan is a compile and a stable one is a lookup; the partials and planes
// grow to the widest plan, dropping the captured graph that names the old
// buffers. It compiles and allocates, so it runs outside any submission.
// Callers hold g.mu.
func (g *devTier) pagedPlanFor(bs *blockScratch, keys int) (*pagedVariant, error) {
	pg := bs.pkv
	var plan kernels.DecodePlan
	if g.DecodePlanOff != 0 {
		// The plan with the pieces DecodePlanOff names put back as they were:
		// 1 the split count, 2 the merge (below), 4 the vector V loads.
		s := pg.shape
		p := kernels.ChooseDecodePlan(g.dev.API(), s, keys, g.dev.Slots())
		s = p.Shape
		if p.Path != kernels.PathFlashKV {
			s = pg.shape
		}
		if g.DecodePlanOff&4 != 0 {
			s.VecV = false
		}
		if g.DecodePlanOff&1 != 0 {
			s.Warps, s.Splits = pg.shape.Warps, 1
			c := kernels.FlashKVChunk(s)
			slots := g.dev.Slots()
			if slots <= 0 {
				slots = 16384
			}
			sp := 1 << int(math.Round(math.Log2(math.Sqrt(max(1, float64(keys)/8)))))
			sp = max(1, min(sp, max(1, slots/1280/max(1, kernels.FlashKVGroups(s)))))
			s.Splits = max(sp, (keys+c-1)/c)
		}
		plan = kernels.DecodePlan{Path: kernels.PathFlashKV, Shape: s}
	} else if g.StagedDecode {
		plan = kernels.DecodePlan{Path: kernels.PathStaged, Shape: kernels.PagedStagedPlan(pg.shape, keys, g.dev.Slots())}
	} else if f := g.FlashSplit; f > 0 {
		s := pg.shape
		s.Splits = 1
		c := kernels.FlashKVChunk(s)
		s.Splits = max(f, (keys+c-1)/c)
		plan = kernels.DecodePlan{Path: kernels.PathFlashKV, Shape: s}
	} else {
		plan = kernels.ChooseDecodePlan(g.dev.API(), pg.shape, keys, g.dev.Slots())
	}
	// FlashDecodeKV reduces across a warp and needs the device to promise 32
	// lanes; the staged kernels have one-lane forms. A device that cannot
	// promise the width (Vulkan unpinned on an 8..32 part, a software
	// rasteriser) runs the staged path whatever the plan chose, rather than
	// declining the block (RULE 8).
	if plan.Path == kernels.PathFlashKV && g.pickLanes() != ir.SubgroupLanes {
		plan = kernels.DecodePlan{Path: kernels.PathStaged, Shape: kernels.PagedStagedPlan(pg.shape, keys, g.dev.Slots())}
	}
	// A staged plan's planes grow with its keys: past the pass width it is
	// built for the pass, and a deeper call goes in passes (pagedStreamPrep).
	if pk := g.stagedPassKeys(bs); plan.Path == kernels.PathStaged && pk > 0 && keys > pk {
		keys = pk
		plan.Shape = kernels.PagedStagedPlan(pg.shape, keys, g.dev.Slots())
	}
	v, err := g.pagedBuild(bs, plan)
	// The split count only divides the work, and the partials are rows x
	// heads x splits x the value's width: a 530-row prompt chunk at 8 splits
	// wanted ~280 MB of them on a card its weights fill, and the whole prompt
	// went a row at a time. A staged plan that does not fit halves its splits
	// rather than refusing; it retries on any build error (no backend reports
	// out-of-memory but CUDA), at most log2(splits) extra builds.
	for err != nil && plan.Path == kernels.PathStaged && plan.Shape.Splits > 1 {
		plan.Shape = kernels.PagedStagedAt(plan.Shape, keys, plan.Shape.Splits/2)
		v, err = g.pagedBuild(bs, plan)
	}
	return v, err
}

// pagedBuild is bs's variant for plan: a lookup once built, else the kernels
// compiled and the partials and planes grown to it. Callers hold g.mu, outside
// any submission.
func (g *devTier) pagedBuild(bs *blockScratch, plan kernels.DecodePlan) (*pagedVariant, error) {
	pg := bs.pkv
	if v, ok := pg.vars[plan]; ok {
		return v, nil
	}
	s := plan.Shape
	v := &pagedVariant{plan: plan}
	fail := func(err error) (*pagedVariant, error) {
		v.close()
		return nil, fmt.Errorf("paged attention (%s, %d split(s)): %w", plan.Path, s.Splits, err)
	}
	comp := func(dst *backend.Kernel, k *ir.Kernel, err error) error {
		if err == nil {
			*dst, err = g.compileShared(k)
		}
		return err
	}
	vd := s.Dim
	if s.MLA > 0 {
		vd = s.MLA
	}
	v.mergeT = s.Rows * s.Heads * vd
	merged := s.Splits > 1
	if plan.Path == kernels.PathStaged {
		merged = true
		lanes := g.pickLanes()
		if lanes != ir.SubgroupLanes {
			lanes = 1
		}
		// A row-major MLA row is hundreds of floats a key: a warp across its
		// dimensions: one thread walking all of them was several times slower
		// than a warp-wide score. Head rows of 64-256 keep a thread a slot.
		scoreLanes := 1
		if s.MLA > 0 && lanes == ir.SubgroupLanes {
			scoreLanes = ir.SubgroupLanes
		}
		k, err := kernels.PagedStagedScores(s, scoreLanes)
		if err = comp(&v.scores, k, err); err == nil {
			k, err = kernels.PagedStagedSoftmax(s, lanes)
			err = comp(&v.soft, k, err)
		}
		if err == nil {
			k, err = kernels.PagedStagedAcc(s)
			err = comp(&v.acc, k, err)
		}
		if err != nil {
			return fail(err)
		}
		items := s.Rows * s.Heads * s.Splits
		v.scoreT, v.accT, v.softG = kernels.PagedStagedScoreItems(s)*scoreLanes, kernels.PagedStagedAccThreads(s), (items+63)/64
		if lanes == ir.SubgroupLanes {
			v.softG = (items + 1) / 2
		}
		n, have := kernels.PagedStagedPlane(s), pg.planeFloats
		if err := g.prefillBuf(&pg.sc, &have, n); err != nil {
			return fail(err)
		}
		have = pg.planeFloats
		if err := g.prefillBuf(&pg.wt, &have, n); err != nil {
			return fail(err)
		}
		pg.planeFloats = max(pg.planeFloats, n)
	} else {
		k, err := kernels.FlashDecodeKV(s)
		if err := comp(&v.k, k, err); err != nil {
			return fail(err)
		}
		v.groups, v.width = kernels.FlashKVGroups(s), kernels.FlashKVWidth(s)
	}
	// The merge and its partials run at the value's width: a head, or MLA's
	// latent (the row's prefix), not the whole row the scores read.
	ms := mergeShape(s)
	if merged && g.DecodePlanOff&2 != 0 {
		// The group-per-item merge: a 32-lane group per (row, head).
		k, err := kernels.FlashAttentionMerge(ms)
		if err := comp(&v.merge, k, err); err != nil {
			return fail(err)
		}
		v.mergeItems = s.Rows * s.Heads
		if err := g.prefillBuf(&pg.part, &pg.partFloats, kernels.FlashPartialFloats(ms)); err != nil {
			return fail(err)
		}
	} else if merged {
		k, err := kernels.FlashAttentionMergeWide(ms)
		if err := comp(&v.merge, k, err); err != nil {
			return fail(err)
		}
		if err := g.prefillBuf(&pg.part, &pg.partFloats, kernels.FlashPartialFloats(ms)); err != nil {
			return fail(err)
		}
	}
	pg.vars[plan] = v
	return v, nil
}

// mergeShape is s at its value's width, which is what the partials hold and
// the merge writes: MLA's latent, the row's first MLA floats, where the
// scores read the whole row.
func mergeShape(s kernels.FlashShape) kernels.FlashShape {
	if s.MLA > 0 {
		s.Dim, s.MLA = s.MLA, 0
	}
	return s
}

// pagedDecode launches one layer's attention through the call's variant: q,
// the layer's pools and table, the descriptors desc, into out; sink is the
// layer's sinks (nil without). la launches at a thread count in groups of 128.
func (pg *pagedScratch) pagedDecode(s backend.Session, lc *launcher,
	q, k, v, n, out, tab, desc, sink backend.Buf) error {
	vr := pg.cur
	dst := out
	if vr.merge != nil {
		dst = pg.part
	}
	// The argument lists are arrays on the stack, sliced to the arity: a
	// slice literal grown by append for the sinks was a heap object per layer
	// of every gpt-oss token.
	if vr.k != nil {
		all := [8]backend.Buf{q, k, v, n, dst, tab, desc, sink}
		args := all[:7]
		if sink != nil {
			args = all[:8]
		}
		if err := lc.launch(vr.k, vr.groups, vr.width, args...); err != nil {
			return err
		}
	} else {
		lc.la(vr.scores, vr.scoreT, q, k, n, pg.sc, tab, desc)
		if err := lc.launch(vr.soft, vr.softG, 64, pg.sc, n, pg.wt, pg.part, tab, desc); err != nil {
			return err
		}
		lc.la(vr.acc, vr.accT, pg.wt, v, n, pg.part, tab, desc)
	}
	if vr.merge != nil {
		all := [3]backend.Buf{pg.part, out, sink}
		ma := all[:2]
		if sink != nil {
			ma = all[:3]
		}
		if vr.mergeItems > 0 {
			return lc.launch(vr.merge, vr.mergeItems, 32, ma...)
		}
		lc.la(vr.merge, vr.mergeT, ma...)
	}
	return nil
}

// seqEnd is one sequence a call writes and reads: every position below end
// must have a page. start is the first position the call writes, or -1 for a
// reservation that writes nothing yet: a windowed layer releases its pages
// behind start's window, and only a call that writes knows where that is.
type seqEnd struct {
	s          seqID
	end, start int
}

// seqEnds collapses a call's rows into dst, one entry per sequence: its last
// position plus one. A prompt chunk's rows are one sequence, so the pages are
// checked once and not once a row in every layer -- 28 x 512 map reads on a
// 512-row prompt, twice. A row of a sequence already seen extends that entry
// (a chunk's rows are in order; a batch's are distinct sequences).
func seqEnds(dst []seqEnd, rows []pagedRow) []seqEnd {
	dst = dst[:0]
	for _, r := range rows {
		if r.pad {
			continue
		}
		if n := len(dst); n > 0 && dst[n-1].s == r.s {
			dst[n-1].end = max(dst[n-1].end, r.pos+1)
			dst[n-1].start = min(dst[n-1].start, r.pos)
			continue
		}
		dst = append(dst, seqEnd{r.s, r.pos + 1, r.pos})
	}
	return dst
}

// heldLayers calls f for every pool layer session sid holds history
// on: a session takes pages only where it runs.
func (g *devTier) heldLayers(sid uint64, f func(li int, l *kvLayerPool) error) error {
	if g.kvp == nil {
		return nil
	}
	for li, l := range g.kvp.layers {
		// A block's entries (DeepSeek V4) are held where the block is.
		bi := li
		if l.rate > 0 {
			bi = l.of
		}
		if bl := g.layers[bi]; bl != nil {
			if kvp := bl.kv[sid]; kvp != nil && kvp.paged {
				if err := f(li, l); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// pagedAppend gives every sequence the pages its positions need in every
// layer the session holds, before a submission: a layer that grows may page
// weights out (roomOrReclaim) and drops the captured graph, neither of which
// can happen once blocks are paged in for a submission. A layer that cannot
// grow refuses with ErrKVCapacity, and what was already taken stays taken.
// Callers hold g.mu.
func (g *devTier) pagedAppend(sid uint64, seqs []seqEnd) error {
	kp := g.kvPages()
	for _, se := range seqs {
		if se.start >= 0 {
			if err := g.rewind(sid, kp, se.s, se.start); err != nil {
				return err
			}
		}
	}
	err := g.heldLayers(sid, func(li int, l *kvLayerPool) error {
		for _, se := range seqs {
			need := (l.positions(se.end) + kp.p - 1) / kp.p
			// A named page this call writes into is copied first (kvshare.go).
			if l.rate == 0 {
				if err := g.cowPages(sid, kp, l, se.s, se.start, se.end); err != nil {
					return err
				}
			}
			if have := len(l.owned[se.s]); need > have {
				if err := g.allocPages(sid, kp, l, se.s, need-have); err != nil {
					return err
				}
			}
			if l.win > 0 && se.start >= 0 && !g.KVKeepWindow {
				if err := g.windowPages(sid, kp, li, l, se); err != nil {
					return err
				}
			}
		}
		return nil
	})
	if err == nil {
		err = g.streamRoom(sid)
	}
	if ferr := g.flushTabs(); err == nil {
		err = ferr
	}
	return err
}

// windowPages keeps a windowed layer's pages to what se's rows read: a page
// the call writes into that was released (a row restarting at a lower
// position) gets a fresh id, a page it reads that was released is refused --
// its history is gone -- and every page behind the window of
// se.start-nn.KVWindowSlack is released, so the layer holds the window and
// not the context. Callers hold g.mu, between submissions.
func (g *devTier) windowPages(sid uint64, kp *kvPool, li int, l *kvLayerPool, se seqEnd) error {
	s, P := se.s, kp.p
	need := (se.end + P - 1) / P
	ev := kp.evicted[s]
	for j := se.start / P; j < need; j++ {
		if l.owned[s][j] == 0 && j >= ev {
			if err := g.allocAt(sid, kp, l, s, j); err != nil {
				return err
			}
		}
	}
	fault := pagedFaulted("winrelease")
	for j := l.keyStart(se.start) / P; j < se.start/P && !fault; j++ {
		if l.owned[s][j] == 0 && (j >= ev || l.home[s][j] == nil) {
			return fmt.Errorf("block %d: sequence %v reads position %d's window from page %d, released "+
				"behind it", li, s, se.start, j)
		}
	}
	first := min(l.keyStart(max(0, se.start-nn.KVWindowSlack))/P, len(l.owned[s]))
	if fault {
		// The violation: the page the window starts in goes too, the one being
		// written into when the window lies inside it.
		first = min(l.keyStart(se.start)/P+1, len(l.owned[s]))
	}
	if l.rel == nil {
		l.rel = map[seqID]int{}
	}
	for j := l.rel[s]; j < first; j++ {
		if id := l.owned[s][j]; id != 0 {
			l.releaseID(id)
			l.owned[s][j] = 0
			g.tabPend = append(g.tabPend, tabWrite{l.tab, kp.ranges[s].off + j, l.owned[s][j : j+1]})
			g.KVWindowReleased++
		}
		if j < len(l.home[s]) {
			l.home[s][j] = nil // at home, and behind the window all the same
		}
	}
	l.rel[s] = max(l.rel[s], first)
	return nil
}

// allocAt gives sequence s a fresh id for its page j in layer l, which holds
// none (it was released behind a window). Callers hold g.mu, outside any
// submission.
func (g *devTier) allocAt(sid uint64, kp *kvPool, l *kvLayerPool, s seqID, j int) error {
	if len(l.free) < 1 {
		l.reclaimNamed(1)
	}
	if len(l.free) < 1 {
		if err := g.growKVLayer(kp, l, l.n+1-len(l.free)); err != nil && !g.evictForRoom(sid, kp, l, 1) {
			return err
		}
	}
	id := l.free[len(l.free)-1]
	l.free = l.free[:len(l.free)-1]
	l.owned[s][j] = id
	g.tabPend = append(g.tabPend, tabWrite{l.tab, kp.ranges[s].off + j, l.owned[s][j : j+1]})
	if l.rel[s] > j {
		l.rel[s] = j
	}
	return nil
}

// pagedAppendRows is pagedAppend for a call's rows, from outside g.mu.
func (g *devTier) pagedAppendRows(sid uint64, rows []pagedRow) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.paged {
		return true
	}
	g.pgSeqs = seqEnds(g.pgSeqs, rows)
	if err := g.pagedAppend(sid, g.pgSeqs); err != nil {
		g.LastErr = err.Error()
		return false
	}
	return true
}

// pagedPrep readies bs for a call's rows: the attention variant for their
// keys, and a check that every row's page is there (pagedAppend took them).
// Compiling and allocating are driver calls, so it runs before the
// submission's session. Callers hold g.mu.
func (g *devTier) pagedPrep(sid uint64, bs *blockScratch, rows []pagedRow) error {
	kp, pg := g.kvPages(), bs.pkv
	held := false
	g.pgSeqs = seqEnds(g.pgSeqs, rows)
	err := g.heldLayers(sid, func(li int, l *kvLayerPool) error {
		held = true
		for _, se := range g.pgSeqs {
			if (l.positions(se.end)-1)/kp.p >= len(l.owned[se.s]) {
				return fmt.Errorf("block %d: sequence %v holds %d page(s) and writes position %d",
					li, se.s, len(l.owned[se.s]), se.end-1)
			}
		}
		return nil
	})
	if err != nil || !held {
		pg.cur = nil // no attention block of this session here: nothing to stage
		return err
	}
	// The plan follows the key count rounded to the recording grain, so a
	// captured graph serves every token of a grain rather than recapturing
	// each time the square root moves (graphKey.pagedVar).
	keys := ds4KeyCap(bs, msaKeys(&bs.p, idxKeys(&bs.p, (pagedKeys(rows)+scoreGrain-1)/scoreGrain*scoreGrain)))
	pg.cur, err = g.pagedPlanFor(bs, keys)
	return err
}

// pagedRow is one query row of a call: its sequence and its position.
// A padding row has pad set and reads and writes only the dummy page.
type pagedRow struct {
	s   seqID
	pos int
	pad bool
}

// pagedStage fills pg's descriptor slabs for a call's rows, full and
// windowed: each row's sequence run, the same offset in every layer's arena.
// Callers hold g.mu and have run pagedPrep for the same rows.
func (g *devTier) pagedStage(pg *pagedScratch, p *nn.LayerPlan, rows []pagedRow) {
	kp := g.kvPages()
	pg.desc, pg.descW = pg.desc[:0], pg.descW[:0]
	for _, r := range rows {
		if r.pad {
			pg.desc = append(pg.desc, 0, 0, 0, 0)
			if p.SWAWindow > 0 {
				pg.descW = append(pg.descW, 0, 0, 0, 0)
			}
			continue
		}
		o := uint32(kp.ranges[r.s].off)
		end := uint32(r.pos + 1)
		// A row of a bidirectional run attends to the run's end (bidir.go);
		// its window still starts from its own position, as the reference's
		// does, which the start word here carries apart from the end.
		last := end
		if e := bidirCount(g.bidir, r.pos); e > 0 && !p.BidirOff {
			last = uint32(e)
		}
		pg.desc = append(pg.desc, o, 0, last, uint32(r.pos))
		if p.SWAWindow > 0 {
			// The windowed kernels' mask (kernels.windowed), as a start: the
			// last W keys, or a chunk of W aligned keys.
			var lo uint32
			if w := uint32(p.SWAWindow); p.SWAChunked {
				lo = (end - 1) / w * w
			} else {
				lo = end - min(end, w)
			}
			pg.descW = append(pg.descW, o, lo, last, uint32(r.pos))
		}
	}
}

// pagedRows fills dst with a call's rows: one sequence's positions pos.. for a
// decode or a prefill chunk, or a LayersRows step's sequences, with R-nrow
// padding rows. dst is reused, so a token allocates nothing.
func (g *devTier) pagedRows(sid uint64, dst []pagedRow, R, nrow, pos int, rag *ragStep) []pagedRow {
	rows := slices.Grow(dst[:0], R)[:R]
	for r := range rows {
		switch {
		case rag != nil && r < len(rag.pos):
			base := rag.slot[r] - rag.pos[r]
			if pagedFaulted("alias") {
				base = rag.slot[0] - rag.pos[0]
			}
			s := sid
			if rag.sid != nil && !pagedFaulted("session") {
				s = rag.sid[r]
			}
			rows[r] = pagedRow{s: seqID{s, base}, pos: rag.pos[r]}
		case rag == nil && r < nrow:
			rows[r] = pagedRow{s: seqID{sid, 0}, pos: pos + r}
		default:
			rows[r] = pagedRow{pad: true}
		}
	}
	return rows
}

// pagedKeys is the most keys any of a call's rows attends.
func pagedKeys(rows []pagedRow) int {
	n := 1
	for _, r := range rows {
		if !r.pad {
			n = max(n, r.pos+1)
		}
	}
	return n
}

// pagedLayer gives block li its place in the pool, and session sid its
// seat there: the layer's buffers and table arena, and its working set -- the
// dummy and one page per attached session, since a session with no page
// cannot run a token. That is a first frame, not a reservation of context:
// past it the layer grows as sequences write, and a growth that does not fit
// pages weights out or refuses (ErrKVCapacity). The session's history on the
// block is the marker kvPair{paged: true}. Callers hold g.mu.
func (g *devTier) pagedLayer(sid uint64, l *layer, li int, p *nn.LayerPlan) error {
	kp := g.kvPages()
	seats := 1
	for o, kvp := range l.kv {
		if kvp.paged && o != sid {
			seats++
		}
	}
	if err := g.addKVLayer(kp, li, g.kvGeomOf(p), 1+seats); err != nil {
		return err
	}
	// The window paged attention applies to this block (pagedStage's descW).
	kp.layers[li].win, kp.layers[li].chunked = p.Window(li), p.SWAChunked
	// DeepSeek V4's rows and entries are read through their tables by the
	// gathers, which a streamed pass does not serve: they never leave.
	// So does a selection over the whole history (the lightning indexer,
	// MiniMax-M3's and QSA's block selections): a streamed pass cannot serve
	// it, so its pages cannot go home either.
	kp.layers[li].noEvict = p.DS4 != nil || keepsHistory(p)
	if d := p.DS4; d != nil && p.Comp != nn.DS4CompNone {
		ek := entKey(li)
		geom := kvGeom{kvRow: d.EntRowAt(p.Comp, p.HeadDim), mla: true}
		if err := g.addKVLayer(kp, ek, geom, 1+seats); err != nil {
			return err
		}
		e := kp.layers[ek]
		e.rate, e.of, e.noEvict = d.RateAt(p.Comp), li, true
	}
	if l.kv == nil {
		l.kv = map[uint64]*kvPair{}
	}
	fresh := l.kv[sid] == nil
	l.kv[sid] = &kvPair{paged: true}
	for _, k := range []int{li, entKey(li)} {
		if _, ok := kp.layers[k]; !ok {
			continue
		}
		if err := g.pagedSeat(kp, k, l); err != nil {
			if fresh {
				delete(l.kv, sid)
			}
			return err
		}
	}
	// A history that cannot leave the card is held at the context asked for,
	// charged here with the block, so the budget declines the block at
	// placement rather than a call failing for want of a page mid-prompt.
	if keepsHistory(p) {
		if want := contextPages(kp, g.capped(p).MaxSeq, seats); kp.layers[li].n < want {
			if err := g.growKVLayer(kp, kp.layers[li], want); err != nil {
				if fresh {
					delete(l.kv, sid)
				}
				return err
			}
		}
	}
	return nil
}

// selectsHistory reports a block whose attention selects from its whole
// history: the lightning indexer, MiniMax-M3's and QSA's block selections.
func selectsHistory(p *nn.LayerPlan) bool { return p.IdxHeads != 0 || p.QSA() }

// keepsHistory is selectsHistory as the pool acts on it: such a block's pages
// never leave the card and its context is charged with it. The row fault
// evictselection withdraws both, the violation the gates demand fail.
func keepsHistory(p *nn.LayerPlan) bool {
	return selectsHistory(p) && !rowsFaulted("evictselection")
}

// contextPages is the pages a block that holds its context needs for seats
// sequences of the plan's longest context, and the dummy.
func contextPages(kp *kvPool, maxSeq, seats int) int {
	return 1 + seats*((maxSeq+kp.p-1)/kp.p)
}

// pagedSeat raises layer li's floor to its attached sessions' working set and
// grows it there. Callers hold g.mu.
func (g *devTier) pagedSeat(kp *kvPool, li int, bl *layer) error {
	l := kp.layers[li]
	seats := 0
	for _, kvp := range bl.kv {
		if kvp.paged {
			seats++
		}
	}
	l.floor = 1 + seats
	if l.n < l.floor {
		return g.growKVLayer(kp, l, l.floor)
	}
	return nil
}

// pagedLayerBytes is what placing another attention block costs: a layer at
// the working set a session brings, and its table arena.
func (g *devTier) pagedLayerBytes(p *nn.LayerPlan) uint64 {
	geom := g.kvGeomOf(p)
	words := 256
	if g.kvp != nil {
		words = g.kvp.tabWords
	}
	pages := uint64(1 + g.sessions())
	if keepsHistory(p) && g.kvp != nil {
		// Held at the context asked for (pagedLayer).
		pages = uint64(contextPages(g.kvp, g.capped(p).MaxSeq, int(g.sessions())))
	}
	n := uint64(geom.kWords(g.kvPage())+geom.vWords(g.kvPage()))*4*pages + uint64(words)*4 +
		kvStageBytes(geom, g.kvPage())
	if d := p.DS4; d != nil && p.Comp != nn.DS4CompNone {
		// The block's entries, a pool of their own (ds4.go).
		e := kvGeom{kvRow: d.EntRowAt(p.Comp, p.HeadDim), mla: true}
		n += uint64(e.kWords(g.kvPage()))*4*uint64(1+g.sessions()) + uint64(words)*4
	}
	return n
}

// dropKVLayer frees layer li's pages, table and buffers and refunds them. The
// sequences' runs stay while another layer holds pages of them. Callers hold
// g.mu.
func (g *devTier) dropKVLayer(kp *kvPool, li int) {
	if li >= 0 {
		g.dropKVLayer(kp, entKey(li)) // a DeepSeek V4 block's entries go with it
	}
	l, ok := kp.layers[li]
	if !ok {
		return
	}
	g.freeKVLayer(l)
	delete(kp.layers, li)
	for s := range l.owned {
		kp.dropRange(s)
	}
}

// pagedLeave gives back session sid's pages in layer li to the layer's free
// list, so what the session held there is room again -- the next sequence's
// pages, or given back to the budget by kvCompact when something else needs
// it: the point of a session yielding one block while another session keeps
// it. Callers hold g.mu, between submissions: the released pages are fenced
// until every submission that could read them has completed (fence).
func (g *devTier) pagedLeave(li int, sid uint64, bl *layer) {
	if g.kvp == nil {
		return
	}
	if li >= 0 {
		g.pagedLeave(entKey(li), sid, bl) // a DeepSeek V4 block's entries go with it
	}
	l := g.kvp.layers[li]
	if l == nil {
		return
	}
	for s := range l.owned {
		if s.sid == sid {
			l.releasePages(s)
			g.kvp.dropRange(s)
			g.roomGen.Add(1) // free pages are room (kvCompact)
		}
	}
	l.fence()
	if bl != nil {
		// The seat goes with the session: pagedSeat recounts.
		if err := g.pagedSeat(g.kvp, li, bl); err != nil {
			g.LastErr = err.Error()
		}
	}
}

// kvCompact makes room for n bytes out of the pool's free pages: every layer
// but keep is compacted to what its sequences hold and its floor, one at a
// time until n fits, and it reports whether n fits. Released pages otherwise
// stay in the pool -- the pages one request gives back are the next one's, and
// a pool that shrank on every release grew again on every prompt, an
// allocation and a copy a layer a request. keep is a layer being grown, whose
// ids its caller is still counting. Callers hold g.mu, outside any submission.
func (g *devTier) kvCompact(n uint64, keep *kvLayerPool) bool {
	if g.room(n) {
		return true
	}
	if g.kvp == nil {
		return false
	}
	for _, l := range g.kvp.layers {
		if l == nil || l == keep {
			continue
		}
		l.fence()
		if err := g.shrinkKVLayer(g.kvp, l); err != nil {
			g.LastErr = err.Error()
		}
		if g.room(n) {
			return true
		}
	}
	return false
}

// trimSeq keeps the pages sequence s needs for pos positions in every layer
// and releases the rest, reporting whether any went. Callers hold g.mu.
func (kp *kvPool) trimSeq(s seqID, pos int) bool {
	keep := (pos + kp.p - 1) / kp.p
	trimmed := false
	for _, l := range kp.layers {
		keep := keep
		if l.rate > 0 {
			// Entries: the ones the kept positions closed.
			keep = (pos/l.rate + kp.p - 1) / kp.p
		}
		ids := l.owned[s]
		if keep >= len(ids) {
			continue
		}
		l.fenceIDs(ids[keep:])
		if keep == 0 {
			delete(l.owned, s)
		} else {
			l.owned[s] = ids[:keep]
		}
		if l.rel[s] > keep {
			l.rel[s] = keep
		}
		trimmed = true
	}
	kp.forgetFrom(s, keep)
	if kp.written[s] > pos {
		kp.written[s] = pos
	}
	kp.dropRange(s)
	return trimmed
}

// pagedMigrate moves layer li's history for session sid's sequence
// between host and device: k and v are [pos][kvRow] float32 on the host. To
// the device each page is laid out on the host in the layer's layout and
// written at its offset; from the device the sequence's pages are gathered
// into one buffer by a generated kernel and read home. Callers hold g.mu.
func (g *devTier) pagedMigrate(sid uint64, li, base int, k, v []float32, pos int, toDevice bool) error {
	kp := g.kvPages()
	pl, ok := kp.layers[li]
	if !ok {
		return fmt.Errorf("block %d has no pages on this device", li)
	}
	s := seqID{sid, base}
	kvRow, P := pl.geom.kvRow, kp.p
	g.KVPositionsMigrated += pos
	if pos*kvRow > len(k) || pos*kvRow > len(v) {
		return fmt.Errorf("%d positions of %d floats do not fit the host's %d/%d", pos, kvRow, len(k), len(v))
	}
	npages := (pos + P - 1) / P
	// The sequence's oldest pages may be at home (kvevict.go): the same count
	// in every layer, so a layer joining keeps them there too.
	// Sent up, the host's position is the history: past it the device's
	// record is a sequence that has since restarted lower. Read home, pos is
	// only how much of the history is wanted (a prefix a store saves), and the
	// device keeps the rest.
	if toDevice {
		if err := g.rewind(sid, kp, s, pos); err != nil {
			return err
		}
	}
	e := min(kp.evicted[s], npages)
	if toDevice && e > 0 && len(pl.owned[s]) == 0 {
		if err := g.ensureRange(kp, s, e); err != nil {
			return err
		}
		pl.owned[s] = make([]uint32, e)
		if err := pl.tab.WriteAt(kp.ranges[s].off*4, u32view(pl.owned[s])); err != nil {
			return err
		}
	}
	if toDevice {
		if have := len(pl.owned[s]); npages > have {
			if err := g.allocPages(sid, kp, pl, s, npages-have); err != nil {
				return err
			}
		}
		if err := g.flushTabs(); err != nil {
			return err
		}
	}
	ids := pl.owned[s]
	if len(ids) < npages {
		return fmt.Errorf("the sequence holds %d page(s) and %d positions need %d", len(ids), pos, npages)
	}
	kw, vw := pl.geom.kWords(P), pl.geom.vWords(P)
	if toDevice {
		kPage := make([]float32, kw)
		vPage := make([]float32, P*kvRow)
		for j := 0; j < npages; j++ {
			clear(kPage)
			clear(vPage)
			for t := j * P; t < min(pos, (j+1)*P); t++ {
				if pl.geom.mla {
					// MLA's row is row-major: the latent and its key, as the
					// host holds them.
					copy(kPage[(t%P)*kvRow:], k[t*kvRow:(t+1)*kvRow])
					continue
				}
				for e := 0; e < kvRow; e++ {
					kPage[e*P+t%P] = k[t*kvRow+e]
				}
				copy(vPage[(t%P)*kvRow:], v[t*kvRow:(t+1)*kvRow])
			}
			var vb []byte
			if vw > 0 { // MLA has none: the value is the key row's prefix
				vb = f32b(vPage)
				if pl.geom.f16 {
					vb = packF16(vPage)
				}
			}
			if ids[j] == 0 {
				if pl.home == nil {
					pl.home = map[seqID][][]float32{}
				}
				page := append(append(make([]float32, 0, kw+vw), kPage...), f32of(vb)...)
				pl.home[s] = append(pl.home[s][:j], page)
				continue
			}
			if err := pl.k.WriteAt(int(ids[j])*kw*4, f32b(kPage)); err != nil {
				return err
			}
			if vw == 0 {
				continue
			}
			if err := pl.v.WriteAt(int(ids[j])*vw*4, vb); err != nil {
				return err
			}
		}
		return nil
	}
	kAll, err := g.readPages(pl, pl.k, ids[e:npages], kw)
	if err != nil {
		return err
	}
	var vAll []float32
	if vw > 0 {
		if vAll, err = g.readPages(pl, pl.v, ids[e:npages], vw); err != nil {
			return err
		}
	}
	if e > 0 {
		// The pages at home ahead of the ones read from the card. A windowed
		// layer's page released behind its window has no copy: zeros, which
		// nothing reads (the host trusts this history only from its window).
		hk, hv := make([]float32, 0, npages*kw), make([]float32, 0, npages*vw)
		for _, page := range pl.home[s][:e] {
			if page == nil {
				hk, hv = append(hk, make([]float32, kw)...), append(hv, make([]float32, vw)...)
				continue
			}
			hk, hv = append(hk, page[:kw]...), append(hv, page[kw:]...)
		}
		kAll, vAll = append(hk, kAll...), append(hv, vAll...)
	}
	for t := 0; t < pos; t++ {
		j, o := t/P, t%P
		if pl.geom.mla {
			// Row-major, and the value is the row's prefix. The host's latent
			// pages alias k and v and are written from both, so both carry
			// the row: a v left zero overwrites the k just restored.
			row := kAll[j*kw+o*kvRow : j*kw+(o+1)*kvRow]
			copy(k[t*kvRow:(t+1)*kvRow], row)
			copy(v[t*kvRow:(t+1)*kvRow], row)
			continue
		}
		for e := 0; e < kvRow; e++ {
			k[t*kvRow+e] = kAll[j*kw+e*P+o]
		}
		if pl.geom.f16 {
			unpackF16(v[t*kvRow:(t+1)*kvRow], vAll[j*vw+o*kvRow/2:j*vw+(o+1)*kvRow/2])
		} else {
			copy(v[t*kvRow:(t+1)*kvRow], vAll[j*vw+o*kvRow:j*vw+(o+1)*kvRow])
		}
	}
	return nil
}

// readPages gathers pages ids of one of layer l's buffers, words each, into
// one staging buffer on the device and reads it home. When the card refuses
// that buffer the pages come home a page at a time through the layer's
// reserve (kvLayerPool.stage): a park or an eviction is how a full card makes
// room, so it must not need room of its own.
func (g *devTier) readPages(l *kvLayerPool, src backend.Buf, ids []uint32, words int) ([]float32, error) {
	if len(ids) == 0 || words == 0 {
		return nil, nil
	}
	out := make([]float32, len(ids)*words)
	dst, err := g.dev.Alloc(len(ids) * words * 4)
	if err != nil {
		if l.stage == nil {
			return nil, err
		}
		g.KVStageReads++
		return out, g.readStaged(l, src, ids, words, out)
	}
	defer dst.Free()
	if err := g.gatherInto(src, dst, ids, words); err != nil {
		return nil, err
	}
	return out, dst.Read(f32b(out))
}

// readStaged is readPages through l's reserve, a page a pass, with the gather
// makeStage compiled: nothing is allocated or compiled on the device.
func (g *devTier) readStaged(l *kvLayerPool, src backend.Buf, ids []uint32, words int, out []float32) error {
	k, err := g.gatherKernel(words, 1)
	if err != nil {
		return err
	}
	for i := range ids {
		if err := l.stageTab.Write(u32view(ids[i : i+1])); err != nil {
			return err
		}
		if err := k.Launch((words+127)/128, 128, src, l.stage, l.stageTab); err != nil {
			return err
		}
		if err := l.stage.Read(f32b(out[i*words : (i+1)*words])); err != nil {
			return err
		}
	}
	return nil
}

// PerSequenceKV reports that this device pages its history (nn.SeqKVDevice).
func (g *devTier) PerSequenceKV() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.paged
}

// MigrateKVSeq is migrateKVSeq for the zero session: a caller that never attached.
func (g *devTier) MigrateKVSeq(li, base int, k, v []float32, pos int, toDevice bool) bool {
	return g.migrateKVSeq(0, li, base, k, v, pos, toDevice)
}

// migrateKVSeq moves layer li's history for session sid's sequence at
// base between host and device (nn.SeqKVDevice).
func (g *devTier) migrateKVSeq(sid uint64, li, base int, k, v []float32, pos int, toDevice bool) bool {
	sv := g.as(sid)
	if sv == nil {
		return false
	}
	defer sv.done()
	g = sv
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.paged {
		g.LastErr = "MigrateKVSeq: this device keeps the contiguous cache"
		return false
	}
	// A KV-sharing block has no history of its own to move: it reads its
	// source's, which moves with the source.
	if l := g.layers[li]; pos <= 0 || l != nil && l.kvSrc != li {
		return true
	}
	if err := g.pagedMigrate(sid, li, base, k, v, pos, toDevice); err != nil {
		g.LastErr = fmt.Sprintf("MigrateKVSeq: block %d, sequence at %d: %v", li, base, err)
		return false
	}
	return true
}

// MigrateEnt is migrateEnt for the zero session: a caller that never attached.
func (g *devTier) MigrateEnt(li, base int, ent []float32, n int, toDevice bool) bool {
	return g.migrateEnt(0, li, base, ent, n, toDevice)
}

// migrateEnt moves DeepSeek V4 block li's entries for session sid's
// sequence at base between host and device (nn.EntDevice): n entries of the
// block's entry row, [n][row] float32.
func (g *devTier) migrateEnt(sid uint64, li, base int, ent []float32, n int, toDevice bool) bool {
	v := g.as(sid)
	if v == nil {
		return false
	}
	defer v.done()
	g = v
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.paged {
		g.LastErr = "MigrateEnt: this device keeps the contiguous cache"
		return false
	}
	if n <= 0 {
		return true
	}
	if err := g.pagedMigrate(sid, entKey(li), base, ent, ent, n, toDevice); err != nil {
		g.LastErr = fmt.Sprintf("MigrateEnt: block %d, sequence at %d: %v", li, base, err)
		return false
	}
	return true
}

// ReserveKVSeqs is reserveKVSeqs for the zero session: a caller that never attached.
func (g *devTier) ReserveKVSeqs(bases, ends []int) bool {
	return g.reserveKVSeqs(0, bases, ends)
}

// reserveKVSeqs takes the pages session sid's sequences at bases need
// for positions up to ends (nn.SeqKVDevice).
func (g *devTier) reserveKVSeqs(sid uint64, bases, ends []int) bool {
	// No scratch is read: the session's pages and table run are all it
	// touches, so it takes no lane.
	g = g.bareView(sid)
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.paged {
		g.LastErr = "ReserveKVSeqs: this device keeps the contiguous cache"
		return false
	}
	g.pgSeqs = g.pgSeqs[:0]
	for i, b := range bases {
		g.pgSeqs = append(g.pgSeqs, seqEnd{seqID{sid, b}, ends[i], -1})
	}
	if err := g.pagedAppend(sid, g.pgSeqs); err != nil {
		g.LastErr = err.Error()
		return false
	}
	return true
}

// packF16 is f32 values as packed binary16, two to a word: the pool's f16 V,
// rounded by the host's generated narrowing (nn.NarrowF16).
func packF16(v []float32) []byte {
	b := make([]byte, 2*len(v))
	if len(v) > 0 {
		nn.NarrowF16(unsafe.Slice((*uint16)(unsafe.Pointer(&b[0])), len(v)), v)
	}
	return b
}

// unpackF16 widens packed binary16 words into dst through the host's
// generated widening (nn.WidenF16).
func unpackF16(dst, words []float32) {
	if len(dst) > 0 {
		nn.WidenF16(dst, unsafe.Slice((*uint16)(unsafe.Pointer(&words[0])), len(dst)))
	}
}

// pagedHold gives session sid's sequence the pages pos positions of
// its history need, in every layer it holds, growing a layer that is short.
// It takes them rather than counting free ones as room: every caller reserves
// positions it is about to write, and a free page is exactly what kvCompact
// gives back when another layer grows -- counted and not taken, it was gone by
// the time the token asked for it. Callers hold g.mu.
func (g *devTier) pagedHold(sid uint64, pos int) error {
	g.pgSeqs = append(g.pgSeqs[:0], seqEnd{seqID{sid, 0}, pos, -1})
	return g.pagedAppend(sid, g.pgSeqs)
}

// kvPoolBytes is what the pool holds against the budget: every layer's
// buffers and tables, the dummy pages included. Callers hold g.mu.
func (g *devTier) kvPoolBytes() uint64 {
	if g.kvp == nil {
		return 0
	}
	return g.kvp.charged()
}

// pagedHeldBy is what session sid's pages cost on this device, over every
// layer it holds.
func (g *devTier) pagedHeldBy(sid uint64) uint64 {
	if g.kvp == nil {
		return 0
	}
	var n uint64
	for _, l := range g.kvp.layers {
		for s, ids := range l.owned {
			if s.sid == sid {
				// Resident pages: an id of 0 is at home or released.
				for _, id := range ids {
					if id != 0 {
						n += g.kvp.pageBytes(l)
					}
				}
			}
		}
	}
	return n
}
