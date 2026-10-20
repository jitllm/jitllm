package tier

import (
	"slices"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// A ragged call's prompt runs attend as a prefill through a run table: each
// sequence with more than one row in the call gets its rows into the table
// in position order, padded to whole query tiles with its last row, and the
// staged prefill's run-table form (compose.Prefill.Map) attends them all,
// each slot's partition base its run's first key aligned down to 64. Where
// every real row is in such a run (and the padding rows a run of their own
// attending nothing) it is the call's whole attention.

// ragRun is a call's run table and each descriptor set's bases and chunk.
type ragRun struct {
	n          int // the table's slots; 0 for no run-table attention
	slots      []uint32
	order      []int
	base       [2][]uint32
	chunk      [2]int
	pMap       backend.Buf
	pBase      [2]backend.Buf
	mapCap     int
	baseCap    [2]int
	sq, kt, qt int
}

// ragStaged reports whether a ragged call's runs attend through the staged
// run table here: a device with neither the int8 matrix instruction nor the
// collective tiles, outside PTX (whose matrix forms take a chunk).
func (g *devTier) ragStaged(bs *blockScratch) bool {
	mma, _ := backend.LowersIntMMA(g.dev)
	if bs.p.MLA() {
		// A latent row has no flash or matrix form but sm_70's pair: the
		// staged FMA tiles serve it everywhere else.
		return !(g.composeTarget().API == compose.APIPTX && !mma)
	}
	if _, _, ok := g.prefillTile(bs); ok || g.flash16Serves(bs) {
		return false
	}
	if g.composeTarget().API == compose.APIPTX || g.tilesOK() {
		return true
	}
	return !mma
}

// f16MMA is the k of the binary16 m16n8 instruction a PTX device lowers:
// 16, else 8, else 0.
func (g *devTier) f16MMA() int {
	if g.f16K == 0 {
		g.f16K = -1
		if g.composeTarget().API != compose.APIPTX {
			return 0
		}
		for _, k := range []int{16, 8} {
			if ok, _ := backend.LowersMMA(g.dev, ir.MMAShape{M: 16, N: 8, K: k, Kind: ir.MMAF16}); ok {
				g.f16K = k
				break
			}
		}
	}
	return max(g.f16K, 0)
}

// flash16Serves reports whether a prompt's paged attention is the one
// kernel on the binary16 m16n8 instruction here: a device that lowers it,
// a head a whole number of 32 up to 256, rows a whole number of its row
// block (16 a subgroup, four subgroups, two sharing a head past 128).
func (g *devTier) flash16Serves(bs *blockScratch) bool {
	p := &bs.p
	if g.composeTarget().API != compose.APIPTX || g.kb.noAttnMMA || p.MLA() || g.f16MMA() == 0 {
		return false
	}
	hd, split := p.HeadDim, 1
	if hd > 128 {
		split = 2
	}
	return p.NKVHead > 0 && p.NHead%p.NKVHead == 0 && hd%32 == 0 && hd >= 32 && hd <= 256 &&
		bs.rows%(16*4/split) == 0 && p.AttnSoftcap >= 0 && !g.KVF16
}

// prefillTile is the collective-tile flash prefill's blocking for bs's
// rows (subgroups of eight query rows, keys a trip), false where the device
// lowers no tile, the attention is latent or has sinks (whose merge the
// form does not take), or no blocking serves its head width.
func (g *devTier) prefillTile(bs *blockScratch) (sg, tk int, ok bool) {
	p := &bs.p
	if !g.tileF32() || p.MLA() || p.AttnSinks || g.KVF16 || g.flash16Serves(bs) {
		return 0, 0, false
	}
	return flashTileOf(p.HeadDim, bs.rows)
}

// prefillRuns reports whether a chunk's or a ragged call's runs attend
// as a prefill here: staged, or on the collective tiles.
func (g *devTier) prefillRuns(bs *blockScratch) bool {
	_, _, tile := g.prefillTile(bs)
	return tile || g.flash16Serves(bs) || g.ragStaged(bs)
}

// flashTileOf is the collective-tile flash prefill's blocking for a head
// width over rows rows, ok where it serves them: the blocking with the most query rows, then the
// most keys, whose workgroup memory -- binary16 K and float32 V a key tile,
// the scores and the output a query tile -- fits 32 KiB and whose staging
// divides among its threads, must be four subgroups of rows that tile the
// chunk, its key tile a divisor of 32.
func flashTileOf(hd, rows int) (sg, tk int, ok bool) {
	for _, t := range [][2]int{{4, 32}, {4, 16}, {2, 32}, {2, 16}, {2, 8}, {1, 32}, {1, 16}} {
		sg, tk := t[0], t[1]
		qb, threads := 8*sg, 32*sg
		if compose.FlashTileMem(hd, sg, tk) <= 32<<10 && (tk*hd/2)%threads == 0 && (tk*hd/4)%threads == 0 {
			return sg, tk, sg >= 4 && rows%qb == 0 && 32%tk == 0
		}
	}
	return 0, 0, false
}

// ragStage builds the call's run table from its rows and descriptors, or
// leaves it empty where some real row is its sequence's only one, a span
// passes the pass width, or the device takes another form. Callers hold
// g.mu, outside a submission.
func (g *devTier) ragStage(bs *blockScratch, pk *pagedScratch) {
	rr := &pk.rag
	rr.n = 0
	if !g.prefillRuns(bs) || len(pk.desc) == 0 {
		return
	}
	rr.sq, rr.kt, rr.qt = g.tiles.qtile, g.tiles.ktile, ragAccTile
	wg := lcmInt(rr.sq, rr.qt)
	sg, _, tile := g.prefillTile(bs)
	switch {
	case tile:
		wg = 8 * sg // a workgroup's query rows, one sequence's
	case g.flash16Serves(bs):
		wg, tile = g.flash16(bs, 0, false).QRows(), true
	}
	rows := pk.rows
	rr.order = rr.order[:0]
	for r, row := range rows {
		if !row.pad {
			rr.order = append(rr.order, r)
		}
	}
	slices.SortFunc(rr.order, func(a, b int) int {
		ra, rb := rows[a], rows[b]
		switch {
		case ra.s.sid != rb.s.sid:
			if ra.s.sid < rb.s.sid {
				return -1
			}
			return 1
		case ra.s.base != rb.s.base:
			return ra.s.base - rb.s.base
		}
		return ra.pos - rb.pos
	})
	rr.slots = rr.slots[:0]
	for i := 0; i < len(rr.order); {
		j := i + 1
		for j < len(rr.order) && rows[rr.order[j]].s == rows[rr.order[i]].s {
			j++
		}
		if j-i < 2 {
			rr.slots = rr.slots[:0]
			return
		}
		for k := 0; k < (j-i+wg-1)/wg*wg; k++ {
			rr.slots = append(rr.slots, uint32(rr.order[min(i+k, j-1)]))
		}
		i = j
	}
	if len(rr.slots) == 0 {
		return
	}
	first, pads := len(rr.slots), 0
	for r, row := range rows {
		if row.pad {
			rr.slots = append(rr.slots, uint32(r))
			pads++
		}
	}
	for pads > 0 && (len(rr.slots)-first)%wg != 0 {
		rr.slots = append(rr.slots, rr.slots[len(rr.slots)-1])
	}
	n := len(rr.slots)
	for i, desc := range [2][]uint32{pk.desc, pk.descW} {
		rr.base[i] = rr.base[i][:0]
		if len(desc) == 0 {
			continue
		}
		span, cur := 1, uint32(0)
		for slot, r := range rr.slots {
			row := rows[r]
			if slot == 0 || rows[rr.slots[slot-1]].s != row.s || rows[rr.slots[slot-1]].pad != row.pad {
				cur = desc[int(r)*kernels.PRowWords+kernels.PRowStart] &^ 63
			}
			rr.base[i] = append(rr.base[i], cur)
			if end := int(desc[int(r)*kernels.PRowWords+kernels.PRowEnd]); end > int(cur) {
				span = max(span, end-int(cur))
			}
		}
		if span > ragPassW && !tile {
			return
		}
		c := 64
		for c < span {
			c *= 2
		}
		rr.chunk[i] = c
		if !g.regrow(&rr.pBase[i], &rr.baseCap[i], n, 4, false, false) {
			return
		}
	}
	if !g.regrow(&rr.pMap, &rr.mapCap, n, 4, false, false) || !tile && !g.stagedPlanes(bs, pk, n, max(rr.chunk[0], rr.chunk[1])) {
		return
	}
	rr.n = n
}

// stagedPlanes grows the staged prefill's score and weight planes to rows
// slots of every head over chunk keys, outside any submission: inside one
// they may not grow. Callers hold g.mu.
func (g *devTier) stagedPlanes(bs *blockScratch, pk *pagedScratch, rows, chunk int) bool {
	plane := rows * bs.p.NHead * chunk
	return g.regrow(&pk.sc, &pk.scCap, plane, 4, true, true) && g.regrow(&pk.wt, &pk.wtCap, plane, 4, true, true)
}

// ragAccTile is the run table's accumulate (and softmax reader) query tile,
// ragPassW the widest span one launch of the table attends, and accMMANT
// the matrix accumulate's 8-query tiles a subgroup.
const (
	ragAccTile = 8
	ragPassW   = 4096
	accMMANT   = 2
)

// prefillAt is a layer's prefill launches: the collective-tile flash where
// it serves, else the staged three.
func (g *devTier) prefillAt(bs *blockScratch, rows, c int, mapped bool, set int) []genLaunch {
	if sg, tk, ok := g.prefillTile(bs); ok {
		return g.tilePrefill(bs, rows, mapped, set, sg, tk)
	}
	if g.flash16Serves(bs) {
		return g.flash16Prefill(bs, rows, mapped, set)
	}
	return g.stagedPrefill(bs, rows, c, mapped, set)
}

// flash16 is the binary16 m16n8 flash prefill over rows rows of bs's
// attention, on the k the device's instruction takes.
func (g *devTier) flash16(bs *blockScratch, rows int, mapped bool) compose.Flash16 {
	p := &bs.p
	return compose.Flash16{Heads: p.NHead, KVHeads: p.NKVHead, Dim: p.HeadDim, Rows: rows, Page: g.kvPage(),
		Scale: g.scoreScale(p), Softcap: p.AttnSoftcap, Map: mapped, K8: g.f16MMA() == 8}
}

// flash16Prefill is the binary16 m16n8 flash prefill's one launch.
func (g *devTier) flash16Prefill(bs *blockScratch, rows int, mapped bool, set int) []genLaunch {
	s := g.flash16(bs, rows, mapped)
	if rows%s.QRows() != 0 {
		return nil
	}
	u32 := compose.BufU32
	rb := &rowBuild{r: compose.Row{Width: s.Threads(), Rows: s.Groups()}}
	q, k, v := rb.buf(compose.BufRow, gaIn0), rb.buf(compose.BufRow, gaKPool), rb.buf(compose.BufRow, gaVPool)
	n, out := rb.buf(u32, gaCount), rb.buf(compose.BufOut, gaOut0)
	tab, desc := rb.buf(u32, gaTab), rb.buf(u32, prefillDesc(mapped, set))
	mp := -1
	if mapped {
		mp = rb.buf(u32, gaRagMap)
	}
	rb.r.PagedFlash16(s, q, k, v, n, out, tab, desc, mp)
	ls := g.launches(rb)
	for i := range ls {
		ls[i].whole = true
	}
	return ls
}

// prefillDesc is the descriptor plane a prefill reads: the chunk's or the
// run table's, the windowed one for set 1.
func prefillDesc(mapped bool, set int) genArg {
	switch {
	case !mapped && set == 1:
		return gaPreDescW
	case !mapped:
		return gaPreDesc
	case set == 1:
		return gaPRowW
	}
	return gaPRow
}

// tilePrefill is the collective-tile flash prefill's one launch: a
// workgroup a query tile of a head, its whole grid at any rows (a padding
// row's descriptor attends nothing).
func (g *devTier) tilePrefill(bs *blockScratch, rows int, mapped bool, set, sg, tk int) []genLaunch {
	p := &bs.p
	prow := prefillDesc(mapped, set)
	s := compose.FlashTile{Heads: p.NHead, KVHeads: p.NKVHead, Dim: p.HeadDim, Rows: rows, Page: g.kvPage(),
		Scale: g.scoreScale(p), Softcap: p.AttnSoftcap, Map: mapped, SG: sg, TK: tk}
	if rows%s.QRows() != 0 {
		return nil
	}
	u32 := compose.BufU32
	rb := &rowBuild{r: compose.Row{Width: s.Threads(), Rows: s.Groups()}}
	q, k, v := rb.buf(compose.BufRow, gaIn0), rb.buf(compose.BufRow, gaKPool), rb.buf(compose.BufRow, gaVPool)
	n, out := rb.buf(u32, gaCount), rb.buf(compose.BufOut, gaOut0)
	tab, desc := rb.buf(u32, gaTab), rb.buf(u32, prow)
	mp := -1
	if mapped {
		mp = rb.buf(u32, gaRagMap)
	}
	rb.r.PagedFlashTile(s, q, k, v, n, out, tab, desc, mp)
	ls := g.launches(rb)
	for i := range ls {
		ls[i].whole = true
	}
	return ls
}

// stagedPrefill is the staged prefill's three launches for a layer: the
// scores, the softmax and the accumulate into the attention's output. With
// mapped each slot's row and base come from the run table, else the rows
// are the chunk's and the base row 0's; set picks the windowed descriptors.
func (g *devTier) stagedPrefill(bs *blockScratch, rows, c int, mapped bool, set int) []genLaunch {
	p := &bs.p
	pk := bs.pkv
	sq, kt, qt := g.tiles.qtile, g.tiles.ktile, ragAccTile
	prow, pbase := gaPRow, gaRagBase
	switch {
	case !mapped && set == 1:
		prow = gaPreDescW
	case !mapped:
		prow = gaPreDesc
	case set == 1:
		prow, pbase = gaPRowW, gaRagBaseW
	}
	s := compose.Prefill{Rows: rows, Heads: p.NHead, KVHeads: p.NKVHead, Dim: p.HeadDim, Page: g.kvPage(),
		Chunk: c, Scale: g.scoreScale(p), Softcap: p.AttnSoftcap, Sink: p.AttnSinks, Map: mapped}
	vd, vpool := p.HeadDim, gaVPool
	if p.MLA() {
		// A latent row: one KV head of the latent and its key, the value
		// its prefix, both in the one pool.
		s.KVHeads, s.Dim, s.MLA = 1, p.KVLoraRank+p.NRot, p.KVLoraRank
		vd, vpool = p.KVLoraRank, gaKPool
	}
	plane := rows * p.NHead * c
	if g.genInSub && (pk.scCap < plane || pk.wtCap < plane) ||
		!g.regrow(&pk.sc, &pk.scCap, plane, 4, true, true) || !g.regrow(&pk.wt, &pk.wtCap, plane, 4, true, true) {
		return nil
	}
	// A chunk's scores, and where the device lowers the binary16 m16n8
	// instruction its accumulate too, on the matrix instruction where the
	// head and the rows take its 16.
	t := g.composeTarget()
	mmaScores := !mapped && t.API == compose.APIPTX && !g.NoMMA && !g.kb.noAttnMMA && s.MLA == 0 &&
		p.HeadDim%16 == 0 && rows%16 == 0
	mmaAcc := mmaScores && g.f16MMA() > 0 && rows%(8*accMMANT) == 0
	if mmaAcc {
		qt = 8 * accMMANT
	}
	group := lcmInt(sq, qt)
	if !mapped {
		group = rows
	}
	u32 := compose.BufU32
	tables := func(rb *rowBuild) (int, int) {
		if !mapped {
			return -1, -1
		}
		return rb.buf(u32, gaRagMap), rb.buf(u32, pbase)
	}
	var sc *rowBuild
	if mmaScores {
		nt := g.tiles.attnNT
		sc = &rowBuild{r: compose.Row{Width: 32 * s.ScoresMMAWarps(nt), Rows: 1}}
		q, k, n := sc.buf(compose.BufRow, gaIn0), sc.buf(compose.BufRow, gaKPool), sc.buf(u32, gaCount)
		out, tab, desc := sc.buf(compose.BufOut, gaScores), sc.buf(u32, gaTab), sc.buf(u32, prow)
		sc.r.PrefillScoresMMA(s, nt, q, k, n, out, tab, desc)
	} else {
		sc = &rowBuild{r: compose.Row{Width: c, Rows: rows * p.NHead, SpanGroup: group}}
		q, k := sc.buf(compose.BufRow, gaIn0), sc.buf(compose.BufRow, gaKPool)
		out, tab, desc := sc.buf(compose.BufOut, gaScores), sc.buf(u32, gaTab), sc.buf(u32, prow)
		mp, bt := tables(sc)
		sc.r.PrefillScores(s, q, k, out, tab, desc, mp, bt)
	}

	sm := &rowBuild{r: compose.Row{Width: 1, Rows: rows * p.NHead}}
	a, w, part, sdesc := sm.buf(compose.BufRow, gaScores), sm.buf(compose.BufOut, gaWts), sm.buf(compose.BufOut, gaParts), sm.buf(u32, prow)
	sink := -1
	if p.AttnSinks {
		sink = sm.buf(compose.BufRow, gaSink)
	}
	smp, sbt := tables(sm)
	sm.r.PrefillSoftmax(s, a, w, part, sdesc, sink, smp, sbt, qt)

	var ac *rowBuild
	if mmaAcc {
		mt := 4
		for mt > 1 && s.Dim%(16*mt) != 0 {
			mt /= 2
		}
		ac = &rowBuild{r: compose.Row{Width: 32 * s.AccMMAWarps(mt, accMMANT), Rows: 1}}
		aw, v, n := ac.buf(compose.BufRow, gaWts), ac.buf(compose.BufRow, vpool), ac.buf(u32, gaCount)
		aout, atab, adesc := ac.buf(compose.BufOut, gaOut0), ac.buf(u32, gaTab), ac.buf(u32, prow)
		ac.r.PrefillAccMMA(s, mt, accMMANT, aw, v, n, aout, atab, adesc)
	} else {
		ac = &rowBuild{r: compose.Row{Width: vd, Rows: rows * p.NHead, SpanGroup: group}}
		aw, v, aout := ac.buf(compose.BufRow, gaWts), ac.buf(compose.BufRow, vpool), ac.buf(compose.BufOut, gaOut0)
		atab, adesc := ac.buf(u32, gaTab), ac.buf(u32, prow)
		amp, abt := tables(ac)
		ac.r.PrefillAcc(s, aw, v, aout, atab, adesc, amp, abt)
	}

	facts := []compose.Facts{
		{Target: t, Step: compose.StepChunk, Tok: sq, RowTile: kt},
		{Target: t},
		{Target: t, Step: compose.StepChunk, Tok: qt, RowTile: 1},
	}
	if mmaScores {
		facts[0] = compose.Facts{Target: t}
	}
	if mmaAcc {
		facts[2] = compose.Facts{Target: t}
	}
	var ls []genLaunch
	for i, f := range facts {
		rb := []*rowBuild{sc, sm, ac}[i]
		kk := g.rowKernWith(&rb.r, f)
		if kk.k == nil {
			return nil
		}
		l := genLaunch{k: kk.k, groups: kk.groups, width: kk.group, n: uint8(len(rb.args))}
		copy(l.args[:], rb.args)
		ls = append(ls, l)
	}
	return ls
}

// ragAttendAt is ragAttend for this call's table, generated once a width,
// chunk and descriptor set and kept.
func (g *devTier) ragAttendAt(o *devOp, prog *devProgram, bs *blockScratch) []genLaunch {
	rr := &bs.pkv.rag
	win := false
	if l := g.curLayer; l != nil && l.windowed {
		win = true
	}
	set := 0
	if win {
		set = 1
	}
	k := attendKey{op: o.Src[0], rows: -rr.n, keys: 2*rr.chunk[set] + set}
	if ls, ok := prog.attend[k]; ok {
		return ls
	}
	ls := g.prefillAt(bs, rr.n, rr.chunk[set], true, set)
	if ls != nil {
		if prog.attend == nil {
			prog.attend = map[attendKey][]genLaunch{}
		}
		prog.attend[k] = ls
	}
	return ls
}

// prePass is a prompt chunk's staged prefill: one sequence's rows in
// position order, every padding row empty at the last real row's end in row
// 0's table, in descriptor copies of its own; the chunk the span from row
// 0's key aligned down covers.
type prePass struct {
	on    bool
	desc  [2][]uint32
	chunk [2]int
	pDesc [2]backend.Buf
	caps  [2]int
}

// preStage readies a chunk's staged prefill, or leaves it off where the
// device takes another form or a span passes the pass width. Callers hold
// g.mu, outside a submission.
func (g *devTier) preStage(bs *blockScratch, pk *pagedScratch, nrow int) {
	pp := &pk.pre
	pp.on = false
	if nrow < 1 || !g.prefillRuns(bs) || len(pk.desc) == 0 {
		return
	}
	_, _, tile := g.prefillTile(bs)
	tile = tile || g.flash16Serves(bs)
	W := kernels.PRowWords
	for i, src := range [2][]uint32{pk.desc, pk.descW} {
		pp.desc[i] = append(pp.desc[i][:0], src...)
		if len(src) == 0 {
			continue
		}
		end := src[(nrow-1)*W+kernels.PRowEnd]
		for r := nrow; r < len(src)/W; r++ {
			o := r * W
			pp.desc[i][o+kernels.PRowTab], pp.desc[i][o+kernels.PRowStart], pp.desc[i][o+kernels.PRowEnd] = src[kernels.PRowTab], end, end
		}
		span := int(end) - int(src[kernels.PRowStart]&^63)
		if span > ragPassW && !tile {
			return
		}
		c := 64
		for c < span {
			c *= 2
		}
		pp.chunk[i] = c
		if !g.regrow(&pp.pDesc[i], &pp.caps[i], len(src), 4, false, false) || !tile && !g.stagedPlanes(bs, pk, bs.rows, c) {
			return
		}
	}
	pp.on = true
}

// preAttendAt is a chunk's staged prefill launches, kept a chunk and set.
func (g *devTier) preAttendAt(o *devOp, prog *devProgram, bs *blockScratch, rows int) []genLaunch {
	pp := &bs.pkv.pre
	set := 0
	if l := g.curLayer; l != nil && l.windowed {
		set = 1
	}
	k := attendKey{op: o.Src[0], rows: -(1 << 20) - rows, keys: 2*pp.chunk[set] + set}
	if ls, ok := prog.attend[k]; ok {
		return ls
	}
	ls := g.prefillAt(bs, rows, pp.chunk[set], false, set)
	if ls != nil {
		if prog.attend == nil {
			prog.attend = map[attendKey][]genLaunch{}
		}
		prog.attend[k] = ls
	}
	return ls
}
