package tier

import (
	"slices"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/compose"
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
	if g.composeTarget().API == compose.APIPTX {
		return false
	}
	if g.tilesOK() {
		return !flashTileServes(bs.p.HeadDim, bs.rows)
	}
	ok, _ := backend.LowersIntMMA(g.dev)
	return !ok
}

// flashTileServes reports whether a collective-tile flash prefill serves a
// head width over rows rows: the blocking with the most query rows, then the
// most keys, whose workgroup memory -- binary16 K and float32 V a key tile,
// the scores and the output a query tile -- fits 32 KiB and whose staging
// divides among its threads, must be four subgroups of rows that tile the
// chunk, its key tile a divisor of 32.
func flashTileServes(hd, rows int) bool {
	for _, t := range [][2]int{{4, 32}, {4, 16}, {2, 32}, {2, 16}, {2, 8}, {1, 32}, {1, 16}} {
		sg, tk := t[0], t[1]
		qb, threads := 8*sg, 32*sg
		mem := tk*(hd+8)*2 + tk*(hd+4)*4 + qb*(tk+4)*4 + qb*(hd+4)*4
		if mem <= 32<<10 && (tk*hd/2)%threads == 0 && (tk*hd/4)%threads == 0 {
			return sg >= 4 && rows%qb == 0 && 32%tk == 0
		}
	}
	return false
}

// ragStage builds the call's run table from its rows and descriptors, or
// leaves it empty where some real row is its sequence's only one, a span
// passes the pass width, or the device takes another form. Callers hold
// g.mu, outside a submission.
func (g *devTier) ragStage(bs *blockScratch, pk *pagedScratch) {
	rr := &pk.rag
	rr.n = 0
	if !g.ragStaged(bs) || len(pk.desc) == 0 {
		return
	}
	rr.sq, rr.kt, rr.qt = g.tiles.qtile, g.tiles.ktile, ragAccTile
	wg := lcmInt(rr.sq, rr.qt)
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
		if span > ragPassW {
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
	if !g.regrow(&rr.pMap, &rr.mapCap, n, 4, false, false) {
		return
	}
	rr.n = n
}

// ragAccTile is the run table's accumulate (and softmax reader) query tile,
// and ragPassW the widest span one launch of the table attends.
const (
	ragAccTile = 8
	ragPassW   = 4096
)

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
		Chunk: c, Scale: float32(p.Scale()), Softcap: p.AttnSoftcap, Sink: p.AttnSinks, Map: mapped}
	plane := rows * p.NHead * c
	if g.genInSub && (pk.scCap < plane || pk.wtCap < plane) ||
		!g.regrow(&pk.sc, &pk.scCap, plane, 4, true, true) || !g.regrow(&pk.wt, &pk.wtCap, plane, 4, true, true) {
		return nil
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
	sc := &rowBuild{r: compose.Row{Width: c, Rows: rows * p.NHead, SpanGroup: group}}
	q, k := sc.buf(compose.BufRow, gaIn0), sc.buf(compose.BufRow, gaKPool)
	out, tab, desc := sc.buf(compose.BufOut, gaScores), sc.buf(u32, gaTab), sc.buf(u32, prow)
	mp, bt := tables(sc)
	sc.r.PrefillScores(s, q, k, out, tab, desc, mp, bt)

	sm := &rowBuild{r: compose.Row{Width: 1, Rows: rows * p.NHead}}
	a, w, part, sdesc := sm.buf(compose.BufRow, gaScores), sm.buf(compose.BufOut, gaWts), sm.buf(compose.BufOut, gaParts), sm.buf(u32, prow)
	sink := -1
	if p.AttnSinks {
		sink = sm.buf(compose.BufRow, gaSink)
	}
	smp, sbt := tables(sm)
	sm.r.PrefillSoftmax(s, a, w, part, sdesc, sink, smp, sbt, qt)

	ac := &rowBuild{r: compose.Row{Width: p.HeadDim, Rows: rows * p.NHead, SpanGroup: group}}
	aw, v, aout := ac.buf(compose.BufRow, gaWts), ac.buf(compose.BufRow, gaVPool), ac.buf(compose.BufOut, gaOut0)
	atab, adesc := ac.buf(u32, gaTab), ac.buf(u32, prow)
	amp, abt := tables(ac)
	ac.r.PrefillAcc(s, aw, v, aout, atab, adesc, amp, abt)

	t := g.composeTarget()
	var ls []genLaunch
	for i, f := range []compose.Facts{
		{Target: t, Step: compose.StepChunk, Tok: sq, RowTile: kt},
		{Target: t},
		{Target: t, Step: compose.StepChunk, Tok: qt, RowTile: 1},
	} {
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
	ls := g.stagedPrefill(bs, rr.n, rr.chunk[set], true, set)
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
	if nrow < 1 || !g.ragStaged(bs) || len(pk.desc) == 0 {
		return
	}
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
		if span > ragPassW {
			return
		}
		c := 64
		for c < span {
			c *= 2
		}
		pp.chunk[i] = c
		if !g.regrow(&pp.pDesc[i], &pp.caps[i], len(src), 4, false, false) {
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
	ls := g.stagedPrefill(bs, rows, pp.chunk[set], false, set)
	if ls != nil {
		if prog.attend == nil {
			prog.attend = map[attendKey][]genLaunch{}
		}
		prog.attend[k] = ls
	}
	return ls
}
