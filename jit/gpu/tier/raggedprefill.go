package tier

import (
	"slices"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// A ragged step's prompt chunks attend as a prefill
// (docs/design/ragged-prefill.md). A step's rows are several sequences', and
// a prompt chunk among them used to attend through the decode plan a row at a
// time, every row reading its sequence's whole history on its own. Here each
// sequence with more than one row in the step gets its rows into a run
// table, in position order and padded to whole workgroups with its last row,
// and one launch of the flash prefill's run-table form (kernels'
// FlashPrefill70Shape.Map) attends them all, q and the output at the step's
// own rows. The decoding rows -- one row of their sequence -- keep the decode
// plan, run first over descriptors in which the chunk rows are empty, and the
// table's launch then writes the chunk rows over whatever that left.
//
// The table is data the launch reads (pMap), staged with the descriptors, so
// a recording serves every step of the same padded width (graphKey.ragMap).

// ragTable is a ragged call's run table: the step rows of each sequence with
// more than one row, padded per workgroup. Its slot count is a multiple of wg.
// alone reports that every real row is in it -- no sequence has a single
// row, which the decode plan would take -- and then the call's padding rows
// are in it too, a run of their own whose descriptors attend nothing, so the
// table's launch writes every row and the decode plan need not run. It is
// built into dst and returned; dst is reused, so a step allocates nothing
// once the widest table has been made.
func ragTable(dst []uint32, rows []pagedRow, wg int, order []int) (_ []uint32, _ []int, alone bool) {
	dst = dst[:0]
	order = order[:0]
	for r, row := range rows {
		if !row.pad {
			order = append(order, r)
		}
	}
	// By sequence, then position: each sequence's rows a contiguous run.
	slices.SortFunc(order, func(a, b int) int {
		ra, rb := rows[a], rows[b]
		if ra.s != rb.s {
			if ra.s.sid != rb.s.sid {
				if ra.s.sid < rb.s.sid {
					return -1
				}
				return 1
			}
			return ra.s.base - rb.s.base
		}
		return ra.pos - rb.pos
	})
	alone = true
	for i := 0; i < len(order); {
		j := i + 1
		for j < len(order) && rows[order[j]].s == rows[order[i]].s {
			j++
		}
		if j-i > 1 {
			for k := 0; k < (j-i+wg-1)/wg*wg; k++ {
				dst = append(dst, uint32(order[min(i+k, j-1)]))
			}
		} else {
			alone = false
		}
		i = j
	}
	if alone && len(dst) > 0 {
		first, n := len(dst), 0
		for r, row := range rows {
			if row.pad {
				dst = append(dst, uint32(r))
				n++
			}
		}
		for n > 0 && (len(dst)-first)%wg != 0 {
			dst = append(dst, dst[len(dst)-1])
		}
	}
	return dst, order, alone
}

// ragPrep readies bs's ragged call for the run-table launch, or reports false
// (the decode plan takes every row) where the call has no run of two rows or
// the device has no run-table form: the flash prefill must be one with a
// table (FlashPrefill80 and its m16n8k8 twin, FlashPrefillTile), with no
// sinks and no streamed history. It builds the table and the decode descriptors with
// the table's rows emptied, compiles the launch at the table's width, and
// grows the table's buffer. Callers hold g.mu and have run pagedStage.
func (g *devTier) ragPrep(bs *blockScratch) bool {
	pk := bs.pkv
	pk.ragRows = 0
	pf := pk.pf
	if pf == nil || pk.st != nil || g.NoRagPrefill {
		return false
	}
	// The staged form's run table: the FMA kernels (prefillTiled), the
	// direct form; its query tile is the scores' and the accumulate's both.
	staged := pf.ragBuild == nil
	if staged && (pf.flash != nil || pf.mode != prefillTiled) {
		return false
	}
	wg := pf.ragWG
	if staged {
		wg = lcm(pf.sq, pf.qt)
	}
	pk.ragMap, pk.ragOrder, pk.ragAlone = ragTable(pk.ragMap, pk.rows, wg, pk.ragOrder)
	if len(pk.ragMap) == 0 {
		return false
	}
	n := len(pk.ragMap)
	pk.ragStaged = staged
	if staged {
		if !g.ragStagedPrep(bs, n) {
			return false
		}
	} else if !g.ragFlashPrep(pf, n) {
		return false
	}
	if pk.pMapBytes < 4*n {
		if pk.pMap != nil {
			pk.pMap.Free()
		}
		b, err := g.allocBytes(make([]byte, 4*n))
		if err != nil {
			pk.pMap, pk.pMapBytes = nil, 0
			g.LastErr = "ragged prefill: " + err.Error()
			return false
		}
		pk.pMap, pk.pMapBytes = b, 4*n
	}
	return g.ragDecDescs(pk, n)
}

// ragFlashPrep compiles the flash form's run-table launch at n slots, and
// with sinks its mapped merge.
func (g *devTier) ragFlashPrep(pf *pagedPrefill, n int) bool {
	if pf.ragK == nil {
		pf.ragK = map[int]backend.Kernel{}
	}
	if pf.ragK[n] == nil {
		fs := pf.fs
		fs.Map, fs.Rows = true, n
		k, err := pf.ragBuild(fs)
		if err != nil {
			g.LastErr = "ragged prefill: " + err.Error()
			return false
		}
		c, err := g.compileShared(k)
		if err != nil {
			g.LastErr = "ragged prefill: " + err.Error()
			return false
		}
		pf.ragK[n] = c
	}
	// Sinks: the table's launch writes each row's one split's partials, and
	// the wide merge takes the same table to add the sink and write the
	// rows (FlashShape.Map).
	if pf.fs.Splits > 0 && pf.ragMerge[n] == nil {
		ms := kernels.FlashShape{Heads: pf.fs.Heads, KVHeads: pf.fs.KVHeads, Dim: pf.fs.Dim, Rows: n,
			Splits: pf.fs.Splits, Sink: true, Map: true}
		k, err := kernels.FlashAttentionMergeWide(ms)
		if err != nil {
			g.LastErr = "ragged prefill: " + err.Error()
			return false
		}
		c, err := g.compileShared(k)
		if err != nil {
			g.LastErr = "ragged prefill: " + err.Error()
			return false
		}
		if pf.ragMerge == nil {
			pf.ragMerge = map[int]backend.Kernel{}
		}
		pf.ragMerge[n] = c
	}
	return true
}

// ragDecDescs builds the decode plan's descriptors with the table's rows
// emptied, and n slots' worth of nothing else. Callers hold g.mu.
func (g *devTier) ragDecDescs(pk *pagedScratch, n int) bool {
	for _, dec := range []struct {
		buf  *backend.Buf
		src  []uint32
		dst  *[]uint32
		size *int
	}{{&pk.pRowDec, pk.desc, &pk.descDec, &pk.pRowDecBytes}, {&pk.pRowDecW, pk.descW, &pk.descDecW, &pk.pRowDecWBytes}} {
		if len(dec.src) == 0 {
			*dec.dst = (*dec.dst)[:0]
			continue
		}
		d := append((*dec.dst)[:0], dec.src...)
		// Empty at the row's own end: the decode plan attends nothing there,
		// and the table's launch writes the row.
		for _, r := range pk.ragMap {
			w := int(r) * kernels.PRowWords
			d[w+kernels.PRowStart] = d[w+kernels.PRowEnd]
		}
		*dec.dst = d
		if *dec.size < 4*len(d) {
			if *dec.buf != nil {
				(*dec.buf).Free()
			}
			b, err := g.allocBytes(make([]byte, 4*len(d)))
			if err != nil {
				*dec.buf, *dec.size = nil, 0
				g.LastErr = "ragged prefill: " + err.Error()
				return false
			}
			*dec.buf, *dec.size = b, 4*len(d)
		}
	}
	pk.ragRows = n
	return true
}

// ragWrites stages the run table and the decode descriptors (ragPrep's).
func (pk *pagedScratch) ragWrites(w func(backend.Buf, []byte)) {
	if pk.ragRows == 0 {
		return
	}
	w(pk.pMap, u32view(pk.ragMap))
	w(pk.pRowDec, u32view(pk.descDec))
	if len(pk.descDecW) > 0 {
		w(pk.pRowDecW, u32view(pk.descDecW))
	}
	if pk.ragStaged {
		for i := range pk.ragSt {
			if rs := &pk.ragSt[i]; rs.v != nil {
				w(rs.buf, u32view(rs.base))
			}
		}
	}
}

// ragAttend launches the run table's flash prefill over the chunk rows,
// through desc (the full or the windowed set).
// With sinks (one split) it writes the partials and the mapped merge adds
// the sink and writes the rows.
func (pk *pagedScratch) ragAttend(lc *launcher, q, k, v, n, out, tab, desc, sink backend.Buf) error {
	pf := pk.pf
	fs := pf.fs
	fs.Map, fs.Rows = true, pk.ragRows
	if fs.Splits == 0 {
		return lc.launch(pf.ragK[pk.ragRows], pf.ragGroups(fs), pf.ragThreads, q, k, v, n, out, tab, desc, pk.pMap)
	}
	if err := lc.launch(pf.ragK[pk.ragRows], pf.ragGroups(fs), pf.ragThreads, q, k, v, n, pf.part, tab, desc, pk.pMap); err != nil {
		return err
	}
	lc.la(pf.ragMerge[pk.ragRows], pk.ragRows*fs.Heads*fs.Dim, pf.part, out, sink, pk.pMap)
	return nil
}

// ragFree gives back the run table's buffers. They are a few words a row,
// allocated as the recurrent seats' zeros are, outside the budget.
func (pk *pagedScratch) ragFree() {
	for _, b := range []*backend.Buf{&pk.pMap, &pk.pRowDec, &pk.pRowDecW} {
		if *b != nil {
			(*b).Free()
			*b = nil
		}
	}
	pk.pMapBytes, pk.pRowDecBytes, pk.pRowDecWBytes = 0, 0, 0
	for i := range pk.ragSt {
		if rs := &pk.ragSt[i]; rs.buf != nil {
			rs.buf.Free()
			rs.buf, rs.bytes = nil, 0
		}
	}
}

// lcm is the least common multiple of two positive tiles.
func lcm(a, b int) int {
	x, y := a, b
	for y != 0 {
		x, y = y, x%y
	}
	return a / x * b
}

// ragStaged is one descriptor set's staged run table: each slot's partition
// base (its run's first key, 64-aligned) and the chunk every run's span fits,
// the variant at that chunk and the table's width, and the bases' buffer.
type ragStaged struct {
	base  []uint32
	chunk int
	v     *prefillVariant
	buf   backend.Buf
	bytes int
}

// ragStagedPrep builds the staged run table's bases and chunk for each
// descriptor set (the full one and the windowed one), compiles the mapped
// variant and grows the planes. False keeps the decode plan: a span past the
// pass width (the staged run table is the direct form's), or a build that
// failed. Callers hold g.mu.
func (g *devTier) ragStagedPrep(bs *blockScratch, n int) bool {
	pk, pf := bs.pkv, bs.pkv.pf
	pw, _ := g.prefillPass()
	for i, desc := range [2][]uint32{pk.desc, pk.descW} {
		rs := &pk.ragSt[i]
		if len(desc) == 0 {
			rs.v = nil
			continue
		}
		rs.base = rs.base[:0]
		span := 1
		var cur uint32
		for slot, r := range pk.ragMap {
			row := pk.rows[r]
			// A run's first slot sets its base: the slot before it is
			// another sequence's (or none).
			if slot == 0 || pk.rows[pk.ragMap[slot-1]].s != row.s || pk.rows[pk.ragMap[slot-1]].pad != row.pad {
				cur = desc[int(r)*kernels.PRowWords+kernels.PRowStart] &^ 63
			}
			rs.base = append(rs.base, cur)
			if end := int(desc[int(r)*kernels.PRowWords+kernels.PRowEnd]); end > int(cur) {
				span = max(span, end-int(cur))
			}
		}
		if span > pw {
			return false
		}
		c := 64
		for c < span {
			c *= 2
		}
		rs.chunk = c
		v, err := g.ragStagedVariant(bs, pf, c, n)
		if err != nil {
			g.LastErr = "ragged prefill: " + err.Error()
			return false
		}
		rs.v = v
		if rs.bytes < 4*n {
			if rs.buf != nil {
				rs.buf.Free()
			}
			b, err := g.allocBytes(make([]byte, 4*n))
			if err != nil {
				rs.buf, rs.bytes = nil, 0
				g.LastErr = "ragged prefill: " + err.Error()
				return false
			}
			rs.buf, rs.bytes = b, 4*n
		}
	}
	return true
}

// ragStagedVariant is the staged kernels with the run table at chunk keys
// and n slots, the direct form, compiled once; the planes grow to its size.
func (g *devTier) ragStagedVariant(bs *blockScratch, pf *pagedPrefill, chunk, n int) (*prefillVariant, error) {
	key := [2]int{chunk, n}
	if v, ok := pf.ragVars[key]; ok {
		return v, nil
	}
	s := bs.pkv.shape
	s.Rows, s.Chunk, s.Splits, s.Map = n, chunk, 0, true
	s.Group, s.Warps, s.KImm = 0, 0, false
	s.Sink = bs.p.AttnSinks
	v := &prefillVariant{s: s}
	ks, err := kernels.PagedAttnScoresTiled(s, pf.sq, pf.kt)
	var kf, ka *ir.Kernel
	if err == nil {
		kf, err = kernels.PagedPrefillSoftmax(s, pf.lanes, pf.qt)
	}
	if err == nil {
		ka, err = kernels.PagedAttnAccTiled(s, pf.qt)
	}
	if err != nil {
		return nil, err
	}
	v.scoreT = kernels.PagedAttnScoresTiledThreads(s, pf.sq, pf.kt)
	v.accT = kernels.PagedAttnAccTiledThreads(s, pf.qt)
	v.softG, v.softW = kernels.PagedPrefillSoftmaxGroups(s, pf.lanes), kernels.PagedPrefillSoftmaxWidth(pf.lanes)
	for i, k := range []*ir.Kernel{ks, kf, ka} {
		c, err := g.compileShared(k)
		if err != nil {
			v.close()
			return nil, err
		}
		*[]*backend.Kernel{&v.scores, &v.soft, &v.acc}[i] = c
	}
	np := kernels.PagedPrefillPlane(s)
	have := pf.planeFloats
	if err := g.prefillBuf(&pf.sc, &have, np); err != nil {
		v.close()
		return nil, err
	}
	have = pf.planeFloats
	if err := g.prefillBuf(&pf.wt, &have, np); err != nil {
		v.close()
		return nil, err
	}
	pf.planeFloats = max(pf.planeFloats, np)
	if pf.ragVars == nil {
		pf.ragVars = map[[2]int]*prefillVariant{}
	}
	pf.ragVars[key] = v
	return v, nil
}

// ragStagedAttend launches the staged run table over the table's rows: the
// scores, the softmax (with the sink) and the accumulate, which writes out.
// win picks the windowed descriptor set's bases and variant.
func (pk *pagedScratch) ragStagedAttend(lc *launcher, q, k, v, n, out, tab, desc, sink backend.Buf, win bool) error {
	pf := pk.pf
	rs := &pk.ragSt[0]
	if win {
		rs = &pk.ragSt[1]
	}
	vr := rs.v
	lc.la(vr.scores, vr.scoreT, q, k, n, pf.sc, tab, desc, pk.pMap, rs.buf)
	// A fixed array, so a warm step's argument list allocates nothing.
	var argv [9]backend.Buf
	args := append(argv[:0], pf.sc, n, pf.wt, pf.part, tab, desc)
	if vr.s.Sink {
		args = append(args, sink)
	}
	args = append(args, pk.pMap, rs.buf)
	err := lc.launch(vr.soft, vr.softG, vr.softW, args...)
	clear(args)
	if err != nil {
		return err
	}
	lc.la(vr.acc, vr.accT, pf.wt, v, n, out, tab, desc, pk.pMap, rs.buf)
	return nil
}

// ragKey is what a recording keys the ragged call's run table by: the
// table's width, whether the decode plan runs beside it, and the staged
// form's chunk for each descriptor set -- each picks a launch sequence.
func (pk *pagedScratch) ragKey() int {
	if pk.ragRows == 0 {
		return 0
	}
	k := pk.ragRows << 1
	if pk.ragAlone {
		k |= 1
	}
	if pk.ragStaged {
		for i := range pk.ragSt {
			if rs := &pk.ragSt[i]; rs.v != nil {
				k = k*8191 + rs.chunk
			}
		}
	}
	return k
}
