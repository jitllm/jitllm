package tier

import (
	"slices"

	"github.com/jitllm/jitllm/jit/gpu/backend"
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
	// The staged form's run table, on every matrix instruction the staged
	// form takes (the FMA tiles, m16n8k16 with either accumulate, sm_70's
	// m8n8k4) and past the pass width in passes; its query tile is the
	// scores' and the accumulate's both.
	staged := pf.ragBuild == nil
	if staged && pf.flash != nil {
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
			rs := &pk.ragSt[i]
			if rs.v == nil {
				continue
			}
			w(rs.buf, u32view(rs.base))
			n, nd := len(rs.base), len(rs.passDesc)/max(1, rs.passes)
			for j := 0; j < rs.passes; j++ {
				w(rs.pb[j].b, u32view(rs.passBase[j*n:(j+1)*n]))
				w(rs.pd[j].b, u32view(rs.passDesc[j*nd:(j+1)*nd]))
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
		rs := &pk.ragSt[i]
		if rs.buf != nil {
			rs.buf.Free()
			rs.buf, rs.bytes = nil, 0
		}
		for _, rb := range append(rs.pb, rs.pd...) {
			if rb.b != nil {
				rb.b.Free()
			}
		}
		rs.pb, rs.pd, rs.passes = rs.pb[:0], rs.pd[:0], 0
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
// A span past the pass width runs in passes (passes > 0): pass j's bases are
// each run's own plus j pass widths and its descriptors are the set's with
// each table row clipped to its run's pass (passBase, passDesc, and their
// buffers pb, pd), the variant the pass form's (prefillPassChunk keys a split).
type ragStaged struct {
	base  []uint32
	chunk int
	v     *prefillVariant
	buf   backend.Buf
	bytes int

	passes             int
	passBase, passDesc []uint32
	pb, pd             []ragBuf
}

// ragBuf is a run-table buffer and its size in bytes.
type ragBuf struct {
	b backend.Buf
	n int
}

// ragStagedPrep builds the staged run table's bases and chunk for each
// descriptor set (the full one and the windowed one), compiles the mapped
// variant and grows the planes. A span past the pass width takes the passes,
// as the sequence alone does (prefillPrep); false keeps the decode plan: a
// scratch built without the pass form, or a build that failed. Callers hold
// g.mu.
func (g *devTier) ragStagedPrep(bs *blockScratch, n int) bool {
	pk, pf := bs.pkv, bs.pkv.pf
	pw, pc := g.prefillPass()
	for i, desc := range [2][]uint32{pk.desc, pk.descW} {
		rs := &pk.ragSt[i]
		rs.passes = 0
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
		c, splits := 64, 0
		for c < span {
			c *= 2
		}
		if span > pw {
			if pf.passV == nil {
				// The scratch was built for a capacity under the pass width
				// (initPagedPrefill): the decode plan serves.
				return false
			}
			c, splits = pc, pw/pc
			rs.passes = (span + pw - 1) / pw
			g.ragPassStage(pk, rs, desc, pw)
		}
		rs.chunk = c
		v, err := g.ragStagedVariant(bs, pf, c, splits, n)
		if err != nil {
			g.LastErr = "ragged prefill: " + err.Error()
			return false
		}
		rs.v = v
		if !g.ragGrow(&rs.buf, &rs.bytes, 4*n) {
			return false
		}
		for j := 0; j < rs.passes; j++ {
			if len(rs.pb) <= j {
				rs.pb, rs.pd = append(rs.pb, ragBuf{}), append(rs.pd, ragBuf{})
			}
			if !g.ragGrow(&rs.pb[j].b, &rs.pb[j].n, 4*n) || !g.ragGrow(&rs.pd[j].b, &rs.pd[j].n, 4*len(desc)) {
				return false
			}
		}
		if rs.passes > 0 && !g.ragFinPrep(bs, n) {
			return false
		}
	}
	return true
}

// ragPassStage builds rs's passes' bases and descriptors from desc: pass j
// of slot i attends its row's keys in [base+j*pw, base+(j+1)*pw), base its
// run's, which leaves a run whose span ends before the pass empty there.
func (g *devTier) ragPassStage(pk *pagedScratch, rs *ragStaged, desc []uint32, pw int) {
	rs.passBase, rs.passDesc = rs.passBase[:0], rs.passDesc[:0]
	for j := 0; j < rs.passes; j++ {
		at := len(rs.passDesc)
		rs.passDesc = append(rs.passDesc, desc...)
		for slot, r := range pk.ragMap {
			lo := rs.base[slot] + uint32(j*pw)
			rs.passBase = append(rs.passBase, lo)
			if pagedFaulted("passclip") {
				continue
			}
			o := at + int(r)*kernels.PRowWords
			for _, f := range []int{kernels.PRowStart, kernels.PRowEnd} {
				rs.passDesc[o+f] = min(max(rs.passDesc[o+f], lo), lo+uint32(pw))
			}
		}
	}
}

// ragFinPrep compiles the mapped merge that finishes a passed run table:
// the running partial at each slot's row, with the sink, into the output.
func (g *devTier) ragFinPrep(bs *blockScratch, n int) bool {
	pf := bs.pkv.pf
	if pf.ragFin[n] != nil {
		return true
	}
	ms := mergeShape(kernels.FlashShape{Heads: bs.p.NHead, KVHeads: bs.pkv.shape.KVHeads, Dim: bs.pkv.shape.Dim,
		MLA: bs.pkv.shape.MLA, Rows: n, Splits: 1})
	ms.Sink, ms.Map = bs.p.AttnSinks, true
	k, err := kernels.FlashAttentionMergeWide(ms)
	var c backend.Kernel
	if err == nil {
		c, err = g.compileShared(k)
	}
	if err != nil {
		g.LastErr = "ragged prefill: " + err.Error()
		return false
	}
	if pf.ragFin == nil {
		pf.ragFin = map[int]backend.Kernel{}
	}
	pf.ragFin[n] = c
	return true
}

// ragGrow makes *b hold at least n bytes; the run table's buffers are a few
// words a row, allocated as the recurrent seats' zeros are, outside the
// budget.
func (g *devTier) ragGrow(b *backend.Buf, have *int, n int) bool {
	if *b != nil && *have >= n {
		return true
	}
	if *b != nil {
		(*b).Free()
		*b, *have = nil, 0
	}
	nb, err := g.allocBytes(make([]byte, n))
	if err != nil {
		g.LastErr = "ragged prefill: " + err.Error()
		return false
	}
	*b, *have = nb, n
	return true
}

// ragStagedVariant is the staged kernels with the run table at chunk keys
// a split, splits splits (0: the direct form) and n slots, on the set's
// matrix instruction, compiled once; the planes grow to its size.
func (g *devTier) ragStagedVariant(bs *blockScratch, pf *pagedPrefill, chunk, splits, n int) (*prefillVariant, error) {
	key := [3]int{chunk, splits, n}
	if v, ok := pf.ragVars[key]; ok {
		return v, nil
	}
	s := bs.pkv.shape
	s.Rows, s.Chunk, s.Splits, s.Map = n, chunk, splits, true
	s.Group, s.Warps, s.KImm = 0, 0, false
	s.Sink = bs.p.AttnSinks && splits == 0
	v, err := g.buildVariant(pf, s)
	if err != nil {
		return nil, err
	}
	if pf.ragVars == nil {
		pf.ragVars = map[[3]int]*prefillVariant{}
	}
	pf.ragVars[key] = v
	return v, nil
}

// ragStagedAttend launches the staged run table over the table's rows: the
// scores, the softmax (with the sink) and the accumulate, which writes out;
// with passes each pass writes its partials at the slots' rows, folded into
// the running partial as the sequence alone folds them (prefillAttn), and the
// mapped merge writes out. win picks the windowed descriptor set's bases and
// variant; own says desc is the set's own, which the passes clip -- the
// "ragtab" violation hands every pass the emptied set as it is.
func (pk *pagedScratch) ragStagedAttend(lc *launcher, q, k, v, n, out, tab, desc, sink backend.Buf, win, own bool) error {
	pf := pk.pf
	rs := &pk.ragSt[0]
	if win {
		rs = &pk.ragSt[1]
	}
	vr := rs.v
	one := func(desc, base, dst backend.Buf) error {
		lc.la(vr.scores, vr.scoreT, q, k, n, pf.sc, tab, desc, pk.pMap, base)
		args := []backend.Buf{pf.sc, n, pf.wt, pf.part, tab, desc}
		if vr.s.Sink {
			args = append(args, sink)
		}
		args = append(args, pk.pMap, base)
		if err := lc.launch(vr.soft, vr.softG, vr.softW, args...); err != nil {
			return err
		}
		lc.la(vr.acc, vr.accT, pf.wt, v, n, dst, tab, desc, pk.pMap, base)
		return nil
	}
	if rs.passes == 0 {
		return one(desc, rs.buf, out)
	}
	// The fold runs over the scratch's rows (the partials are at the rows),
	// as the pass form's merges were built.
	ps := pf.passV.s
	vd := mergeShape(ps).Dim
	rg := ps.Rows * ps.Heads * ((vd + 31) / 32 * 32)
	for j := 0; j < rs.passes; j++ {
		d := rs.pd[j].b
		if !own {
			d = desc
		}
		if err := one(d, rs.pb[j].b, pf.part); err != nil {
			return err
		}
		if j == 0 {
			lc.la(pf.mFirst, rg, pf.part, pf.run[0])
		} else {
			lc.la(pf.mFold, rg, pf.part, pf.run[(j+1)%2], pf.run[j%2])
		}
	}
	fin := []backend.Buf{pf.run[(rs.passes+1)%2], out}
	if sink != nil {
		fin = append(fin, sink)
	}
	fin = append(fin, pk.pMap)
	lc.la(pf.ragFin[pk.ragRows], pk.ragRows*ps.Heads*vd, fin...)
	return nil
}

// ragKey is what a recording keys the ragged call's run table by: the
// table's width, whether the decode plan runs beside it, and the staged
// form's chunk and passes for each descriptor set -- each picks a launch
// sequence.
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
				k = (k*8191+rs.chunk)*31 + rs.passes
			}
		}
	}
	return k
}

// ragPasses is how many passes the staged run table's descriptor set win
// runs, 0 for one launch.
func (pk *pagedScratch) ragPasses(win bool) int {
	if win {
		return pk.ragSt[1].passes
	}
	return pk.ragSt[0].passes
}
