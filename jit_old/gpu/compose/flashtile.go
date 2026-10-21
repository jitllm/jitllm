package compose

import (
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// FlashTile is a prompt chunk's paged attention on the collective 8x8
// tiles: SG subgroups of eight query rows a workgroup, TK keys a trip, the
// scores never leaving the workgroup. Q and K go through workgroup memory as
// binary16, V and the probabilities as float32; four lanes a row take the
// running maximum and sum, and the output lives in workgroup memory, scaled
// by exp(m_old - m_new) before each trip's P V accumulates into it.
type FlashTile struct {
	Heads, KVHeads, Dim, Rows int
	Page                      int
	Scale, Softcap            float32
	Window                    int
	// Splits > 0 cuts a workgroup's key tiles into that many runs, one
	// workgroup each, writing the merge's partials; 0 writes the rows.
	Splits int
	// Map reads each query slot's row from a run table (the last buffer).
	Map    bool
	SG, TK int
}

// QRows is a workgroup's query rows; Threads its width.
func (s FlashTile) QRows() int   { return 8 * s.SG }
func (s FlashTile) Threads() int { return 32 * s.SG }

// Groups is the workgroups a launch runs.
func (s FlashTile) Groups() int { return s.Heads * (s.Rows / s.QRows()) * max(1, s.Splits) }

// FlashTileMem is the workgroup memory a blocking takes at head width hd:
// binary16 K rows padded by eight halves, float32 V rows padded by four, the
// scores (the probabilities over them) and the output as float32.
func FlashTileMem(hd, sg, tk int) int {
	qb := 8 * sg
	return tk*(hd+8)*2 + tk*(hd+4)*4 + qb*(tk+4)*4 + qb*(hd+4)*4
}

// PagedFlashTile is s over the group (Threads() elements, Groups() rows, a
// row a workgroup): q the rows' query heads, k and v the paged pools ([dim]
// [page] and [page][dim] a page), n unread, out the rows or the partials,
// tab the page ids, desc the row descriptors, mp the run table (-1 without
// Map).
func (g *Row) PagedFlashTile(s FlashTile, q, k, v, n, out, tab, desc, mp int) {
	u, ub := g.UConst, g.UBin
	add := func(a, b int) int { return ub(OpAdd, a, b) }
	addc := func(a, c int) int {
		if c == 0 {
			return a
		}
		return add(a, u(c))
	}
	mul := func(a, c int) int { return ub(OpMul, a, u(c)) }
	div := func(a, c int) int { return ub(OpDiv, a, u(c)) }
	rem := func(a, c int) int { return ub(OpRem, a, u(c)) }
	umin, umax, sub := func(a, b int) int { return ub(OpMin, a, b) }, func(a, b int) int { return ub(OpMax, a, b) },
		func(a, b int) int { return ub(OpSub, a, b) }
	f := g.Const
	hd, qb, tk, nth := s.Dim, s.QRows(), s.TK, s.Threads()
	P, S := s.Page, max(1, s.Splits)
	gqa := s.Heads / s.KVHeads
	kvDim := s.KVHeads * hd
	ks, vs, ss, os := hd+8, hd+4, tk+4, hd+4
	kpl, dpl := tk/4, hd/4
	negInf, zero := g.bits(u(0xFF800000)), f(0)
	shK := g.Shared(tk*ks/2, true)
	shV := g.Shared(tk*vs, true)
	shS := g.Shared(qb*ss, false)
	shO := g.Shared(qb*os, true)
	rowOf := func(slot int) int { return slot }
	if s.Map {
		rowOf = func(slot int) int { return g.LoadAt(mp, slot) }
	}
	descAt := func(row, w int) int { return g.LoadAt(desc, addc(mul(row, kernels.PRowWords), w)) }

	tid, wg := g.Column(), g.RowIdx()
	spl := rem(wg, S)
	wg = div(wg, S)
	h := rem(wg, s.Heads)
	kvBase := mul(div(h, gqa), hd)
	row0 := mul(div(wg, s.Heads), qb)
	first, last := rowOf(row0), rowOf(addc(row0, qb-1))
	lo, hi, tabOff := descAt(first, kernels.PRowStart), descAt(last, kernels.PRowEnd), descAt(first, kernels.PRowTab)
	live := g.Less(lo, hi)
	a := mul(div(lo, 32), 32)
	total := g.Select(live, div(add(sub(hi, a), u(tk-1)), tk), u(0))
	kStart, ntiles := a, total
	if S > 1 {
		per := div(addc(total, S-1), S)
		t0 := umin(ub(OpMul, spl, per), total)
		kStart, ntiles = add(a, mul(t0, tk)), sub(umin(add(t0, per), total), t0)
	}

	// The queries as binary16 in the output's memory, each subgroup's hd/8
	// tiles into registers, then the output cleared.
	tok := -1
	for it := 0; it < qb*hd/2/nth; it++ {
		w := addc(tid, nth*it)
		qi, p := div(w, hd/2), rem(w, hd/2)
		src := add(mul(add(mul(rowOf(add(row0, qi)), s.Heads), h), hd), ub(OpShl, p, u(1)))
		e0, e1 := g.LoadVec(q, src, 2, 0), g.LoadVec(q, src, 2, 1)
		tok = g.Write(shO, add(mul(qi, ks/2), p), g.Bin(OpPackF16, e0, e1), tok)
	}
	tok = g.Barrier(tok)
	sg := g.Subgroup()
	aT := ir.TileType{Rows: 8, Cols: 8, Elem: ir.TileF16, Use: ir.TileA}
	bT := ir.TileType{Rows: 8, Cols: 8, Elem: ir.TileF16, Use: ir.TileB}
	aP := ir.TileType{Rows: 8, Cols: 8, Elem: ir.TileF32, Use: ir.TileA}
	bV := ir.TileType{Rows: 8, Cols: 8, Elem: ir.TileF32, Use: ir.TileB}
	cT := ir.TileType{Rows: 8, Cols: 8, Elem: ir.TileF32, Use: ir.TileAcc}
	sgQ := mul(sg, 8*ks)
	qt := make([]int, hd/8)
	for dk := range qt {
		qt[dk] = g.TileLoad(aT, shO, addc(sgQ, 8*dk), u(ks), false, tok)
		tok = qt[dk]
	}
	tok = g.Barrier(tok)
	for it := 0; it < qb*os/nth; it++ {
		tok = g.Write(shO, addc(tid, nth*it), u(0), tok)
	}

	lane := ub(OpAnd, tid, u(31))
	rl, part := ub(OpShr, lane, u(2)), ub(OpAnd, lane, u(3))
	row := add(mul(sg, 8), rl)
	rrow := rowOf(add(row0, row))
	kst, cnt := descAt(rrow, kernels.PRowStart), descAt(rrow, kernels.PRowEnd)
	var winLo int
	if s.Window > 0 {
		winLo = sub(cnt, umin(cnt, u(s.Window)))
	}
	keyL, dimL := mul(part, kpl), mul(part, dpl)
	sRow, oRow := add(mul(row, ss), keyL), add(mul(row, os), dimL)
	sgS, sgO := mul(sg, 8*ss), mul(sg, 8*os)
	tok = g.Barrier(tok)
	var kOff, vOff []int
	for it := 0; it < tk*hd/2/nth; it++ {
		w := addc(tid, nth*it)
		key, dp := rem(w, tk), div(w, tk)
		kOff = append(kOff, add(mul(add(kvBase, ub(OpShl, dp, u(1))), P), key))
	}
	for it := 0; it < tk*hd/4/nth; it++ {
		w := addc(tid, nth*it)
		dq, key := rem(w, hd/4), div(w, hd/4)
		vOff = append(vOff, add(mul(key, kvDim), add(kvBase, ub(OpShl, dq, u(2)))))
	}
	kspan, hlo := sub(cnt, kst), sub(hi, lo)

	// The run's tiles a page at a time: the page's id read once, the tiles
	// inside it needing no lookup.
	kEnd := add(kStart, mul(ntiles, tk))
	pg0 := div(kStart, P)
	npg := g.Select(g.Less(u(0), ntiles), addc(sub(div(sub(kEnd, u(1)), P), pg0), 1), u(0))
	cp := g.LoopN(npg)
	mo, lo0, pg := g.Fold(cp, negInf), g.Fold(cp, zero), g.Fold(cp, pg0)
	t0 := umax(kStart, mul(pg, P))
	pid := g.LoadAt(tab, add(tabOff, div(t0, P)))
	t1 := umin(kEnd, mul(addc(pg, 1), P))
	ct := g.LoopN(div(sub(t1, t0), tk))
	m, l, k0 := g.Fold(ct, mo), g.Fold(ct, lo0), g.Fold(ct, t0)
	off := rem(k0, P)
	kTile := add(mul(pid, P*kvDim), off)
	vTile := mul(add(mul(pid, P), off), kvDim)

	// K and V staged, after the last trip's products read them.
	bt := g.Barrier(g.After(k0, tok))
	for it := range kOff {
		w := addc(tid, nth*it)
		key, dp := rem(w, tk), div(w, tk)
		e := add(kTile, kOff[it])
		klo, khi := g.LoadAt(k, e), g.LoadAt(k, addc(e, P))
		bt = g.Write(shK, add(mul(key, ks/2), dp), g.Bin(OpPackF16, klo, khi), bt)
	}
	for it := range vOff {
		w := addc(tid, nth*it)
		dq, key := rem(w, hd/4), div(w, hd/4)
		ok := g.Less(sub(add(k0, key), lo), hlo)
		at := add(vTile, vOff[it])
		var vw [4]int
		for x := range vw {
			vw[x] = g.word(g.Select(ok, g.LoadVec(v, at, 4, x), zero))
		}
		bt = g.WriteVec(shV, add(mul(key, vs), ub(OpShl, dq, u(2))), bt, vw[:]...)
	}
	bt = g.Barrier(bt)

	// S = Q K^T for the subgroup's eight rows.
	for j := 0; j < tk/8; j++ {
		acc := g.TileSplat(cT, g.After(zero, bt))
		for dk := 0; dk < hd/8; dk++ {
			kb := g.TileLoad(bT, shK, u(8*j*ks+8*dk), u(ks), true, bt)
			acc = g.TileMMA(qt[dk], kb, acc)
		}
		bt = g.WriteTile(shS, addc(sgS, 8*j), u(ss), acc, cT, false, bt)
	}
	bt = g.Barrier(bt)

	// The transform, the running maximum and sum, P over the scores, O
	// rescaled.
	val := make([]int, kpl)
	mx := negInf
	for i := range val {
		x := g.Bin(OpMul, g.LoadAt(shS, g.After(addc(sRow, i), bt)), f(s.Scale))
		if s.Softcap > 0 {
			e := g.exp(g.Bin(OpMul, f(2/s.Softcap), x))
			x = g.Bin(OpMul, f(s.Softcap), g.Bin(OpSub, f(1), g.Bin(OpDiv, f(2), g.Bin(OpAdd, e, f(1)))))
		}
		key := add(k0, addc(keyL, i))
		x = g.Select(g.Less(sub(key, kst), kspan), x, negInf)
		if s.Window > 0 {
			x = g.Select(g.Less(key, winLo), negInf, x)
		}
		val[i] = x
		mx = g.Bin(OpMax, mx, x)
	}
	for _, mk := range []int{1, 2} {
		mx = g.Bin(OpMax, mx, g.Shuffle(mx, mk))
	}
	mn := g.Bin(OpMax, m, mx)
	mf := g.Bin(OpMax, mn, f(-1e30))
	alpha := g.exp(g.Bin(OpSub, m, mf))
	sum := zero
	for i := 0; i < kpl; i += 2 {
		p0, p1 := g.exp(g.Bin(OpSub, val[i], mf)), g.exp(g.Bin(OpSub, val[i+1], mf))
		sum = g.Bin(OpAdd, sum, g.Bin(OpAdd, p0, p1))
		bt = g.WriteVec(shS, addc(sRow, i), bt, p0, p1)
	}
	for _, mk := range []int{1, 2} {
		sum = g.Bin(OpAdd, sum, g.Shuffle(sum, mk))
	}
	lNext := g.Bin(OpAdd, g.Bin(OpMul, l, alpha), sum)
	for d := 0; d < dpl; d++ {
		at := addc(oRow, d)
		o := g.bits(g.LoadAt(shO, g.After(at, bt)))
		bt = g.Write(shO, at, g.word(g.Bin(OpMul, o, alpha)), bt)
	}
	bt = g.Barrier(bt)

	// O += P V for the subgroup's rows.
	pt := make([]int, tk/8)
	for kk := range pt {
		pt[kk] = g.TileLoad(aP, shS, addc(sgS, 8*kk), u(ss), false, bt)
	}
	for dt := 0; dt < hd/8; dt++ {
		oOff := addc(sgO, 8*dt)
		acc := g.TileLoad(cT, shO, oOff, u(os), false, bt)
		for kk := 0; kk < tk/8; kk++ {
			vb := g.TileLoad(bV, shV, u(8*kk*vs+8*dt), u(vs), false, bt)
			acc = g.TileMMA(pt[kk], vb, acc)
		}
		bt = g.WriteTile(shO, oOff, u(os), acc, cT, false, bt)
	}
	g.SetFold(k0, g.After(addc(k0, tk), bt))
	g.SetFold(m, mn)
	g.SetFold(l, lNext)
	g.SetFold(mo, m)
	g.SetFold(lo0, l)
	g.SetFold(pg, addc(pg, 1))

	// From the folds: a block whose widest count is zero runs no trip.
	end := g.Barrier(g.After(mo, lo0))
	if s.Splits > 0 {
		padded := (hd + 31) / 32 * 32
		pb := mul(add(mul(add(mul(rrow, s.Heads), h), S), spl), padded+64)
		mr := g.After(g.Bin(OpMax, mo, f(-1e30)), end)
		at := add(pb, mul(part, 8))
		for r := 0; r < 8; r++ {
			g.StoreAt(out, mr, addc(at, padded+r))
			g.StoreAt(out, lo0, addc(at, padded+32+r))
		}
		for d := 0; d < dpl; d++ {
			g.StoreAt(out, g.bits(g.LoadAt(shO, g.After(addc(oRow, d), end))), addc(add(pb, dimL), d))
		}
		return
	}
	inv := g.Bin(OpDiv, f(1), g.Bin(OpMax, lo0, f(1e-30)))
	dst := add(mul(add(mul(rrow, s.Heads), h), hd), dimL)
	for d := 0; d < dpl; d++ {
		o := g.bits(g.LoadAt(shO, g.After(addc(oRow, d), end)))
		g.StoreAt(out, g.Bin(OpMul, o, inv), addc(dst, d))
	}
}
