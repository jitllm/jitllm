package compose

import (
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// Flash16 is a prompt chunk's paged attention in one kernel on the binary16
// m16n8 instruction: queries the A rows (sixteen a subgroup, held in
// registers), a tile's keys the B columns, so the scores' fragment of two
// 8-key n-tiles is the A fragment of the 16-key step of P V and the
// probabilities never leave registers. A row's maximum and sum reduce over
// its quad. K is staged as binary16 key rows and V as dim rows of key pairs,
// both padded four words. Past a 128-wide head two subgroups share a row
// block's dims and the queries go through workgroup memory.
type Flash16 struct {
	Heads, KVHeads, Dim, Rows, Page int
	Scale, Softcap                  float32
	// Splits > 0 writes the merge's partials, a run of key tiles each.
	Splits int
	// Map reads each query slot's row from a run table; K8 takes each k16
	// product as two m16n8k8.
	Map, K8 bool
}

func (s Flash16) dimSplit() int {
	if s.Dim > 128 {
		return 2
	}
	return 1
}

// QRows is a workgroup's query rows.
func (s Flash16) QRows() int { return 16 * 4 / s.dimSplit() }

// Threads is a workgroup's width; Groups the workgroups a launch runs.
func (s Flash16) Threads() int { return 128 }
func (s Flash16) Groups() int  { return s.Heads * (s.Rows / s.QRows()) * max(1, s.Splits) }

// PagedFlash16 is s over the group (Threads() elements, Groups() rows, a row
// a workgroup): q the rows' query heads, k and v the paged pools, n unread,
// out the rows or the partials, tab the page ids, desc the row descriptors,
// mp the run table (-1 without Map).
func (g *Row) PagedFlash16(s Flash16, q, k, v, n, out, tab, desc, mp int) {
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
	sub := func(a, b int) int { return ub(OpSub, a, b) }
	umin, umax := func(a, b int) int { return ub(OpMin, a, b) }, func(a, b int) int { return ub(OpMax, a, b) }
	f := g.Const
	hd, P, S := s.Dim, s.Page, max(1, s.Splits)
	ds := s.dimSplit()
	tk := 32
	if ds > 1 {
		tk = 16
	}
	const qw = 16
	rowsWG := s.QRows()
	gqa := s.Heads / s.KVHeads
	kvDim := s.KVHeads * hd
	qs, vs := hd/2+4, tk/2+4
	nd, ksteps := hd/8/ds, hd/16
	sh := ir.MMAShape{M: 16, N: 8, K: 16, Kind: ir.MMAF16}
	mma := func(a, bf, acc []int) []int {
		if !s.K8 {
			return g.MMA(sh, a, bf, acc)
		}
		h := ir.MMAShape{M: 16, N: 8, K: 8, Kind: ir.MMAF16}
		return g.MMA(h, a[2:4], bf[1:2], g.MMA(h, a[0:2], bf[0:1], acc))
	}
	rowOf := func(slot int) int { return slot }
	if s.Map {
		rowOf = func(slot int) int { return g.LoadAt(mp, slot) }
	}
	descAt := func(row, w int) int { return g.LoadAt(desc, addc(mul(row, kernels.PRowWords), w)) }
	negInf, zero := g.bits(u(0xFF800000)), f(0)
	shK := g.Shared(tk*qs, true)
	shV := g.Shared(hd*vs, true)
	shQ := -1
	if ds > 1 {
		shQ = g.Shared(rowsWG*qs, true)
	}

	tid, wg := g.Column(), g.RowIdx()
	lane := ub(OpAnd, tid, u(31))
	warp := ub(OpShr, tid, u(5))
	lg, lt := ub(OpShr, lane, u(2)), ub(OpAnd, lane, u(3))
	spl := rem(wg, S)
	wg = div(wg, S)
	h := rem(wg, s.Heads)
	qb := div(wg, s.Heads)
	kvBase := mul(div(h, gqa), hd)
	row0 := mul(qb, rowsWG)
	qwarp, dh := div(warp, ds), rem(warp, ds)
	dimBase := mul(dh, hd/ds)
	wrow0 := add(row0, mul(qwarp, qw))
	first, last := rowOf(row0), rowOf(addc(row0, rowsWG-1))
	lo, hi, tabOff := descAt(first, kernels.PRowStart), descAt(last, kernels.PRowEnd), descAt(first, kernels.PRowTab)
	live := g.Less(lo, hi)
	a32 := mul(div(lo, 32), 32)
	total := g.Select(live, div(add(sub(hi, a32), u(tk-1)), tk), u(0))
	kStart, ntiles := a32, total
	if S > 1 {
		per := div(addc(total, S-1), S)
		t0 := umin(ub(OpMul, spl, per), total)
		kStart, ntiles = add(a32, mul(t0, tk)), sub(umin(add(t0, per), total), t0)
	}
	hlo := sub(hi, lo)

	var rows, kst, kend [2]int
	for x := 0; x < 2; x++ {
		rows[x] = rowOf(add(wrow0, addc(lg, 8*x)))
		kst[x], kend[x] = descAt(rows[x], kernels.PRowStart), descAt(rows[x], kernels.PRowEnd)
	}
	qa := make([][]int, ksteps)
	qAt := func(x int) int {
		return add(mul(add(mul(rows[x], s.Heads), h), hd), ub(OpShl, lt, u(1)))
	}
	q0, q1 := qAt(0), qAt(1)
	pk := func(base, off int) int {
		at := addc(base, off)
		return g.Bin(OpPackF16, g.LoadVec(q, at, 2, 0), g.LoadVec(q, at, 2, 1))
	}
	for ks := range qa {
		if ds > 1 {
			break
		}
		qa[ks] = []int{pk(q0, 16*ks), pk(q1, 16*ks), pk(q0, 16*ks+8), pk(q1, 16*ks+8)}
	}
	tok := -1
	for it := 0; ds > 1 && it < rowsWG*hd/2/s.Threads(); it++ {
		w := addc(tid, s.Threads()*it)
		qi, p := div(w, hd/2), rem(w, hd/2)
		src := add(mul(add(mul(rowOf(add(row0, qi)), s.Heads), h), hd), ub(OpShl, p, u(1)))
		e0, e1 := g.LoadVec(q, src, 2, 0), g.LoadVec(q, src, 2, 1)
		tok = g.Write(shQ, add(mul(qi, qs), p), g.Bin(OpPackF16, e0, e1), tok)
	}
	qRow := add(mul(add(mul(qwarp, qw), lg), qs), lt)

	kEnd := add(kStart, mul(ntiles, tk))
	pg0 := div(kStart, P)
	npg := g.Select(g.Less(u(0), ntiles), addc(sub(div(sub(kEnd, u(1)), P), pg0), 1), u(0))
	cp := g.LoopN(npg)
	type state struct {
		o    [][]int
		m, l [2]int
	}
	newState := func(c int, from *state) state {
		var t state
		for nn := 0; nn < nd; nn++ {
			o := make([]int, 4)
			for i := range o {
				init := zero
				if from != nil {
					init = from.o[nn][i]
				}
				o[i] = g.Fold(c, init)
			}
			t.o = append(t.o, o)
		}
		for x := 0; x < 2; x++ {
			mi, li := negInf, zero
			if from != nil {
				mi, li = from.m[x], from.l[x]
			}
			t.m[x], t.l[x] = g.Fold(c, mi), g.Fold(c, li)
		}
		return t
	}
	outer := newState(cp, nil)
	pg := g.Fold(cp, pg0)
	t0 := umax(kStart, mul(pg, P))
	pid := g.LoadAt(tab, add(tabOff, div(t0, P)))
	t1 := umin(kEnd, mul(addc(pg, 1), P))
	ct := g.LoopN(div(sub(t1, t0), tk))
	st := newState(ct, &outer)
	k0 := g.Fold(ct, t0)
	o, m, l := st.o, st.m, st.l

	bt := k0
	if tok >= 0 {
		bt = g.After(k0, tok)
	}
	bt = g.Barrier(bt)
	for it := 0; it < tk*hd/2/s.Threads(); it++ {
		w := addc(tid, s.Threads()*it)
		key := ub(OpAnd, w, u(tk-1))
		p := div(w, tk)
		e := add(add(mul(pid, P*kvDim), mul(add(kvBase, ub(OpShl, p, u(1))), P)), add(rem(k0, P), key))
		klo, khi := g.LoadAt(k, e), g.LoadAt(k, addc(e, P))
		bt = g.Write(shK, add(mul(key, qs), p), g.Bin(OpPackF16, klo, khi), bt)
	}
	for it := 0; it < tk/2*hd/4/s.Threads(); it++ {
		w := addc(tid, s.Threads()*it)
		dq, kp := rem(w, hd/4), div(w, hd/4)
		p0 := add(k0, ub(OpShl, kp, u(1)))
		p1 := addc(p0, 1)
		ok0, ok1 := g.Less(sub(p0, lo), hlo), g.Less(sub(p1, lo), hlo)
		col := add(kvBase, ub(OpShl, dq, u(2)))
		vAt := func(p, x int) int {
			return g.LoadVec(v, add(mul(add(mul(pid, P), rem(p, P)), kvDim), col), 4, x)
		}
		base := add(mul(ub(OpShl, dq, u(2)), vs), kp)
		for x := 0; x < 4; x++ {
			bt = g.Write(shV, addc(base, x*vs), g.Bin(OpPackF16, g.Select(ok0, vAt(p0, x), zero), g.Select(ok1, vAt(p1, x), zero)), bt)
		}
	}
	bt = g.Barrier(bt)

	qfrag := func(ks int) []int {
		if ds == 1 {
			return qa[ks]
		}
		w := 8 * ks
		ld := func(off int) int { return g.LoadAt(shQ, g.After(addc(qRow, off), bt)) }
		return []int{ld(w), ld(w + 8*qs), ld(w + 4), ld(w + 8*qs + 4)}
	}
	sacc := make([][]int, tk/8)
	for j := range sacc {
		sacc[j] = []int{zero, zero, zero, zero}
		kRow := add(mul(addc(lg, 8*j), qs), lt)
		for ks := 0; ks < ksteps; ks++ {
			bf := []int{g.LoadAt(shK, g.After(addc(kRow, 8*ks), bt)), g.LoadAt(shK, g.After(addc(kRow, 8*ks+4), bt))}
			sacc[j] = mma(qfrag(ks), bf, sacc[j])
		}
	}
	kq := add(k0, ub(OpShl, lt, u(1)))
	val := make([][]int, len(sacc))
	for j := range sacc {
		val[j] = make([]int, 4)
		for x := 0; x < 4; x++ {
			vv := g.Bin(OpMul, sacc[j][x], f(s.Scale))
			if s.Softcap > 0 {
				e := g.exp(g.Bin(OpMul, f(2/s.Softcap), vv))
				vv = g.Bin(OpMul, f(s.Softcap), g.Bin(OpSub, f(1), g.Bin(OpDiv, f(2), g.Bin(OpAdd, e, f(1)))))
			}
			key := addc(kq, 8*j+x&1)
			r := x >> 1
			val[j][x] = g.Select(g.Less(sub(key, kst[r]), sub(kend[r], kst[r])), vv, negInf)
		}
	}
	var mNext, lNext, alpha [2]int
	pf := make([][]int, len(sacc))
	for j := range pf {
		pf[j] = make([]int, 4)
	}
	for r := 0; r < 2; r++ {
		t := negInf
		for j := range val {
			t = g.Bin(OpMax, t, g.Bin(OpMax, val[j][2*r], val[j][2*r+1]))
		}
		for _, mk := range []int{1, 2} {
			t = g.Bin(OpMax, t, g.Shuffle(t, mk))
		}
		mn := g.Bin(OpMax, m[r], t)
		mf := g.Bin(OpMax, mn, f(-1e30))
		alpha[r] = g.exp(g.Bin(OpSub, m[r], mf))
		sum := g.Bin(OpMul, l[r], alpha[r])
		for j := range val {
			for x := 2 * r; x < 2*r+2; x++ {
				pf[j][x] = g.exp(g.Bin(OpSub, val[j][x], mf))
				sum = g.Bin(OpAdd, sum, pf[j][x])
			}
		}
		mNext[r], lNext[r] = mn, sum
	}
	od := make([][]int, nd)
	for nn := range od {
		od[nn] = make([]int, 4)
		for x := 0; x < 4; x++ {
			od[nn][x] = g.Bin(OpMul, o[nn][x], alpha[x>>1])
		}
	}
	for kk := 0; kk < tk/16; kk++ {
		a := []int{
			g.Bin(OpPackF16, pf[2*kk][0], pf[2*kk][1]), g.Bin(OpPackF16, pf[2*kk][2], pf[2*kk][3]),
			g.Bin(OpPackF16, pf[2*kk+1][0], pf[2*kk+1][1]), g.Bin(OpPackF16, pf[2*kk+1][2], pf[2*kk+1][3]),
		}
		for nn := 0; nn < nd; nn++ {
			vRow := add(mul(addc(add(lg, dimBase), 8*nn), vs), lt)
			bf := []int{g.LoadAt(shV, g.After(addc(vRow, 8*kk), bt)), g.LoadAt(shV, g.After(addc(vRow, 8*kk+4), bt))}
			od[nn] = mma(a, bf, od[nn])
		}
	}
	for nn := range o {
		for x := range o[nn] {
			g.SetFold(o[nn][x], od[nn][x])
			g.SetFold(outer.o[nn][x], o[nn][x])
		}
	}
	for r := 0; r < 2; r++ {
		g.SetFold(m[r], mNext[r])
		g.SetFold(l[r], lNext[r])
		g.SetFold(outer.m[r], m[r])
		g.SetFold(outer.l[r], l[r])
	}
	g.SetFold(k0, addc(k0, tk))
	g.SetFold(pg, addc(pg, 1))
	o, m, l = outer.o, outer.m, outer.l

	var lsum [2]int
	for r := 0; r < 2; r++ {
		t := l[r]
		for _, mk := range []int{1, 2} {
			t = g.Bin(OpAdd, t, g.Shuffle(t, mk))
		}
		lsum[r] = t
	}
	dimOf := func(nn, x int) int {
		return addc(add(ub(OpShl, lt, u(1)), dimBase), 8*nn+x&1)
	}
	if s.Splits > 0 {
		stride := hd + 64
		pbOf := func(qr int) int {
			return mul(add(mul(add(mul(qr, s.Heads), h), S), spl), stride)
		}
		padded := (hd + 31) / 32 * 32
		for r := 0; r < 2; r++ {
			mr := g.Bin(OpMax, m[r], f(-1e30))
			at := add(pbOf(rows[r]), mul(lt, 8))
			for i := 0; i < 8; i++ {
				g.StoreAt(out, mr, addc(at, padded+i))
				g.StoreAt(out, lsum[r], addc(at, padded+32+i))
			}
		}
		for nn := 0; nn < nd; nn++ {
			for x := 0; x < 4; x++ {
				g.StoreAt(out, o[nn][x], add(pbOf(rows[x>>1]), dimOf(nn, x)))
			}
		}
		return
	}
	var inv [2]int
	for r := 0; r < 2; r++ {
		inv[r] = g.Bin(OpDiv, f(1), g.Bin(OpMax, lsum[r], f(1e-30)))
	}
	for nn := 0; nn < nd; nn++ {
		for x := 0; x < 4; x++ {
			at := add(mul(add(mul(rows[x>>1], s.Heads), h), hd), dimOf(nn, x))
			g.StoreAt(out, g.Bin(OpMul, o[nn][x], inv[x>>1]), at)
		}
	}
}
