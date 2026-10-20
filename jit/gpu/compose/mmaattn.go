package compose

import (
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// The staged prefill's scores on the subgroup matrix instruction: a
// subgroup a tile of 16 query rows by 8*nt keys over the head in steps of
// 16, A the queries and B a page's transposed K -- the column-major operand
// as it stands -- both as binary16 pairs, the sums float32. A tile none of
// the subgroup's rows attends is not computed.

// ScoresMMAWarps is the subgroups PrefillScoresMMA runs.
func (s Prefill) ScoresMMAWarps(nt int) int {
	return s.Rows / 16 * s.Heads * (s.splits() * s.Chunk / (8 * nt))
}

// PrefillScoresMMA is s's score plane on the m16n8k16 binary16 instruction,
// a thread an element of the group (ScoresMMAWarps()*32 of them): q the
// rows' query heads, k the paged K pool, n unread, out the plane, tab and
// desc the page table and the row descriptors.
func (g *Row) PrefillScoresMMA(s Prefill, nt, q, k, n, out, tab, desc int) {
	u, ub := g.UConst, g.UBin
	add := func(a, b int) int { return ub(OpAdd, a, b) }
	mul := func(a, c int) int { return ub(OpMul, a, u(c)) }
	W := s.splits() * s.Chunk
	P, kvRow, gqa := s.Page, s.KVHeads*s.Dim, s.Heads/s.KVHeads
	sh := ir.MMAShape{M: 16, N: 8, K: 16, Kind: ir.MMAF16}
	na, nb, nc := sh.Frags()
	ktiles := W / (8 * nt)
	tid := g.Add(Node{Kind: NThread})
	lane := ub(OpAnd, tid, u(31))
	gr := ub(OpShr, lane, u(2))
	two := mul(ub(OpAnd, lane, u(3)), 2)
	warp := ub(OpMin, ub(OpDiv, tid, u(32)), u(ktiles*s.Heads*(s.Rows/16)-1))
	kt := ub(OpRem, warp, u(ktiles))
	rest := ub(OpDiv, warp, u(ktiles))
	h := ub(OpRem, rest, u(s.Heads))
	qrow := mul(ub(OpDiv, rest, u(s.Heads)), 16)
	sp := g.spanOf(s, desc, -1, -1, qrow, add(qrow, u(15)))
	qbase := add(mul(add(mul(add(qrow, gr), s.Heads), h), s.Dim), two)
	kvh := mul(ub(OpDiv, h, u(gqa)), s.Dim)
	slot0 := mul(kt, 8*nt)
	T := add(sp.base, slot0)
	Tc := g.clamp(sp, T, 8*nt)
	pid := g.LoadAt(tab, add(sp.tab, ub(OpDiv, Tc, u(P))))
	kbase := add(add(add(mul(pid, P*kvRow), mul(add(kvh, two), P)), ub(OpRem, Tc, u(P))), gr)
	rows := [2]int{add(qrow, gr), add(qrow, add(gr, u(8)))}
	var ks, ke [2]int
	for i, r := range rows {
		ks[i], ke[i] = g.descOf(desc, r, kernels.PRowStart), g.descOf(desc, r, kernels.PRowEnd)
	}
	live := g.Select(g.Less(T, sp.hi), g.Select(g.Less(sp.lo, add(T, u(8*nt))), u(1), u(0)), u(0))
	c := g.LoopN(live)
	it := g.Iter(c)
	zero := g.Const(0)
	accs := make([][]int, nt)
	for j := range accs {
		accs[j] = make([]int, nc)
		for i := range accs[j] {
			accs[j][i] = zero
		}
	}
	at := func(base, off int) int {
		x := add(base, ub(OpMul, it, u(0)))
		if off != 0 {
			x = add(x, u(off))
		}
		return x
	}
	for kk := 0; kk < s.Dim; kk += 16 {
		af := make([]int, na)
		for i := range af {
			off := kk
			if i >= 2 {
				off += 8
			}
			if i%2 == 1 {
				off += 8 * s.Heads * s.Dim
			}
			af[i] = g.Bin(OpPackF16, g.LoadAt(q, at(qbase, off)), g.LoadAt(q, at(qbase, off+1)))
		}
		for j := 0; j < nt; j++ {
			bf := make([]int, nb)
			for i := range bf {
				e := kk
				if i >= 1 {
					e += 8
				}
				o := e*P + j*8
				bf[i] = g.Bin(OpPackF16, g.LoadAt(k, at(kbase, o)), g.LoadAt(k, at(kbase, o+P)))
			}
			accs[j] = g.MMA(sh, af, bf, accs[j])
		}
	}
	negInf := g.bits(g.After(u(0xFF800000), it))
	for j := 0; j < nt; j++ {
		for ci := 0; ci < nc; ci++ {
			ri := ci / 2
			slot := add(add(slot0, two), u(j*8+ci%2))
			ok := g.Less(g.After(ub(OpSub, add(sp.base, slot), ks[ri]), it), ub(OpSub, ke[ri], ks[ri]))
			o := add(mul(add(mul(rows[ri], s.Heads), h), W), slot)
			g.StoreAt(out, g.Select(ok, g.capped(accs[j][ci], s.Scale, s.Softcap), negInf), o)
		}
	}
}

// AccMMAWarps is the subgroups PrefillAccMMA runs.
func (s Prefill) AccMMAWarps(mt, nt int) int {
	return s.valueDim() / (16 * mt) * s.Heads * s.Rows / (8 * nt) * s.splits()
}

// PrefillAccMMA is s's weighted sum of V on the m16n8k8 binary16
// instruction, a thread an element of the group (AccMMAWarps()*32): V^T the
// A operand and the weights' transpose the B, 16*mt dims by 8*nt queries a
// subgroup, over its split's keys eight a step. The first and last steps
// are masked (V and the weights zero outside the split's keys, every load
// clamped into its pages); the steps between walk a page at a time, the
// page's id read once and V and the weights by pointer. a the weights, v
// the paged V pool, n unread, out the partials (the rows where Splits is
// 0), tab and desc the page table and the row descriptors.
func (g *Row) PrefillAccMMA(s Prefill, mt, nt, a, v, n, out, tab, desc int) {
	u, ub := g.UConst, g.UBin
	add := func(x, y int) int { return ub(OpAdd, x, y) }
	addc := func(x, c int) int {
		if c == 0 {
			return x
		}
		return add(x, u(c))
	}
	mul := func(x, c int) int { return ub(OpMul, x, u(c)) }
	sub := func(x, y int) int { return ub(OpSub, x, y) }
	umin, umax := func(x, y int) int { return ub(OpMin, x, y) }, func(x, y int) int { return ub(OpMax, x, y) }
	vd := s.valueDim()
	P, C, S := s.Page, s.Chunk, s.splits()
	kvRow, gqa := s.KVHeads*s.Dim, s.Heads/s.KVHeads
	sh := ir.MMAShape{M: 16, N: 8, K: 8, Kind: ir.MMAF16}
	dblocks, qgroups := vd/(16*mt), s.Rows/(8*nt)
	tid := g.Add(Node{Kind: NThread})
	warp := umin(ub(OpShr, tid, u(5)), u(dblocks*S*s.Heads*qgroups-1))
	lane := ub(OpAnd, tid, u(31))
	lg := ub(OpShr, lane, u(2))
	q2 := ub(OpShl, ub(OpAnd, lane, u(3)), u(1))
	db := ub(OpRem, warp, u(dblocks))
	rest := ub(OpDiv, warp, u(dblocks))
	spl := ub(OpRem, rest, u(S))
	rest = ub(OpDiv, rest, u(S))
	h := ub(OpRem, rest, u(s.Heads))
	qryW := mul(ub(OpDiv, rest, u(s.Heads)), 8*nt)
	sp := g.spanOf(s, desc, -1, -1, qryW, addc(qryW, 8*nt-1))
	dimW := mul(db, 16*mt)
	col := add(dimW, lg)
	if s.MLA == 0 {
		col = add(mul(ub(OpDiv, h, u(gqa)), s.Dim), col)
	}
	c0 := add(sp.base, mul(spl, C))
	st := umin(umax(sp.lo, c0), addc(c0, C))
	en := umax(umin(sp.hi, addc(c0, C)), st)
	a8 := add(c0, mul(ub(OpDiv, sub(st, c0), u(8)), 8))
	nsteps := g.Select(g.Less(st, en), ub(OpDiv, addc(sub(en, a8), 7), u(8)), u(0))
	mid := sub(umax(nsteps, u(2)), u(2))
	wRow := add(mul(add(mul(add(mul(add(qryW, lg), s.Heads), h), S), spl), C), q2)
	wStep := 8 * s.Heads * S * C
	zero := g.Const(0)
	vAt := func(pid, t, e, imm int) int {
		return g.LoadAt(v, addc(add(mul(add(mul(pid, P), ub(OpRem, t, u(P))), kvRow), e), imm))
	}
	type frags = [][][]int
	stepAt := func(pos, vp, wp int, d frags, masked bool) {
		var ok [2]int
		var slot, mpid, mpos int
		if masked {
			slot = umin(sub(pos, c0), u(C-8))
			for x := range ok {
				ok[x] = g.Less(sub(addc(add(pos, q2), x), st), sub(en, st))
			}
			mpos = g.clamp(sp, pos, 8)
			mpid = g.LoadAt(tab, add(sp.tab, ub(OpDiv, mpos, u(P))))
		}
		af := make([][]int, mt)
		for i := range af {
			af[i] = make([]int, 2)
			for hh := 0; hh < 2; hh++ {
				var e [2]int
				for x := range e {
					imm := 16*i + 8*hh
					if !masked {
						e[x] = g.LoadAt(v, addc(vp, imm+x*kvRow))
						continue
					}
					e[x] = g.Select(ok[x], vAt(mpid, addc(add(mpos, q2), x), col, imm), zero)
				}
				af[i][hh] = g.Bin(OpPackF16, e[0], e[1])
			}
		}
		bf := make([][]int, nt)
		for j := range bf {
			wa := wp
			if masked {
				wa = add(wRow, slot)
			}
			wa = addc(wa, j*wStep)
			e := [2]int{g.LoadVec(a, wa, 2, 0), g.LoadVec(a, wa, 2, 1)}
			if masked {
				for x := range e {
					e[x] = g.Select(ok[x], e[x], zero)
				}
			}
			bf[j] = []int{g.Bin(OpPackF16, e[0], e[1])}
		}
		for i := 0; i < mt; i++ {
			for j := 0; j < nt; j++ {
				d[i][j] = g.MMA(sh, af[i], bf[j], d[i][j])
			}
		}
	}
	fresh := func(init frags, f func(int) int) frags {
		o := make(frags, mt)
		for i := range o {
			o[i] = make([][]int, nt)
			for j := range o[i] {
				o[i][j] = make([]int, 4)
				for x := range o[i][j] {
					o[i][j][x] = f(init[i][j][x])
				}
			}
		}
		return o
	}
	acc := make(frags, mt)
	for i := range acc {
		acc[i] = make([][]int, nt)
		for j := range acc[i] {
			acc[i][j] = []int{zero, zero, zero, zero}
		}
	}
	stepAt(a8, 0, 0, acc, true)
	m0 := addc(a8, 8)
	m1 := add(m0, mul(mid, 8))
	pg0 := ub(OpDiv, m0, u(P))
	npg := g.Select(g.Less(u(0), mid), addc(sub(ub(OpDiv, sub(m1, u(1)), u(P)), pg0), 1), u(0))
	cp := g.LoopN(npg)
	ph := fresh(acc, func(x int) int { return g.Fold(cp, x) })
	pg := g.Fold(cp, pg0)
	pid := g.LoadAt(tab, add(sp.tab, pg))
	from := func(t int) int {
		return add(m0, mul(ub(OpDiv, addc(sub(umax(m0, t), m0), 7), u(8)), 8))
	}
	s0 := from(mul(pg, P))
	s1 := umin(m1, from(mul(addc(pg, 1), P)))
	vp0 := add(mul(add(add(mul(pid, P), ub(OpRem, s0, u(P))), q2), kvRow), col)
	wp0 := add(wRow, sub(s0, c0))
	ci := g.LoopN(ub(OpDiv, sub(s1, s0), u(8)))
	ih := fresh(ph, func(x int) int { return g.Fold(ci, x) })
	vp, wp := g.Fold(ci, vp0), g.Fold(ci, wp0)
	d := fresh(ih, func(x int) int { return x })
	stepAt(0, vp, wp, d, false)
	for i := range ih {
		for j := range ih[i] {
			for x := range ih[i][j] {
				g.SetFold(ih[i][j][x], d[i][j][x])
				g.SetFold(ph[i][j][x], ih[i][j][x])
			}
		}
	}
	g.SetFold(vp, addc(vp, 8*kvRow))
	g.SetFold(wp, addc(wp, 8))
	g.SetFold(pg, addc(pg, 1))
	d = fresh(ph, func(x int) int { return x })
	stepAt(add(a8, mul(addc(mid, 1), 8)), 0, 0, d, true)
	padded := (vd + 31) / 32 * 32
	for i := 0; i < mt; i++ {
		for j := 0; j < nt; j++ {
			for x := 0; x < 4; x++ {
				dim := addc(add(dimW, lg), 16*i+8*(x>>1))
				q := addc(add(qryW, q2), 8*j+x&1)
				o := add(mul(add(mul(add(mul(q, s.Heads), h), S), spl), padded+64), dim)
				if s.Splits == 0 {
					o = add(mul(add(mul(q, s.Heads), h), vd), dim)
				}
				g.StoreAt(out, d[i][j][x], o)
			}
		}
	}
}
