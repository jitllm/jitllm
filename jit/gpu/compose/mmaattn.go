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
