package kernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/ir"
)

// PagedAttnAccMMAWarps is how many warps PagedAttnAccMMA launches (groups of
// 128 threads).
func PagedAttnAccMMAWarps(s FlashShape, mt, nt int) int {
	return stagedValue(s) / (16 * mt) * s.Heads * s.Rows / (8 * nt) * splitsOf(s)
}

// PagedAttnAccMMA is PagedAttnAccMMA70 on the m16n8k8 binary16 instruction
// (sm_75 on): V^T the A operand and the weights' transpose the B, 16*mt dims
// by 8*nt queries a warp, over the keys of its split its queries attend, eight
// keys a step. Lane (g, q) holds dims g and g+8 of each m-tile at keys 2q and
// 2q+1 of the step, and query g of each n-tile at the same two keys, so its
// two weights are one 8-byte load.
//
// Eight keys from a multiple of eight lie in one page, as PagedAttnAccTiled's
// blocks do. The first and last steps are masked -- V selected to 0 outside
// the split's part of the tile's keys (0 * NaN is NaN), the weights with it
// -- with every load clamped into the tile's pages and the split's chunk; the
// steps between are wholly inside, walked a page at a time with the page id
// read once. Packed f16 V is read a half at a time. It writes the numerators
// into the merge's partials, or in the direct form (Splits 0) the output,
// [row][head][value]. The softmax is built with qt = 8*nt, which must divide
// the scores kernel's 16-row query tile. Parameters: pA (the weights), pV, pN
// (not read), pPart (the output in the direct form), pTab, pRow.
func PagedAttnAccMMA(s FlashShape, mt, nt int) (*ir.Kernel, error) {
	if err := prefillCheck("PagedAttnAccMMA", s); err != nil {
		return nil, err
	}
	vd := stagedValue(s)
	if mt < 1 || nt < 1 || vd%(16*mt) != 0 || s.Rows%(8*nt) != 0 || s.Page%8 != 0 || s.Chunk%8 != 0 {
		return nil, fmt.Errorf("kernels: PagedAttnAccMMA: %+v mt=%d nt=%d", s, mt, nt)
	}
	P, C, S := s.Page, s.Chunk, splitsOf(s)
	kvRow, gqa := s.KVHeads*s.Dim, s.Heads/s.KVHeads
	sh := ir.MMAShape{M: 16, N: 8, K: 8, Kind: ir.MMAF16}
	b := ir.New(fmt.Sprintf("pagedaccmma_m%dn%d", mt, nt), [3]int{128, 1, 1})
	pA, pV := b.Param("pA", ir.F32), b.Param("pV", ir.F32)
	b.Param("pN", ir.U32)
	pPart := b.Param("pPart", ir.F32)
	pTab, pRow := b.Param("pTab", ir.U32), b.Param("pRow", ir.U32)
	c := func(v int) ir.Value { return b.Const(ir.U32, int64(v)) }
	add := func(a, x ir.Value) ir.Value { return b.Add(ir.U32, a, x) }
	mul := func(a, x ir.Value) ir.Value { return b.Mul(ir.U32, a, x) }
	dblocks, qgroups := vd/(16*mt), s.Rows/(8*nt)
	tid := add(mul(b.CTAID(), b.NTID()), b.TID())
	warp := b.Min(ir.U32, b.Shr(ir.U32, tid, c(5)), c(dblocks*S*s.Heads*qgroups-1))
	lane := b.And(ir.U32, tid, c(31))
	lg := b.Shr(ir.U32, lane, c(2))
	q2 := b.Shl(ir.U32, b.And(ir.U32, lane, c(3)), c(1)) // the lane's first key of a step
	db := urem(b, warp, dblocks)
	rest := udiv(b, warp, dblocks)
	spl := urem(b, rest, S)
	rest = udiv(b, rest, S)
	h := urem(b, rest, s.Heads)
	qryW := mul(udiv(b, rest, s.Heads), c(8*nt))
	sp := prefillSpanOf(b, pRow, qryW, add(qryW, c(8*nt-1)))
	dimW := mul(db, c(16*mt))
	col := add(dimW, lg) // + 8x + 16i: the lane's A rows, dims of V
	if s.MLAValueTail {
		col = add(col, c(s.Dim-s.MLA))
	}
	if s.MLA == 0 {
		col = add(mul(udiv(b, h, gqa), c(s.Dim)), col)
	}
	// The split's part of the group's keys, [st, en), from the 8-aligned step
	// holding st: n steps.
	c0 := add(sp.base, mul(spl, c(C)))
	st := b.Min(ir.U32, b.Max(ir.U32, sp.lo, c0), add(c0, c(C)))
	en := b.Max(ir.U32, b.Min(ir.U32, sp.hi, add(c0, c(C))), st)
	a8 := add(c0, mul(udiv(b, b.Sub(ir.U32, st, c0), 8), c(8)))
	n := b.Select(ir.U32, b.Lt(ir.U32, st, en), udiv(b, add(b.Sub(ir.U32, en, a8), c(7)), 8), c(0))
	mid := b.Sub(ir.U32, b.Max(ir.U32, n, c(2)), c(2))
	// Query qryW+g's weights of this split: ((q*Heads + h)*S + spl)*C, 8j
	// queries on, at the slot, the lane's two keys q2 on.
	wRow := add(mul(add(mul(add(mul(add(qryW, lg), c(s.Heads)), h), c(S)), spl), c(C)), q2)
	wStep := int64(8 * s.Heads * S * C)
	zero := b.ConstF32(0)
	// vAt is element e+imm of position t's V row, t in page pid.
	vAt := func(pid, t, e ir.Value, imm int) ir.Value {
		idx := add(mul(add(mul(pid, c(P)), pageRem(b, t, P)), c(kvRow)), e)
		if !s.F16 {
			return b.Load(ir.F32, pV, idx, int64(imm))
		}
		ix := add(idx, c(imm))
		word := b.Bitcast(ir.U32, b.Load(ir.F32, pV, b.Shr(ir.U32, ix, c(1)), 0))
		return b.CvtF16H(b.Shr(ir.U32, word, mul(b.And(ir.U32, ix, c(1)), c(16))))
	}
	// stepAt adds the 8 keys from pos. Masked: the lane's keys pos+q2 and
	// pos+q2+1 are kept in [st, en), V read at the clamped step's page, the
	// weights' slot clamped into the chunk. Unmasked: the lane's first V row
	// starts at element vp and its weights at wp, walked by pointer.
	stepAt := func(pos, vp, wp ir.Value, d [][][]ir.Value, masked bool) {
		var ok [2]ir.Value
		var slot, mpid, mpos ir.Value
		if masked {
			slot = b.Min(ir.U32, b.Sub(ir.U32, pos, c0), c(C-8))
			for x := range ok {
				ok[x] = b.Lt(ir.U32, b.Sub(ir.U32, add(add(pos, q2), c(x)), st), b.Sub(ir.U32, en, st))
				if pagedFault == "vmask" {
					ok[x] = sp.in(b, add(add(pos, q2), c(x)))
				}
			}
			mpos = sp.clamp(b, pos, 8)
			mpid = pageOf(b, pTab, sp.tab, mpos, P)
		}
		var vw, vsh ir.Value
		if !masked && s.F16 {
			// A lane's elements all have vp's parity: kvRow, 8 and 16 are even.
			vw, vsh = b.Shr(ir.U32, vp, c(1)), mul(b.And(ir.U32, vp, c(1)), c(16))
		}
		af := make([][]ir.Value, mt)
		for i := range af {
			af[i] = make([]ir.Value, 2)
			for hh := 0; hh < 2; hh++ {
				var e [2]ir.Value
				for x := range e {
					imm := 16*i + 8*hh
					if !masked {
						imm += x * kvRow
						if s.F16 {
							word := b.Bitcast(ir.U32, b.Load(ir.F32, pV, vw, int64(imm/2)))
							e[x] = b.CvtF16H(b.Shr(ir.U32, word, vsh))
						} else {
							e[x] = b.Load(ir.F32, pV, vp, int64(imm))
						}
						continue
					}
					e[x] = b.Select(ir.F32, ok[x], vAt(mpid, add(add(mpos, q2), c(x)), col, imm), zero)
				}
				af[i][hh] = b.PackF16(e[0], e[1])
			}
		}
		bf := make([][]ir.Value, nt)
		for j := range bf {
			wa := wp
			if masked {
				wa = add(wRow, slot)
			}
			e := b.LoadV(ir.F32, pA, wa, int64(j)*wStep, 2)
			if masked {
				for x := range e {
					e[x] = b.Select(ir.F32, ok[x], e[x], zero)
				}
			}
			bf[j] = []ir.Value{b.PackF16(e[0], e[1])}
		}
		for i := 0; i < mt; i++ {
			for j := 0; j < nt; j++ {
				d[i][j] = b.MMA(sh, af[i], bf[j], d[i][j])
			}
		}
	}
	fresh := func(init [][][]ir.Value, f func(ir.Value) ir.Value) [][][]ir.Value {
		out := make([][][]ir.Value, mt)
		for i := range out {
			out[i] = make([][]ir.Value, nt)
			for j := range out[i] {
				out[i][j] = make([]ir.Value, 4)
				for x := range out[i][j] {
					out[i][j][x] = f(init[i][j][x])
				}
			}
		}
		return out
	}
	acc := make([][][]ir.Value, mt)
	for i := range acc {
		acc[i] = make([][]ir.Value, nt)
		for j := range acc[i] {
			acc[i][j] = []ir.Value{zero, zero, zero, zero}
		}
	}
	// The first step, masked.
	stepAt(a8, 0, 0, acc, true)
	// The steps between, [m0, m1), unmasked: a run of them per page, the
	// page's id read once and V and the weights walked by pointer inside it.
	m0 := add(a8, c(8))
	m1 := add(m0, mul(mid, c(8)))
	pg0 := pageDiv(b, m0, P)
	npg := b.Select(ir.U32, b.Lt(ir.U32, c(0), mid), add(b.Sub(ir.U32, pageDiv(b, b.Sub(ir.U32, m1, c(1)), P), pg0), c(1)), c(0))
	b.LoopN(npg)
	ph := fresh(acc, func(x ir.Value) ir.Value { return b.Phi(ir.F32, x) })
	pg := b.Phi(ir.U32, pg0)
	pid := b.Load(ir.U32, pTab, add(sp.tab, pg), 0)
	// The page's part of [m0, m1), in steps of 8 from m0 (exactly the page's
	// span, since m0 and a page edge are both multiples of 8 from c0).
	from := func(t ir.Value) ir.Value {
		return add(m0, mul(udiv(b, add(b.Sub(ir.U32, b.Max(ir.U32, m0, t), m0), c(7)), 8), c(8)))
	}
	s0 := from(mul(pg, c(P)))
	s1 := b.Min(ir.U32, m1, from(mul(add(pg, c(1)), c(P))))
	vp0 := add(mul(add(add(mul(pid, c(P)), pageRem(b, s0, P)), q2), c(kvRow)), col)
	wp0 := add(wRow, b.Sub(ir.U32, s0, c0))
	b.LoopN(udiv(b, b.Sub(ir.U32, s1, s0), 8))
	ih := fresh(ph, func(x ir.Value) ir.Value { return b.Phi(ir.F32, x) })
	vp := b.Phi(ir.U32, vp0)
	wp := b.Phi(ir.U32, wp0)
	d := fresh(ih, func(x ir.Value) ir.Value { return x })
	stepAt(0, vp, wp, d, false)
	for i := range ih {
		for j := range ih[i] {
			for x := range ih[i][j] {
				b.SetPhi(ih[i][j][x], d[i][j][x])
			}
		}
	}
	b.SetPhi(vp, add(vp, c(8*kvRow)))
	b.SetPhi(wp, add(wp, c(8)))
	b.EndLoop()
	for i := range ph {
		for j := range ph[i] {
			for x := range ph[i][j] {
				b.SetPhi(ph[i][j][x], ih[i][j][x])
			}
		}
	}
	b.SetPhi(pg, add(pg, c(1)))
	b.EndLoop()
	// The last step, masked; from the phis, which hold their inits when a
	// loop runs no step.
	d = fresh(ph, func(x ir.Value) ir.Value { return x })
	stepAt(add(a8, mul(add(mid, c(1)), c(8))), 0, 0, d, true)
	// D component x: dim g (+8 when x >= 2), query 2q + x%2.
	padded := (vd + 31) / 32 * 32
	for i := 0; i < mt; i++ {
		for j := 0; j < nt; j++ {
			for x := 0; x < 4; x++ {
				dim := add(add(dimW, lg), c(16*i+8*(x>>1)))
				q := add(add(qryW, q2), c(8*j+x&1))
				o := add(mul(add(mul(add(mul(q, c(s.Heads)), h), c(S)), spl), c(padded+64)), dim)
				if s.Splits == 0 {
					o = add(mul(add(mul(q, c(s.Heads)), h), c(vd)), dim)
				}
				b.Store(pPart, o, d[i][j][x], 0)
			}
		}
	}
	return b.Done()
}
