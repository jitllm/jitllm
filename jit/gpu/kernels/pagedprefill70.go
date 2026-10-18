package kernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// PagedAttnScoresMMA70Warps is how many warps PagedAttnScoresMMA70 launches
// (groups of 128 threads).
func PagedAttnScoresMMA70Warps(s FlashShape, mt, nt int) int {
	return s.Rows / (8 * nt) * s.Heads * (splitsOf(s) * s.Chunk / (32 * mt))
}

// PagedAttnScoresMMA70 is AttnScoresMMA70 over paged KV: sm_70's m8n8k4 with
// the keys the A operand (a quad-pair's eight rows) and the queries the B, 32*mt
// keys by 8*nt queries a warp. Each 32-key part of the tile lies in one page
// and reads its id once. Transposed K is a page's [kvRow][P], its stride P;
// s.MLA > 0 is the row-major MLA region, a key's four dims one 16-byte load. A
// tile none of the warp's queries attends is not computed. Operands binary16,
// sums float32, the scale (and softcap) on the result. Parameters: pQ, pK, pN
// (not read), pOut (the plane), pTab, pRow.
func PagedAttnScoresMMA70(s FlashShape, mt, nt int) (*ir.Kernel, error) {
	if err := prefillCheck("PagedAttnScoresMMA70", s); err != nil {
		return nil, err
	}
	W, P, kvRow := splitsOf(s)*s.Chunk, s.Page, s.KVHeads*s.Dim
	rowMajor := s.MLA > 0
	if mt < 1 || nt < 1 || s.Dim%4 != 0 || s.Rows%(8*nt) != 0 || W%(32*mt) != 0 || rowMajor && kvRow%4 != 0 {
		return nil, fmt.Errorf("kernels: PagedAttnScoresMMA70: %+v mt=%d nt=%d", s, mt, nt)
	}
	gqa := s.Heads / s.KVHeads
	b := ir.New(fmt.Sprintf("pagedscoresmma70_m%dn%d", mt, nt), [3]int{128, 1, 1})
	pQ, pK := b.Param("pQ", ir.F32), b.Param("pK", ir.F32)
	b.Param("pN", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	pTab, pRow := b.Param("pTab", ir.U32), b.Param("pRow", ir.U32)
	c := func(v int) ir.Value { return b.Const(ir.U32, int64(v)) }
	add := func(a, x ir.Value) ir.Value { return b.Add(ir.U32, a, x) }
	mul := func(a, x ir.Value) ir.Value { return b.Mul(ir.U32, a, x) }
	ktiles, qgroups := W/(32*mt), s.Rows/(8*nt)
	tid := add(mul(b.CTAID(), b.NTID()), b.TID())
	warp := b.Min(ir.U32, b.Shr(ir.U32, tid, c(5)), c(ktiles*s.Heads*qgroups-1))
	v := voltaLane{b, b.And(ir.U32, tid, c(31))}
	kt := urem(b, warp, ktiles)
	rest := udiv(b, warp, ktiles)
	h := urem(b, rest, s.Heads)
	qryW := mul(udiv(b, rest, s.Heads), c(8*nt)) // + 8j + row
	sp := prefillSpanOf(b, pRow, qryW, add(qryW, c(8*nt-1)))
	q8 := b.Shl(ir.U32, v.quad(), c(3))
	tile := mul(kt, c(32*mt))
	keyW := add(tile, q8) // a slot: + 32i + row
	kvh := mul(udiv(b, h, gqa), c(s.Dim))
	laneKey := add(q8, v.row())
	kBase := make([]ir.Value, mt)
	for i := range kBase {
		T := sp.clamp(b, add(sp.base, add(tile, c(32*i))), 32)
		pid := pageOf(b, pTab, sp.tab, T, P)
		if rowMajor {
			kBase[i] = add(mul(add(add(mul(pid, c(P)), pageRem(b, T, P)), laneKey), c(kvRow)), kvh)
		} else {
			kBase[i] = add(add(mul(pid, c(P*kvRow)), mul(kvh, c(P))), add(pageRem(b, T, P), laneKey))
		}
	}
	qBase := mul(add(mul(add(qryW, v.row()), c(s.Heads)), h), c(s.Dim))
	T0 := add(sp.base, tile)
	live := b.Select(ir.U32, b.Lt(ir.U32, T0, sp.hi), b.Select(ir.U32, b.Lt(ir.U32, sp.lo, add(T0, c(32*mt))), c(1), c(0)), c(0))
	b.LoopN(live)
	zero := b.ConstF32(0)
	acc := make([][][]ir.Value, mt)
	for i := range acc {
		acc[i] = make([][]ir.Value, nt)
		for j := range acc[i] {
			acc[i][j] = make([]ir.Value, 8)
			for x := range acc[i][j] {
				acc[i][j][x] = zero
			}
		}
	}
	for st := 0; st < s.Dim/4; st++ {
		af := make([][]ir.Value, mt)
		for i := range af {
			var e []ir.Value
			if rowMajor {
				e = b.LoadV(ir.F32, pK, kBase[i], int64(4*st), 4)
			} else {
				e = make([]ir.Value, 4)
				for x := range e {
					e[x] = b.Load(ir.F32, pK, kBase[i], int64(4*st+x)*int64(P))
				}
			}
			af[i] = []ir.Value{b.PackF16(e[0], e[1]), b.PackF16(e[2], e[3])}
		}
		bf := make([][]ir.Value, nt)
		for j := range bf {
			e := b.LoadV(ir.F32, pQ, qBase, int64(8*j*s.Heads*s.Dim+4*st), 4)
			bf[j] = []ir.Value{b.PackF16(e[0], e[1]), b.PackF16(e[2], e[3])}
		}
		for i := 0; i < mt; i++ {
			for j := 0; j < nt; j++ {
				acc[i][j] = b.MMA(ir.MMAVolta(), af[i], bf[j], acc[i][j])
			}
		}
	}
	negInf := b.Bitcast(ir.F32, c(0xFF800000))
	dRow := []ir.Value{add(keyW, v.dRow(0)), add(keyW, v.dRow(2))}
	dCol := []ir.Value{add(qryW, v.dCol(0)), add(qryW, v.dCol(1)), add(qryW, v.dCol(4)), add(qryW, v.dCol(5))}
	for j := 0; j < nt; j++ {
		var ks, ke, row [4]ir.Value
		for x := range ks {
			q := add(dCol[x], c(8*j))
			ks[x], ke[x] = pagedDesc(b, pRow, q, PRowStart), pagedDesc(b, pRow, q, PRowEnd)
			row[x] = mul(add(mul(q, c(s.Heads)), h), c(W))
		}
		for i := 0; i < mt; i++ {
			for comp := 0; comp < 8; comp++ {
				slot := add(dRow[(comp>>1)&1], c(32*i))
				x := (comp & 1) + 2*(comp>>2)
				ok := rowAttends(b, add(sp.base, slot), ks[x], ke[x])
				b.Store(pOut, add(row[x], slot), b.Select(ir.F32, ok, softcapped(b, acc[i][j][comp], s), negInf), 0)
			}
		}
	}
	b.EndLoop()
	return b.Done(), nil
}

// PagedAttnAccMMA70Warps is how many warps PagedAttnAccMMA70 launches (groups
// of 128 threads).
func PagedAttnAccMMA70Warps(s FlashShape, mt, nt int) int {
	return stagedValue(s) / (32 * mt) * s.Heads * s.Rows / (8 * nt) * splitsOf(s)
}

// PagedAttnAccMMA70 is AttnAccMMA70 over paged KV, one split at a time: V^T
// the A operand (a lane's row one dim at four consecutive keys) and the
// weights' transpose the B, 32*mt dims by 8*nt queries a warp, over the keys
// of its split its queries attend. Four keys from a multiple of four lie in
// one page. The first and last steps are masked -- V selected to 0 outside the
// split's part of the tile's keys (0 * NaN is NaN), the weights with it --
// with every load clamped into the tile's pages and the split's chunk; the
// steps between are wholly inside and read a page id each, a step ahead.
// Packed f16 V is read a half at a time. It writes the numerators into the
// merge's partials, or in the direct form (Splits 0) the output,
// [row][head][value]. The softmax is built with qt = 8*nt. Parameters: pA
// (the weights), pV, pN (not read), pPart (the output in the direct form),
// pTab, pRow.
func PagedAttnAccMMA70(s FlashShape, mt, nt int) (*ir.Kernel, error) {
	if err := prefillCheck("PagedAttnAccMMA70", s); err != nil {
		return nil, err
	}
	vd := stagedValue(s)
	if mt < 1 || nt < 1 || vd%(32*mt) != 0 || s.Rows%(8*nt) != 0 {
		return nil, fmt.Errorf("kernels: PagedAttnAccMMA70: %+v mt=%d nt=%d", s, mt, nt)
	}
	P, C, S := s.Page, s.Chunk, splitsOf(s)
	kvRow, gqa := s.KVHeads*s.Dim, s.Heads/s.KVHeads
	b := ir.New(fmt.Sprintf("pagedaccmma70_m%dn%d", mt, nt), [3]int{128, 1, 1})
	pA, pV := b.Param("pA", ir.F32), b.Param("pV", ir.F32)
	b.Param("pN", ir.U32)
	pPart := b.Param("pPart", ir.F32)
	pTab, pRow := b.Param("pTab", ir.U32), b.Param("pRow", ir.U32)
	c := func(v int) ir.Value { return b.Const(ir.U32, int64(v)) }
	add := func(a, x ir.Value) ir.Value { return b.Add(ir.U32, a, x) }
	mul := func(a, x ir.Value) ir.Value { return b.Mul(ir.U32, a, x) }
	dblocks, qgroups := vd/(32*mt), s.Rows/(8*nt)
	tid := add(mul(b.CTAID(), b.NTID()), b.TID())
	warp := b.Min(ir.U32, b.Shr(ir.U32, tid, c(5)), c(dblocks*S*s.Heads*qgroups-1))
	v := voltaLane{b, b.And(ir.U32, tid, c(31))}
	db := urem(b, warp, dblocks)
	rest := udiv(b, warp, dblocks)
	spl := urem(b, rest, S)
	rest = udiv(b, rest, S)
	h := urem(b, rest, s.Heads)
	qryW := mul(udiv(b, rest, s.Heads), c(8*nt))
	sp := prefillSpanOf(b, pRow, qryW, add(qryW, c(8*nt-1)))
	q8 := b.Shl(ir.U32, v.quad(), c(3))
	dimW := add(mul(db, c(32*mt)), q8)
	col := add(dimW, v.row()) // + 32i: the lane's A row, a dim of V
	if s.MLAValueTail {
		col = add(col, c(s.Dim-s.MLA))
	}
	if s.MLA == 0 {
		col = add(mul(udiv(b, h, gqa), c(s.Dim)), col)
	}
	// The split's part of the group's keys, [st, en), from the 4-aligned step
	// holding st: n steps.
	c0 := add(sp.base, mul(spl, c(C)))
	st := b.Min(ir.U32, b.Max(ir.U32, sp.lo, c0), add(c0, c(C)))
	en := b.Max(ir.U32, b.Min(ir.U32, sp.hi, add(c0, c(C))), st)
	a4 := add(c0, mul(udiv(b, b.Sub(ir.U32, st, c0), 4), c(4)))
	n := b.Select(ir.U32, b.Lt(ir.U32, st, en), udiv(b, add(b.Sub(ir.U32, en, a4), c(3)), 4), c(0))
	mid := b.Sub(ir.U32, b.Max(ir.U32, n, c(2)), c(2))
	// Query qryW+row's weights of this split: ((q*Heads + h)*S + spl)*C, 8j
	// queries on, at the slot.
	wRow := mul(add(mul(add(mul(add(qryW, v.row()), c(s.Heads)), h), c(S)), spl), c(C))
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
	// stepAt adds the 4 keys from pos. Masked: key pos+x is kept in [st, en),
	// its V read at its own clamped position's page, the weights' slot clamped
	// into the chunk. Unmasked: the keys' V row starts at element vp and their
	// weights at wp, both walked by pointer as the contiguous kernel does.
	stepAt := func(pos, vp, wp ir.Value, d [][][]ir.Value, masked bool) {
		var ok [4]ir.Value
		var slot ir.Value
		if masked {
			slot = b.Sub(ir.U32, pos, c0)
			for x := range ok {
				ok[x] = b.Lt(ir.U32, b.Sub(ir.U32, add(pos, c(x)), st), b.Sub(ir.U32, en, st))
				if pagedFault == "vmask" {
					ok[x] = sp.in(b, add(pos, c(x)))
				}
			}
			slot = b.Min(ir.U32, slot, c(C-4))
		}
		var vw, vsh ir.Value
		if !masked && s.F16 {
			// A lane's elements all have vp's parity: kvRow and 32 are even.
			vw, vsh = b.Shr(ir.U32, vp, c(1)), mul(b.And(ir.U32, vp, c(1)), c(16))
		}
		// A masked step's four keys lie in one page, as every 4-aligned step
		// does: its page is the clamped step's, and each key is read at its
		// own offset in it -- in bounds, and selected away when outside.
		var mpid, mpos ir.Value
		if masked {
			mpos = sp.clamp(b, pos, 4)
			mpid = pageOf(b, pTab, sp.tab, mpos, P)
		}
		af := make([][]ir.Value, mt)
		for i := range af {
			e := make([]ir.Value, 4)
			for x := range e {
				if !masked {
					imm := x*kvRow + 32*i
					if s.F16 {
						word := b.Bitcast(ir.U32, b.Load(ir.F32, pV, vw, int64(imm/2)))
						e[x] = b.CvtF16H(b.Shr(ir.U32, word, vsh))
					} else {
						e[x] = b.Load(ir.F32, pV, vp, int64(imm))
					}
					continue
				}
				e[x] = b.Select(ir.F32, ok[x], vAt(mpid, add(mpos, c(x)), col, 32*i), zero)
			}
			af[i] = []ir.Value{b.PackF16(e[0], e[1]), b.PackF16(e[2], e[3])}
		}
		bf := make([][]ir.Value, nt)
		for j := range bf {
			wa := wp
			if masked {
				wa = add(wRow, slot)
			}
			e := b.LoadV(ir.F32, pA, wa, int64(j)*wStep, 4)
			if masked {
				for x := range e {
					e[x] = b.Select(ir.F32, ok[x], e[x], zero)
				}
			}
			bf[j] = []ir.Value{b.PackF16(e[0], e[1]), b.PackF16(e[2], e[3])}
		}
		for i := 0; i < mt; i++ {
			for j := 0; j < nt; j++ {
				d[i][j] = b.MMA(ir.MMAVolta(), af[i], bf[j], d[i][j])
			}
		}
	}
	fresh := func(init [][][]ir.Value, f func(ir.Value) ir.Value) [][][]ir.Value {
		out := make([][][]ir.Value, mt)
		for i := range out {
			out[i] = make([][]ir.Value, nt)
			for j := range out[i] {
				out[i][j] = make([]ir.Value, 8)
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
			acc[i][j] = []ir.Value{zero, zero, zero, zero, zero, zero, zero, zero}
		}
	}
	// The first step, masked.
	stepAt(a4, 0, 0, acc, true)
	// The steps between, [m0, m1), unmasked: a run of them per page, the
	// page's id read once and V and the weights walked by pointer inside it.
	m0 := add(a4, c(4))
	m1 := add(m0, mul(mid, c(4)))
	pg0 := pageDiv(b, m0, P)
	npg := b.Select(ir.U32, b.Lt(ir.U32, c(0), mid), add(b.Sub(ir.U32, pageDiv(b, b.Sub(ir.U32, m1, c(1)), P), pg0), c(1)), c(0))
	b.LoopN(npg)
	ph := fresh(acc, func(x ir.Value) ir.Value { return b.Phi(ir.F32, x) })
	pg := b.Phi(ir.U32, pg0)
	pid := b.Load(ir.U32, pTab, add(sp.tab, pg), 0)
	// The page's part of [m0, m1), in steps of 4 from m0 (exactly the page's
	// span, since m0 and a page edge are both multiples of 4 from c0).
	from := func(t ir.Value) ir.Value {
		return add(m0, mul(udiv(b, add(b.Sub(ir.U32, b.Max(ir.U32, m0, t), m0), c(3)), 4), c(4)))
	}
	s0 := from(mul(pg, c(P)))
	s1 := b.Min(ir.U32, m1, from(mul(add(pg, c(1)), c(P))))
	vp0 := add(mul(add(mul(pid, c(P)), pageRem(b, s0, P)), c(kvRow)), col)
	wp0 := add(wRow, b.Sub(ir.U32, s0, c0))
	b.LoopN(udiv(b, b.Sub(ir.U32, s1, s0), 4))
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
	b.SetPhi(vp, add(vp, c(4*kvRow)))
	b.SetPhi(wp, add(wp, c(4)))
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
	stepAt(add(a4, mul(add(mid, c(1)), c(4))), 0, 0, d, true)
	padded := (vd + 31) / 32 * 32
	dRow := []ir.Value{add(dimW, v.dRow(0)), add(dimW, v.dRow(2))}
	dCol := []ir.Value{add(qryW, v.dCol(0)), add(qryW, v.dCol(1)), add(qryW, v.dCol(4)), add(qryW, v.dCol(5))}
	for i := 0; i < mt; i++ {
		for j := 0; j < nt; j++ {
			for comp := 0; comp < 8; comp++ {
				dim := add(dRow[(comp>>1)&1], c(32*i))
				q := add(dCol[(comp&1)+2*(comp>>2)], c(8*j))
				o := add(mul(add(mul(add(mul(q, c(s.Heads)), h), c(S)), spl), c(padded+64)), dim)
				if s.Splits == 0 {
					o = add(mul(add(mul(q, c(s.Heads)), h), c(vd)), dim)
				}
				b.Store(pPart, o, d[i][j][comp], 0)
			}
		}
	}
	return b.Done(), nil
}
