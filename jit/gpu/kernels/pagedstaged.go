package kernels

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/jit/ir"
)

// The staged decode over paged KV, in bounded chunks (docs/design/
// device-kv-paging.md): three launches and a merge in place of
// scores/softmax/accumulate over a context-wide score plane.
//
//	PagedStagedScores   scores of each split's keys into a plane of Chunk
//	                    slots per (row, head, split)
//	PagedStagedSoftmax  each split's maximum and denominator, into the
//	                    merge's partials, and its UNNORMALISED weights
//	PagedStagedAcc      each split's weighted sum of V, into the partials
//	FlashAttentionMerge the splits combined; sinks and the normalisation
//	                    applied there, once
//
// A split's keys are FlashDecodeKV's partition: the row's keys from the
// 32-aligned position at or below keyStart, divided into s.Splits runs of
// whole 32-key tiles, each at most s.Chunk long -- so the caller picks
// Splits*Chunk >= keyEnd (the plane is Rows*Heads*Splits*Chunk floats, a
// function of the split count, never of a capacity). Keys below keyStart are
// masked; an empty row has no keys, and every split of it is (max -MaxFloat32,
// denominator 0), which the merge turns into 0.
//
// Every kernel takes pTab and pRow after its existing parameters. The shape's
// fields are FlashShape's; Chunk is the per-split bound, and MLA > 0 selects
// the MLA layout: one row-major region [P][kvRow] (KVHeads 1, Dim the whole
// row) whose first MLA elements are the value, so the accumulate and the
// merge run at width MLA.

// stagedCheck validates a staged paged shape.
func stagedCheck(what string, s FlashShape) error {
	if s.Page == 0 {
		return fmt.Errorf("kernels: %s: a staged paged shape needs a Page", what)
	}
	if err := flashPaged(what, s); err != nil {
		return err
	}
	if s.Heads < 1 || s.KVHeads < 1 || s.Heads%s.KVHeads != 0 || s.Dim < 1 || s.Rows < 1 || s.Splits < 1 ||
		s.Chunk < 32 || s.Chunk%32 != 0 || s.Softcap < 0 {
		return fmt.Errorf("kernels: %s: invalid shape %+v", what, s)
	}
	if s.MLA > 0 && (s.KVHeads != 1 || s.MLA > s.Dim || s.F16) {
		return fmt.Errorf("kernels: %s: MLA wants one KV head, a value within the row and f32: %+v", what, s)
	}
	if s.F16 && s.Dim%2 != 0 {
		return fmt.Errorf("kernels: %s: packed f16 V needs an even Dim: %+v", what, s)
	}
	return nil
}

// PagedStagedPlane is the score and weight planes' size in floats.
func PagedStagedPlane(s FlashShape) int { return s.Rows * s.Heads * s.Splits * s.Chunk }

// stagedValue is the width the accumulate and the merge run at.
func stagedValue(s FlashShape) int {
	if s.MLA > 0 {
		return s.MLA
	}
	return s.Dim
}

// stagedSplit is split sp's keys [st, en) of a row with keys [ks, ke), and
// the last key the row may address. See FlashDecodeKV's partitions.
type stagedSplit struct{ tab, ks, st, en, last, live ir.Value }

func pagedSplitOf(b *ir.Builder, pRow, row, sp ir.Value, splits int) stagedSplit {
	u := func(x int) ir.Value { return b.Const(ir.U32, int64(x)) }
	tab, ks, ke := pagedDesc(b, pRow, row, PRowTab), pagedDesc(b, pRow, row, PRowStart), pagedDesc(b, pRow, row, PRowEnd)
	a := b.Mul(ir.U32, b.Div(ir.U32, ks, u(32)), u(32))
	if pagedFault == "unaligned" {
		a = ks
	}
	live := b.Lt(ir.U32, ks, ke)
	span := b.Select(ir.U32, live, b.Sub(ir.U32, ke, a), u(0))
	per := b.Div(ir.U32, b.Add(ir.U32, span, u(splits-1)), u(splits))
	per = b.Mul(ir.U32, b.Div(ir.U32, b.Add(ir.U32, per, u(31)), u(32)), u(32))
	st := b.Min(ir.U32, ke, b.Add(ir.U32, a, b.Mul(ir.U32, sp, per)))
	en := b.Min(ir.U32, ke, b.Add(ir.U32, st, per))
	en = b.Select(ir.U32, live, en, st)
	last := b.Sub(ir.U32, b.Max(ir.U32, ke, u(1)), u(1))
	return stagedSplit{tab: tab, ks: ks, st: st, en: en, last: last, live: live}
}

// key is where key st+slot of the split is read: clamped into [keyStart,
// last] so a masked slot addresses the row's own pages, and to 0 for an empty
// row, whose table is the dummy page.
func (p stagedSplit) key(b *ir.Builder, slot ir.Value) ir.Value {
	t := b.Min(ir.U32, b.Max(ir.U32, b.Add(ir.U32, p.st, slot), p.ks), p.last)
	return b.Select(ir.U32, p.live, t, b.Const(ir.U32, 0))
}

// valid is whether slot holds one of the row's keys.
func (p stagedSplit) valid(b *ir.Builder, slot ir.Value) ir.Value {
	return b.Lt(ir.U32, b.Sub(ir.U32, b.Add(ir.U32, p.st, slot), p.ks), b.Sub(ir.U32, p.en, p.ks))
}

// stagedGroup is how many query heads of one KV head a staged thread serves,
// so one K or V load feeds that many products: s.Group when set, else one per
// 64 slots of a chunk up to 8 -- the grouping divides the threads, and a short
// chunk (a short context) has too few to divide (at 512 keys one head a
// thread wins, at 4096 two to four). Rounded down to a
// divisor of the KV head's group.
func stagedGroup(s FlashShape) int {
	gqa := s.Heads / s.KVHeads
	g := min(8, gqa, max(1, s.Chunk/64))
	if s.Group > 0 {
		g = min(s.Group, gqa)
	}
	for gqa%g != 0 {
		g--
	}
	return g
}

// PagedStagedScoreItems is how many work items PagedStagedScores covers: a
// thread each at lanes 1 (groups of 128), a warp each at lanes 32 (four to a
// group of 128).
func PagedStagedScoreItems(s FlashShape) int {
	return s.Rows * (s.Heads / stagedGroup(s)) * s.Splits * s.Chunk
}

// PagedStagedAccThreads is how many threads PagedStagedAcc takes (groups of
// 128).
func PagedStagedAccThreads(s FlashShape) int {
	per := stagedValue(s)
	if s.F16 {
		per /= 2
	}
	return s.Rows * (s.Heads / stagedGroup(s)) * s.Splits * per
}

// PagedStagedScores writes the scaled (and softcapped) q.k of every slot of
// every split into pOut, [row][head][split][Chunk], -MaxFloat32 where the slot
// holds no key of the row. A thread (lanes 1) or a warp across the dimensions
// (lanes 32, for a row-major MLA row, on a device that guarantees the width)
// takes one slot for stagedGroup(s) query heads of one KV head, so a K load
// serves all of them; a transposed page's keys are adjacent, so a warp's K
// loads coalesce. Parameters: pQ, pK, pN (not read), pOut, pTab, pRow.
func PagedStagedScores(s FlashShape, lanes int) (*ir.Kernel, error) {
	if err := stagedCheck("PagedStagedScores", s); err != nil {
		return nil, err
	}
	if lanes != 1 && lanes != 32 {
		return nil, fmt.Errorf("kernels: PagedStagedScores: lanes=%d, want 1 or 32", lanes)
	}
	P, C, S := s.Page, s.Chunk, s.Splits
	kvRow := s.KVHeads * s.Dim
	gqa, G := s.Heads/s.KVHeads, stagedGroup(s)
	b := ir.New("pagedscores", [3]int{max(lanes, 128), 1, 1})
	pQ, pK := b.Param("pQ", ir.F32), b.Param("pK", ir.F32)
	b.Param("pN", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	pTab, pRow := b.Param("pTab", ir.U32), b.Param("pRow", ir.U32)
	u := func(x int) ir.Value { return b.Const(ir.U32, int64(x)) }
	add := func(a, c ir.Value) ir.Value { return b.Add(ir.U32, a, c) }
	mul := func(a, c ir.Value) ir.Value { return b.Mul(ir.U32, a, c) }
	// Work items (row, head group, split, slot), slot fastest.
	total := s.Rows * (s.Heads / G) * S * C
	var flat, lane ir.Value
	if lanes == 1 {
		flat = b.Min(ir.U32, add(mul(b.CTAID(), b.NTID()), b.TID()), u(total-1))
	} else {
		// 128 threads: four warps, a slot each.
		flat = b.Min(ir.U32, add(mul(b.CTAID(), u(4)), b.Div(ir.U32, b.TID(), u(32))), u(total-1))
		lane = b.Rem(ir.U32, b.TID(), u(32))
	}
	slot := b.Rem(ir.U32, flat, u(C))
	rest := b.Div(ir.U32, flat, u(C))
	sp := b.Rem(ir.U32, rest, u(S))
	rest = b.Div(ir.U32, rest, u(S))
	hg := b.Rem(ir.U32, rest, u(s.Heads/G)) // head group; its first head is hg*G
	row := b.Div(ir.U32, rest, u(s.Heads/G))
	ps := pagedSplitOf(b, pRow, row, sp, S)
	t := ps.key(b, slot)
	pid := pageOf(b, pTab, ps.tab, t, P)
	h0 := mul(hg, u(G))
	kvh := mul(b.Div(ir.U32, h0, u(gqa)), u(s.Dim))
	rh0 := add(mul(row, u(s.Heads)), h0) // (row*Heads + first head)
	qb := mul(rh0, u(s.Dim))
	// The position's K: element d at kb + d*step.
	var kb ir.Value
	step := 1
	if s.MLA > 0 {
		kb = mul(add(mul(pid, u(P)), pageRem(b, t, P)), u(kvRow))
	} else if s.F16K {
		// Words, a pair of dimensions each: pair d/2 at kb + (d/2)*P.
		kb = add(add(mul(pid, u(P*kvRow/2)), mul(b.Shr(ir.U32, kvh, u(1)), u(P))), pageRem(b, t, P))
		step = P
	} else {
		kb = add(add(mul(pid, u(P*kvRow)), mul(kvh, u(P))), pageRem(b, t, P))
		step = P
	}
	// kAt2 is dimensions d and d+1 (d even) of binary16 K from base kb.
	kAt2 := func(kb ir.Value, d int) (ir.Value, ir.Value) {
		return kPair(b, b.Load(ir.F32, pK, kb, int64(d/2*step)))
	}
	dots := make([]ir.Value, G)
	if lanes == 1 {
		// Four chains for one head; one per head (G independent) otherwise.
		chains := max(1, 4/G)
		acc := make([][]ir.Value, G)
		for g := range acc {
			acc[g] = make([]ir.Value, chains)
			for c := range acc[g] {
				acc[g][c] = b.ConstF32(0)
			}
		}
		// More than 512 products a thread go in a loop over blocks of 32
		// dimensions, the rest unrolled after it. One head is best unrolled at
		// short contexts, and four heads are best in the loop at long ones.
		const blk = 32
		unrolled := 0 // the dimensions the loop does not cover start here
		if nb := s.Dim / blk; G*s.Dim > 512 && nb > 1 {
			unrolled = nb * blk
			b.Loop(int64(nb))
			ki := b.Phi(ir.U32, kb)
			qi := b.Phi(ir.U32, qb)
			ph := make([][]ir.Value, G)
			for g := range ph {
				ph[g] = make([]ir.Value, chains)
				for c := range ph[g] {
					ph[g][c] = b.Phi(ir.F32, acc[g][c])
				}
			}
			nx := make([][]ir.Value, G)
			for g := range nx {
				nx[g] = append([]ir.Value(nil), ph[g]...)
			}
			for d := 0; d < blk; d++ {
				var kv ir.Value
				if s.F16K {
					if d%2 == 1 {
						continue
					}
					lo, hi := kAt2(ki, d)
					for g := range nx {
						nx[g][d%chains] = b.Fma(b.Load(ir.F32, pQ, qi, int64(g*s.Dim+d)), lo, nx[g][d%chains])
						nx[g][(d+1)%chains] = b.Fma(b.Load(ir.F32, pQ, qi, int64(g*s.Dim+d+1)), hi, nx[g][(d+1)%chains])
					}
					continue
				}
				kv = b.Load(ir.F32, pK, ki, int64(d*step))
				for g := range nx {
					nx[g][d%chains] = b.Fma(b.Load(ir.F32, pQ, qi, int64(g*s.Dim+d)), kv, nx[g][d%chains])
				}
			}
			for g := range ph {
				for c := range ph[g] {
					b.SetPhi(ph[g][c], nx[g][c])
				}
			}
			kAdv := blk * step
			if s.F16K {
				kAdv = blk / 2 * step
			}
			b.SetPhi(ki, add(ki, u(kAdv)))
			b.SetPhi(qi, add(qi, u(blk)))
			b.EndLoop()
			acc = ph
		}
		for d := unrolled; d < s.Dim; d++ {
			if s.F16K {
				if (d-unrolled)%2 == 1 {
					continue
				}
				lo, hi := kAt2(kb, d)
				for g := range acc {
					acc[g][d%chains] = b.Fma(b.Load(ir.F32, pQ, qb, int64(g*s.Dim+d)), lo, acc[g][d%chains])
					acc[g][(d+1)%chains] = b.Fma(b.Load(ir.F32, pQ, qb, int64(g*s.Dim+d+1)), hi, acc[g][(d+1)%chains])
				}
				continue
			}
			kv := b.Load(ir.F32, pK, kb, int64(d*step))
			for g := range acc {
				acc[g][d%chains] = b.Fma(b.Load(ir.F32, pQ, qb, int64(g*s.Dim+d)), kv, acc[g][d%chains])
			}
		}
		for g := range dots {
			dots[g] = acc[g][0]
			for c := 1; c < chains; c++ {
				dots[g] = b.Add(ir.F32, dots[g], acc[g][c])
			}
		}
	} else {
		// Lanes stride the dimensions; a lane past the last one multiplies a
		// clamped element by zero.
		acc := make([]ir.Value, G)
		for g := range acc {
			acc[g] = b.ConstF32(0)
		}
		for i := 0; i*32 < s.Dim; i++ {
			d := add(lane, u(i*32))
			dc := b.Min(ir.U32, d, u(s.Dim-1))
			var kv ir.Value
			if s.F16K {
				w := b.Bitcast(ir.U32, b.Load(ir.F32, pK, add(kb, mul(b.Shr(ir.U32, dc, u(1)), u(step))), 0))
				kv = kHalf(b, w, dc)
			} else {
				kv = b.Load(ir.F32, pK, add(kb, mul(dc, u(step))), 0)
			}
			if (i+1)*32 > s.Dim {
				kv = b.Select(ir.F32, b.Lt(ir.U32, d, u(s.Dim)), kv, b.ConstF32(0))
			}
			for g := range acc {
				acc[g] = b.Fma(b.Load(ir.F32, pQ, add(qb, dc), int64(g*s.Dim)), kv, acc[g])
			}
		}
		for g := range dots {
			dots[g] = butterfly(b, acc[g], func(x, y ir.Value) ir.Value { return b.Add(ir.F32, x, y) })
		}
	}
	valid := ps.valid(b, slot)
	// Head h0+g's slot: ((rh0+g)*S + sp)*C + slot.
	ob := add(mul(add(mul(rh0, u(S)), sp), u(C)), slot)
	for g, dot := range dots {
		score := b.Mul(ir.F32, dot, b.ConstF32(s.Scale))
		if s.Softcap > 0 {
			e := b.Exp(b.Mul(ir.F32, b.ConstF32(2/s.Softcap), score))
			score = b.Mul(ir.F32, b.ConstF32(s.Softcap), b.Sub(ir.F32, b.ConstF32(1), b.Div(ir.F32, b.ConstF32(2), b.Add(ir.F32, e, b.ConstF32(1)))))
		}
		b.Store(pOut, ob, b.Select(ir.F32, valid, score, b.ConstF32(-math.MaxFloat32)), int64(g*S*C))
	}
	return b.Done()
}

// PagedStagedSoftmax reduces each split's slots: its maximum m and its
// denominator l go to the merge's partials (FlashAttentionMerge's layout, at
// the value width), and exp(score - m) -- unnormalised, 0 where the slot holds
// no key -- to pOut. The sink is not applied here: the merge applies it once.
// lanes 32 is a warp per split on a device that guarantees the width, 1 a
// thread. Parameters: pA (the scores), pN (not read), pOut (the weights),
// pPart, pTab (not read), pRow.
func PagedStagedSoftmax(s FlashShape, lanes int) (*ir.Kernel, error) {
	if err := stagedCheck("PagedStagedSoftmax", s); err != nil {
		return nil, err
	}
	if lanes != 1 && lanes != 32 {
		return nil, fmt.Errorf("kernels: PagedStagedSoftmax: lanes=%d, want 1 or 32", lanes)
	}
	C, S := s.Chunk, s.Splits
	items := s.Rows * s.Heads * S
	b := ir.New("pagedsoftmax", [3]int{max(lanes, 64), 1, 1})
	pA := b.Param("pA", ir.F32)
	b.Param("pN", ir.U32)
	pOut, pPart := b.Param("pOut", ir.F32), b.Param("pPart", ir.F32)
	b.Param("pTab", ir.U32)
	pRow := b.Param("pRow", ir.U32)
	u := func(x int) ir.Value { return b.Const(ir.U32, int64(x)) }
	add := func(a, c ir.Value) ir.Value { return b.Add(ir.U32, a, c) }
	var item, lane, step ir.Value
	if lanes == 32 {
		// The group is two warps, a split each.
		item = b.Min(ir.U32, add(b.Mul(ir.U32, b.CTAID(), u(2)), b.Div(ir.U32, b.TID(), u(32))), u(items-1))
		lane, step = b.Rem(ir.U32, b.TID(), u(32)), u(32)
	} else {
		item = b.Min(ir.U32, add(b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()), u(items-1))
		lane, step = u(0), u(1)
	}
	sp := b.Rem(ir.U32, item, u(S))
	row := b.Div(ir.U32, item, u(s.Heads*S))
	ps := pagedSplitOf(b, pRow, row, sp, S)
	keys := b.Sub(ir.U32, ps.en, ps.st)
	base := b.Mul(ir.U32, item, u(C))
	// This lane's slots: lane, lane+step, ... below keys.
	cnt := b.Div(ir.U32, add(b.Sub(ir.U32, b.Max(ir.U32, keys, lane), lane), b.Sub(ir.U32, step, u(1))), step)
	minus := b.ConstF32(-math.MaxFloat32)
	b.LoopN(cnt)
	mx := b.Phi(ir.F32, minus)
	mi := b.Phi(ir.U32, lane)
	b.SetPhi(mx, b.Max(ir.F32, mx, b.Load(ir.F32, pA, add(base, mi), 0)))
	b.SetPhi(mi, add(mi, step))
	b.EndLoop()
	m := mx
	if lanes == 32 {
		m = butterfly(b, mx, func(x, y ir.Value) ir.Value { return b.Max(ir.F32, x, y) })
	}
	// The weights cover every slot of the chunk, 0 past the split's keys: the
	// accumulate reads whole blocks.
	cntC := b.Div(ir.U32, add(b.Sub(ir.U32, u(C), lane), b.Sub(ir.U32, step, u(1))), step)
	zero := b.ConstF32(0)
	b.LoopN(cntC)
	sum := b.Phi(ir.F32, zero)
	si := b.Phi(ir.U32, lane)
	w := b.Select(ir.F32, ps.valid(b, si), b.Exp(b.Sub(ir.F32, b.Load(ir.F32, pA, add(base, si), 0), m)), zero)
	b.Store(pOut, add(base, si), w, 0)
	b.SetPhi(sum, b.Add(ir.F32, sum, w))
	b.SetPhi(si, add(si, step))
	b.EndLoop()
	l := sum
	if lanes == 32 {
		l = butterfly(b, sum, func(x, y ir.Value) ir.Value { return b.Add(ir.F32, x, y) })
	}
	if pagedFault == "normpart" {
		// The violation: weights normalised per split, so the merge's
		// rescaling by l is applied to already-normalised partials.
		inv := b.Div(ir.F32, b.ConstF32(1), b.Max(ir.F32, l, b.ConstF32(1e-30)))
		b.LoopN(cnt)
		ni := b.Phi(ir.U32, lane)
		b.Store(pOut, add(base, ni), b.Mul(ir.F32, inv, b.Select(ir.F32, ps.valid(b, ni), b.Exp(b.Sub(ir.F32, b.Load(ir.F32, pA, add(base, ni), 0), m)), zero)), 0)
		b.SetPhi(ni, add(ni, step))
		b.EndLoop()
	}
	vd := stagedValue(s)
	padded := (vd + 31) / 32 * 32
	pb := b.Mul(ir.U32, item, u(padded+64))
	if lanes == 32 {
		b.Store(pPart, add(pb, lane), m, int64(padded))
		b.Store(pPart, add(pb, lane), l, int64(padded+32))
	} else {
		for r := 0; r < 32; r++ {
			b.Store(pPart, pb, m, int64(padded+r))
			b.Store(pPart, pb, l, int64(padded+32+r))
		}
	}
	return b.Done()
}

// PagedStagedAcc is each split's weighted sum of V: a thread per (row, head
// group, split, value element) -- two adjacent elements for packed f16 --
// serving stagedGroup(s) query heads, so one V load feeds that many products.
// It walks the split's keys in blocks of accBlock, which start at a multiple of
// it from a 32-aligned partition and so lie in one page: a block's page id is
// read once, an iteration ahead. A key outside the row's live range (below
// keyStart, past the split's end) contributes a V of 0 -- the load stays in
// the block's page and its value is selected away, since 0*NaN is NaN -- and
// the softmax wrote its weight as 0. It writes the numerators into the
// merge's partials. Parameters: pA (the weights), pV, pN (not read), pPart,
// pTab, pRow.
func PagedStagedAcc(s FlashShape) (*ir.Kernel, error) {
	if err := stagedCheck("PagedStagedAcc", s); err != nil {
		return nil, err
	}
	const accBlock = 8
	P, C, S := s.Page, s.Chunk, s.Splits
	kvRow := s.KVHeads * s.Dim
	gqa, G := s.Heads/s.KVHeads, stagedGroup(s)
	vd := stagedValue(s)
	per := vd // elements a thread owns: one, or a word of two
	if s.F16 {
		per = vd / 2
	}
	b := ir.New("pagedacc", [3]int{128, 1, 1})
	pA, pV := b.Param("pA", ir.F32), b.Param("pV", ir.F32)
	b.Param("pN", ir.U32)
	pPart := b.Param("pPart", ir.F32)
	pTab, pRow := b.Param("pTab", ir.U32), b.Param("pRow", ir.U32)
	u := func(x int) ir.Value { return b.Const(ir.U32, int64(x)) }
	add := func(a, c ir.Value) ir.Value { return b.Add(ir.U32, a, c) }
	mul := func(a, c ir.Value) ir.Value { return b.Mul(ir.U32, a, c) }
	flat := b.Min(ir.U32, add(mul(b.CTAID(), b.NTID()), b.TID()), u(s.Rows*(s.Heads/G)*S*per-1))
	j := b.Rem(ir.U32, flat, u(per))
	rest := b.Div(ir.U32, flat, u(per))
	sp := b.Rem(ir.U32, rest, u(S))
	rest = b.Div(ir.U32, rest, u(S))
	hg := b.Rem(ir.U32, rest, u(s.Heads/G))
	row := b.Div(ir.U32, rest, u(s.Heads/G))
	h0 := mul(hg, u(G))
	ps := pagedSplitOf(b, pRow, row, sp, S)
	keys := b.Sub(ir.U32, ps.en, ps.st)
	blocks := b.Div(ir.U32, add(keys, u(accBlock-1)), u(accBlock))
	// The element (f32) or word (f16) this thread reads within a position's V
	// row: its KV head's slice, or the MLA row's prefix.
	col := j
	if s.MLAValueTail {
		col = add(j, u(s.Dim-s.MLA))
	}
	if s.MLA == 0 {
		hb := mul(b.Div(ir.U32, h0, u(gqa)), u(s.Dim))
		if s.F16 {
			col = add(b.Shr(ir.U32, hb, u(1)), j)
		} else {
			col = add(hb, j)
		}
	}
	rowW := kvRow // a position's V row in loaded units
	if s.F16 {
		rowW = kvRow / 2
	}
	// Block `it` starts at st + it*accBlock; the page prefetched for the next
	// one is clamped into the row (and to position 0 for an empty row, whose
	// table is the dummy page).
	lastTile := mul(b.Div(ir.U32, ps.last, u(32)), u(32))
	blockPage := func(it ir.Value) ir.Value {
		t := b.Min(ir.U32, add(ps.st, mul(it, u(accBlock))), lastTile)
		return pageOf(b, pTab, ps.tab, b.Select(ir.U32, ps.live, t, u(0)), P)
	}
	u0, zero := u(0), b.ConstF32(0)
	pid0 := blockPage(u0)
	// Weights of head h0+g: ((row*Heads + h0+g)*S + sp)*C + slot.
	wb := mul(add(mul(add(mul(row, u(s.Heads)), h0), u(S)), sp), u(C))
	// Key st+k is live when (st + k - keyStart) < (en - keyStart), unsigned.
	d0 := b.Sub(ir.U32, ps.st, ps.ks)
	span := b.Sub(ir.U32, ps.en, ps.ks)
	nacc := G
	if s.F16 {
		nacc = 2 * G
	}
	b.LoopN(blocks)
	it := b.Phi(ir.U32, u0)
	pid := b.Phi(ir.U32, pid0)
	acc := make([]ir.Value, nacc)
	for i := range acc {
		acc[i] = b.Phi(ir.F32, zero)
	}
	next := append([]ir.Value(nil), acc...)
	k0 := mul(it, u(accBlock))
	bs := add(ps.st, k0)
	vb := add(mul(add(mul(pid, u(P)), pageRem(b, bs, P)), u(rowW)), col)
	dk := add(d0, k0)
	wk := add(wb, k0)
	for kk := 0; kk < accBlock; kk++ {
		live := b.Lt(ir.U32, add(dk, u(kk)), span)
		raw := b.Load(ir.F32, pV, vb, int64(kk*rowW))
		var v0, v1 ir.Value
		if s.F16 {
			word := b.Bitcast(ir.U32, raw)
			v0 = b.Select(ir.F32, live, b.CvtF16H(word), zero)
			v1 = b.Select(ir.F32, live, b.CvtF16H(b.Shr(ir.U32, word, u(16))), zero)
		} else {
			v0 = b.Select(ir.F32, live, raw, zero)
		}
		for g := 0; g < G; g++ {
			w := b.Load(ir.F32, pA, wk, int64(g*S*C+kk))
			if s.F16 {
				next[2*g] = b.Fma(w, v0, next[2*g])
				next[2*g+1] = b.Fma(w, v1, next[2*g+1])
			} else {
				next[g] = b.Fma(w, v0, next[g])
			}
		}
	}
	for i := range acc {
		b.SetPhi(acc[i], next[i])
	}
	b.SetPhi(pid, blockPage(add(it, u(1))))
	b.SetPhi(it, add(it, u(1)))
	b.EndLoop()
	padded := (vd + 31) / 32 * 32
	// Head h0+g's partial: ((row*Heads + h0+g)*S + sp)*(padded+64).
	pb := mul(add(mul(add(mul(row, u(s.Heads)), h0), u(S)), sp), u(padded+64))
	for g := 0; g < G; g++ {
		o := int64(g * S * (padded + 64))
		if s.F16 {
			b.Store(pPart, add(pb, mul(j, u(2))), acc[2*g], o)
			b.Store(pPart, add(pb, mul(j, u(2))), acc[2*g+1], o+1)
		} else {
			b.Store(pPart, add(pb, j), acc[g], o)
		}
	}
	return b.Done()
}
