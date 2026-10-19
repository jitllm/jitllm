package compose

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// The decode attention over a paged history, formed: the loop nests of one
// query row's attention over its keys, each key's K and V read through the
// history's pages (the paged ABI's fragments: a row's descriptor, a
// position's page through the table, its slot in the page, the binary16 K
// pair). The op's numbers -- heads, KV heads, head width, rows, the page, the
// value and key formats, a sink, a softcap, the score scale -- and the
// schedule the decode plan chose for the step's key count (splits, the group
// of query heads a workgroup serves, warps, the staged chunk, the K address
// form and vector V loads) are one kernels.FlashShape, the plan's
// (kernels.ChooseDecodePlan). Two schedules, the plan's two paths:
//
//	Attend         one workgroup per (row, query head group, key split):
//	               scores, the maximum, the weights and their sum, and the
//	               values in one kernel; normalised, or a split's partials
//	StagedScores   the staged path's three kernels over chunks of keys, a
//	StagedSoftmax  score plane and a weight plane between them, writing the
//	StagedAcc      merge's partials
//	Merge          the splits' partials combined, the sink added once
//
// Each is the arithmetic of the menu kernel the plan named (FlashDecodeKV,
// PagedStagedScores/Softmax/Acc, FlashAttentionMerge and its wide form) in
// the same order, so a formed attention computes the same bits.

// Attend is one paged decode attention under its plan's flash schedule.
//
// Parameters: pQ, pK (the K pool), pV (the V pool), pN (not read), pOut (the
// output, or the partials where S.Splits > 1), pTab, pRow, then pSink where
// S.Sink.
type Attend struct{ S kernels.FlashShape }

// Encode is the attention's numbers and schedule, for the kernel cache.
func (a *Attend) Encode() []uint32 { return append([]uint32{20}, encShape(a.S)...) }

// Width is the workgroup; Groups the grid.
func (a *Attend) Width() int { return kernels.FlashKVWidth(a.S) }

// Groups is the workgroups the attention launches.
func (a *Attend) Groups() int { return kernels.FlashKVGroups(a.S) }

// encShape is a plan's numbers as integers.
func encShape(s kernels.FlashShape) []uint32 {
	return []uint32{uint32(s.Heads), uint32(s.Dim), uint32(s.KVHeads), uint32(s.Rows), uint32(s.KStride),
		uint32(s.Window), b2u(s.Chunked), uint32(s.Splits), uint32(s.Group), uint32(s.Warps), f2u(s.Scale),
		f2u(s.Softcap), b2u(s.F16), b2u(s.Sink), b2u(s.Map), uint32(s.Page), uint32(s.Chunk), uint32(s.MLA),
		b2u(s.MLAValueTail), b2u(s.KImm), b2u(s.VecV), b2u(s.F16K)}
}

// Emit forms the kernel.
func (a *Attend) Emit() (*ir.Kernel, error) {
	s := a.S
	if s.Page == 0 {
		return nil, fmt.Errorf("compose: Attend: a history that is not paged: %+v", s)
	}
	if err := kernels.FlashKVCheck(s); err != nil {
		return nil, err
	}
	T := kernels.FlashKVWidth(s)
	W, Wd, Wk, ds := kernels.FlashKVGeom(s)
	// gqa below is the heads THIS workgroup serves, a slice of its KV head's.
	gqa, splits, C := kernels.FlashKVGroup(s), max(1, s.Splits), kernels.FlashKVChunk(s)
	slices := s.Heads / s.KVHeads / gqa
	P := s.Page
	b := ir.New("composed_attend", [3]int{T, 1, 1})
	q, k, v := b.Param("pQ", ir.F32), b.Param("pK", ir.F32), b.Param("pV", ir.F32)
	b.Param("pN", ir.U32)
	out := b.Param("pOut", ir.F32)
	pTab, pRow := b.Param("pTab", ir.U32), b.Param("pRow", ir.U32)
	var sink ir.Value
	if s.Sink {
		sink = b.Param("pSink", ir.F32)
	}
	// The score plane is [head][slice][C]: each dimension slice's partial
	// dots, and row [head][0] the finished scores.
	sS := b.Shared("s", ir.F32, gqa*Wd*C)
	sMax := b.Shared("mx", ir.F32, gqa*W*32)
	sSum := b.Shared("sum", ir.F32, gqa*W*32)
	u := func(x int) ir.Value { return b.Const(ir.U32, int64(x)) }
	add := func(a, c ir.Value) ir.Value { return b.Add(ir.U32, a, c) }
	mul := func(a, c ir.Value) ir.Value { return b.Mul(ir.U32, a, c) }
	zero, minus, u0 := b.ConstF32(0), b.ConstF32(-math.MaxFloat32), b.Const(ir.U32, 0)
	tid := b.TID()
	wl, warp := b.Rem(ir.U32, tid, u(32)), b.Div(ir.U32, tid, u(32))
	grp := b.CTAID()
	// A launch serves s.Rows descriptors, each a whole decode: the row is the
	// workgroup's major index and qrow its first query head's offset.
	var qrow ir.Value
	row := u0
	if s.Rows > 1 {
		per := s.Heads / gqa * splits
		row = b.Min(ir.U32, b.Div(ir.U32, grp, u(per)), u(s.Rows-1))
		grp = b.Rem(ir.U32, grp, u(per))
		qrow = mul(row, u(s.Heads))
	}
	// The workgroup's first query head; its KV head follows from it.
	h0 := mul(b.Min(ir.U32, b.Div(ir.U32, grp, u(splits)), u(s.Heads/gqa-1)), u(gqa))
	kvh := b.Div(ir.U32, h0, u(gqa*slices))
	sp := b.Rem(ir.U32, grp, u(splits))

	// The row's keys are [keyStart, keyEnd). Its partitions start at the
	// 32-aligned position at or below keyStart, so every 32-key tile lies in
	// one page, and the keys below keyStart are masked. An empty row (keyEnd
	// <= keyStart, a padded row) has no keys and reads nothing.
	tabOff, keyStart := kernels.PagedDesc(b, pRow, row, kernels.PRowTab), kernels.PagedDesc(b, pRow, row, kernels.PRowStart)
	count := kernels.PagedDesc(b, pRow, row, kernels.PRowEnd)
	start := mul(b.Div(ir.U32, keyStart, u(32)), u(32))
	span := b.Select(ir.U32, b.Lt(ir.U32, keyStart, count), b.Sub(ir.U32, count, start), u0)
	per := b.Div(ir.U32, add(span, u(splits-1)), u(splits))
	per = mul(b.Div(ir.U32, add(per, u(31)), u(32)), u(32))
	st := b.Min(ir.U32, count, add(start, mul(sp, per)))
	en := b.Min(ir.U32, count, add(st, per))
	keys := b.Sub(ir.U32, en, st)
	trips := b.Div(ir.U32, add(keys, u(T-1)), u(T))
	lastKey := b.Sub(ir.U32, b.Max(ir.U32, count, u(1)), u(1))
	kvbase := mul(kvh, u(s.Dim))
	at := func(g int, slot ir.Value) ir.Value { return add(slot, u(g*Wd*C)) }
	kvRow := s.KVHeads * s.Dim
	// A tile past the last key (a key group's surplus) reads the last key's
	// tile, whose page the row owns; its slots are never valid.
	lastTile := mul(b.Div(ir.U32, lastKey, u(32)), u(32))

	// 1. Partial scores. Lanes run over keys (a 32-key tile) and warps over
	// Wd slices of the dimensions times Wk groups of tiles, the query slice
	// held in registers.
	wd, wk := b.Rem(ir.U32, warp, u(Wd)), b.Div(ir.U32, warp, u(Wd))
	d0 := mul(wd, u(ds))
	qr := make([][]ir.Value, gqa)
	for g := range qr {
		qr[g] = make([]ir.Value, ds)
		for i := range qr[g] {
			di := add(d0, u(i))
			qh := add(h0, u(g))
			if qrow != 0 {
				qh = add(qh, qrow)
			}
			qv := b.Load(ir.F32, q, add(mul(qh, u(s.Dim)), b.Min(ir.U32, di, u(s.Dim-1))), 0)
			if ds*Wd != s.Dim {
				qv = b.Select(ir.F32, b.Lt(ir.U32, di, u(s.Dim)), qv, zero)
			}
			qr[g][i] = qv
		}
	}
	tiles := b.Div(ir.U32, add(keys, u(32*Wk-1)), u(32*Wk))
	// A slice past the last dimension (ds*Wd > Dim) reads a clamped row and
	// multiplies it by the zeroed query.
	lastDim := u(s.Dim - 1)
	// A tile's page id is loaded one tile ahead, so the K loads never wait on
	// the table: tileAt is warp group wk's j'th tile, clamped to the row's
	// last.
	tileAt := func(j ir.Value) ir.Value {
		return b.Min(ir.U32, add(st, mul(add(mul(j, u(Wk)), wk), u(32))), lastTile)
	}
	const unroll = 4 // step 5's keys a warp takes an iteration
	block := 32%(unroll*W) == 0
	blockAt := func(it ir.Value) ir.Value { return b.Min(ir.U32, add(st, mul(it, u(unroll*W))), lastTile) }
	pid0 := kernels.PageOf(b, pTab, tabOff, tileAt(u0), P)
	var vpid0 ir.Value
	// Step 5's first page too, here, so its load is not on the path after
	// the reductions.
	if block {
		vpid0 = kernels.PageOf(b, pTab, tabOff, blockAt(u0), P)
	}
	b.LoopN(tiles)
	j := b.Phi(ir.U32, u0)
	pidPhi := b.Phi(ir.U32, pid0)
	slot := add(mul(add(mul(j, u(Wk)), wk), u(32)), wl)
	kd := make([]ir.Value, ds)
	// One page id for the whole 32-key tile, which lies in one page because
	// it starts at a multiple of 32; its lane's key is tile%P + lane there.
	tb := tileAt(j)
	pid := pidPhi
	b.SetPhi(pidPhi, kernels.PageOf(b, pTab, tabOff, tileAt(add(j, u(1))), P))
	kp := add(add(mul(pid, u(P*kvRow)), kernels.PageRem(b, tb, P)), wl)
	if s.F16K {
		kp = add(add(mul(pid, u(P*kvRow/2)), kernels.PageRem(b, tb, P)), wl)
	}
	// Two forms of the same addresses (FlashShape.KImm): one base plus i*P
	// immediates, or each dimension's index computed. Binary16 K reads a word
	// a pair of dimensions where the slice is whole pairs, else each
	// dimension's half.
	switch {
	case s.F16K && ds%2 == 0 && ds*Wd == s.Dim:
		var kb ir.Value
		if s.KImm {
			kb = add(kp, mul(b.Shr(ir.U32, add(kvbase, d0), u(1)), u(P)))
		}
		for i := 0; i < ds; i += 2 {
			var w ir.Value
			if s.KImm {
				w = b.Load(ir.F32, k, kb, int64(i/2*P))
			} else {
				w = b.Load(ir.F32, k, add(kp, mul(b.Shr(ir.U32, add(kvbase, add(d0, u(i))), u(1)), u(P))), 0)
			}
			kd[i], kd[i+1] = kernels.KPair(b, w)
		}
	case s.F16K:
		for i := range kd {
			e := add(kvbase, b.Min(ir.U32, add(d0, u(i)), lastDim))
			w := b.Bitcast(ir.U32, b.Load(ir.F32, k, add(kp, mul(b.Shr(ir.U32, e, u(1)), u(P))), 0))
			kd[i] = kernels.KHalf(b, w, e)
		}
	case s.KImm && ds*Wd == s.Dim:
		kb := add(kp, mul(add(kvbase, d0), u(P)))
		for i := range kd {
			kd[i] = b.Load(ir.F32, k, kb, int64(i*P))
		}
	default:
		for i := range kd {
			di := b.Min(ir.U32, add(d0, u(i)), lastDim)
			kd[i] = b.Load(ir.F32, k, add(kp, mul(add(kvbase, di), u(P))), 0)
		}
	}
	const chains = 4
	cslot := b.Min(ir.U32, slot, u(C-1))
	for g := range qr {
		var dot [chains]ir.Value
		for c := range dot {
			dot[c] = zero
		}
		for i := range kd {
			dot[i%chains] = b.Fma(qr[g][i], kd[i], dot[i%chains])
		}
		sum := b.Add(ir.F32, b.Add(ir.F32, dot[0], dot[1]), b.Add(ir.F32, dot[2], dot[3]))
		b.Store(sS, add(at(g, cslot), mul(wd, u(C))), sum, 0)
	}
	b.SetPhi(j, add(j, u(1)))
	b.EndLoop()
	b.Barrier()

	// 2. The scores: each thread folds its keys' Wd partials into row [g][0].
	b.LoopN(trips)
	j1 := b.Phi(ir.U32, u0)
	mx := make([]ir.Value, gqa)
	for g := range mx {
		mx[g] = b.Phi(ir.F32, minus)
	}
	slot1 := add(mul(j1, u(T)), tid)
	// Below en and not below keyStart: key-keyStart wraps for those, and en
	// >= keyStart whenever the partition has keys (it starts within 32).
	valid := b.Lt(ir.U32, b.Sub(ir.U32, add(st, slot1), keyStart), b.Sub(ir.U32, en, keyStart))
	cslot1 := b.Min(ir.U32, slot1, u(C-1))
	for g := range mx {
		dotv := b.Load(ir.F32, sS, at(g, cslot1), 0)
		for w := 1; w < Wd; w++ {
			dotv = b.Add(ir.F32, dotv, b.Load(ir.F32, sS, at(g, cslot1), int64(w*C)))
		}
		score := attnScore(b, s, dotv)
		score = b.Select(ir.F32, valid, score, minus)
		b.Store(sS, at(g, cslot1), score, 0)
		b.SetPhi(mx[g], b.Max(ir.F32, mx[g], score))
	}
	b.SetPhi(j1, add(j1, u(1)))
	b.EndLoop()

	// 3. The group maximum: a butterfly per warp, then every warp's in turn.
	// Each lane writes its own replica, so no two lanes store to one address.
	for g := range mx {
		wm := kernels.Butterfly(b, mx[g], func(x, y ir.Value) ir.Value { return b.Max(ir.F32, x, y) })
		b.Store(sMax, add(mul(add(u(g*W), warp), u(32)), wl), wm, 0)
	}
	b.Barrier()
	m := make([]ir.Value, gqa)
	for g := range m {
		m[g] = minus
		for w := 0; w < W; w++ {
			m[g] = b.Max(ir.F32, m[g], b.Load(ir.F32, sMax, u((g*W+w)*32), 0))
		}
	}

	// 4. The weights, over the same keys, and their sums.
	b.LoopN(trips)
	j2 := b.Phi(ir.U32, u0)
	sum := make([]ir.Value, gqa)
	for g := range sum {
		sum[g] = b.Phi(ir.F32, zero)
	}
	slot2 := add(mul(j2, u(T)), tid)
	valid2 := b.Lt(ir.U32, add(st, slot2), en)
	cslot2 := b.Min(ir.U32, slot2, u(C-1))
	for g := range sum {
		at := add(cslot2, u(g*Wd*C))
		w := b.Select(ir.F32, valid2, b.Exp(b.Sub(ir.F32, b.Load(ir.F32, sS, at, 0), m[g])), zero)
		b.Store(sS, at, w, 0)
		b.SetPhi(sum[g], b.Add(ir.F32, sum[g], w))
	}
	b.SetPhi(j2, add(j2, u(1)))
	b.EndLoop()
	for g := range sum {
		ws := kernels.Butterfly(b, sum[g], func(x, y ir.Value) ir.Value { return b.Add(ir.F32, x, y) })
		b.Store(sSum, add(mul(add(u(g*W), warp), u(32)), wl), ws, 0)
	}
	b.Barrier() // also publishes step 4's weights
	l := make([]ir.Value, gqa)
	for g := range l {
		l[g] = zero
		for w := 0; w < W; w++ {
			l[g] = b.Add(ir.F32, l[g], b.Load(ir.F32, sSum, u((g*W+w)*32), 0))
		}
	}

	// 5. The values. A warp takes every W-th key and its lanes the dimensions
	// (lane, lane+32, ...), so each key is one coalesced V row per warp; the
	// W warps' sums meet in shared memory afterwards.
	nd := (s.Dim + 31) / 32
	dims := make([]ir.Value, nd)
	vw := kernels.FlashKVVecV(s)
	for r := range dims {
		if vw > 1 {
			// Component r%vw of the lane's (r/vw)'th vector.
			c := r % vw
			dims[r] = add(mul(add(wl, u(r/vw*32)), u(vw)), u(c))
		} else {
			dims[r] = b.Min(ir.U32, add(wl, u(r*32)), u(s.Dim-1))
		}
	}
	pairs := b.Div(ir.U32, add(keys, u(unroll*W-1)), u(unroll*W)) // iterations per warp
	// An iteration's unroll*W keys are consecutive from a multiple of it;
	// where that divides 32 they lie in one 32-key tile, so in one page, whose
	// id is loaded an iteration ahead. Otherwise each key is translated alone.
	// A dead key (past the partition's end) reads the partition's last key,
	// which is in the iteration's own tile.
	lastIn := b.Sub(ir.U32, b.Max(ir.U32, en, u(1)), u(1))
	b.LoopN(pairs)
	it := b.Phi(ir.U32, u0)
	var vpid ir.Value
	if block {
		vpid = b.Phi(ir.U32, vpid0)
	}
	acc := make([][]ir.Value, gqa)
	for g := range acc {
		acc[g] = make([]ir.Value, nd)
		for r := range acc[g] {
			acc[g][r] = b.Phi(ir.F32, zero)
		}
	}
	next := make([][]ir.Value, gqa)
	for g := range next {
		next[g] = append([]ir.Value(nil), acc[g]...)
	}
	if vpid != 0 {
		b.SetPhi(vpid, kernels.PageOf(b, pTab, tabOff, blockAt(add(it, u(1))), P))
	}
	// Every load of the iteration's keys first, then the products.
	type keyLoads struct{ ws, vv []ir.Value }
	loads := make([]keyLoads, unroll)
	for half := range loads {
		kk := add(mul(add(mul(it, u(unroll)), u(half)), u(W)), warp) // key index in the partition
		live := b.Lt(ir.U32, kk, keys)
		// Clamped into [keyStart, the partition's last key]: a masked key's
		// weight is 0, and 0*V must not meet a V nobody wrote.
		t := b.Min(ir.U32, b.Max(ir.U32, add(st, kk), keyStart), lastIn)
		pid := vpid
		if !block {
			pid = kernels.PageOf(b, pTab, tabOff, t, P)
		}
		vb := add(mul(add(mul(pid, u(P)), kernels.PageRem(b, t, P)), u(kvRow)), kvbase)
		for r := 0; vw > 1 && r < nd; r += vw {
			at := add(vb, mul(add(wl, u(r/vw*32)), u(vw)))
			loads[half].vv = append(loads[half].vv, b.LoadV(ir.F32, v, at, 0, vw)...)
		}
		for _, dim := range dims {
			if vw > 1 {
				break
			}
			idx := add(vb, dim)
			var vv ir.Value
			if s.F16 {
				word := b.Bitcast(ir.U32, b.Load(ir.F32, v, b.Shr(ir.U32, idx, u(1)), 0))
				word = b.Shr(ir.U32, word, mul(b.And(ir.U32, idx, u(1)), u(16)))
				vv = b.CvtF16H(word)
			} else {
				vv = b.Load(ir.F32, v, idx, 0)
			}
			loads[half].vv = append(loads[half].vv, vv)
		}
		wslot := b.Min(ir.U32, kk, u(C-1))
		for g := 0; g < gqa; g++ {
			loads[half].ws = append(loads[half].ws, b.Select(ir.F32, live, b.Load(ir.F32, sS, add(wslot, u(g*Wd*C)), 0), zero))
		}
	}
	for _, ld := range loads {
		for r := range dims {
			for g := range next {
				next[g][r] = b.Fma(ld.ws[g], ld.vv[r], next[g][r])
			}
		}
	}
	for g := range acc {
		for r := range acc[g] {
			b.SetPhi(acc[g][r], next[g][r])
		}
	}
	b.SetPhi(it, add(it, u(1)))
	b.EndLoop()
	// Every warp is done reading the weights: the score plane holds the
	// per-warp sums now, [warp][g][dim].
	b.Barrier()
	for g := range acc {
		for r, dim := range dims {
			b.Store(sS, add(mul(add(mul(warp, u(gqa)), u(g)), u(s.Dim)), dim), acc[g][r], 0)
		}
	}
	b.Barrier()
	// Each thread folds the W sums of dimensions tid, tid+T, ... for every head.
	nt := (s.Dim + T - 1) / T
	odims := make([]ir.Value, nt)
	for r := range odims {
		odims[r] = b.Min(ir.U32, add(tid, u(r*T)), u(s.Dim-1))
	}
	fin := make([][]ir.Value, gqa)
	for g := range fin {
		fin[g] = make([]ir.Value, nt)
		for r, dim := range odims {
			x := zero
			for w := 0; w < W; w++ {
				x = b.Add(ir.F32, x, b.Load(ir.F32, sS, add(u((w*gqa+g)*s.Dim), dim), 0))
			}
			fin[g][r] = x
		}
	}
	acc, dims = fin, odims

	// 6. Out: normalised here, or the merge's partials.
	for g := 0; g < gqa; g++ {
		head := add(h0, u(g))
		if qrow != 0 {
			head = add(head, qrow)
		}
		if splits > 1 {
			padded := (s.Dim + 31) / 32 * 32
			base := mul(add(mul(head, u(splits)), sp), u(padded+64))
			lg := l[g]
			for r, dim := range dims {
				b.Store(out, add(base, dim), acc[g][r], 0)
			}
			b.Store(out, add(base, wl), m[g], int64(padded))
			b.Store(out, add(base, wl), lg, int64(padded+32))
			continue
		}
		factor, denom := b.ConstF32(1), l[g]
		if s.Sink {
			sh := head
			if qrow != 0 {
				sh = add(h0, u(g))
			}
			sv := b.Load(ir.F32, sink, sh, 0)
			final := b.Max(ir.F32, m[g], sv)
			factor = b.Exp(b.Sub(ir.F32, m[g], final))
			denom = b.Fma(l[g], factor, b.Exp(b.Sub(ir.F32, sv, final)))
		}
		factor = b.Div(ir.F32, factor, b.Max(ir.F32, denom, b.ConstF32(1e-30)))
		for r, dim := range dims {
			b.Store(out, add(mul(head, u(s.Dim)), dim), b.Mul(ir.F32, acc[g][r], factor), 0)
		}
	}
	return b.Done(), nil
}

// attnScore is a dot product scaled, and softcapped where the plan states a
// cap: cap*tanh(score/cap) as 1 - 2/(e^(2x/cap)+1).
func attnScore(b *ir.Builder, s kernels.FlashShape, dot ir.Value) ir.Value {
	score := b.Mul(ir.F32, dot, b.ConstF32(s.Scale))
	if s.Softcap > 0 {
		e := b.Exp(b.Mul(ir.F32, b.ConstF32(2/s.Softcap), score))
		score = b.Mul(ir.F32, b.ConstF32(s.Softcap), b.Sub(ir.F32, b.ConstF32(1), b.Div(ir.F32, b.ConstF32(2), b.Add(ir.F32, e, b.ConstF32(1)))))
	}
	return score
}

// StagedScores is the staged path's scores: the scaled (and softcapped) q.k
// of every slot of every split into a plane of S.Chunk slots per (row, head,
// split), -MaxFloat32 where the slot holds no key of the row. A thread
// (Lanes 1) or a warp across the dimensions (Lanes 32, a row-major latent
// row) takes one slot for kernels.StagedGroup(S) query heads of one KV head.
//
// Parameters: pQ, pK, pN (not read), pOut (the plane), pTab, pRow.
type StagedScores struct {
	S     kernels.FlashShape
	Lanes int
}

// Encode is the scores' numbers and schedule.
func (a *StagedScores) Encode() []uint32 {
	return append([]uint32{21, uint32(a.Lanes)}, encShape(a.S)...)
}

// Width is the workgroup.
func (a *StagedScores) Width() int { return max(a.Lanes, 128) }

// Groups is the workgroups the scores launch.
func (a *StagedScores) Groups() int {
	return (kernels.PagedStagedScoreItems(a.S)*a.Lanes + 127) / 128
}

// Emit forms the kernel.
func (a *StagedScores) Emit() (*ir.Kernel, error) {
	s, lanes := a.S, a.Lanes
	if err := kernels.StagedCheck("StagedScores", s); err != nil {
		return nil, err
	}
	if lanes != 1 && lanes != 32 {
		return nil, fmt.Errorf("compose: StagedScores: lanes=%d, want 1 or 32", lanes)
	}
	P, C, S := s.Page, s.Chunk, s.Splits
	kvRow := s.KVHeads * s.Dim
	gqa, G := s.Heads/s.KVHeads, kernels.StagedGroup(s)
	b := ir.New("composed_scores", [3]int{max(lanes, 128), 1, 1})
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
	ps := kernels.PagedSplitOf(b, pRow, row, sp, S)
	t := ps.Key(b, slot)
	pid := kernels.PageOf(b, pTab, ps.Tab(), t, P)
	h0 := mul(hg, u(G))
	kvh := mul(b.Div(ir.U32, h0, u(gqa)), u(s.Dim))
	rh0 := add(mul(row, u(s.Heads)), h0) // (row*Heads + first head)
	qb := mul(rh0, u(s.Dim))
	// The position's K: element d at kb + d*step.
	var kb ir.Value
	step := 1
	switch {
	case s.MLA > 0:
		kb = mul(add(mul(pid, u(P)), kernels.PageRem(b, t, P)), u(kvRow))
	case s.F16K:
		// Words, a pair of dimensions each: pair d/2 at kb + (d/2)*P.
		kb = add(add(mul(pid, u(P*kvRow/2)), mul(b.Shr(ir.U32, kvh, u(1)), u(P))), kernels.PageRem(b, t, P))
		step = P
	default:
		kb = add(add(mul(pid, u(P*kvRow)), mul(kvh, u(P))), kernels.PageRem(b, t, P))
		step = P
	}
	// kAt2 is dimensions d and d+1 (d even) of binary16 K from base kb.
	kAt2 := func(kb ir.Value, d int) (ir.Value, ir.Value) {
		return kernels.KPair(b, b.Load(ir.F32, pK, kb, int64(d/2*step)))
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
		// dimensions, the rest unrolled after it.
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
				kv := b.Load(ir.F32, pK, ki, int64(d*step))
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
				kv = kernels.KHalf(b, w, dc)
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
			dots[g] = kernels.Butterfly(b, acc[g], func(x, y ir.Value) ir.Value { return b.Add(ir.F32, x, y) })
		}
	}
	valid := ps.Valid(b, slot)
	// Head h0+g's slot: ((rh0+g)*S + sp)*C + slot.
	ob := add(mul(add(mul(rh0, u(S)), sp), u(C)), slot)
	for g, dot := range dots {
		score := attnScore(b, s, dot)
		b.Store(pOut, ob, b.Select(ir.F32, valid, score, b.ConstF32(-math.MaxFloat32)), int64(g*S*C))
	}
	return b.Done(), nil
}

// StagedSoftmax is the staged path's softmax of each split's slots: its
// maximum and denominator into the merge's partials, and its unnormalised
// weights -- 0 where the slot holds no key -- to the weight plane. Lanes 32
// is a warp a split, 1 a thread.
//
// Parameters: pA (the scores), pN (not read), pOut (the weights), pPart,
// pTab (not read), pRow.
type StagedSoftmax struct {
	S     kernels.FlashShape
	Lanes int
}

// Encode is the softmax's numbers and schedule.
func (a *StagedSoftmax) Encode() []uint32 {
	return append([]uint32{22, uint32(a.Lanes)}, encShape(a.S)...)
}

// Width is the workgroup.
func (a *StagedSoftmax) Width() int { return max(a.Lanes, 64) }

// Groups is the workgroups the softmax launches: two warps a group, a split
// each, or a thread a split.
func (a *StagedSoftmax) Groups() int {
	items := a.S.Rows * a.S.Heads * a.S.Splits
	if a.Lanes == 32 {
		return (items + 1) / 2
	}
	return (items + 63) / 64
}

// Emit forms the kernel.
func (a *StagedSoftmax) Emit() (*ir.Kernel, error) {
	s, lanes := a.S, a.Lanes
	if err := kernels.StagedCheck("StagedSoftmax", s); err != nil {
		return nil, err
	}
	if lanes != 1 && lanes != 32 {
		return nil, fmt.Errorf("compose: StagedSoftmax: lanes=%d, want 1 or 32", lanes)
	}
	C, S := s.Chunk, s.Splits
	items := s.Rows * s.Heads * S
	b := ir.New("composed_softmax", [3]int{max(lanes, 64), 1, 1})
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
	ps := kernels.PagedSplitOf(b, pRow, row, sp, S)
	keys := b.Sub(ir.U32, ps.En(), ps.St())
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
		m = kernels.Butterfly(b, mx, func(x, y ir.Value) ir.Value { return b.Max(ir.F32, x, y) })
	}
	// The weights cover every slot of the chunk, 0 past the split's keys: the
	// accumulate reads whole blocks.
	cntC := b.Div(ir.U32, add(b.Sub(ir.U32, u(C), lane), b.Sub(ir.U32, step, u(1))), step)
	zero := b.ConstF32(0)
	b.LoopN(cntC)
	sum := b.Phi(ir.F32, zero)
	si := b.Phi(ir.U32, lane)
	w := b.Select(ir.F32, ps.Valid(b, si), b.Exp(b.Sub(ir.F32, b.Load(ir.F32, pA, add(base, si), 0), m)), zero)
	b.Store(pOut, add(base, si), w, 0)
	b.SetPhi(sum, b.Add(ir.F32, sum, w))
	b.SetPhi(si, add(si, step))
	b.EndLoop()
	l := sum
	if lanes == 32 {
		l = kernels.Butterfly(b, sum, func(x, y ir.Value) ir.Value { return b.Add(ir.F32, x, y) })
	}
	vd := kernels.StagedValue(s)
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
	return b.Done(), nil
}

// StagedAcc is the staged path's weighted sum of V a split: a thread per
// (row, head group, split, value element) -- two adjacent elements of a
// binary16 V -- serving kernels.StagedGroup(S) query heads, walking the
// split's keys in blocks of eight that lie in one page, into the merge's
// partials.
//
// Parameters: pA (the weights), pV, pN (not read), pPart, pTab, pRow.
type StagedAcc struct{ S kernels.FlashShape }

// Encode is the accumulate's numbers and schedule.
func (a *StagedAcc) Encode() []uint32 { return append([]uint32{23}, encShape(a.S)...) }

// Width is the workgroup; Groups the grid.
func (a *StagedAcc) Width() int { return 128 }

// Groups is the workgroups the accumulate launches.
func (a *StagedAcc) Groups() int { return (kernels.PagedStagedAccThreads(a.S) + 127) / 128 }

// Emit forms the kernel.
func (a *StagedAcc) Emit() (*ir.Kernel, error) {
	s := a.S
	if err := kernels.StagedCheck("StagedAcc", s); err != nil {
		return nil, err
	}
	const accBlock = 8
	P, C, S := s.Page, s.Chunk, s.Splits
	kvRow := s.KVHeads * s.Dim
	gqa, G := s.Heads/s.KVHeads, kernels.StagedGroup(s)
	vd := kernels.StagedValue(s)
	per := vd // elements a thread owns: one, or a word of two
	if s.F16 {
		per = vd / 2
	}
	b := ir.New("composed_acc", [3]int{128, 1, 1})
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
	ps := kernels.PagedSplitOf(b, pRow, row, sp, S)
	keys := b.Sub(ir.U32, ps.En(), ps.St())
	blocks := b.Div(ir.U32, add(keys, u(accBlock-1)), u(accBlock))
	// The element (f32) or word (f16) this thread reads within a position's V
	// row: its KV head's slice, or the latent row's prefix.
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
	// Block it starts at st + it*accBlock; the page prefetched for the next
	// one is clamped into the row (and to position 0 for an empty row, whose
	// table is the dummy page).
	lastTile := mul(b.Div(ir.U32, ps.Last(), u(32)), u(32))
	blockPage := func(it ir.Value) ir.Value {
		t := b.Min(ir.U32, add(ps.St(), mul(it, u(accBlock))), lastTile)
		return kernels.PageOf(b, pTab, ps.Tab(), b.Select(ir.U32, ps.Live(), t, u(0)), P)
	}
	u0, zero := u(0), b.ConstF32(0)
	pid0 := blockPage(u0)
	// Weights of head h0+g: ((row*Heads + h0+g)*S + sp)*C + slot.
	wb := mul(add(mul(add(mul(row, u(s.Heads)), h0), u(S)), sp), u(C))
	// Key st+k is live when (st + k - keyStart) < (en - keyStart), unsigned.
	d0 := b.Sub(ir.U32, ps.St(), ps.Ks())
	span := b.Sub(ir.U32, ps.En(), ps.Ks())
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
	bs := add(ps.St(), k0)
	vb := add(mul(add(mul(pid, u(P)), kernels.PageRem(b, bs, P)), u(rowW)), col)
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
	return b.Done(), nil
}

// Merge combines the splits' unnormalised partials (each a maximum, a
// denominator and a numerator row) into the attention's output, the sink
// added once: Wide is a thread an output element, else a 32-lane group a
// (row, head). S is at the value's width (a latent attention's value).
//
// Parameters: pPart, pOut, then pSink where S.Sink, then pMap where S.Map
// (Wide only: slot i merges and writes row pMap[i]).
type Merge struct {
	S    kernels.FlashShape
	Wide bool
}

// Encode is the merge's numbers and schedule.
func (a *Merge) Encode() []uint32 {
	return append([]uint32{24, b2u(a.Wide)}, encShape(a.S)...)
}

// Width is the workgroup.
func (a *Merge) Width() int {
	if a.Wide {
		return 128
	}
	return 32
}

// Groups is the workgroups the merge launches.
func (a *Merge) Groups() int {
	if a.Wide {
		return (a.S.Rows*a.S.Heads*a.S.Dim + 127) / 128
	}
	return a.S.Rows * a.S.Heads
}

// Emit forms the kernel.
func (a *Merge) Emit() (*ir.Kernel, error) {
	s := a.S
	if s.Splits < 1 || s.Heads < 1 || s.Rows < 1 || s.Dim < 1 || s.Map && !a.Wide {
		return nil, fmt.Errorf("compose: Merge: invalid shape %+v", s)
	}
	if !a.Wide {
		return mergeItems(s), nil
	}
	b := ir.New("composed_merge", [3]int{128, 1, 1})
	p, out := b.Param("pPart", ir.F32), b.Param("pOut", ir.F32)
	var sink ir.Value
	if s.Sink {
		sink = b.Param("pSink", ir.F32)
	}
	u := func(x int) ir.Value { return b.Const(ir.U32, int64(x)) }
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()), u(s.Rows*s.Heads*s.Dim-1))
	item := b.Div(ir.U32, flat, u(s.Dim))
	dim := b.Rem(ir.U32, flat, u(s.Dim))
	if s.Map {
		// The slot's row, through the table: the item and the output element
		// are that row's.
		pMap := b.Param("pMap", ir.U32)
		head := b.Rem(ir.U32, item, u(s.Heads))
		row := b.Load(ir.U32, pMap, b.Div(ir.U32, item, u(s.Heads)), 0)
		item = b.Add(ir.U32, b.Mul(ir.U32, row, u(s.Heads)), head)
		flat = b.Add(ir.U32, b.Mul(ir.U32, item, u(s.Dim)), dim)
	}
	padded := (s.Dim + 31) / 32 * 32
	stride := int64(padded + 64)
	base := b.Mul(ir.U32, item, u(s.Splits*(padded+64)))
	mx := b.ConstF32(-math.MaxFloat32)
	ms, ns := make([]ir.Value, s.Splits), make([]ir.Value, s.Splits)
	for j := range ms {
		ms[j] = b.Load(ir.F32, p, base, int64(j)*stride+int64(padded))
		ns[j] = b.Load(ir.F32, p, base, int64(j)*stride+int64(padded+32))
		mx = b.Max(ir.F32, mx, ms[j])
	}
	var sv ir.Value
	if s.Sink {
		sv = b.Load(ir.F32, sink, b.Rem(ir.U32, item, u(s.Heads)), 0)
		mx = b.Max(ir.F32, mx, sv)
	}
	sum, acc := b.ConstF32(0), b.ConstF32(0)
	src := b.Add(ir.U32, base, dim)
	for j := range ms {
		w := b.Exp(b.Sub(ir.F32, ms[j], mx))
		sum = b.Fma(ns[j], w, sum)
		acc = b.Fma(b.Load(ir.F32, p, src, int64(j)*stride), w, acc)
	}
	if s.Sink {
		sum = b.Add(ir.F32, sum, b.Exp(b.Sub(ir.F32, sv, mx)))
	}
	inv := b.Div(ir.F32, b.ConstF32(1), b.Max(ir.F32, sum, b.ConstF32(1e-30)))
	b.Store(out, flat, b.Mul(ir.F32, acc, inv), 0)
	return b.Done(), nil
}

// mergeItems is the merge as a 32-lane group a (row, head), each lane
// walking the head's dimensions lane, lane+32, ....
func mergeItems(s kernels.FlashShape) *ir.Kernel {
	b := ir.New("composed_mergeitems", [3]int{32, 1, 1})
	p, out := b.Param("pPart", ir.F32), b.Param("pOut", ir.F32)
	var sink ir.Value
	if s.Sink {
		sink = b.Param("pSink", ir.F32)
	}
	u := func(x int) ir.Value { return b.Const(ir.U32, int64(x)) }
	add := func(a, c ir.Value) ir.Value { return b.Add(ir.U32, a, c) }
	lane, item := b.TID(), b.CTAID()
	padded := (s.Dim + 31) / 32 * 32
	base := add(b.Mul(ir.U32, item, u(s.Splits*(padded+64))), lane)
	mx := b.ConstF32(-math.MaxFloat32)
	ms, ns := make([]ir.Value, s.Splits), make([]ir.Value, s.Splits)
	for j := range ms {
		ms[j] = b.Load(ir.F32, p, base, int64(j*(padded+64)+padded))
		ns[j] = b.Load(ir.F32, p, base, int64(j*(padded+64)+padded+32))
		mx = b.Max(ir.F32, mx, ms[j])
	}
	var sv ir.Value
	if s.Sink {
		sv = b.Load(ir.F32, sink, b.Rem(ir.U32, item, u(s.Heads)), 0)
		mx = b.Max(ir.F32, mx, sv)
	}
	sum := b.ConstF32(0)
	weights := make([]ir.Value, s.Splits)
	for j := range weights {
		weights[j] = b.Exp(b.Sub(ir.F32, ms[j], mx))
		sum = b.Fma(ns[j], weights[j], sum)
	}
	if s.Sink {
		sum = b.Add(ir.F32, sum, b.Exp(b.Sub(ir.F32, sv, mx)))
	}
	inv := b.Div(ir.F32, b.ConstF32(1), b.Max(ir.F32, sum, b.ConstF32(1e-30)))
	for i := 0; i < (s.Dim+31)/32; i++ {
		acc := b.ConstF32(0)
		// Ragged output lanes use the last valid dimension, including its
		// partial.
		dim := b.Min(ir.U32, add(lane, u(i*32)), u(s.Dim-1))
		src := add(b.Mul(ir.U32, item, u(s.Splits*(padded+64))), dim)
		for j := range weights {
			acc = b.Fma(b.Load(ir.F32, p, src, int64(j*(padded+64))), weights[j], acc)
		}
		b.Store(out, add(b.Mul(ir.U32, item, u(s.Dim)), dim), b.Mul(ir.F32, acc, inv), 0)
	}
	return b.Done()
}
