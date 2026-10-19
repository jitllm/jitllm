package kernels

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// FlashKVWidth is FlashDecodeKV's workgroup width for s: s.Warps warps, or
// when it is 0 one warp per 32 dimensions, four to eight.
func FlashKVWidth(s FlashShape) int { return 32 * flashKVWarps(s) }

// flashKVWarps defaults to one warp per 32-dimension slice, which measured
// best on a large NVIDIA part for both 128- and 256-wide heads.
func flashKVWarps(s FlashShape) int {
	if s.Warps > 0 {
		return s.Warps
	}
	return min(8, max(4, (s.Dim+31)/32))
}

// flashKVGeom splits the W warps of the score pass into wd slices of the
// dimensions (ds each, at most 32 where W allows) times wk groups of keys.
func flashKVGeom(s FlashShape) (w, wd, wk, ds int) {
	w = flashKVWarps(s)
	wd = min(w, (s.Dim+31)/32)
	for w%wd != 0 {
		wd++
	}
	return w, wd, w / wd, (s.Dim + wd - 1) / wd
}

// FlashKVChunk is how many keys one FlashDecodeKV workgroup can hold for s:
// its score plane is one row of partial dot products per (query head,
// dimension slice), kept under 32 KiB with the reductions where the shape
// allows. A caller needs Splits*FlashKVChunk(s) >= every count the kernel
// will see.
func FlashKVChunk(s FlashShape) int {
	gqa, T := flashKVGroup(s), FlashKVWidth(s)
	w, wd, _, _ := flashKVGeom(s)
	// Whole workgroup widths: the score pass strides the plane by that.
	c := min(2048, ((32<<10)/4-2*gqa*w*32)/(gqa*wd)) / T * T
	// The score plane doubles as the warps' value sums at the end:
	// w*gqa*Dim floats in gqa*wd*c.
	return max(c, T, (w*s.Dim/wd+T-1)/T*T)
}

// flashKVVecV is how many consecutive V elements a lane loads at once in
// FlashDecodeKV's value pass: 4 or 2 where s.VecV asks and an f32 row divides
// into whole vectors across the warp, else 1.
func flashKVVecV(s FlashShape) int {
	if !s.VecV || s.F16 {
		return 1
	}
	for _, w := range []int{4, 2} {
		if s.Dim%(32*w) == 0 && (s.KVHeads*s.Dim)%w == 0 {
			return w
		}
	}
	return 1
}

// flashKVGroup is how many query heads one workgroup serves.
func flashKVGroup(s FlashShape) int {
	gqa := max(1, s.Heads/max(1, s.KVHeads))
	if s.Group > 0 && s.Group < gqa {
		gqa = s.Group
	}
	// Each head costs the warps' value sums (w*Dim floats, which the score
	// plane must hold) and two reduction rows, and ds query values in every
	// lane's registers: shrink the group to a divisor that fits 32 KiB and 96
	// registers of queries, or one head. The register cap is measured: more
	// query values per lane lost to the staged path.
	w, _, _, ds := flashKVGeom(s)
	g := gqa
	for g > 1 && (gqa%g != 0 || g*(w*s.Dim+64*w) > (32<<10)/4 || g*ds > 96) {
		g--
	}
	return g
}

// FlashKVGroups is FlashDecodeKV's workgroup count for s.
// A paged shape serves s.Rows descriptors, each its own set.
func FlashKVGroups(s FlashShape) int {
	return max(1, s.Rows) * s.Heads / flashKVGroup(s) * max(1, s.Splits)
}

// FlashDecodeKV is decode attention (s.Rows == 1) as one workgroup per KV head
// and key partition, serving every query head of its GQA group.
//
// FlashAttention runs a 32-lane group per query head walking its keys
// serially, which is slow on a large part. Here lanes run over keys and warps
// over slices of the dimensions for the scores (one K load serves the group's query
// heads), the maximum and the sum are two block reductions, and lanes run over
// dimensions for the values (V loaded once per key for every head). It
// replaces the staged decode's four launches (scores, softmax, split
// accumulate, reduce) with one, plus FlashAttentionMerge where s.Splits > 1,
// whose partial layout this writes. See
// docs/engineering-history/gpu-kernels.md.
//
// Parameters are FlashAttention's: Q, K, V, causal count, output (or the
// partials), and the optional per-head sinks. Surplus threads clamp and
// recompute, as everywhere in this package.
func FlashDecodeKV(s FlashShape) (*ir.Kernel, error) {
	T := FlashKVWidth(s)
	W, Wd, Wk, ds := flashKVGeom(s)
	if (s.Page == 0 && s.Rows != 1) || s.Rows < 1 || s.Splits < 0 || s.Heads < 1 || s.KVHeads < 1 || s.Heads%s.KVHeads != 0 || s.Dim < 1 ||
		s.KStride < 0 || s.Window < 0 || s.Softcap < 0 || (s.F16 && s.Dim%2 != 0) {
		return nil, fmt.Errorf("kernels: FlashDecodeKV: invalid shape %+v", s)
	}
	if err := flashPaged("FlashDecodeKV", s); err != nil {
		return nil, err
	}
	if s.Group > 0 && (s.Heads/s.KVHeads)%s.Group != 0 {
		return nil, fmt.Errorf("kernels: FlashDecodeKV: group %d does not divide %d query heads a KV head", s.Group, s.Heads/s.KVHeads)
	}
	// gqa below is the heads THIS workgroup serves, a slice of its KV head's.
	gqa, splits, C := flashKVGroup(s), max(1, s.Splits), FlashKVChunk(s)
	slices := s.Heads / s.KVHeads / gqa
	name, P := "flashkv", s.Page
	if P > 0 {
		name = "flashkvpaged"
	}
	b := ir.New(name, [3]int{T, 1, 1})
	q, k, v, n, out := b.Param("pQ", ir.F32), b.Param("pK", ir.F32), b.Param("pV", ir.F32), b.Param("pN", ir.U32), b.Param("pOut", ir.F32)
	var pTab, pRow, sink ir.Value
	if P > 0 {
		pTab, pRow = b.Param("pTab", ir.U32), b.Param("pRow", ir.U32)
	}
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
	// A paged launch serves s.Rows descriptors, each a whole decode: the row
	// is the workgroup's major index and qrow its first query head's offset.
	var row, qrow ir.Value
	if P > 0 {
		row = u0
		if s.Rows > 1 {
			per := s.Heads / gqa * splits
			row = b.Min(ir.U32, b.Div(ir.U32, grp, u(per)), u(s.Rows-1))
			grp = b.Rem(ir.U32, grp, u(per))
			qrow = mul(row, u(s.Heads))
		}
	}
	// The workgroup's first query head; its KV head follows from it.
	h0 := mul(b.Min(ir.U32, b.Div(ir.U32, grp, u(splits)), u(s.Heads/gqa-1)), u(gqa))
	kvh := b.Div(ir.U32, h0, u(gqa*slices))
	sp := b.Rem(ir.U32, grp, u(splits))

	var count, start, span, tabOff, keyStart ir.Value
	if P > 0 {
		// The row's keys are [keyStart, keyEnd). Its partitions start at the
		// 32-aligned position at or below keyStart, so every 32-key tile lies in
		// one page, and the keys below keyStart are masked. An empty row
		// (keyEnd <= keyStart, a padded row) has no keys and reads nothing.
		tabOff, keyStart = pagedDesc(b, pRow, row, PRowTab), pagedDesc(b, pRow, row, PRowStart)
		count = pagedDesc(b, pRow, row, PRowEnd)
		if n := benchKnob("kv-constdesc"); n > 0 {
			// The ceiling of hoisting the descriptor: one row whose table
			// starts at 0 and whose keys are [0, n), baked.
			tabOff, keyStart, count = u0, u0, u(n)
		}
		start = mul(b.Div(ir.U32, keyStart, u(32)), u(32))
		if pagedFault == "unaligned" {
			start = keyStart
		}
		span = b.Select(ir.U32, b.Lt(ir.U32, keyStart, count), b.Sub(ir.U32, count, start), u0)
	} else {
		count = b.Load(ir.U32, n, u(0), 0)
		start = u(0)
		if s.Window > 0 && s.Chunked {
			last := b.Sub(ir.U32, b.Max(ir.U32, count, u(1)), u(1))
			start = mul(b.Div(ir.U32, last, u(s.Window)), u(s.Window))
		} else if s.Window > 0 {
			start = b.Sub(ir.U32, count, b.Min(ir.U32, count, u(s.Window)))
		}
		// This partition's keys: whole 32-key tiles, at most C of them.
		span = b.Sub(ir.U32, count, start)
	}
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
	var lastTile ir.Value
	if P > 0 {
		// A tile past the last key (a key group's surplus) reads the last key's
		// tile, whose page the row owns; its slots are never valid.
		lastTile = mul(b.Div(ir.U32, lastKey, u(32)), u(32))
	}

	// 1. Partial scores. Lanes run over keys (a 32-key tile) and warps over
	// Wd slices of the dimensions times Wk groups of tiles, the query slice held
	// in registers: a lane issues its slice's ds K loads at once (coalesced
	// across the lanes, since K is [dimension][position]) and a key's dot is
	// Wd short chains. One thread per key over every dimension left most SMs
	// idle.
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
	// A paged tile's page id is loaded one tile ahead, so the K loads never
	// wait on the table: tileAt is warp group wk's j'th tile, clamped to the
	// row's last (a surplus tile's slots are never valid).
	tileAt := func(j ir.Value) ir.Value {
		return b.Min(ir.U32, add(st, mul(add(mul(j, u(Wk)), wk), u(32))), lastTile)
	}
	const unroll = 4 // step 5's keys a warp takes an iteration
	block := 32%(unroll*W) == 0
	blockAt := func(it ir.Value) ir.Value { return b.Min(ir.U32, add(st, mul(it, u(unroll*W))), lastTile) }
	var pid0, vpid0 ir.Value
	if P > 0 {
		pid0 = pageOf(b, pTab, tabOff, tileAt(u0), P)
		// Step 5's first page too, here, so its load is not on the path after
		// the reductions.
		if block {
			vpid0 = pageOf(b, pTab, tabOff, blockAt(u0), P)
		}
	}
	b.LoopN(tiles)
	j := b.Phi(ir.U32, u0)
	var pidPhi ir.Value
	if P > 0 {
		pidPhi = b.Phi(ir.U32, pid0)
	}
	slot := add(mul(add(mul(j, u(Wk)), wk), u(32)), wl)
	kd := make([]ir.Value, ds)
	if P > 0 {
		// One page id for the whole 32-key tile, which lies in one page because
		// it starts at a multiple of 32; its lane's key is tile%P + lane there.
		tb := tileAt(j)
		pid := pidPhi
		b.SetPhi(pidPhi, pageOf(b, pTab, tabOff, tileAt(add(j, u(1))), P))
		kp := add(add(mul(pid, u(P*kvRow)), pageRem(b, tb, P)), wl)
		if s.F16K {
			kp = add(add(mul(pid, u(P*kvRow/2)), pageRem(b, tb, P)), wl)
		}
		// Two forms of the same addresses, because the drivers disagree (see
		// FlashShape.KImm): one base plus i*P immediates, or each dimension's
		// index computed. Binary16 K reads a word a pair of dimensions where
		// the slice is whole pairs (d0 = wd*ds is even), else each
		// dimension's half.
		if s.F16K && ds%2 == 0 && ds*Wd == s.Dim {
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
				kd[i], kd[i+1] = kPair(b, w)
			}
		} else if s.F16K {
			for i := range kd {
				e := add(kvbase, b.Min(ir.U32, add(d0, u(i)), lastDim))
				w := b.Bitcast(ir.U32, b.Load(ir.F32, k, add(kp, mul(b.Shr(ir.U32, e, u(1)), u(P))), 0))
				kd[i] = kHalf(b, w, e)
			}
		} else if s.KImm && ds*Wd == s.Dim {
			kb := add(kp, mul(add(kvbase, d0), u(P)))
			for i := range kd {
				kd[i] = b.Load(ir.F32, k, kb, int64(i*P))
			}
		} else {
			for i := range kd {
				di := b.Min(ir.U32, add(d0, u(i)), lastDim)
				kd[i] = b.Load(ir.F32, k, add(kp, mul(add(kvbase, di), u(P))), 0)
			}
		}
	} else {
		safe := b.Min(ir.U32, add(st, slot), lastKey)
		for i := range kd {
			di := b.Min(ir.U32, add(d0, u(i)), lastDim)
			var ki ir.Value
			if s.KStride > 0 {
				ki = add(mul(add(kvbase, di), u(s.KStride)), safe)
			} else {
				ki = add(add(mul(safe, u(s.KVHeads*s.Dim)), kvbase), di)
			}
			kd[i] = b.Load(ir.F32, k, ki, 0)
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
	var valid ir.Value
	if P > 0 {
		// Below en and not below keyStart: key-keyStart wraps for those, and en
		// >= keyStart whenever the partition has keys (it starts within 32).
		valid = b.Lt(ir.U32, b.Sub(ir.U32, add(st, slot1), keyStart), b.Sub(ir.U32, en, keyStart))
	} else {
		valid = b.Lt(ir.U32, add(st, slot1), en)
	}
	cslot1 := b.Min(ir.U32, slot1, u(C-1))
	for g := range mx {
		dotv := b.Load(ir.F32, sS, at(g, cslot1), 0)
		for w := 1; w < Wd; w++ {
			dotv = b.Add(ir.F32, dotv, b.Load(ir.F32, sS, at(g, cslot1), int64(w*C)))
		}
		score := b.Mul(ir.F32, dotv, b.ConstF32(s.Scale))
		if s.Softcap > 0 {
			e := b.Exp(b.Mul(ir.F32, b.ConstF32(2/s.Softcap), score))
			score = b.Mul(ir.F32, b.ConstF32(s.Softcap), b.Sub(ir.F32, b.ConstF32(1), b.Div(ir.F32, b.ConstF32(2), b.Add(ir.F32, e, b.ConstF32(1)))))
		}
		score = b.Select(ir.F32, valid, score, minus)
		b.Store(sS, at(g, cslot1), score, 0)
		b.SetPhi(mx[g], b.Max(ir.F32, mx[g], score))
	}
	b.SetPhi(j1, add(j1, u(1)))
	b.EndLoop()

	// 3. The group maximum: a butterfly per warp, then every warp's in turn.
	// Each lane writes its own replica, so no two lanes store to one address.
	for g := range mx {
		wm := butterfly(b, mx[g], func(x, y ir.Value) ir.Value { return b.Max(ir.F32, x, y) })
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
		ws := butterfly(b, sum[g], func(x, y ir.Value) ir.Value { return b.Add(ir.F32, x, y) })
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
	// (lane, lane+32, ...), so each key is one coalesced V row per warp and a
	// lane has nd loads in flight; two keys an iteration double that. The W
	// warps' sums meet in shared memory afterwards. (One thread per dimension
	// over every key had one dependent load an iteration and no latency hiding.)
	nd := (s.Dim + 31) / 32
	dims := make([]ir.Value, nd)
	vw := flashKVVecV(s)
	for r := range dims {
		if vw > 1 {
			// Component r%vw of the lane's (r/vw)'th vector.
			c := r % vw
			if pagedFault == "vecv" {
				c = vw - 1 - c
			}
			dims[r] = add(mul(add(wl, u(r/vw*32)), u(vw)), u(c))
		} else {
			dims[r] = b.Min(ir.U32, add(wl, u(r*32)), u(s.Dim-1))
		}
	}
	pairs := b.Div(ir.U32, add(keys, u(unroll*W-1)), u(unroll*W)) // iterations per warp
	// A paged iteration's unroll*W keys are consecutive from a multiple of it;
	// where that divides 32 they lie in one 32-key tile, so in one page, whose
	// id is loaded an iteration ahead. Otherwise each key is translated alone.
	var lastIn ir.Value
	if P > 0 {
		// A dead key (past the partition's end) reads the partition's last key,
		// which is in the iteration's own tile.
		lastIn = b.Sub(ir.U32, b.Max(ir.U32, en, u(1)), u(1))
	}
	b.LoopN(pairs)
	it := b.Phi(ir.U32, u0)
	var vpid ir.Value
	if P > 0 && block {
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
		b.SetPhi(vpid, pageOf(b, pTab, tabOff, blockAt(add(it, u(1))), P))
	}
	// Every load of the iteration's keys first, then the products.
	type keyLoads struct{ ws, vv []ir.Value }
	loads := make([]keyLoads, unroll)
	for half := range loads {
		kk := add(mul(add(mul(it, u(unroll)), u(half)), u(W)), warp) // key index in the partition
		live := b.Lt(ir.U32, kk, keys)
		var vb ir.Value
		if P > 0 {
			// Clamped into [keyStart, the partition's last key]: a masked key's
			// weight is 0, and 0*V must not meet a V nobody wrote.
			t := b.Min(ir.U32, b.Max(ir.U32, add(st, kk), keyStart), lastIn)
			pid := vpid
			if !block {
				pid = pageOf(b, pTab, tabOff, t, P)
			}
			vb = add(mul(add(mul(pid, u(P)), pageRem(b, t, P)), u(kvRow)), kvbase)
		} else {
			vb = add(mul(b.Min(ir.U32, add(st, kk), lastKey), u(s.KVHeads*s.Dim)), kvbase)
		}
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
			switch pagedFault {
			case "normpart":
				inv := b.Div(ir.F32, b.ConstF32(1), b.Max(ir.F32, lg, b.ConstF32(1e-30)))
				for r := range acc[g] {
					acc[g][r] = b.Mul(ir.F32, acc[g][r], inv)
				}
			case "sinkchunk":
				if s.Sink {
					lg = b.Add(ir.F32, lg, b.Exp(b.Sub(ir.F32, b.Load(ir.F32, sink, add(h0, u(g)), 0), m[g])))
				}
			}
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
