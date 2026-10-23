package kernels

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/jit/ir"
)

// FlashShape specializes online attention. KStride == 0 means row-major K;
// otherwise K is [kv dimension][position]. V is row-major, optionally F16.
// Each query has its own causal count: pN[0] for Rows=1, pN[1+row] otherwise.
// Window=0 uses all preceding keys. Sink is a per-head logit with zero value.
// Chunked makes Window an aligned chunk (Llama 4): a query whose causal count
// is n attends keys floor((n-1)/Window)*Window .. n-1 instead of the last
// Window.
type FlashShape struct {
	Heads, Dim, KVHeads, Rows, KStride, Window int
	Chunked                                    bool
	// Splits partitions keys across workgroups; 0 and 1 select one group.
	// Splits > 1 writes partials consumed by FlashAttentionMerge.
	Splits int
	// Group is how many of a KV head's query heads one FlashDecodeKV
	// workgroup serves; 0 is all of them. It must divide Heads/KVHeads.
	Group int
	// Warps is FlashDecodeKV's workgroup width in warps; 0 picks from Dim.
	Warps          int
	Scale, Softcap float32
	F16, Sink      bool
	// Map is FlashAttentionMergeWide's run table, pMap after its other
	// parameters: Rows is the table's slots, and slot i merges and writes
	// row pMap[i] (FlashPrefill70Shape.Map, whose partials are at the rows).
	Map bool
	// Page > 0 reads K and V from paged pools of Page positions a page
	// (docs/design/device-kv-paging.md; paged.go has the layout). The kernel
	// takes pTab and pRow after its output, before the sinks; every row's key
	// range and table come from its descriptor, so pN is not read, KStride and
	// Window must be 0 (the window is the descriptor's keyStart), and Rows is
	// how many descriptors -- rows of any sequences -- one launch serves.
	Page int
	// Chunk is the staged paged kernels' per-split key bound (a multiple of
	// 32); Splits*Chunk must cover every row's keys. See pagedstaged.go.
	Chunk int
	// MLA > 0 is the staged paged kernels' MLA layout: K one row-major region
	// of Dim a position, V its first MLA elements.
	MLA int
	// MLAValueTail reads the MLA value from the row's last MLA elements
	// instead of its first: a deliberate violation of the latent's one
	// load-bearing layout fact, for the MLA gate (tier.MLAFaultValueHeadMajor
	// on the paged path, where one V load serves every head and a head-major
	// read cannot be expressed). Never set otherwise.
	MLAValueTail bool
	// KImm makes paged FlashDecodeKV address a tile's K as one base plus
	// constant offsets rather than an index per dimension. Same addresses;
	// the drivers disagree on which is fast, so the tier sets it per API.
	// It helps Metal, hurts NVIDIA's Vulkan driver badly at short contexts,
	// and is neutral on PTX.
	KImm bool
	// VecV makes FlashDecodeKV's value pass load each lane's V elements as
	// 2- or 4-wide vectors where an f32 row allows. PTX and MSL lower vector
	// loads; SPIR-V does not, so the tier sets it per API.
	VecV bool
	// F16K reads a paged K pool of binary16, two dimensions a word: element e
	// of position t is the (e&1) half of word pid*P*kvRow/2 + (e/2)*P + t%P
	// (PagedCopyRowsTF16 and PagedRoPERowsTF16 write it). Paged only, an even
	// Dim, and not MLA. It halves K's bytes, which are half of what a decode
	// token reads from the cache.
	F16K bool
}

// kHalf is the binary16 half of word w holding dimension e (e's parity).
func kHalf(b *ir.Builder, w, e ir.Value) ir.Value {
	if pagedFault == "f16k" {
		e = b.Add(ir.U32, e, b.Const(ir.U32, 1))
	}
	return b.CvtF16H(b.Shr(ir.U32, w, b.Mul(ir.U32, b.And(ir.U32, e, b.Const(ir.U32, 1)), b.Const(ir.U32, 16))))
}

// kPair is the two dimensions a binary16 K word holds, low first.
func kPair(b *ir.Builder, w ir.Value) (lo, hi ir.Value) {
	w = b.Bitcast(ir.U32, w)
	if pagedFault == "f16k" {
		return b.CvtF16H(b.Shr(ir.U32, w, b.Const(ir.U32, 16))), b.CvtF16H(w)
	}
	return b.CvtF16H(w), b.CvtF16H(b.Shr(ir.U32, w, b.Const(ir.U32, 16)))
}

// flashPaged checks a paged shape's own constraints.
func flashPaged(what string, s FlashShape) error {
	if s.Page == 0 {
		if s.F16K {
			return fmt.Errorf("kernels: %s: binary16 K is a paged layout: %+v", what, s)
		}
		return nil
	}
	if s.F16K && (s.Dim%2 != 0 || s.MLA > 0) {
		return fmt.Errorf("kernels: %s: binary16 K needs an even Dim and no MLA: %+v", what, s)
	}
	if err := checkPage(what, s.Page); err != nil {
		return err
	}
	if s.KStride != 0 || s.Window != 0 || s.Chunked {
		return fmt.Errorf("kernels: %s: a paged shape has no KStride and no Window; "+
			"the page is the stride and the window is the descriptor's keyStart: %+v", what, s)
	}
	return nil
}

// FlashAttention generates one 32-lane workgroup per query/head. It retains
// only an online maximum, normalization sum and value accumulator, and stages
// 32 probabilities in shared memory. No context-sized score plane is written.
// Parameters: Q, K, V, causal counts, output, and optional per-head sinks.
func FlashAttention(s FlashShape) (*ir.Kernel, error) {
	if s.Splits < 0 || s.Heads < 1 || s.KVHeads < 1 || s.Heads%s.KVHeads != 0 || s.Dim < 1 || s.Rows < 1 || s.KStride < 0 || s.Window < 0 || s.Softcap < 0 || (s.F16 && s.Dim%2 != 0) {
		return nil, fmt.Errorf("kernels: FlashAttention: invalid shape %+v", s)
	}
	if err := flashPaged("FlashAttention", s); err != nil {
		return nil, err
	}
	name, P := "k", s.Page
	if P > 0 {
		name = "flashpaged"
	}
	b := ir.New(name, [3]int{32, 1, 1})
	q, k, v, n, out := b.Param("pQ", ir.F32), b.Param("pK", ir.F32), b.Param("pV", ir.F32), b.Param("pN", ir.U32), b.Param("pOut", ir.F32)
	var pTab, pRow, sink ir.Value
	if P > 0 {
		pTab, pRow = b.Param("pTab", ir.U32), b.Param("pRow", ir.U32)
	}
	if s.Sink {
		sink = b.Param("pSink", ir.F32)
	}
	prob := b.Shared("prob", ir.F32, 32)
	u := func(x int) ir.Value { return b.Const(ir.U32, int64(x)) }
	add := func(a, c ir.Value) ir.Value { return b.Add(ir.U32, a, c) }
	mul := func(a, c ir.Value) ir.Value { return b.Mul(ir.U32, a, c) }
	lane := b.TID()
	splits := max(1, s.Splits)
	group := b.CTAID()
	item := b.Min(ir.U32, b.Div(ir.U32, group, u(splits)), u(s.Rows*s.Heads-1))
	head := b.Rem(ir.U32, item, u(s.Heads))
	row := b.Div(ir.U32, item, u(s.Heads))
	var count, start, tabOff, keyStart, live ir.Value
	if P > 0 {
		// The row's keys are [keyStart, keyEnd), walked in 32-key tiles from the
		// 32-aligned position at or below keyStart, so a tile lies in one page;
		// an empty row (a padded one) walks none.
		tabOff, keyStart = pagedDesc(b, pRow, row, PRowTab), pagedDesc(b, pRow, row, PRowStart)
		start = mul(b.Div(ir.U32, keyStart, u(32)), u(32))
		if pagedFault == "unaligned" {
			start = keyStart
		}
		end := pagedDesc(b, pRow, row, PRowEnd)
		live = b.Lt(ir.U32, keyStart, end)
		count = b.Select(ir.U32, live, end, start)
	} else {
		ni := u(0)
		if s.Rows > 1 {
			ni = add(row, u(1))
		}
		count = b.Load(ir.U32, n, ni, 0)
		start = u(0)
	}
	if s.Window > 0 && s.Chunked {
		last := b.Sub(ir.U32, b.Max(ir.U32, count, u(1)), u(1))
		start = mul(b.Div(ir.U32, last, u(s.Window)), u(s.Window))
	} else if s.Window > 0 {
		start = b.Sub(ir.U32, count, b.Min(ir.U32, count, u(s.Window)))
	}
	if splits > 1 {
		span := b.Sub(ir.U32, count, start)
		tileCount := b.Div(ir.U32, add(span, u(31)), u(32))
		chunk := mul(b.Div(ir.U32, add(tileCount, u(splits-1)), u(splits)), u(32))
		start = b.Min(ir.U32, count, add(start, mul(b.Rem(ir.U32, group, u(splits)), chunk)))
		count = b.Min(ir.U32, count, add(start, chunk))
	}
	tiles := b.Div(ir.U32, add(b.Sub(ir.U32, count, start), u(31)), u(32))
	last := b.Sub(ir.U32, b.Max(ir.U32, count, u(1)), u(1))
	qbase := mul(item, u(s.Dim))
	kvbase := mul(b.Div(ir.U32, head, u(s.Heads/s.KVHeads)), u(s.Dim))
	zero := b.ConstF32(0)
	minus := b.ConstF32(-math.MaxFloat32)
	dims := make([]ir.Value, (s.Dim+31)/32)
	for i := range dims {
		dims[i] = b.Min(ir.U32, add(lane, u(i*32)), u(s.Dim-1))
	}
	// A paged tile's page id is loaded a tile ahead: tilePage is the page of
	// the tile at pos, clamped into the row (position 0, the dummy page's, for
	// an empty row).
	var tilePage func(pos ir.Value) ir.Value
	var pid0 ir.Value
	if P > 0 {
		lastTile := mul(b.Div(ir.U32, last, u(32)), u(32))
		tilePage = func(pos ir.Value) ir.Value {
			return pageOf(b, pTab, tabOff, b.Select(ir.U32, live, b.Min(ir.U32, pos, lastTile), u(0)), P)
		}
		pid0 = tilePage(start)
	}
	b.LoopN(tiles)
	pos := b.Phi(ir.U32, start)
	mx := b.Phi(ir.F32, minus)
	sum := b.Phi(ir.F32, zero)
	accum := make([]ir.Value, len(dims))
	for i := range accum {
		accum[i] = b.Phi(ir.F32, zero)
	}
	var pid ir.Value
	if P > 0 {
		pid = b.Phi(ir.U32, pid0)
	}
	key := add(pos, lane)
	var kb, vt, lo, hi ir.Value
	ks := 1
	kvRow := s.KVHeads * s.Dim
	if P > 0 {
		// One page id for the tile; a lane past the row's last key reads the
		// same page, in bounds, and its score is masked.
		b.SetPhi(pid, tilePage(add(pos, u(32))))
		off := pageRem(b, pos, P)
		kb = add(add(mul(pid, u(P*kvRow)), mul(kvbase, u(P))), add(off, lane))
		if s.F16K {
			// Words: kvbase (a multiple of the even Dim) is a whole pair.
			kb = add(add(mul(pid, u(P*kvRow/2)), mul(b.Shr(ir.U32, kvbase, u(1)), u(P))), add(off, lane))
		}
		ks = P
		// The tile's V rows start at vt; key pos+j reads row j clamped into
		// [lo, hi], the tile's part of [keyStart, last], which is in this page.
		vt = add(mul(add(mul(pid, u(P)), off), u(kvRow)), kvbase)
		lo = b.Sub(ir.U32, b.Max(ir.U32, keyStart, pos), pos)
		hi = b.Sub(ir.U32, last, pos)
	} else {
		safe := b.Min(ir.U32, key, last)
		kb = add(mul(safe, u(s.KVHeads*s.Dim)), kvbase)
		if s.KStride > 0 {
			kb = add(mul(kvbase, u(s.KStride)), safe)
			ks = s.KStride
		}
	}
	score := zero
	for d := 0; d < s.Dim && !s.F16K; d++ {
		score = b.Fma(b.Load(ir.F32, q, qbase, int64(d)), b.Load(ir.F32, k, kb, int64(d*ks)), score)
	}
	for d := 0; d < s.Dim && s.F16K; d += 2 {
		lo, hi := kPair(b, b.Load(ir.F32, k, kb, int64(d/2*ks)))
		score = b.Fma(b.Load(ir.F32, q, qbase, int64(d)), lo, score)
		score = b.Fma(b.Load(ir.F32, q, qbase, int64(d+1)), hi, score)
	}
	score = b.Mul(ir.F32, score, b.ConstF32(s.Scale))
	if s.Softcap > 0 {
		e := b.Exp(b.Mul(ir.F32, b.ConstF32(2/s.Softcap), score))
		score = b.Mul(ir.F32, b.ConstF32(s.Softcap), b.Sub(ir.F32, b.ConstF32(1), b.Div(ir.F32, b.ConstF32(2), b.Add(ir.F32, e, b.ConstF32(1)))))
	}
	var valid ir.Value
	if P > 0 {
		valid = b.Lt(ir.U32, b.Sub(ir.U32, key, keyStart), b.Sub(ir.U32, count, keyStart))
	} else {
		valid = b.Lt(ir.U32, key, count)
	}
	score = b.Select(ir.F32, valid, score, minus)
	tileMax := butterfly(b, score, func(x, y ir.Value) ir.Value { return b.Max(ir.F32, x, y) })
	nextMax := b.Max(ir.F32, mx, tileMax)
	correction := b.Exp(b.Sub(ir.F32, mx, nextMax))
	weight := b.Select(ir.F32, valid, b.Exp(b.Sub(ir.F32, score, nextMax)), zero)
	tileSum := butterfly(b, weight, func(x, y ir.Value) ir.Value { return b.Add(ir.F32, x, y) })
	b.Store(prob, lane, weight, 0)
	b.Barrier()
	next := make([]ir.Value, len(accum))
	for i := range next {
		next[i] = b.Mul(ir.F32, accum[i], correction)
	}
	// Clamp every tail address before loading. Masking a probability alone does
	// not make a read of unwritten (possibly NaN) KV safe.
	loadV := func(vb, dim ir.Value) ir.Value {
		idx := add(vb, dim)
		if s.F16 {
			word := b.Bitcast(ir.U32, b.Load(ir.F32, v, b.Shr(ir.U32, idx, u(1)), 0))
			word = b.Shr(ir.U32, word, mul(b.And(ir.U32, idx, u(1)), u(16)))
			return b.CvtF16H(word)
		}
		return b.Load(ir.F32, v, idx, 0)
	}
	// vRow is where key pos+j's V row starts, clamped into the keys the row
	// may address.
	vRow := func(j int) ir.Value {
		if P > 0 {
			return add(vt, mul(b.Min(ir.U32, b.Max(ir.U32, u(j), lo), hi), u(kvRow)))
		}
		return add(mul(b.Min(ir.U32, add(pos, u(j)), last), u(kvRow)), kvbase)
	}
	// Every V load of a group of keys (64 values a lane) before its products,
	// so they are in flight together. Interleaved, sm_86 issued them a few at
	// a time: the paged kernel lost about a third at 4096 keys interleaved, and
	// the contiguous one lost far more on 256-wide heads, where a lane holds
	// eight accumulators.
	grp := min(32, max(1, 64/len(dims)))
	if g := benchKnob("flash-vgroup"); g > 0 {
		grp = g
	}
	if P > 0 || benchKnob("flash-interleaved") == 0 {
		for j0 := 0; j0 < 32; j0 += grp {
			vv := make([][]ir.Value, min(grp, 32-j0))
			for jj := range vv {
				vb := vRow(j0 + jj)
				for _, dim := range dims {
					vv[jj] = append(vv[jj], loadV(vb, dim))
				}
			}
			for jj := range vv {
				wj := j0 + jj
				if pagedFault == "vgroup" {
					wj = j0
				}
				w := b.Load(ir.F32, prob, u(wj), 0)
				for i := range dims {
					next[i] = b.Fma(w, vv[jj][i], next[i])
				}
			}
		}
	} else {
		for j := 0; j < 32; j++ {
			vb := vRow(j)
			w := b.Load(ir.F32, prob, u(j), 0)
			for i, dim := range dims {
				next[i] = b.Fma(w, loadV(vb, dim), next[i])
			}
		}
	}
	b.Barrier() // no lane overwrites probabilities while a peer still reads
	for i := range accum {
		b.SetPhi(accum[i], next[i])
	}
	b.SetPhi(sum, b.Fma(sum, correction, tileSum))
	b.SetPhi(mx, nextMax)
	b.SetPhi(pos, add(pos, u(32)))
	b.EndLoop()
	if splits > 1 {
		padded := (s.Dim + 31) / 32 * 32
		base := mul(group, u(padded+64))
		switch pagedFault {
		case "normpart":
			inv := b.Div(ir.F32, b.ConstF32(1), b.Max(ir.F32, sum, b.ConstF32(1e-30)))
			for i := range accum {
				accum[i] = b.Mul(ir.F32, accum[i], inv)
			}
		case "sinkchunk":
			if s.Sink {
				sum = b.Add(ir.F32, sum, b.Exp(b.Sub(ir.F32, b.Load(ir.F32, sink, head, 0), mx)))
			}
		}
		for i := range accum {
			b.Store(out, add(base, add(lane, u(i*32))), accum[i], 0)
		}
		// Replicated metadata gives each lane its own address, avoiding racing stores.
		b.Store(out, add(base, lane), mx, int64(padded))
		b.Store(out, add(base, lane), sum, int64(padded+32))
		return b.Done()
	}
	factor := b.ConstF32(1)
	denom := sum
	if s.Sink {
		sv := b.Load(ir.F32, sink, head, 0)
		finalMax := b.Max(ir.F32, mx, sv)
		factor = b.Exp(b.Sub(ir.F32, mx, finalMax))
		denom = b.Fma(sum, factor, b.Exp(b.Sub(ir.F32, sv, finalMax)))
	}
	factor = b.Div(ir.F32, factor, b.Max(ir.F32, denom, b.ConstF32(1e-30)))
	for i, dim := range dims {
		b.Store(out, add(qbase, dim), b.Mul(ir.F32, accum[i], factor), 0)
	}
	return b.Done()
}

// FlashPartialFloats is the scratch size for a split invocation, in floats.
func FlashPartialFloats(s FlashShape) int {
	return s.Rows * s.Heads * max(1, s.Splits) * (((s.Dim+31)/32)*32 + 64)
}

// FlashAttentionMerge combines the unnormalized partials using their maxima.
// Parameters: partials, output, and optional per-head sink logits. The sink is
// added once here, never once per key partition. One split is a merge too: the
// staged paged path's partials are unnormalised at every split count.
func FlashAttentionMerge(s FlashShape) (*ir.Kernel, error) {
	if s.Splits < 1 || s.Heads < 1 || s.Rows < 1 || s.Dim < 1 {
		return nil, fmt.Errorf("kernels: FlashAttentionMerge: invalid shape %+v", s)
	}
	b := ir.New("k", [3]int{32, 1, 1})
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
		// Ragged output lanes use the last valid dimension, including its partial.
		dim := b.Min(ir.U32, add(lane, u(i*32)), u(s.Dim-1))
		src := add(b.Mul(ir.U32, item, u(s.Splits*(padded+64))), dim)
		for j := range weights {
			acc = b.Fma(b.Load(ir.F32, p, src, int64(j*(padded+64))), weights[j], acc)
		}
		b.Store(out, add(b.Mul(ir.U32, item, u(s.Dim)), dim), b.Mul(ir.F32, acc, inv), 0)
	}
	return b.Done()
}

// FlashAttentionMergeWide is FlashAttentionMerge with a thread per output
// element instead of a 32-lane group per (row, head): the same partials, the
// same parameters and the same arithmetic, launched as
// ceil(Rows*Heads*Dim/128) groups of 128. The group-per-item form keeps
// Rows*Heads warps busy, each lane walking Dim/32 elements times every split,
// which on a model with few heads (Qwen3.5-0.8B: 8) is the merge's whole
// latency; this spreads it over every element.
func FlashAttentionMergeWide(s FlashShape) (*ir.Kernel, error) {
	if s.Splits < 1 || s.Heads < 1 || s.Rows < 1 || s.Dim < 1 {
		return nil, fmt.Errorf("kernels: FlashAttentionMergeWide: invalid shape %+v", s)
	}
	name := "flashmergewide"
	if s.Map {
		name += "_map"
	}
	b := ir.New(name, [3]int{128, 1, 1})
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
		// The slot's row, through the table: the item and the output
		// element are that row's.
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
	return b.Done()
}
