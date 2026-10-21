package mainkernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit_old/gpu/ir"
)

// FlashPrefill70Shape is one batched attention geometry for FlashPrefill70.
type FlashPrefill70Shape struct {
	Heads, KVHeads, Dim int
	Rows                int // the chunk's rows, a multiple of FlashPrefill70Rows
	KStride             int // the transposed K cache's position stride
	Scale, Softcap      float32
	Window              int // a sliding window of keys, 0 for none
	// Page > 0 reads K and V from paged pools of Page positions a page
	// (docs/design/device-kv-paging.md, paged.go): the kernel takes pTab and
	// pRow after pOut, every row's keys come from its descriptor (pN is not
	// read), KStride and Window must be 0, and a workgroup's rows are one
	// sequence's in position order (see pagedprefill.go). Splits > 0 divides
	// a workgroup's key tiles into that many runs, one workgroup each, whose
	// partials in pOut FlashAttentionMerge combines -- and applies sinks to,
	// which is why one split is not the same as none: 0 writes the output.
	Page, Splits int
	// F16 reads a paged V packed as binary16 two to a word.
	F16 bool
	// Map reads each query slot's row from a run table, pMap after pRow: slot
	// i is row pMap[i] of pQ, pOut and the descriptors. It is how one launch
	// takes the prompt chunks of a ragged step (docs/design/ragged-prefill.md):
	// each sequence's rows, in position order, fill whole workgroups of slots,
	// the last padded with that sequence's last row, so a workgroup's rows are
	// one sequence's as the paged form demands, and q and the output stay at
	// the step's own rows. A padded slot computes its row's output again, the
	// same arithmetic, and stores the same values. Paged only. With splits
	// the partials are at the rows, and FlashAttentionMergeWide takes the
	// same table (FlashShape.Map).
	Map bool
}

// flashPrefillSplits is how many key runs a flash prefill workgroup's rows
// are divided into.
func flashPrefillSplits(s FlashPrefill70Shape) int { return max(1, s.Splits) }

// flashPrefillCheck validates a paged flash prefill shape.
func flashPrefillCheck(what string, s FlashPrefill70Shape) error {
	if s.Page == 0 {
		if s.Splits > 1 || s.F16 || s.Map {
			return fmt.Errorf("kernels: %s: splits and f16 V are the paged form's", what)
		}
		return nil
	}
	if err := checkPage(what, s.Page); err != nil {
		return err
	}
	if s.KStride != 0 || s.Window != 0 || s.Splits < 0 || s.F16 && s.KVHeads*s.Dim%4 != 0 {
		return fmt.Errorf("kernels: %s: a paged shape has no KStride and no Window (the window is "+
			"the descriptor's keyStart), and packed f16 V needs whole 16-byte rows: %+v", what, s)
	}
	return nil
}

// FlashPrefill70Rows is how many query rows one FlashPrefill70 workgroup takes:
// flashWarps warps of sixteen.
const FlashPrefill70Rows = 16 * flashWarps

// FlashPrefill70Threads is FlashPrefill70's workgroup width.
const FlashPrefill70Threads = 32 * flashWarps

// flashWarps is the warps a workgroup, each sixteen query rows, all sharing one
// staged K and V tile. The tile is the traffic: every workgroup reads its
// whole causal width of K and V, so the query rows a workgroup covers divide
// what the kernel reads from memory.
const flashWarps = 4

// FlashPrefill70Groups is how many workgroups FlashPrefill70 launches.
func FlashPrefill70Groups(s FlashPrefill70Shape) int {
	return s.Heads * (s.Rows / FlashPrefill70Rows) * flashPrefillSplits(s)
}

// FlashPrefill70 is the batched attention of a prompt chunk in one kernel on
// sm_70's m8n8k4 (ir.MMAVolta()): scores, a running softmax and the weighted sum
// of V per 32-key tile, with the probabilities never leaving the workgroup.
//
// It replaces AttnScoresMMA70, the softmax and AttnAccMMA70, which wrote and
// re-read a rows x heads x positions float32 score matrix between them. It
// follows llama.cpp's flash_attn_ext_f16 (ggml-cuda/fattn-mma-f16.cuh, whose
// Volta path is m8n8k4 too): K, V and the queries staged in shared memory as
// binary16, S = K Q^T per tile, the row maximum and sum carried across tiles,
// P = exp(S - max) through shared memory into O^T += V^T P^T, one divide at
// the end.
//
// The products are the other way round from the textbook, as AttnScoresMMA70
// explains: keys (and, for the second product, dims) are the A rows a
// quad-pair owns and queries the B columns every quad-pair shares, so a D
// fragment is keys (or dims) by queries. A query's scores in one tile then sit
// in 16 lanes (those agreeing on bit 1), two apiece, and its maximum and sum
// reduce over lanes xor 1, 4, 8 and 16.
//
// Operands drop to binary16 and the accumulators stay float32; the
// per-element transform is the replaced kernels': the scale, the attention
// softcap, then -inf for a key at or past the row's causal count or before its
// window. A V row past the workgroup's widest count is staged as zero: the
// cache there holds nothing written, and 0 * NaN is NaN.
//
// Parameters: pQ ([row][head][dim] f32), pK (the transposed f32 cache, as
// AttnScoresMMA70), pV ([position][kv head][dim] f32), pN (element 0 the
// uniform width, 1+r row r's count), pOut ([row][head][dim]). Launch
// FlashPrefill70Groups groups of FlashPrefill70Threads.
func FlashPrefill70(s FlashPrefill70Shape) (*ir.Kernel, error) {
	hd := s.Dim
	switch {
	case s.KVHeads <= 0 || s.Heads%s.KVHeads != 0:
		return nil, fmt.Errorf("kernels: FlashPrefill70: %d heads over %d kv heads", s.Heads, s.KVHeads)
	case hd != 64 && hd != 128:
		return nil, fmt.Errorf("kernels: FlashPrefill70: head width %d (64 or 128)", hd)
	case s.Rows <= 0 || s.Rows%FlashPrefill70Rows != 0:
		return nil, fmt.Errorf("kernels: FlashPrefill70: %d rows, want a multiple of %d", s.Rows, FlashPrefill70Rows)
	case s.Page == 0 && s.KStride <= 0:
		return nil, fmt.Errorf("kernels: FlashPrefill70: needs the transposed K cache")
	case s.Window < 0 || s.Softcap < 0:
		return nil, fmt.Errorf("kernels: FlashPrefill70: window %d softcap %v", s.Window, s.Softcap)
	}
	if err := flashPrefillCheck("FlashPrefill70", s); err != nil {
		return nil, err
	}
	P, S := s.Page, flashPrefillSplits(s)
	const tk = 32 // keys a tile
	const wq = flashWarps
	const qw = 16 // queries a warp: two n-tiles
	gqa := s.Heads / s.KVHeads
	kvDim := s.KVHeads * hd
	qs := hd/2 + 1 // u32 stride of a staged K or Q row: odd, so rows fall in different banks
	const vs = tk/2 + 1
	const ps = tk + 1
	dts := hd / 32 // 32-dim tiles of the output

	name := fmt.Sprintf("flashprefill70_d%d", hd)
	if P > 0 {
		name = fmt.Sprintf("flashprefill70paged_d%d_s%d", hd, s.Splits)
	}
	if s.Map {
		name += "_map"
	}
	b := ir.New(name, [3]int{FlashPrefill70Threads, 1, 1})
	pQ := b.Param("pQ", ir.F32)
	pK := b.Param("pK", ir.F32)
	pV := b.Param("pV", ir.F32)
	pN := b.Param("pN", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	var pTab, pDesc ir.Value
	if P > 0 {
		pTab, pDesc = b.Param("pTab", ir.U32), b.Param("pRow", ir.U32)
	}
	c := func(v int64) ir.Value { return b.Const(ir.U32, v) }
	// rowOf is a query slot's row: itself, or the run table's (Map).
	rowOf := func(slot ir.Value) ir.Value { return slot }
	if s.Map {
		pMap := b.Param("pMap", ir.U32)
		rowOf = func(slot ir.Value) ir.Value { return b.Load(ir.U32, pMap, slot, 0) }
	}
	f := b.ConstF32
	negInf := b.Bitcast(ir.F32, c(0xFF800000))

	shQ := b.Shared("fpQ", ir.U32, wq*qw*qs)
	shK := b.Shared("fpK", ir.U32, tk*qs)
	shV := b.Shared("fpV", ir.U32, hd*vs)
	shP := b.Shared("fpP", ir.F32, wq*qw*ps)

	tid := b.TID()
	wg := b.CTAID()
	lane := b.And(ir.U32, tid, c(31))
	warp := b.Shr(ir.U32, tid, c(5))
	v := voltaLane{b, lane}
	var spl ir.Value
	if P > 0 {
		// The split varies fastest, then the head.
		spl = urem(b, wg, S)
		wg = udiv(b, wg, S)
	}
	h := b.Rem(ir.U32, wg, c(int64(s.Heads)))
	qb := b.Div(ir.U32, wg, c(int64(s.Heads)))
	kvBase := b.Mul(ir.U32, b.Div(ir.U32, h, c(int64(gqa))), c(int64(hd)))
	row0 := b.Mul(ir.U32, qb, c(wq*qw))
	wrow0 := b.Add(ir.U32, row0, b.Mul(ir.U32, warp, c(qw)))
	var maxcnt, ntiles, lastPos ir.Value
	var span prefillSpan
	var kStart, hlo ir.Value
	if P > 0 {
		// The workgroup's keys, [keyStart of its first row, keyEnd of its
		// last), in 32-key tiles from a multiple of 32, which lie in one page
		// each; this split's run of them.
		span = tileSpanOf(b, pDesc, rowOf(row0), rowOf(b.Add(ir.U32, row0, c(wq*qw-1))))
		kStart, ntiles = span.tileRun(b, spl, S, tk)
		hlo = b.Sub(ir.U32, span.hi, span.lo)
	} else {
		// The workgroup's widest causal count: its last row's.
		maxcnt = b.Load(ir.U32, pN, row0, wq*qw)
		ntiles = b.Shr(ir.U32, b.Add(ir.U32, maxcnt, c(tk-1)), c(5))
		lastPos = b.Sub(ir.U32, b.Max(ir.U32, maxcnt, c(1)), c(1))
	}

	// The queries, once: 32 rows of hd dims as binary16 pairs.
	nth := int64(FlashPrefill70Threads)
	for it := int64(0); it < int64(wq*qw*hd/2)/nth; it++ {
		w := b.Add(ir.U32, tid, c(nth*it))
		qi := b.Div(ir.U32, w, c(int64(hd/2)))
		p := b.Rem(ir.U32, w, c(int64(hd/2)))
		src := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, rowOf(b.Add(ir.U32, row0, qi)), c(int64(s.Heads))), h),
			c(int64(hd))), b.Shl(ir.U32, p, c(1)))
		e := b.LoadV(ir.F32, pQ, src, 0, 2)
		b.Store(shQ, b.Add(ir.U32, b.Mul(ir.U32, qi, c(int64(qs))), p), b.PackF16(e[0], e[1]), 0)
	}

	// Per lane: the four query columns of each n-tile it holds, and their counts.
	rowL := v.row()
	quad := v.quad()
	q8 := b.Shl(ir.U32, quad, c(3))
	colOf := func(j, i int) ir.Value { return b.Add(ir.U32, v.dCol(i), c(int64(8*j))) }
	slots := [4]int{0, 1, 4, 5} // accumulators i with distinct columns; i^2 is the same column
	var cnt, kst [2][4]ir.Value // paged: keyEnd and keyStart
	for j := 0; j < 2; j++ {
		for x, i := range slots {
			if P > 0 {
				r := rowOf(b.Add(ir.U32, wrow0, colOf(j, i)))
				kst[j][x], cnt[j][x] = pagedDesc(b, pDesc, r, PRowStart), pagedDesc(b, pDesc, r, PRowEnd)
				continue
			}
			cnt[j][x] = b.Load(ir.U32, pN, b.Add(ir.U32, wrow0, colOf(j, i)), 1)
		}
	}

	zero := b.ConstF32(0)
	zeroU := c(0)
	// The loop state: the output accumulators, and each column's maximum and
	// sum. newState makes its phis, entered with from's values or, from nil,
	// the start of the softmax.
	type state struct {
		o    [][2][]ir.Value
		m, l [2][4]ir.Value
	}
	newState := func(from *state) state {
		var t state
		for dt := 0; dt < dts; dt++ {
			var oj [2][]ir.Value
			for j := 0; j < 2; j++ {
				oj[j] = make([]ir.Value, 8)
				for i := range oj[j] {
					init := zero
					if from != nil {
						init = from.o[dt][j][i]
					}
					oj[j][i] = b.Phi(ir.F32, init)
				}
			}
			t.o = append(t.o, oj)
		}
		for j := 0; j < 2; j++ {
			for x := range slots {
				mi, li := negInf, zero
				if from != nil {
					mi, li = from.m[j][x], from.l[j][x]
				}
				t.m[j][x] = b.Phi(ir.F32, mi)
				t.l[j][x] = b.Phi(ir.F32, li)
			}
		}
		return t
	}
	var kt, k0, pid, pg ir.Value
	var outer, st state
	if P > 0 {
		// The run's tiles a page at a time: an outer loop over the pages it
		// touches reads each page's id once, and the tiles inside it need no
		// lookup.
		kEnd := b.Add(ir.U32, kStart, b.Mul(ir.U32, ntiles, c(tk)))
		pg0 := pageDiv(b, kStart, P)
		npg := b.Select(ir.U32, b.Lt(ir.U32, zeroU, ntiles),
			b.Add(ir.U32, b.Sub(ir.U32, pageDiv(b, b.Sub(ir.U32, kEnd, c(1)), P), pg0), c(1)), zeroU)
		b.LoopN(npg)
		outer = newState(nil)
		pg = b.Phi(ir.U32, pg0)
		t0 := b.Max(ir.U32, kStart, b.Mul(ir.U32, pg, c(int64(P))))
		pid = pageOf(b, pTab, span.tab, t0, P)
		t1 := b.Min(ir.U32, kEnd, b.Mul(ir.U32, b.Add(ir.U32, pg, c(1)), c(int64(P))))
		b.LoopN(udiv(b, b.Sub(ir.U32, t1, t0), tk))
		st = newState(&outer)
		k0 = b.Phi(ir.U32, t0)
	} else {
		b.LoopN(ntiles)
		kt = b.Phi(ir.U32, zeroU)
		st = newState(nil)
		k0 = b.Shl(ir.U32, kt, c(5))
	}
	o, m, l := st.o, st.m, st.l

	// Stage the tile. The barrier first: the last tile's fragments were being
	// read out of these same arrays.
	b.Barrier()
	for it := int64(0); it < int64(tk*hd/2)/nth; it++ {
		w := b.Add(ir.U32, tid, c(nth*it))
		key := b.And(ir.U32, w, c(tk-1)) // consecutive threads, consecutive keys: coalesced
		p := b.Shr(ir.U32, w, c(5))
		var lo, hi ir.Value
		if P > 0 {
			// A page's K is [kvRow][P]: the tile's keys are adjacent.
			e := b.Add(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, pid, c(int64(P*kvDim))),
				b.Mul(ir.U32, b.Add(ir.U32, kvBase, b.Shl(ir.U32, p, c(1))), c(int64(P)))), b.Add(ir.U32, pageRem(b, k0, P), key))
			lo, hi = b.Load(ir.F32, pK, e, 0), b.Load(ir.F32, pK, e, int64(P))
		} else {
			pos := b.Min(ir.U32, b.Add(ir.U32, k0, key), c(int64(s.KStride-1)))
			e := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, kvBase, b.Shl(ir.U32, p, c(1))), c(int64(s.KStride))), pos)
			lo = b.Load(ir.F32, pK, e, 0)
			hi = b.Load(ir.F32, pK, e, int64(s.KStride))
		}
		b.Store(shK, b.Add(ir.U32, b.Mul(ir.U32, key, c(int64(qs))), p), b.PackF16(lo, hi), 0)
	}
	// V four dims at a time: a row's dims are contiguous in the cache, so each
	// item is two 16-byte loads (positions p and p+1) and four packed words.
	for it := int64(0); it < int64(tk/2*hd/4)/nth; it++ {
		w := b.Add(ir.U32, tid, c(nth*it))
		dq := b.Rem(ir.U32, w, c(int64(hd/4))) // consecutive threads, consecutive dims: coalesced
		kp := b.Div(ir.U32, w, c(int64(hd/4)))
		p0 := b.Add(ir.U32, k0, b.Shl(ir.U32, kp, c(1)))
		p1 := b.Add(ir.U32, p0, c(1))
		var ok0, ok1 ir.Value
		var v0, v1 []ir.Value
		if P > 0 {
			// Both positions are in the tile's page; one outside the
			// workgroup's keys is a slot nothing wrote.
			ok0, ok1 = b.Lt(ir.U32, b.Sub(ir.U32, p0, span.lo), hlo), b.Lt(ir.U32, b.Sub(ir.U32, p1, span.lo), hlo)
			if pagedFault == "vmask" {
				ok0, ok1 = span.in(b, p0), span.in(b, p1)
			}
			v0, v1 = pagedV4(b, pV, pid, p0, b.Add(ir.U32, kvBase, b.Shl(ir.U32, dq, c(2))), P, kvDim, s.F16),
				pagedV4(b, pV, pid, p1, b.Add(ir.U32, kvBase, b.Shl(ir.U32, dq, c(2))), P, kvDim, s.F16)
		} else {
			at := func(p ir.Value) ir.Value {
				return b.Add(ir.U32, b.Mul(ir.U32, b.Min(ir.U32, p, lastPos), c(int64(kvDim))),
					b.Add(ir.U32, kvBase, b.Shl(ir.U32, dq, c(2))))
			}
			ok0, ok1 = b.Lt(ir.U32, p0, maxcnt), b.Lt(ir.U32, p1, maxcnt)
			v0 = b.LoadV(ir.F32, pV, at(p0), 0, 4)
			v1 = b.LoadV(ir.F32, pV, at(p1), 0, 4)
		}
		base := b.Add(ir.U32, b.Mul(ir.U32, b.Shl(ir.U32, dq, c(2)), c(vs)), kp)
		for x := 0; x < 4; x++ {
			b.Store(shV, base, b.PackF16(b.Select(ir.F32, ok0, v0[x], zero), b.Select(ir.F32, ok1, v1[x], zero)), int64(x*vs))
		}
	}
	b.Barrier()

	// S^T = K Q^T over the tile: quad-pair q holds keys 8q..8q+7.
	var sacc [2][]ir.Value
	for j := range sacc {
		sacc[j] = make([]ir.Value, 8)
		for i := range sacc[j] {
			sacc[j][i] = zero
		}
	}
	kRow := b.Mul(ir.U32, b.Add(ir.U32, q8, rowL), c(int64(qs)))
	var qRow [2]ir.Value
	for j := 0; j < 2; j++ {
		qRow[j] = b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, warp, c(qw)), b.Add(ir.U32, rowL, c(int64(8*j)))), c(int64(qs)))
	}
	for st := 0; st < hd/4; st++ {
		af := []ir.Value{b.Load(ir.U32, shK, kRow, int64(2*st)), b.Load(ir.U32, shK, kRow, int64(2*st+1))}
		for j := 0; j < 2; j++ {
			bf := []ir.Value{b.Load(ir.U32, shQ, qRow[j], int64(2*st)), b.Load(ir.U32, shQ, qRow[j], int64(2*st+1))}
			sacc[j] = b.MMA(ir.MMAVolta(), af, bf, sacc[j])
		}
	}

	// The transform, the running maximum and sum, P into shared memory.
	sc := f(s.Scale)
	pRow := func(j, i int) ir.Value {
		return b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, warp, c(qw)), colOf(j, i)), c(ps))
	}
	var mNext, lNext [2][4]ir.Value
	alpha := [2][4]ir.Value{}
	for j := 0; j < 2; j++ {
		var val [8]ir.Value
		for i := 0; i < 8; i++ {
			x := b.Mul(ir.F32, sacc[j][i], sc)
			if s.Softcap > 0 {
				e := b.Exp(b.Mul(ir.F32, f(2/s.Softcap), x))
				x = b.Mul(ir.F32, f(s.Softcap), b.Sub(ir.F32, f(1), b.Div(ir.F32, f(2), b.Add(ir.F32, e, f(1)))))
			}
			key := b.Add(ir.U32, k0, b.Add(ir.U32, q8, v.dRow(i)))
			ct := cnt[j][slotOf(i)]
			var valid ir.Value
			if P > 0 {
				valid = rowAttends(b, key, kst[j][slotOf(i)], ct)
			} else {
				valid = b.Lt(ir.U32, key, ct)
			}
			x = b.Select(ir.F32, valid, x, negInf)
			if s.Window > 0 {
				lo := b.Sub(ir.U32, ct, b.Min(ir.U32, ct, c(int64(s.Window))))
				x = b.Select(ir.F32, b.Lt(ir.U32, key, lo), negInf, x)
			}
			val[i] = x
		}
		for x, i0 := range slots {
			i1 := i0 ^ 2
			t := b.Max(ir.F32, val[i0], val[i1])
			for _, mk := range []int64{1, 4, 8, 16} {
				t = b.Max(ir.F32, t, b.ShuffleXor(ir.F32, t, mk))
			}
			mn := b.Max(ir.F32, m[j][x], t)
			// A column with no key yet stays at -inf; exp against a finite
			// stand-in keeps -inf - -inf out of it.
			mf := b.Max(ir.F32, mn, f(-1e30))
			a := b.Exp(b.Sub(ir.F32, m[j][x], mf))
			p0 := b.Exp(b.Sub(ir.F32, val[i0], mf))
			p1 := b.Exp(b.Sub(ir.F32, val[i1], mf))
			mNext[j][x] = mn
			lNext[j][x] = b.Add(ir.F32, b.Mul(ir.F32, l[j][x], a), b.Add(ir.F32, p0, p1))
			alpha[j][x] = a
			b.Store(shP, b.Add(ir.U32, pRow(j, i0), b.Add(ir.U32, q8, v.dRow(i0))), p0, 0)
			b.Store(shP, b.Add(ir.U32, pRow(j, i1), b.Add(ir.U32, q8, v.dRow(i1))), p1, 0)
		}
	}
	b.Barrier()

	// O^T += V^T P^T: quad-pair q holds dims 32*dt + 8q .. +7.
	var dRowV [4]ir.Value
	for dt := 0; dt < dts; dt++ {
		dRowV[dt] = b.Mul(ir.U32, b.Add(ir.U32, c(int64(32*dt)), b.Add(ir.U32, q8, rowL)), c(vs))
	}
	var pbRow [2]ir.Value
	for j := 0; j < 2; j++ {
		pbRow[j] = b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, warp, c(qw)), b.Add(ir.U32, rowL, c(int64(8*j)))), c(ps))
	}
	var od [][2][]ir.Value
	for dt := 0; dt < dts; dt++ {
		var oj [2][]ir.Value
		for j := 0; j < 2; j++ {
			oj[j] = make([]ir.Value, 8)
			for i := 0; i < 8; i++ {
				oj[j][i] = b.Mul(ir.F32, o[dt][j][i], alpha[j][slotOf(i)])
			}
		}
		od = append(od, oj)
	}
	for st := 0; st < tk/4; st++ {
		var bf [2][]ir.Value
		for j := 0; j < 2; j++ {
			e := make([]ir.Value, 4)
			for x := range e {
				e[x] = b.Load(ir.F32, shP, pbRow[j], int64(4*st+x))
			}
			bf[j] = []ir.Value{b.PackF16(e[0], e[1]), b.PackF16(e[2], e[3])}
		}
		for dt := 0; dt < dts; dt++ {
			af := []ir.Value{b.Load(ir.U32, shV, dRowV[dt], int64(2*st)), b.Load(ir.U32, shV, dRowV[dt], int64(2*st+1))}
			for j := 0; j < 2; j++ {
				od[dt][j] = b.MMA(ir.MMAVolta(), af, bf[j], od[dt][j])
			}
		}
	}
	for dt := 0; dt < dts; dt++ {
		for j := 0; j < 2; j++ {
			for i := 0; i < 8; i++ {
				b.SetPhi(o[dt][j][i], od[dt][j][i])
			}
		}
	}
	for j := 0; j < 2; j++ {
		for x := range slots {
			b.SetPhi(m[j][x], mNext[j][x])
			b.SetPhi(l[j][x], lNext[j][x])
		}
	}
	if P > 0 {
		b.SetPhi(k0, b.Add(ir.U32, k0, c(tk)))
		b.EndLoop()
		for dt := range o {
			for j := range o[dt] {
				for i := range o[dt][j] {
					b.SetPhi(outer.o[dt][j][i], o[dt][j][i])
				}
			}
		}
		for j := 0; j < 2; j++ {
			for x := range slots {
				b.SetPhi(outer.m[j][x], m[j][x])
				b.SetPhi(outer.l[j][x], l[j][x])
			}
		}
		b.SetPhi(pg, b.Add(ir.U32, pg, c(1)))
		b.EndLoop()
		o, m, l = outer.o, outer.m, outer.l
	} else {
		b.SetPhi(kt, b.Add(ir.U32, kt, c(1)))
		b.EndLoop()
	}

	// From the phis: a workgroup whose widest count is zero runs no tile, and
	// a body value is then a register nothing wrote.
	if P > 0 && s.Splits > 0 {
		// This run's partials for FlashAttentionMerge: the unnormalised sums,
		// and each query's maximum (the finite stand-in its exponentials were
		// taken against) and sum. The 16 lanes holding a query column (all but
		// lane bit 1) write two of its 32 replicas each.
		stride := int64(hd + 64)
		rep := b.Add(ir.U32, b.And(ir.U32, lane, c(1)), b.Shl(ir.U32, b.Shr(ir.U32, lane, c(2)), c(1)))
		pbOf := func(qr ir.Value) ir.Value {
			return b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, qr, c(int64(s.Heads))), h),
				c(int64(S))), spl), c(stride))
		}
		for j := 0; j < 2; j++ {
			for x, i := range slots {
				t := l[j][x]
				for _, mk := range []int64{1, 4, 8, 16} {
					t = b.Add(ir.F32, t, b.ShuffleXor(ir.F32, t, mk))
				}
				partialML(b, pOut, pbOf(rowOf(b.Add(ir.U32, wrow0, colOf(j, i)))), b.Max(ir.F32, m[j][x], f(-1e30)), t, rep, hd, 16)
			}
		}
		for dt := 0; dt < dts; dt++ {
			for j := 0; j < 2; j++ {
				for i := 0; i < 8; i++ {
					dim := b.Add(ir.U32, c(int64(32*dt)), b.Add(ir.U32, q8, v.dRow(i)))
					b.Store(pOut, b.Add(ir.U32, pbOf(rowOf(b.Add(ir.U32, wrow0, colOf(j, i)))), dim), o[dt][j][i], 0)
				}
			}
		}
		return b.Done(), nil
	}
	var lsum [2][4]ir.Value
	for j := 0; j < 2; j++ {
		for x := range slots {
			t := l[j][x]
			for _, mk := range []int64{1, 4, 8, 16} {
				t = b.Add(ir.F32, t, b.ShuffleXor(ir.F32, t, mk))
			}
			lsum[j][x] = b.Div(ir.F32, f(1), b.Max(ir.F32, t, f(1e-30)))
		}
	}
	for dt := 0; dt < dts; dt++ {
		for j := 0; j < 2; j++ {
			for i := 0; i < 8; i++ {
				dim := b.Add(ir.U32, c(int64(32*dt)), b.Add(ir.U32, q8, v.dRow(i)))
				qr := rowOf(b.Add(ir.U32, wrow0, colOf(j, i)))
				at := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, qr, c(int64(s.Heads))), h), c(int64(hd))), dim)
				b.Store(pOut, at, b.Mul(ir.F32, o[dt][j][i], lsum[j][slotOf(i)]), 0)
			}
		}
	}
	return b.Done(), nil
}

// slotOf is which of the four column slots accumulator i's column is: i and
// i^2 share one (see voltaLane.dCol).
func slotOf(i int) int {
	i &^= 2
	switch i {
	case 0:
		return 0
	case 1:
		return 1
	case 4:
		return 2
	}
	return 3
}
