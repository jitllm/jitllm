package kernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// The ops between the matvecs, so the residual stream stays on the device
// instead of returning to the host after every matvec. A cross-workgroup
// reduction is a second kernel in the same submission: the launch is the
// barrier, with no host round trip.

// quantGroup is Quantize's workgroup: 128 threads, 16 blocks of 8.
const quantGroup = 128

// QuantizeThreads is how many threads a Quantize over nb 32-element blocks
// launches: one per packed output word, eight per block.
func QuantizeThreads(nb int) int { return nb * 8 }

// Quantize packs k floats as int8 with a scale per 32-element block (taken
// over an amax window of `window` elements) and a sum per 16. Launch it with
// QuantizeThreads(k/32) threads.
//
// Eight threads a block each own four values and one packed word; the block's
// (or window's) maximum and the per-16 sums meet in shared memory.
//
// It quantizes exactly as jit/cpu's generated quantizer does: the scale and its
// reciprocal by two correctly rounded divisions (quantScales), then half away
// from zero, built as trunc(f + copysign(0.5, f)) with no branch. Any
// difference is a slightly different int8 near a boundary, which diverges deep
// models (TestQuantizeOnTheRoundingBoundary).
func Quantize(k, window int) (*ir.Kernel, error) {
	return quantize(k, window, nil)
}

// QuantizeAct is ActMul followed by Quantize in one launch: element i is
// act(g[i]) * u[i] as ActMul computes it, written to pOut as ActMul writes it,
// and quantized exactly as Quantize quantizes pOut -- the same bits as the
// two kernels, without the float row's second read and the second launch.
// Parameters pG, pU, pOut, pA, pAX; launch QuantizeThreads(k/32) threads.
func QuantizeAct(k, window int, kind ActKind) (*ir.Kernel, error) {
	if err := actMulKind(kind); err != nil {
		return nil, err
	}
	return quantize(k, window, &kind)
}

func quantize(k, window int, kind *ActKind) (*ir.Kernel, error) {
	if k%32 != 0 {
		return nil, fmt.Errorf("kernels: Quantize: k=%d is not a multiple of 32", k)
	}
	nb := k / 32
	// The amax window must match the CPU's (see cpu.QuantizeQ8Window) or the
	// tiers disagree.
	per := window / 32
	if per < 1 {
		per = 1
	}
	const wpb = 8 // words (threads) per 32-element block
	span := per * wpb
	if span > quantGroup {
		// A window wider than one group's tile loops (quantwide.go).
		return quantizeWide(k, window, kind)
	}
	if quantGroup%span != 0 {
		return nil, fmt.Errorf("kernels: Quantize: window %d does not tile a %d-thread group", window, quantGroup)
	}
	name := "quantize"
	if kind != nil {
		name = "quantizeact"
	}
	b := ir.New(name, [3]int{quantGroup, 1, 1})
	var pX, pG, pU, pOut ir.Value
	if kind == nil {
		pX = b.Param("pX", ir.F32) // activations
	} else {
		pG, pU, pOut = b.Param("pG", ir.F32), b.Param("pU", ir.F32), b.Param("pOut", ir.F32)
	}
	pA := b.Param("pA", ir.U32)   // packed int8, 4 per word
	pAX := b.Param("pAX", ir.F32) // [scale(nb) | sum16(2nb)]
	shMax := b.Shared("amax", ir.F32, quantGroup)
	shSum := b.Shared("sum", ir.F32, quantGroup)

	// Every index comes from the clamped word, never from TID: a surplus
	// thread clamped onto the last word must read the same shared slots as the
	// real owner to write the same bits (RULE 13). It still stores at its own
	// TID slot, which nothing reads.
	word := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(nb*wpb-1)))
	local := b.Sub(ir.U32, word, b.Mul(ir.U32, b.CTAID(), b.NTID()))
	blk := b.Div(ir.U32, word, b.Const(ir.U32, wpb))
	base := b.Mul(ir.U32, word, b.Const(ir.U32, 4))
	var v [4]ir.Value
	for i := range v {
		if kind == nil {
			v[i] = b.Load(ir.F32, pX, base, int64(i))
			continue
		}
		at := b.Add(ir.U32, base, b.Const(ir.U32, int64(i)))
		v[i] = actMulValue(b, *kind, b.Load(ir.F32, pG, at, 0), func() ir.Value { return b.Load(ir.F32, pU, at, 0) })
		b.Store(pOut, at, v[i], 0)
	}
	b.Store(shMax, b.TID(), amaxOf(b, v[:]), 0)
	b.Barrier()
	wbase := b.Mul(ir.U32, b.Div(ir.U32, local, b.Const(ir.U32, int64(span))), b.Const(ir.U32, int64(span)))
	amax := b.ConstF32(0)
	for i := 0; i < span; i++ {
		amax = b.Max(ir.F32, amax, b.Load(ir.F32, shMax, wbase, int64(i)))
	}

	half := b.Const(ir.U32, 0x3F000000) // 0.5f
	signMask := b.Const(ir.U32, 0x80000000)
	// d and inv as the host derives them; an all-zero block has inv 0 and
	// quantizes to zeros.
	d, inv := quantScales(b, amax)
	sum := b.Const(ir.I32, 0)
	var packed ir.Value
	for i := range v {
		f := b.Mul(ir.F32, v[i], inv)
		sgn := b.And(ir.U32, b.Bitcast(ir.U32, f), signMask)
		rnd := b.Bitcast(ir.F32, b.Add(ir.U32, sgn, half)) // copysign(0.5, f)
		q := b.CvtI32(b.Add(ir.F32, f, rnd))
		sum = b.Add(ir.I32, sum, q)
		byteI := b.And(ir.U32, q, b.Const(ir.U32, 0xFF))
		if i == 0 {
			packed = byteI
		} else {
			packed = b.Add(ir.U32, packed, b.Shl(ir.U32, byteI, b.Const(ir.U32, int64(8*i))))
		}
	}
	b.Store(pA, word, packed, 0)
	b.Store(shSum, b.TID(), b.CvtF32(sum), 0)
	b.Barrier()
	// A 16-element sum is four words; the int8 sums are exact in f32.
	qbase := b.Mul(ir.U32, b.Div(ir.U32, local, b.Const(ir.U32, 4)), b.Const(ir.U32, 4))
	s16 := b.Load(ir.F32, shSum, qbase, 0)
	for i := 1; i < 4; i++ {
		s16 = b.Add(ir.F32, s16, b.Load(ir.F32, shSum, qbase, int64(i)))
	}
	b.Store(pAX, blk, d, 0)
	b.Store(pAX, b.Div(ir.U32, word, b.Const(ir.U32, 4)), s16, int64(nb))
	return b.Done(), nil
}

// NormPart is the first half of RMSNorm: partial sums of squares.
func NormPart(k, parts int) (*ir.Kernel, error) { return NormPartRows(k, parts, 1) }

// NormPartRows is NormPart over `rows` independent vectors, as a prefill chunk
// needs: the rows do not interact, so it is the same kernel with the thread
// index split into (row, part). Only the clamp grows, so rows == 1 emits the
// original kernel.
func NormPartRows(k, parts, rows int) (*ir.Kernel, error) {
	if k%parts != 0 {
		return nil, fmt.Errorf("kernels: NormPart: %d parts do not divide %d", parts, k)
	}
	if rows < 1 {
		rows = 1
	}
	per := k / parts
	b := ir.New("normpartrows", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	p := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(parts*rows-1)))
	start := b.Mul(ir.U32, p, b.Const(ir.U32, int64(per)))
	if rows > 1 {
		// flat = row*parts + part, and the parts of a row are contiguous, so
		// `start` and the output index need no change; only the clamp grows.
		_ = start
	}
	zero := b.ConstF32(0)
	b.Loop(int64(per))
	acc := b.Phi(ir.F32, zero)
	idx := b.Phi(ir.U32, start)
	x := b.Load(ir.F32, pX, idx, 0)
	b.SetPhi(acc, b.Fma(x, x, acc))
	b.SetPhi(idx, b.Add(ir.U32, idx, b.Const(ir.U32, 1)))
	b.EndLoop()
	b.Store(pOut, p, acc, 0)
	return b.Done(), nil
}

// NormApply finishes RMSNorm: out[i] = x[i] * rsqrt(mean + eps) * w[i].
//
// Every thread re-sums the partials, `parts` loads against the one store it
// makes, which is cheaper than another kernel.
func NormApply(k, parts int, eps float32, addOne bool) (*ir.Kernel, error) {
	return NormApplyRows(k, parts, eps, addOne, 1)
}

// RMSNormRows is a whole RMSNorm in one launch, a 1024-thread workgroup per
// row: each thread sums the squares of k/1024 elements, a shared-memory tree
// reduces them, and the same threads scale and weight their elements. With add
// it first forms x+y, writes the sum to pSum (the residual) and normalises it.
// Params: pX, [pY,] pW, [pSum,] pOut. Launch rows groups of RMSNormGroup.
//
// A 1024-thread group and a tree reduction, as llama.cpp's rms_norm_f32:
// a decode norm is one row, so a 128-thread group ran it on one SM. The tree
// needs no conditional (RULE 13): at every step every thread adds its
// partner's partial, clamped into the group; threads above the live range
// compute sums nobody reads, and a barrier separates the reads from the
// writes.
func RMSNormRows(k, rows int, eps float32, addOne, add bool) (*ir.Kernel, error) {
	return rmsNormRows(k, rows, eps, addOne, add, 0, false)
}

// RMSNormRowsWarp is RMSNormRows reducing its sum of squares with subgroup
// shuffles instead of the shared-memory tree: a butterfly inside each 32-lane
// warp, one partial per warp in threadgroup memory, and a second butterfly over
// the 32 partials. The partial for warp w is stored at tid/32, so subgroups
// must be contiguous runs of 32 threads; the caller asks for this kernel only
// on CUDA, where they are.
func RMSNormRowsWarp(k, rows int, eps float32, addOne, add bool) (*ir.Kernel, error) {
	return rmsNormRows(k, rows, eps, addOne, add, 0, true)
}

// RMSNormQuantRowsWarp is RMSNormQuantRows with RMSNormRowsWarp's reduction.
func RMSNormQuantRowsWarp(k, rows int, eps float32, addOne, add bool, window int) (*ir.Kernel, error) {
	if _, err := RMSNormQuantRows(k, rows, eps, addOne, add, window); err != nil {
		return nil, err
	}
	return rmsNormRows(k, rows, eps, addOne, add, max(window, 32), true)
}

// RMSNormQuantRows is RMSNormRows that also writes what Quantize(rows*k,
// window) would write from its output: packed int8 into pA and scales then
// per-16 sums into pAX, after pOut. The norm's output is still written.
//
// One launch where there were two, with the same bits. A kernel may not read
// a buffer it writes (RULE 13), so the quantize phase recomputes each word's
// four values from the input with the norm's own arithmetic, then quantizes
// exactly as Quantize does, so the result is bit-identical to the two kernels.
func RMSNormQuantRows(k, rows int, eps float32, addOne, add bool, window int) (*ir.Kernel, error) {
	if k%32 != 0 {
		return nil, fmt.Errorf("kernels: RMSNormQuantRows: k=%d is not a multiple of 32", k)
	}
	per := max(window/32, 1)
	// A window of the whole row or wider is one scale a row (quantPhase's row
	// mode); a narrower one must tile the group.
	if span := per * 8; window < k && (span > RMSNormGroup || RMSNormGroup%span != 0 || (k/4)%span != 0) {
		return nil, fmt.Errorf("kernels: RMSNormQuantRows: window %d does not tile k=%d", window, k)
	}
	return rmsNormRows(k, rows, eps, addOne, add, max(window, 32), false)
}

// RMSNormF16TRows is RMSNormRows (RMSNormRowsWarp with warp) that also
// writes what ActF16T(q, rows, k) would write from its output: the normed
// rows as q's token-major binary16 operand, into pB after pOut. One launch
// where a binary16 GEMM's prompt chunk took two, with the same bits: the
// second phase recomputes each sub-block's values with the norm's own
// arithmetic, as RMSNormQuantRows does (RULE 13).
func RMSNormF16TRows(k, rows int, eps float32, addOne, add bool, q Quant, warp bool) (*ir.Kernel, error) {
	if !Volta70OK(q) {
		return nil, fmt.Errorf("kernels: RMSNormF16TRows: %v has no binary16 layout", q)
	}
	qi := qtab[q]
	if k%qi.sub != 0 || qi.sub%8 != 0 {
		return nil, fmt.Errorf("kernels: RMSNormF16TRows: k=%d is not a multiple of %d", k, qi.sub)
	}
	return rmsNormRowsAs(k, rows, eps, addOne, add, 0, warp, &qi)
}

func rmsNormRows(k, rows int, eps float32, addOne, add bool, window int, warp bool) (*ir.Kernel, error) {
	return rmsNormRowsAs(k, rows, eps, addOne, add, window, warp, nil)
}

// rmsNormRowsAs is the RMSNorm family's one emitter: window > 0 adds the
// quantize phase, f16 the binary16 operand phase in that format's layout.
func rmsNormRowsAs(k, rows int, eps float32, addOne, add bool, window int, warp bool, f16 *qinfo) (*ir.Kernel, error) {
	if k < 1 || rows < 1 {
		return nil, fmt.Errorf("kernels: RMSNormRows: k=%d rows=%d", k, rows)
	}
	const g = RMSNormGroup
	name := "rmsnormrows"
	if window > 0 {
		name = "rmsnormquant"
	}
	if f16 != nil {
		name = "rmsnormf16t"
	}
	if warp {
		name += "w"
	}
	b := ir.New(name, [3]int{g, 1, 1})
	pX := b.Param("pX", ir.F32)
	var pY, pSum ir.Value
	if add {
		pY = b.Param("pY", ir.F32)
	}
	pW := b.Param("pW", ir.F32)
	if add {
		pSum = b.Param("pSum", ir.F32)
	}
	pOut := b.Param("pOut", ir.F32)
	var pA, pAX, shMax, shSum, pB ir.Value
	if window > 0 {
		pA = b.Param("pA", ir.U32)
		pAX = b.Param("pAX", ir.F32)
	}
	if f16 != nil {
		pB = b.Param("pB", ir.U32)
	}
	part := b.Shared("ss", ir.F32, g)
	if window > 0 {
		shMax = b.Shared("amax", ir.F32, g)
		shSum = b.Shared("qsum", ir.F32, g)
	}

	r := b.Min(ir.U32, b.CTAID(), b.Const(ir.U32, int64(rows-1)))
	rowBase := b.Mul(ir.U32, r, b.Const(ir.U32, int64(k)))
	last := b.Const(ir.U32, int64(k-1))
	kk := b.Const(ir.U32, int64(k))
	step := b.Const(ir.U32, g)
	zeroF := b.ConstF32(0)
	tid := b.TID()
	iters := int64((k + g - 1) / g)
	value := func(idx ir.Value, store bool) ir.Value {
		at := b.Add(ir.U32, rowBase, idx)
		v := b.Load(ir.F32, pX, at, 0)
		if add {
			v = b.Add(ir.F32, v, b.Load(ir.F32, pY, at, 0))
			if store {
				b.Store(pSum, at, v, 0)
			}
		}
		return v
	}

	b.Loop(iters)
	acc := b.Phi(ir.F32, zeroF)
	i := b.Phi(ir.U32, tid)
	idx := b.Min(ir.U32, i, last)
	v := value(idx, true)
	b.SetPhi(acc, b.Add(ir.F32, acc, b.Select(ir.F32, b.Lt(ir.U32, i, kk), b.Mul(ir.F32, v, v), zeroF)))
	b.SetPhi(i, b.Add(ir.U32, i, step))
	b.EndLoop()

	var ss ir.Value
	if warp {
		// See RMSNormRowsWarp. The group is 1024 threads, so 32 warps and
		// exactly one partial per lane for the second butterfly.
		const lanes = ir.SubgroupLanes
		v := acc
		for m := lanes / 2; m >= 1; m /= 2 {
			v = b.Add(ir.F32, v, b.ShuffleXor(ir.F32, v, int64(m)))
		}
		b.Store(part, b.Shr(ir.U32, tid, b.Const(ir.U32, 5)), v, 0)
		b.Barrier()
		t := b.Load(ir.F32, part, b.And(ir.U32, tid, b.Const(ir.U32, lanes-1)), 0)
		for m := lanes / 2; m >= 1; m /= 2 {
			t = b.Add(ir.F32, t, b.ShuffleXor(ir.F32, t, int64(m)))
		}
		ss = t
	} else {
		b.Store(part, tid, acc, 0)
		b.Barrier()
		top := b.Const(ir.U32, g-1)
		for s := g / 2; s >= 1; s /= 2 {
			mine := b.Load(ir.F32, part, tid, 0)
			other := b.Load(ir.F32, part, b.Min(ir.U32, b.Add(ir.U32, tid, b.Const(ir.U32, int64(s))), top), 0)
			sum := b.Add(ir.F32, mine, other)
			b.Barrier()
			b.Store(part, tid, sum, 0)
			b.Barrier()
		}
		ss = b.Load(ir.F32, part, b.Const(ir.U32, 0), 0)
	}
	mean := b.Add(ir.F32, b.Mul(ir.F32, ss, b.ConstF32(1/float32(k))), b.ConstF32(eps))
	scale := b.Div(ir.F32, b.ConstF32(1), b.Sqrt(mean))

	b.Loop(iters)
	i2 := b.Phi(ir.U32, tid)
	idx2 := b.Min(ir.U32, i2, last)
	w := b.Load(ir.F32, pW, idx2, 0)
	if addOne {
		w = b.Add(ir.F32, w, b.ConstF32(1))
	}
	b.Store(pOut, b.Add(ir.U32, rowBase, idx2), b.Mul(ir.F32, b.Mul(ir.F32, value(idx2, false), scale), w), 0)
	b.SetPhi(i2, b.Add(ir.U32, i2, step))
	b.EndLoop()
	if window > 0 {
		quantPhase(b, k, rows, window, g, r, tid, pA, pAX, shMax, shSum, func(at ir.Value) ir.Value {
			w := b.Load(ir.F32, pW, at, 0)
			if addOne {
				w = b.Add(ir.F32, w, b.ConstF32(1))
			}
			return b.Mul(ir.F32, b.Mul(ir.F32, value(at, false), scale), w)
		})
	}
	if f16 != nil {
		// Output word wd of row r is thread tid's in iteration j, wd = tid +
		// j*g, clamped to the row's last: a surplus thread writes the last
		// word again, the same bits to the same place. A word is two elements
		// (f16WordElems), so every thread of the group has one.
		nw := k / 2
		lastW := b.Const(ir.U32, int64(nw-1))
		b.Loop(int64((nw + g - 1) / g))
		i3 := b.Phi(ir.U32, tid)
		wd := b.Min(ir.U32, i3, lastW)
		e0, e1 := f16WordElems(b, *f16, wd)
		x := func(idx ir.Value) ir.Value {
			w := b.Load(ir.F32, pW, idx, 0)
			if addOne {
				w = b.Add(ir.F32, w, b.ConstF32(1))
			}
			return b.Mul(ir.F32, b.Mul(ir.F32, value(idx, false), scale), w)
		}
		b.Store(pB, b.Add(ir.U32, b.Mul(ir.U32, r, b.Const(ir.U32, int64(nw))), wd), b.PackF16(x(e0), x(e1)), 0)
		b.SetPhi(i3, b.Add(ir.U32, i3, step))
		b.EndLoop()
	}
	return b.Done(), nil
}

// quantPhase is RMSNormQuantRows' second half: Quantize's arithmetic over the
// norm's output, recomputed per word. Word w of row r is elements 4w..4w+3; a
// thread owns word tid + j*RMSNormGroup in iteration j, so a window of
// consecutive words is consecutive threads in one iteration, and every index
// comes from the clamped word as in Quantize (RULE 13).
//
// g is the workgroup width, a multiple of the window's span of words, and
// elem(at) is element at's value as quantized: the norm's output, or the
// input itself where no norm leads (matVec's operand prologue).
func quantPhase(b *ir.Builder, k, rows, window, g int, r, tid, pA, pAX, shMax, shSum ir.Value,
	elem func(at ir.Value) ir.Value) {
	const wpb = 8
	nw := k / 4
	nbAll := rows * k / 32
	span := max(window/32, 1) * wpb
	lastW := b.Const(ir.U32, int64(nw-1))
	half := b.Const(ir.U32, 0x3F000000)
	signMask := b.Const(ir.U32, 0x80000000)
	rowW := b.Mul(ir.U32, r, b.Const(ir.U32, int64(nw)))
	// Row mode: one scale for the whole row (a window at least k wide, the
	// host clamping it at the row's end), its maximum met by a strided scan
	// and a tree before any word is quantized.
	var rowAmax ir.Value
	if window >= k {
		acc := b.ConstF32(0)
		for j := 0; j < (nw+g-1)/g; j++ {
			word := b.Min(ir.U32, b.Add(ir.U32, tid, b.Const(ir.U32, int64(j*g))), lastW)
			base := b.Mul(ir.U32, word, b.Const(ir.U32, 4))
			var v [4]ir.Value
			for i := range v {
				v[i] = elem(b.Add(ir.U32, base, b.Const(ir.U32, int64(i))))
			}
			acc = b.Max(ir.F32, acc, amaxOf(b, v[:]))
		}
		rowAmax = treeMax(b, shMax, tid, acc, g)
	}
	for j := 0; j < (nw+g-1)/g; j++ {
		word := b.Min(ir.U32, b.Add(ir.U32, tid, b.Const(ir.U32, int64(j*g))), lastW)
		local := b.Sub(ir.U32, word, b.Const(ir.U32, int64(j*g)))
		base := b.Mul(ir.U32, word, b.Const(ir.U32, 4))
		var v [4]ir.Value
		for i := range v {
			v[i] = elem(b.Add(ir.U32, base, b.Const(ir.U32, int64(i))))
		}
		amax := rowAmax
		if window < k {
			b.Store(shMax, tid, amaxOf(b, v[:]), 0)
			b.Barrier()
			wbase := b.Mul(ir.U32, b.Div(ir.U32, local, b.Const(ir.U32, int64(span))), b.Const(ir.U32, int64(span)))
			amax = b.ConstF32(0)
			for i := 0; i < span; i++ {
				amax = b.Max(ir.F32, amax, b.Load(ir.F32, shMax, wbase, int64(i)))
			}
		}
		d, inv := quantScales(b, amax)
		sum := b.Const(ir.I32, 0)
		var packed ir.Value
		for i := range v {
			f := b.Mul(ir.F32, v[i], inv)
			sgn := b.And(ir.U32, b.Bitcast(ir.U32, f), signMask)
			rnd := b.Bitcast(ir.F32, b.Add(ir.U32, sgn, half))
			q := b.CvtI32(b.Add(ir.F32, f, rnd))
			sum = b.Add(ir.I32, sum, q)
			byteI := b.And(ir.U32, q, b.Const(ir.U32, 0xFF))
			if i == 0 {
				packed = byteI
			} else {
				packed = b.Add(ir.U32, packed, b.Shl(ir.U32, byteI, b.Const(ir.U32, int64(8*i))))
			}
		}
		gword := b.Add(ir.U32, rowW, word)
		b.Store(pA, gword, packed, 0)
		b.Store(shSum, tid, b.CvtF32(sum), 0)
		b.Barrier()
		qbase := b.Mul(ir.U32, b.Div(ir.U32, local, b.Const(ir.U32, 4)), b.Const(ir.U32, 4))
		s16 := b.Load(ir.F32, shSum, qbase, 0)
		for i := 1; i < 4; i++ {
			s16 = b.Add(ir.F32, s16, b.Load(ir.F32, shSum, qbase, int64(i)))
		}
		b.Store(pAX, b.Div(ir.U32, gword, b.Const(ir.U32, wpb)), d, 0)
		b.Store(pAX, b.Div(ir.U32, gword, b.Const(ir.U32, 4)), s16, int64(nbAll))
		// The next iteration's shared writes must not overtake this one's reads.
		b.Barrier()
	}
}

// RMSNormGroup is RMSNormRows' workgroup size: one group per row.
const RMSNormGroup = 1024

// NormApplyRows is NormApply over `rows` independent vectors.
//
// Each row reads its own `parts` partials at row*parts; reading row 0's
// for every row normalises the chunk by one token's magnitude, silently.
func NormApplyRows(k, parts int, eps float32, addOne bool, rows int) (*ir.Kernel, error) {
	if rows < 1 {
		rows = 1
	}
	b := ir.New("normapplyrows", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pW := b.Param("pW", ir.F32)
	pPart := b.Param("pPart", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(k*rows-1)))
	zeroU := b.Const(ir.U32, 0)
	if rows > 1 {
		// This thread's row, and the base of its partials.
		zeroU = b.Mul(ir.U32, b.Div(ir.U32, i, b.Const(ir.U32, int64(k))),
			b.Const(ir.U32, int64(parts)))
	}
	sum := b.ConstF32(0)
	for p := 0; p < parts; p++ {
		sum = b.Add(ir.F32, sum, b.Load(ir.F32, pPart, zeroU, int64(p)))
	}
	mean := b.Add(ir.F32, b.Mul(ir.F32, sum, b.ConstF32(1/float32(k))), b.ConstF32(eps))
	scale := b.Div(ir.F32, b.ConstF32(1), b.Sqrt(mean))
	wi := i
	if rows > 1 {
		wi = b.Rem(ir.U32, i, b.Const(ir.U32, int64(k)))
	}
	w := b.Load(ir.F32, pW, wi, 0)
	if addOne {
		// gemma stores its norm weight as (w - 1).
		w = b.Add(ir.F32, w, b.ConstF32(1))
	}
	b.Store(pOut, i, b.Mul(ir.F32, b.Mul(ir.F32, b.Load(ir.F32, pX, i, 0), scale), w), 0)
	return b.Done(), nil
}

// ActMul is the FFN's gate: out[i] = act(gate[i]) * up[i], with the
// activation baked from its kind -- SiLU (llama), the tanh GELU (gemma), or
// gpt-oss's swiglu-oai, which clamps both operands and multiplies by up+1:
//
//	x = min(gate, 7)   y = clamp(up, -7, 7)   out = x*sigma(1.702x) * (y+1)
//
// tanh(z) is built as 1 - 2/(exp(2z)+1) because the IR has no tanh on all
// three backends.
func ActMul(n int, k ActKind) (*ir.Kernel, error) {
	if err := actMulKind(k); err != nil {
		return nil, err
	}
	b := ir.New("actmul", [3]int{128, 1, 1})
	pG := b.Param("pG", ir.F32)
	pU := b.Param("pU", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(n-1)))
	b.Store(pOut, i, actMulValue(b, k, b.Load(ir.F32, pG, i, 0), func() ir.Value { return b.Load(ir.F32, pU, i, 0) }), 0)
	return b.Done(), nil
}

// actMulKind refuses an activation ActMul has no gated form of.
func actMulKind(k ActKind) error {
	switch k {
	case ActSiLU, ActGELU, ActSwiGLUOAI, ActIdentity, ActSwiGLUClamp, ActSitu, ActGELUErf, ActReLU2:
		return nil
	}
	return fmt.Errorf("kernels: ActMul: %v is not a gated activation", k)
}

// actMulValue is ActMul's element: act(g) times u, u loaded by up when it is
// needed. ActMul, QuantizeAct and ActMulF16T share it, so their elements are
// the same bits.
func actMulValue(b *ir.Builder, k ActKind, g ir.Value, up func() ir.Value) ir.Value {
	switch k {
	case ActSwiGLUOAI:
		lim := b.ConstF32(7)
		a := act(b, b.Min(ir.F32, g, lim), ActQuickGELU)
		u := up()
		u = b.Add(ir.F32, b.Max(ir.F32, b.Min(ir.F32, u, lim), b.ConstF32(-7)), b.ConstF32(1))
		return b.Mul(ir.F32, a, u)
	case ActSwiGLUClamp:
		// DeepSeek V4: x = min(gate, 10), y = clamp(up, -10, 10), silu(x)*y.
		lim := b.ConstF32(10)
		a := act(b, b.Min(ir.F32, g, lim), ActSiLU)
		u := b.Max(ir.F32, b.Min(ir.F32, up(), lim), b.ConstF32(-10))
		return b.Mul(ir.F32, a, u)
	case ActSitu:
		// Kimi-K3: c*tanh(x/c) at both bounds, the gate's times sigma(gate),
		// the tanh the host tiers' rational (TanhClamp).
		one := b.ConstF32(1)
		bound := func(x ir.Value, c float32) ir.Value {
			return b.Mul(ir.F32, b.ConstF32(c), tanhRational(b, b.Mul(ir.F32, b.ConstF32(1/c), x)))
		}
		sig := b.Div(ir.F32, one, b.Add(ir.F32, one, b.Exp(b.Sub(ir.F32, b.ConstF32(0), g))))
		a := b.Mul(ir.F32, bound(g, SituBeta), sig)
		return b.Mul(ir.F32, a, bound(up(), SituLinearBeta))
	}
	a := act(b, g, k)
	return b.Mul(ir.F32, a, up())
}

// ActMulWeighted is ActMul for a mixture that weights each expert's input
// (Llama 4): with w the routed weight of the expert element i belongs to,
// out[i] = act(w*g[i]) * (w*u[i]).
//
// Scaling the projections equals scaling the input (gate and up are linear)
// and avoids a per-expert quantization of the shared activation. It is not
// w*act(g)*u, which weights the output as other mixtures do.
//
// per is how many consecutive elements share one weight (the expert width).
// With indexed, element i reads its weight through pIdx[i/per] (the grouped
// prefill's sorted columns, each naming its (token, slot) pair); without it,
// straight from pW[i/per].
func ActMulWeighted(n, per int, k ActKind, indexed bool) (*ir.Kernel, error) {
	switch k {
	case ActSiLU, ActGELU:
	default:
		return nil, fmt.Errorf("kernels: ActMulWeighted: %v is not implemented", k)
	}
	if per <= 0 || n%per != 0 {
		return nil, fmt.Errorf("kernels: ActMulWeighted: per=%d does not tile n=%d", per, n)
	}
	b := ir.New("actmulweighted", [3]int{128, 1, 1})
	pG := b.Param("pG", ir.F32)
	pU := b.Param("pU", ir.F32)
	pW := b.Param("pW", ir.F32)
	var pIdx ir.Value
	if indexed {
		pIdx = b.Param("pIdx", ir.U32)
	}
	pOut := b.Param("pOut", ir.F32)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(n-1)))
	slot := b.Div(ir.U32, i, b.Const(ir.U32, int64(per)))
	if indexed {
		slot = b.Load(ir.U32, pIdx, slot, 0)
	}
	w := b.Load(ir.F32, pW, slot, 0)
	g := b.Mul(ir.F32, w, b.Load(ir.F32, pG, i, 0))
	u := b.Mul(ir.F32, w, b.Load(ir.F32, pU, i, 0))
	b.Store(pOut, i, b.Mul(ir.F32, act(b, g, k), u), 0)
	return b.Done(), nil
}

// ScaleRowsByCount multiplies each row of a block of rows x width floats by a
// factor looked up from its causal count: row r takes
// pTab[min((cnt(r)-1+off)/div, tabLen-1)], where cnt(r) is pN[0] for a
// one-row block and pN[1+r] otherwise -- the key-count layout every attention
// kernel reads.
//
// It is Llama 4's attention temperature: 1 + s*ln(1 + floor((pos+off)/div)).
// The IR has no logarithm, but the argument is a small integer, so the host
// fills the whole curve into a table once and the kernel only divides.
func ScaleRowsByCount(rows, width, div, off, tabLen int) (*ir.Kernel, error) {
	if rows < 1 || width < 1 || div < 1 || off < 0 || tabLen < 1 {
		return nil, fmt.Errorf("kernels: ScaleRowsByCount rows=%d width=%d div=%d off=%d tab=%d",
			rows, width, div, off, tabLen)
	}
	n := rows * width
	b := ir.New("scalerowsbycount", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pN := b.Param("pN", ir.U32)
	pTab := b.Param("pTab", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	u := func(v int) ir.Value { return b.Const(ir.U32, int64(v)) }
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()), u(n-1))
	ni := u(0)
	if rows > 1 {
		ni = b.Add(ir.U32, b.Div(ir.U32, i, u(width)), u(1))
	}
	cnt := b.Load(ir.U32, pN, ni, 0)
	pos := b.Sub(ir.U32, b.Max(ir.U32, cnt, u(1)), u(1))
	k := b.Min(ir.U32, b.Div(ir.U32, b.Add(ir.U32, pos, u(off)), u(div)), u(tabLen-1))
	b.Store(pOut, i, b.Mul(ir.F32, b.Load(ir.F32, pX, i, 0), b.Load(ir.F32, pTab, k, 0)), 0)
	return b.Done(), nil
}

// SigmoidMul gates one vector by another: out[i] = sigma(g[i]) * v[i].
//
// It is not ActMul with another activation: SiLU is g*sigma(g) of one vector,
// this is sigma(g)*v of two, and routing a gate through ActMul is wrong by a
// factor of g. qwen3next gates its attention output (the second half of a
// double-width q) and its recurrent output this way.
func SigmoidMul(n int) (*ir.Kernel, error) {
	b := ir.New("sigmoidmul", [3]int{128, 1, 1})
	pG := b.Param("pG", ir.F32)
	pV := b.Param("pV", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(n-1)))
	g := b.Load(ir.F32, pG, i, 0)
	one := b.ConstF32(1)
	sig := b.Div(ir.F32, one, b.Add(ir.F32, one, b.Exp(b.Sub(ir.F32, b.ConstF32(0), g))))
	b.Store(pOut, i, b.Mul(ir.F32, sig, b.Load(ir.F32, pV, i, 0)), 0)
	return b.Done(), nil
}

// Softcap is out[i] = c*tanh(x[i]/c), gemma2's attention logit cap, with tanh
// from exp as in act: c*(1 - 2/(exp(2x/c)+1)). exp's overflow gives the
// saturations for free -- a huge x makes the quotient 0 (c), a huge negative
// one makes it 2 (-c).
//
// A masked key stays -inf: c*tanh(-inf/c) is -c, a finite logit softmax
// would weight. Anything below -1e30 passes through.
func Softcap(n int, c float32) (*ir.Kernel, error) {
	if c <= 0 {
		return nil, fmt.Errorf("kernels: Softcap: cap %v", c)
	}
	b := ir.New("softcap", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(n-1)))
	x := b.Load(ir.F32, pX, i, 0)
	one := b.ConstF32(1)
	e := b.Exp(b.Mul(ir.F32, b.ConstF32(2/c), x))
	t := b.Sub(ir.F32, one, b.Div(ir.F32, b.ConstF32(2), b.Add(ir.F32, e, one)))
	y := b.Mul(ir.F32, b.ConstF32(c), t)
	masked := b.Lt(ir.F32, x, b.ConstF32(-1e30))
	b.Store(pOut, i, b.Select(ir.F32, masked, x, y), 0)
	return b.Done(), nil
}

// Clamp is out[i] = min(max(x[i], -c), c), DBRX's clip on q, k and v, the
// device twin of cpu.EmitClamp. Out of place, since no kernel may write what
// it reads.
func Clamp(n int, c float32) (*ir.Kernel, error) {
	if !(c > 0) {
		return nil, fmt.Errorf("kernels: Clamp: bound %v", c)
	}
	b := ir.New("clamp", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(n-1)))
	x := b.Load(ir.F32, pX, i, 0)
	b.Store(pOut, i, b.Min(ir.F32, b.Max(ir.F32, x, b.ConstF32(-c)), b.ConstF32(c)), 0)
	return b.Done(), nil
}

// ClampAt is out[i] = min(max(x[i], b[off]), b[off+1]): Gemma 4's clipped
// linears, whose bounds are a block's own and need not be symmetric, read
// from the block's bounds buffer rather than baked, so one kernel per place
// in the buffer serves every block. Out of place, as Clamp.
func ClampAt(n, off int) (*ir.Kernel, error) {
	if n <= 0 || off < 0 {
		return nil, fmt.Errorf("kernels: ClampAt: %d elements at %d", n, off)
	}
	b := ir.New("clampat", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pB := b.Param("pB", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(n-1)))
	x := b.Load(ir.F32, pX, i, 0)
	lo := b.Load(ir.F32, pB, b.Const(ir.U32, int64(off)), 0)
	hi := b.Load(ir.F32, pB, b.Const(ir.U32, int64(off+1)), 0)
	b.Store(pOut, i, b.Min(ir.F32, b.Max(ir.F32, x, lo), hi), 0)
	return b.Done(), nil
}

// act is x*sigma(x) (SiLU), x*sigma(1.702x) (quick-GELU), max(x, 0)^2
// (squared ReLU) or the tanh approximation of GELU.
func act(b *ir.Builder, g ir.Value, k ActKind) ir.Value {
	one := b.ConstF32(1)
	switch k {
	case ActIdentity:
		return g
	case ActReLU:
		return b.Max(ir.F32, g, b.ConstF32(0))
	case ActReLU2:
		r := b.Max(ir.F32, g, b.ConstF32(0))
		return b.Mul(ir.F32, r, r)
	case ActSiLU, ActQuickGELU:
		z := g
		if k == ActQuickGELU {
			z = b.Mul(ir.F32, b.ConstF32(1.702), g)
		}
		return b.Div(ir.F32, g, b.Add(ir.F32, one, b.Exp(b.Sub(ir.F32, b.ConstF32(0), z))))
	case ActSqrtSoftplus:
		return b.Sqrt(softplus(b, g))
	case ActSqrtSigmoid:
		// sigmoid(sign(x)*sqrt(max(|x|, floor))); see ActSqrtSigmoid.
		ax := b.Max(ir.F32, g, b.Sub(ir.F32, b.ConstF32(0), g))
		m := b.Sqrt(b.Max(ir.F32, ax, b.ConstF32(SqrtSigmoidFloor)))
		sm := b.Select(ir.F32, b.Lt(ir.F32, g, b.ConstF32(0)), b.Sub(ir.F32, b.ConstF32(0), m), m)
		return b.Div(ir.F32, one, b.Add(ir.F32, one, b.Exp(b.Sub(ir.F32, b.ConstF32(0), sm))))
	case ActGELUErf:
		// 0.5*((x + |x|) - |x|*q), q = 1 - erf(|x|/sqrt 2); see ActGELUErf.
		ax := b.Max(ir.F32, g, b.Sub(ir.F32, b.ConstF32(0), g))
		z := b.Mul(ir.F32, b.ConstF32(0.70710678118654752), ax)
		t := b.Div(ir.F32, one, b.Fma(b.ConstF32(GELUErfP), z, one))
		p := b.ConstF32(GELUErfA[4])
		for i := 3; i >= 0; i-- {
			p = b.Fma(p, t, b.ConstF32(GELUErfA[i]))
		}
		q := b.Mul(ir.F32, b.Mul(ir.F32, p, t), b.Exp(b.Sub(ir.F32, b.ConstF32(0), b.Mul(ir.F32, z, z))))
		return b.Mul(ir.F32, b.ConstF32(0.5), b.Sub(ir.F32, b.Add(ir.F32, g, ax), b.Mul(ir.F32, ax, q)))
	}
	const c = 0.7978845608028654 // sqrt(2/pi)
	inner := b.Mul(ir.F32, b.ConstF32(c),
		b.Add(ir.F32, g, b.Mul(ir.F32, b.ConstF32(0.044715), b.Mul(ir.F32, g, b.Mul(ir.F32, g, g)))))
	e := b.Exp(b.Mul(ir.F32, b.ConstF32(2), inner))
	tanh := b.Sub(ir.F32, one, b.Div(ir.F32, b.ConstF32(2), b.Add(ir.F32, e, one)))
	return b.Mul(ir.F32, b.Mul(ir.F32, b.ConstF32(0.5), g), b.Add(ir.F32, one, tanh))
}

// tanhRational is tanh(x) as x*P(x^2)/Q(x^2) on x clamped to +-TanhClamp,
// ActSitu's tanh on every tier (see TanhClamp for why not the exp form).
func tanhRational(b *ir.Builder, x ir.Value) ir.Value {
	x = b.Max(ir.F32, b.Min(ir.F32, x, b.ConstF32(TanhClamp)), b.ConstF32(-TanhClamp))
	x2 := b.Mul(ir.F32, x, x)
	p := b.ConstF32(TanhP[0])
	for _, c := range TanhP[1:] {
		p = b.Fma(p, x2, b.ConstF32(c))
	}
	q := b.ConstF32(TanhQ[0])
	for _, c := range TanhQ[1:] {
		q = b.Fma(q, x2, b.ConstF32(c))
	}
	return b.Div(ir.F32, b.Mul(ir.F32, p, x), q)
}

// softplus is log(1 + exp(z)) without a logarithm (the IR has none), the
// host's |z| form and DeltaGateRows' series:
//
//	softplus(z) = max(z,0) + log(1 + exp(-|z|)),  log(1+u) = 2*atanh(u/(2+u))
//
// with atanh's five terms in Horner order, the coefficients written out as the
// CPU's are.
func softplus(b *ir.Builder, z ir.Value) ir.Value {
	zero, two := b.ConstF32(0), b.ConstF32(2)
	absz := b.Max(ir.F32, z, b.Sub(ir.F32, zero, z))
	e := b.Exp(b.Sub(ir.F32, zero, absz)) // exp(-|z|), in (0,1]
	x := b.Div(ir.F32, e, b.Add(ir.F32, two, e))
	x2 := b.Mul(ir.F32, x, x)
	ser := b.ConstF32(1.0 / 9)
	ser = b.Fma(ser, x2, b.ConstF32(1.0/7))
	ser = b.Fma(ser, x2, b.ConstF32(1.0/5))
	ser = b.Fma(ser, x2, b.ConstF32(1.0/3))
	ser = b.Fma(ser, x2, b.ConstF32(1))
	return b.Add(ir.F32, b.Max(ir.F32, z, zero), b.Mul(ir.F32, two, b.Mul(ir.F32, x, ser)))
}

// Act is the ungated activation: out[i] = act(x[i]), for FFNs with no gate
// (a ViT's), and DeepSeek V4's router gate (ActSqrtSoftplus).
func Act(n int, k ActKind) (*ir.Kernel, error) {
	switch k {
	case ActSiLU, ActGELU, ActQuickGELU, ActReLU2, ActReLU, ActSqrtSoftplus, ActGELUErf, ActSqrtSigmoid:
	default:
		return nil, fmt.Errorf("kernels: Act: %v is not an ungated activation", k)
	}
	b := ir.New("act", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(n-1)))
	b.Store(pOut, i, act(b, b.Load(ir.F32, pX, i, 0), k), 0)
	return b.Done(), nil
}

// LayerNormPartRows is the first of three kernels: partial sums of x.
//
// Three, not two, for the numerics: a one-pass E[x^2]-E[x]^2 cancels in f32
// when the mean is much larger than the variance (a ViT residual), so the mean
// comes before the variance. cpu.EmitLayerNorm reads the vector three times
// for the same reason, and the tiers must agree.
//
// It runs over `rows` independent vectors. The parts of a row are contiguous, so a flat thread index over parts*rows
// already addresses the right slice; only the clamp grows.
func LayerNormPartRows(k, parts, rows int) (*ir.Kernel, error) {
	if k%parts != 0 {
		return nil, fmt.Errorf("kernels: LayerNormPartRows: %d parts do not divide %d", parts, k)
	}
	if rows < 1 {
		rows = 1
	}
	per := k / parts
	b := ir.New("layernormpartrows", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	p := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(parts*rows-1)))
	start := b.Mul(ir.U32, p, b.Const(ir.U32, int64(per)))
	zero := b.ConstF32(0)
	b.Loop(int64(per))
	acc := b.Phi(ir.F32, zero)
	idx := b.Phi(ir.U32, start)
	b.SetPhi(acc, b.Add(ir.F32, acc, b.Load(ir.F32, pX, idx, 0)))
	b.SetPhi(idx, b.Add(ir.U32, idx, b.Const(ir.U32, 1)))
	b.EndLoop()
	b.Store(pOut, p, acc, 0)
	return b.Done(), nil
}

// LayerNormVarRows is the second: partial sums of (x - mean)^2, with the mean
// summed from LayerNormPartRows' output by every thread.
//
// Re-summing `parts` values per thread is cheaper than another kernel, as in
// NormApply.
//
// Each of the `rows` vectors has its own mean; see layerNormMeanAt.
func LayerNormVarRows(k, parts, rows int) (*ir.Kernel, error) {
	if k%parts != 0 {
		return nil, fmt.Errorf("kernels: LayerNormVarRows: %d parts do not divide %d", parts, k)
	}
	if rows < 1 {
		rows = 1
	}
	per := k / parts
	b := ir.New("layernormvarrows", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pSum := b.Param("pSum", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	p := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(parts*rows-1)))
	base := b.Const(ir.U32, 0)
	if rows > 1 {
		base = b.Mul(ir.U32, b.Div(ir.U32, p, b.Const(ir.U32, int64(parts))),
			b.Const(ir.U32, int64(parts)))
	}
	mean := layerNormMeanAt(b, pSum, base, parts, k)
	start := b.Mul(ir.U32, p, b.Const(ir.U32, int64(per)))
	zero := b.ConstF32(0)
	b.Loop(int64(per))
	acc := b.Phi(ir.F32, zero)
	idx := b.Phi(ir.U32, start)
	d := b.Sub(ir.F32, b.Load(ir.F32, pX, idx, 0), mean)
	b.SetPhi(acc, b.Fma(d, d, acc))
	b.SetPhi(idx, b.Add(ir.U32, idx, b.Const(ir.U32, 1)))
	b.EndLoop()
	b.Store(pOut, p, acc, 0)
	return b.Done(), nil
}

// LayerNormApplyRows is the third: out = (x - mean) * rsqrt(var + eps) * w + b.
//
// bias is baked, not branched: whether the file carries a LayerNorm bias is a
// property of the model, and without it the kernel takes one parameter fewer.
//
// Over `rows` independent vectors the weight and bias are shared across rows; the mean and variance are not.
func LayerNormApplyRows(k, parts int, eps float32, bias bool, rows int) (*ir.Kernel, error) {
	if rows < 1 {
		rows = 1
	}
	b := ir.New("layernormapplyrows", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pW := b.Param("pW", ir.F32)
	pSum := b.Param("pSum", ir.F32)
	pVar := b.Param("pVar", ir.F32)
	var pB ir.Value
	if bias {
		pB = b.Param("pB", ir.F32)
	}
	pOut := b.Param("pOut", ir.F32)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(k*rows-1)))
	base := b.Const(ir.U32, 0)
	wi := i
	if rows > 1 {
		base = b.Mul(ir.U32, b.Div(ir.U32, i, b.Const(ir.U32, int64(k))),
			b.Const(ir.U32, int64(parts)))
		wi = b.Rem(ir.U32, i, b.Const(ir.U32, int64(k)))
	}
	mean := layerNormMeanAt(b, pSum, base, parts, k)
	vsum := b.ConstF32(0)
	for p := 0; p < parts; p++ {
		vsum = b.Add(ir.F32, vsum, b.Load(ir.F32, pVar, base, int64(p)))
	}
	varr := b.Add(ir.F32, b.Mul(ir.F32, vsum, b.ConstF32(1/float32(k))), b.ConstF32(eps))
	scale := b.Div(ir.F32, b.ConstF32(1), b.Sqrt(varr))
	v := b.Mul(ir.F32, b.Mul(ir.F32, b.Sub(ir.F32, b.Load(ir.F32, pX, i, 0), mean), scale),
		b.Load(ir.F32, pW, wi, 0))
	if bias {
		v = b.Add(ir.F32, v, b.Load(ir.F32, pB, wi, 0))
	}
	b.Store(pOut, i, v, 0)
	return b.Done(), nil
}

// layerNormMean re-sums the partials and divides by k.
func layerNormMean(b *ir.Builder, pSum ir.Value, parts, k int) ir.Value {
	return layerNormMeanAt(b, pSum, b.Const(ir.U32, 0), parts, k)
}

// layerNormMeanAt is layerNormMean for one row of a batch: base is that row's
// first partial. Reading row 0's for every row normalises every patch by the
// first patch's statistics, which is fluent and wrong.
func layerNormMeanAt(b *ir.Builder, pSum, base ir.Value, parts, k int) ir.Value {
	sum := b.ConstF32(0)
	zero := base
	for p := 0; p < parts; p++ {
		sum = b.Add(ir.F32, sum, b.Load(ir.F32, pSum, zero, int64(p)))
	}
	return b.Mul(ir.F32, sum, b.ConstF32(1/float32(k)))
}

// Add is the residual, three-operand: out[i] = a[i] + b[i].
//
// Not dst += src: surplus threads clamp onto the last element and would
// accumulate twice, and ir.Validate refuses the aliased form. The forward
// pass ping-pongs two residual buffers instead.
func Add(n int) (*ir.Kernel, error) {
	b := ir.New("add", [3]int{128, 1, 1})
	pA := b.Param("pA", ir.F32)
	pB := b.Param("pB", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(n-1)))
	b.Store(pOut, i, b.Add(ir.F32, b.Load(ir.F32, pA, i, 0), b.Load(ir.F32, pB, i, 0)), 0)
	return b.Done(), nil
}

// AddScaled is the residual add of a block output that is scaled first
// (Granite's residual_scale): out[i] = a[i] + alpha*b[i], with alpha read from
// a one-element buffer for Scale's reason. Three-operand for Add's.
//
// One rounding, as the host's axpy rounds it (VFMADD231PS on amd64, FMLA on
// arm64): ir.Fma. A multiply and an add written apart are fused or not at the
// compiler's whim -- ptxas and both Vulkan drivers here fused them, Metal did
// not, and its scaled residual then parted from the host's in the last bit of
// 2.8% of every block's elements (TestAddScaledRoundsOnce).
func AddScaled(n int) (*ir.Kernel, error) {
	b := ir.New("add_scaled", [3]int{128, 1, 1})
	pA := b.Param("pA", ir.F32)
	pB := b.Param("pB", ir.F32)
	pAlpha := b.Param("pAlpha", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	zero := b.Const(ir.U32, 0)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(n-1)))
	b.Store(pOut, i, b.Fma(b.Load(ir.F32, pAlpha, zero, 0), b.Load(ir.F32, pB, i, 0), b.Load(ir.F32, pA, i, 0)), 0)
	return b.Done(), nil
}

// Scale is out[i] = a[i] * alpha, with alpha read from a one-element buffer.
//
// Three-operand for Add's reason: an in-place `x[i] *= alpha` would scale the
// last element once per surplus thread, and ir.Validate refuses it.
//
// alpha arrives as a buffer rather than a baked Const because it is per-model
// (gemma's embedding scale) or per-head (1/sqrt(hd)); baking it would key the
// kernel cache on a float.
func Scale(n int) (*ir.Kernel, error) {
	b := ir.New("scale", [3]int{128, 1, 1})
	pA := b.Param("pA", ir.F32)
	pAlpha := b.Param("pAlpha", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	zero := b.Const(ir.U32, 0)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(n-1)))
	b.Store(pOut, i, b.Mul(ir.F32, b.Load(ir.F32, pA, i, 0), b.Load(ir.F32, pAlpha, zero, 0)), 0)
	return b.Done(), nil
}

// SplitHeadGate deinterleaves a double-width query projection into the query
// and the attention output gate.
//
// The source row is [head][2*headDim]: the first headDim of each head is q
// and the second that head's gate. Reading it as two halves also runs and
// pairs every head with the wrong gate.
//
// Out of place by construction (RULE 13): every output element is a pure
// function of one read-only input element.
func SplitHeadGate(nHead, headDim, rows int) (*ir.Kernel, error) {
	if nHead <= 0 || headDim <= 0 || rows <= 0 {
		return nil, fmt.Errorf("kernels: SplitHeadGate nHead=%d headDim=%d rows=%d",
			nHead, headDim, rows)
	}
	qdim := nHead * headDim
	n := int64(rows * qdim)
	b := ir.New("splitheadgate", [3]int{128, 1, 1})
	pSrc := b.Param("pSrc", ir.F32)   // [rows][nHead][2*headDim]
	pQ := b.Param("pQ", ir.F32)       // [rows][nHead][headDim]
	pGate := b.Param("pGate", ir.F32) // [rows][nHead][headDim]

	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, n-1))
	qw := b.Const(ir.U32, int64(qdim))
	hw := b.Const(ir.U32, int64(headDim))
	r := b.Div(ir.U32, i, qw)
	rem := b.Sub(ir.U32, i, b.Mul(ir.U32, r, qw))
	hh := b.Div(ir.U32, rem, hw)
	j := b.Sub(ir.U32, rem, b.Mul(ir.U32, hh, hw))
	// src = r*2*qdim + hh*2*headDim + j, and the gate is headDim further on.
	src := b.Add(ir.U32,
		b.Add(ir.U32, b.Mul(ir.U32, r, b.Const(ir.U32, int64(2*qdim))),
			b.Mul(ir.U32, hh, b.Const(ir.U32, int64(2*headDim)))), j)
	b.Store(pQ, i, b.Load(ir.F32, pSrc, src, 0), 0)
	b.Store(pGate, i, b.Load(ir.F32, pSrc, src, int64(headDim)), 0)
	return b.Done(), nil
}

// amaxOf is the largest magnitude among v: the sign bit masked off, then Max.
// quantScales is an activation block's scale and its reciprocal from its amax,
// as the host derives them: d = amax/127, then inv = 1/d, and inv 0 where d
// is 0 -- two divisions, not one multiply by 127/amax, which differs in the
// last bit (cpu.QuantizeQ8Window, internal/oracle.QuantizeQ8Window). Both are
// correctly rounded here on every backend (div127, divRN): an inv one ulp off
// moves an element that sits on a rounding boundary to the next int8, and that
// boundary is where a deep model parts from the host.
//
// Every thread of a norm-and-quantize group computes these, so they are kept
// short: 1/d is not a second division after the first. 127/amax, taken beside
// div127, is within a few floats of it, and one Newton step against d,
// y + y*(1 - d*y), brings it within two (the step's own rounding, 2^-23
// relative), which is the reach roundQuotient then rounds over.
func quantScales(b *ir.Builder, amax ir.Value) (d, inv ir.Value) {
	d = div127(b, amax)
	y := b.Div(ir.F32, b.ConstF32(127), amax)
	y = b.Add(ir.F32, y, b.Mul(ir.F32, y, b.Sub(ir.F32, b.ConstF32(1), b.Mul(ir.F32, d, y))))
	inv = b.Select(ir.F32, b.Lt(ir.F32, b.ConstF32(0), d), roundQuotient(b, b.ConstF32(1), d, y, 2),
		b.ConstF32(0))
	return d, inv
}

func amaxOf(b *ir.Builder, v []ir.Value) ir.Value {
	mask := b.Const(ir.U32, 0x7FFFFFFF)
	amax := b.ConstF32(0)
	for _, x := range v {
		a := b.Bitcast(ir.F32, b.And(ir.U32, b.Bitcast(ir.U32, x), mask))
		amax = b.Max(ir.F32, amax, a)
	}
	return amax
}

// WindowMaskRows confines each query row's attention scores to its window:
// rows query rows of heads heads at a score stride of sstride, the score of a
// key outside its row's [lo, hi) replaced by -inf, which the softmax turns
// into a weight of zero. pWin holds lo and hi per row, u32 pairs. It is how a
// vision block whose rows attend inside windows (Qwen2.5-VL) runs on the
// bidirectional attention every other vision block uses: the scores of the
// whole image, then the mask, as llama.cpp's window_mask is added to them.
// Out of place, since no kernel may write what it reads (RULE 13).
func WindowMaskRows(rows, heads, sstride int) (*ir.Kernel, error) {
	if rows < 1 || heads < 1 || sstride < 1 {
		return nil, fmt.Errorf("kernels: WindowMaskRows: %d rows, %d heads, stride %d", rows, heads, sstride)
	}
	n := rows * heads * sstride
	b := ir.New("windowmask", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pWin := b.Param("pWin", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(n-1)))
	j := b.Rem(ir.U32, i, b.Const(ir.U32, int64(sstride)))
	r := b.Div(ir.U32, i, b.Const(ir.U32, int64(heads*sstride)))
	two := b.Mul(ir.U32, r, b.Const(ir.U32, 2))
	lo := b.Load(ir.U32, pWin, two, 0)
	hi := b.Load(ir.U32, pWin, b.Add(ir.U32, two, b.Const(ir.U32, 1)), 0)
	negInf := b.Bitcast(ir.F32, b.Const(ir.U32, 0xFF800000))
	x := b.Load(ir.F32, pX, i, 0)
	// j < lo, or not j < hi: both through Lt, the comparison windowed() masks
	// a sliding window with.
	v := b.Select(ir.F32, b.Lt(ir.U32, j, lo), negInf, x)
	v = b.Select(ir.F32, b.Lt(ir.U32, j, hi), v, negInf)
	b.Store(pOut, i, v, 0)
	return b.Done(), nil
}

// BiasAct is act(x + bias) over rows rows of width: bias is one value a
// column, shared by every row. Mamba-1's convolution output takes its bias
// and SiLU here, as one launch, before x_proj reads it.
func BiasAct(width, rows int, k ActKind) (*ir.Kernel, error) {
	switch k {
	case ActSiLU, ActGELU, ActQuickGELU, ActReLU2, ActReLU, ActIdentity:
	default:
		return nil, fmt.Errorf("kernels: BiasAct: %v is not an ungated activation", k)
	}
	if width <= 0 || rows <= 0 {
		return nil, fmt.Errorf("kernels: BiasAct width=%d rows=%d", width, rows)
	}
	b := ir.New("biasact", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pBias := b.Param("pBias", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(width*rows-1)))
	c := b.Rem(ir.U32, i, b.Const(ir.U32, int64(width)))
	b.Store(pOut, i, act(b, b.Add(ir.F32, b.Load(ir.F32, pX, i, 0), b.Load(ir.F32, pBias, c, 0)), k), 0)
	return b.Done(), nil
}
