package mainkernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit_old/gpu/ir"
)

// ActIn is a decode matvec's operand prologue: each workgroup builds the int8
// activation, its scales and its per-16 sums in threadgroup memory from the
// float row, and the matvec reads them from there. The launch that wrote them
// to global memory before it -- Quantize, QuantizeAct, or RMSNormQuantRows
// with Norm -- goes, and with it a write and a read of the operand and, with
// Norm, of the normed float row.
//
// The bits are the same as the launch it replaces: the norm's sum of squares
// is RMSNormGroup's reduction (the shared-memory tree, or RMSNormRowsWarp's
// butterflies with Warp) run by RMSNormGroup/group virtual threads a thread,
// in the same order; every element is the same product; the amax and the
// per-16 sums are a max and integer sums, which no order changes. What it
// costs is the prologue once per workgroup instead of once per row: every
// group reads the K floats (and the norm weight) from L2 and quantizes them
// itself, so it pays only where few groups share a row, and only with the
// weight stream started under it (Prefetch).
//
// Each thread quantizes whole words, reading their elements from L2, the
// window's maximum and the per-16 sums meeting by subgroup shuffles (a
// barrier only for a window wider than a subgroup, and the norm's reduction),
// so a kernel takes it only where a subgroup is 32 lanes.
type ActIn struct {
	On bool
	// Norm leads the quantization with an RMSNorm by the weight pNW, as
	// RMSNormQuantRows computes it with Eps and AddOne; Warp is
	// RMSNormQuantRowsWarp's reduction.
	Norm   bool
	Eps    float32
	AddOne bool
	Warp   bool
	// Window is the amax window, Quantize's.
	Window int
	// Gated quantizes act(pIn)*pU, QuantizeAct's row, from the gate and up
	// projections' outputs: parameters pIn then pU. Not with Norm.
	Gated bool
	Act   ActKind
	// Prefetch is how many of each thread's weight words the matvec asks the
	// cache for before the prologue (ir.OpPrefetch), so the weight stream
	// runs under the prologue's loads and barriers; -1 is all of them. A
	// segment kernel (MatVecSegmentsIn) takes none.
	Prefetch int
}

func (in ActIn) check(k, group int) error {
	span := actSpan(in)
	switch {
	case in.Gated && in.Norm:
		return fmt.Errorf("kernels: ActIn: a gated operand takes no norm")
	case in.Gated && actMulKind(in.Act) != nil:
		return actMulKind(in.Act)
	case k%32 != 0 || k%(span*4) != 0:
		return fmt.Errorf("kernels: ActIn: k=%d is not a multiple of 32 and of the window %d", k, in.Window)
	case group%ir.SubgroupLanes != 0 || group%span != 0:
		return fmt.Errorf("kernels: ActIn: window %d does not tile a %d-thread group", in.Window, group)
	case (k/4+k/32+k/16+2+group/ir.SubgroupLanes+normSlots(group)+group)*4 > 48<<10:
		// The operand, its scales and sums, the window and norm partials and
		// the matvec's own in-group partials, against the 48 KiB every
		// backend gives a workgroup's static threadgroup memory.
		return fmt.Errorf("kernels: ActIn: k=%d does not fit threadgroup memory", k)
	}
	return nil
}

// normSlots is the norm partials' threadgroup slots: RMSNormGroup's, rounded
// up to whole groups, since a group that does not divide the norm's runs
// ceil(RMSNormGroup/group) virtual threads a thread and the ones past the
// norm's group store into slots the reduction never reads.
func normSlots(group int) int { return (RMSNormGroup + group - 1) / group * group }

// actSpan is the amax window in packed words.
func actSpan(in ActIn) int { return max(in.Window/32, 1) * 8 }

// actParamSet is the prologue's parameters, declared where the kernel's
// parameter order puts them (actParams) and read by actPrologue, which a
// matvec emits later, behind its weight prefetch.
type actParamSet struct{ in, nw, u ir.Value }

func actParams(b *ir.Builder, in ActIn) actParamSet {
	ps := actParamSet{in: b.Param("pIn", ir.F32)}
	if in.Norm {
		ps.nw = b.Param("pNW", ir.F32)
	}
	if in.Gated {
		ps.u = b.Param("pU", ir.F32)
	}
	return ps
}

func actInName(in ActIn) string {
	switch {
	case in.Norm:
		return "_nin"
	case in.Gated:
		return "_ain"
	}
	return "_qin"
}

// actPrologue emits the prologue in a kernel of group threads and returns the
// threadgroup arrays the matvec reads as pA and pAX. It ends in a barrier, so
// every word is written before the matvec reads one.
func actPrologue(b *ir.Builder, k, group int, in ActIn, ps actParamSet) (pA, pAX ir.Value) {
	const lanes = ir.SubgroupLanes
	nw, nb := k/4, k/32
	// One slot past each array takes the stores of the threads past the last
	// word, which compute a value that is not that word's.
	pA = b.Shared("actA", ir.U32, nw+1)
	pAX = b.Shared("actAX", ir.F32, nb+k/16+1)
	span := actSpan(in)
	var shW, part ir.Value
	if span > lanes {
		shW = b.Shared("actWin", ir.F32, group/lanes)
	}
	if in.Norm {
		n := normSlots(group)
		if in.Warp {
			n /= lanes
		}
		part = b.Shared("actSS", ir.F32, n)
	}
	tid := b.TID()

	// The row as it is quantized, element at: loaded once per word in the
	// pass below, from L2 -- a threadgroup copy of the row would not fit
	// beside the operand at the widest k.
	elem := func(at ir.Value) ir.Value { return b.Load(ir.F32, ps.in, at, 0) }
	if in.Gated {
		elem = func(at ir.Value) ir.Value {
			return actMulValue(b, in.Act, b.Load(ir.F32, ps.in, at, 0), func() ir.Value { return b.Load(ir.F32, ps.u, at, 0) })
		}
	}
	if in.Norm {
		scale := normScale(b, k, group, in, ps.in, part, tid)
		elem = func(at ir.Value) ir.Value {
			w := b.Load(ir.F32, ps.nw, at, 0)
			if in.AddOne {
				w = b.Add(ir.F32, w, b.ConstF32(1))
			}
			return b.Mul(ir.F32, b.Mul(ir.F32, b.Load(ir.F32, ps.in, at, 0), scale), w)
		}
	}

	// Pass 2: whole words. Word w is thread tid + j*group's in iteration j, so
	// a window's words are consecutive lanes, and a window is all real words
	// or all past the end (nw and group are multiples of span).
	half := b.Const(ir.U32, 0x3F000000)
	signMask := b.Const(ir.U32, 0x80000000)
	lastW := b.Const(ir.U32, int64(nw-1))
	nwC := b.Const(ir.U32, int64(nw))
	dummyAX := b.Const(ir.U32, int64(nb+k/16))
	for j := 0; j < (nw+group-1)/group; j++ {
		raw := b.Add(ir.U32, tid, b.Const(ir.U32, int64(j*group)))
		word := b.Min(ir.U32, raw, lastW)
		real := b.Lt(ir.U32, raw, nwC)
		base := b.Mul(ir.U32, word, b.Const(ir.U32, 4))
		var v [4]ir.Value
		for i := range v {
			v[i] = elem(b.Add(ir.U32, base, b.Const(ir.U32, int64(i))))
		}
		amax := amaxOf(b, v[:])
		for m := 1; m < min(span, lanes); m <<= 1 {
			amax = b.Max(ir.F32, amax, b.ShuffleXor(ir.F32, amax, int64(m)))
		}
		if span > lanes {
			warp := b.Shr(ir.U32, tid, b.Const(ir.U32, 5))
			b.Store(shW, warp, amax, 0)
			b.Barrier()
			wpw := span / lanes
			wb := b.Mul(ir.U32, b.Div(ir.U32, warp, b.Const(ir.U32, int64(wpw))), b.Const(ir.U32, int64(wpw)))
			amax = b.Load(ir.F32, shW, wb, 0)
			for i := 1; i < wpw; i++ {
				amax = b.Max(ir.F32, amax, b.Load(ir.F32, shW, wb, int64(i)))
			}
			// The next iteration's stores must not overtake these reads.
			b.Barrier()
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
		// A 16-element sum is four words' integer sums, exact in f32 in any
		// order.
		s16 := b.CvtF32(sum)
		s16 = b.Add(ir.F32, s16, b.ShuffleXor(ir.F32, s16, 1))
		s16 = b.Add(ir.F32, s16, b.ShuffleXor(ir.F32, s16, 2))
		b.Store(pA, b.Select(ir.U32, real, word, nwC), packed, 0)
		b.Store(pAX, b.Select(ir.U32, real, b.Shr(ir.U32, word, b.Const(ir.U32, 3)), dummyAX), d, 0)
		b.Store(pAX, b.Select(ir.U32, real, b.Add(ir.U32, b.Shr(ir.U32, word, b.Const(ir.U32, 2)),
			b.Const(ir.U32, int64(nb))), dummyAX), s16, 0)
	}
	b.Barrier()
	return pA, pAX
}

// normScale is rmsNormRowsAs' 1/sqrt(mean(x^2)+eps) for one row, computed by
// group threads as RMSNormGroup would: thread tid stands for the virtual
// threads tid + j*group, each summing its strided elements in the same order
// (unrolled here, a loop there), and the reduction tree (or the two
// butterflies) combines the same partials in the same order.
func normScale(b *ir.Builder, k, group int, in ActIn, pIn, part, tid ir.Value) ir.Value {
	const g = RMSNormGroup
	nv := (g + group - 1) / group
	gC := b.Const(ir.U32, g)
	last := b.Const(ir.U32, int64(k-1))
	kk := b.Const(ir.U32, int64(k))
	zeroF := b.ConstF32(0)
	vt := make([]ir.Value, nv)
	accs := make([]ir.Value, nv)
	for j := range vt {
		vt[j] = b.Add(ir.U32, tid, b.Const(ir.U32, int64(j*group)))
		accs[j] = zeroF
		for m := 0; m < (k+g-1)/g; m++ {
			i := b.Add(ir.U32, vt[j], b.Const(ir.U32, int64(m*g)))
			idx := b.Min(ir.U32, i, last)
			x := b.Load(ir.F32, pIn, idx, 0)
			sq := b.Select(ir.F32, b.Lt(ir.U32, i, kk), b.Mul(ir.F32, x, x), zeroF)
			if nv*group > g {
				// A virtual thread past the norm's group is not one of its
				// threads: it sums nothing, and its slot past g is never read.
				sq = b.Select(ir.F32, b.Lt(ir.U32, vt[j], gC), sq, zeroF)
			}
			accs[j] = b.Add(ir.F32, accs[j], sq)
		}
	}

	var ss ir.Value
	if in.Warp {
		const lanes = ir.SubgroupLanes
		for j, acc := range accs {
			v := acc
			for m := lanes / 2; m >= 1; m /= 2 {
				v = b.Add(ir.F32, v, b.ShuffleXor(ir.F32, v, int64(m)))
			}
			b.Store(part, b.Shr(ir.U32, vt[j], b.Const(ir.U32, 5)), v, 0)
		}
		b.Barrier()
		t := b.Load(ir.F32, part, b.And(ir.U32, tid, b.Const(ir.U32, lanes-1)), 0)
		for m := lanes / 2; m >= 1; m /= 2 {
			t = b.Add(ir.F32, t, b.ShuffleXor(ir.F32, t, int64(m)))
		}
		ss = t
	} else {
		for j, acc := range accs {
			b.Store(part, vt[j], acc, 0)
		}
		b.Barrier()
		top := b.Const(ir.U32, g-1)
		sums := make([]ir.Value, nv)
		for s := g / 2; s >= 1; s /= 2 {
			for j := range sums {
				mine := b.Load(ir.F32, part, vt[j], 0)
				other := b.Load(ir.F32, part, b.Min(ir.U32, b.Add(ir.U32, vt[j], b.Const(ir.U32, int64(s))), top), 0)
				sums[j] = b.Add(ir.F32, mine, other)
			}
			b.Barrier()
			for j := range sums {
				b.Store(part, vt[j], sums[j], 0)
			}
			b.Barrier()
		}
		ss = b.Load(ir.F32, part, b.Const(ir.U32, 0), 0)
	}
	mean := b.Add(ir.F32, b.Mul(ir.F32, ss, b.ConstF32(1/float32(k))), b.ConstF32(in.Eps))
	return b.Div(ir.F32, b.ConstF32(1), b.Sqrt(mean))
}
