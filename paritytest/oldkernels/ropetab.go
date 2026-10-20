package oldkernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// RopeTable builds the rotary table on the device: the fourth transcription of
// jit/cpu's rope table kernel (AVX2, SSE and NEON are the others), so the
// device does not depend on a table built by the host.
//
// The arithmetic is the host's, not a GPU sin/cos: driver transcendentals
// differ by vendor, but a range reduction and two polynomials over constants
// the host folded in float64 are f32 multiplies, adds and bit operations every
// backend runs identically.
//
//	pOut     rows * 2*npairs float32, {cos, sin} per pair, written
//	pPlanes  the per-model folded plane block (ropetab_const.go's layout)
//	pConst   RopeTabConsts()
//	pPos     the valid row count, then rows positions, u32
//
// The row count is runtime because a prefill's last chunk is ragged: the host
// leaves padded rows zero, so this kernel clamps to the valid count and the
// padded rows keep their zeros. A chat prompt shorter than the batch width is
// only a ragged chunk.
//
// One thread per (row, pair); surplus threads clamp onto the last valid pair
// and recompute it from read-only inputs (RULE 13). The {cos, sin} interleave
// that costs a shuffle on amd64 is free here: a thread stores two adjacent
// floats.
func RopeTable(npairs, rows int) (*ir.Kernel, error) {
	if npairs <= 0 {
		return nil, fmt.Errorf("kernels: RopeTable: %d pair(s), want at least one", npairs)
	}
	if rows < 1 {
		rows = 1
	}
	stride := int64(RopeTabStride(npairs))
	b := ir.New("ropetable", [3]int{128, 1, 1})
	pOut := b.Param("pOut", ir.F32)
	pPlanes := b.Param("pPlanes", ir.F32)
	// The constant block is a u32 buffer, bitcast where a float is wanted: some
	// words are integers stored as float bits (the integer one is the smallest
	// denormal), and a GPU may flush denormals, which would silently shift the
	// cosine's quadrant.
	pConst := b.Param("pConst", ir.U32)
	pPos := b.Param("pPos", ir.U32)

	cw := func(w int) ir.Value { return b.Load(ir.U32, pConst, b.Const(ir.U32, int64(w)), 0) }
	cf := func(w int) ir.Value { return b.Bitcast(ir.F32, cw(w)) }

	// The clamp is nvalid*npairs-1, bounded by the baked rows*npairs-1. It is
	// clamped from below too: at a count of zero lim-1 would be 0xFFFFFFFF, an
	// unbounded write rather than a recomputed row zero.
	lim := b.Max(ir.U32, b.Const(ir.U32, int64(npairs)), b.Min(ir.U32,
		b.Mul(ir.U32, b.Load(ir.U32, pPos, b.Const(ir.U32, 0), 0), b.Const(ir.U32, int64(npairs))),
		b.Const(ir.U32, int64(rows*npairs))))
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Sub(ir.U32, lim, b.Const(ir.U32, 1)))
	p, row := flat, ir.Value(b.Const(ir.U32, 0))
	if rows > 1 {
		p = b.Rem(ir.U32, flat, b.Const(ir.U32, int64(npairs)))
		row = b.Div(ir.U32, flat, b.Const(ir.U32, int64(npairs)))
	}
	// +1: element 0 of pPos is the row count.
	pos := b.Load(ir.U32, pPos, row, 1)

	// The position's four base-128 digits, as floats. Each is under 128, so the
	// signed convert is the unsigned one.
	mask := b.Const(ir.U32, int64(1<<RopeTabDigitBits-1))
	var d [RopeTabDigits]ir.Value
	for i := range d {
		x := pos
		if i > 0 {
			x = b.Shr(ir.U32, pos, b.Const(ir.U32, int64(i*RopeTabDigitBits)))
		}
		d[i] = b.CvtF32(b.And(ir.U32, x, mask))
	}

	// Mul-then-add rather than Fma for the two sums: the head sum is exact by
	// construction and the residual's rounding is negligible, while spirv lowers
	// ir.OpFma unfused and ptx/msl fuse it, so avoiding Fma keeps the backends
	// identical.
	plane := func(i int) ir.Value { return b.Load(ir.F32, pPlanes, p, int64(i)*stride) }
	sum := func(base int) ir.Value {
		acc := b.Mul(ir.F32, d[0], plane(base))
		for i := 1; i < RopeTabDigits; i++ {
			acc = b.Add(ir.F32, acc, b.Mul(ir.F32, d[i], plane(base+i)))
		}
		return acc
	}
	s := sum(0)
	c := sum(RopeTabDigits)

	// s mod 4 in integers: the host's x - ((x+M)-M) is reassociated to zero by
	// a SPIR-V driver (see RopeTabConsts). s is a multiple of 2^-12 under 2032,
	// so s*2^12 is an exact integer, the mask is the remainder mod four and the
	// scale back is exact. The remainder lands in [0,4) where the host's lands
	// in [-2,2]; n absorbs the multiple of four.
	grid, rgrid := b.ConstF32(RopeTabGrid), b.ConstF32(1.0/RopeTabGrid)
	si := b.Bitcast(ir.U32, b.CvtI32(b.Mul(ir.F32, s, grid)))
	rem := b.Mul(ir.F32, b.CvtF32(b.And(ir.U32, si, cw(RopeTabMod4MaskWord))), rgrid)

	// The quadrant is chosen from rem+c and the remainder taken from rem alone,
	// which keeps |f| under a half plus |c| (see EmitRopeTable). ir.OpCvtI32
	// truncates, but rem+c+1/2 is never below 7/16, so truncation is the floor
	// and n is never negative.
	n := b.CvtI32(b.Add(ir.F32, b.Add(ir.F32, rem, c), cf(RopeTabHalfWord)))
	fn := b.CvtF32(n)
	f := b.Sub(ir.F32, rem, fn) // exact, |f| <= 1/2 + |c|
	nb := b.Bitcast(ir.U32, n)

	// r = (f + c)*pi/2 with the large term added LAST, so the dominant product
	// carries one rounding rather than three.
	hi, lo := cf(RopeTabPio2HiWord), cf(RopeTabPio2LoWord)
	r := b.Fma(f, hi, b.Fma(f, lo, b.Mul(ir.F32, c, hi)))
	z := b.Mul(ir.F32, r, r)

	// sin(r) = r + (r*z)*(c3 + z*(c5 + z*c7))
	sp := b.Fma(z, cf(RopeTabSinC7Word), cf(RopeTabSinC5Word))
	sp = b.Fma(z, sp, cf(RopeTabSinC3Word))
	sp = b.Fma(b.Mul(ir.F32, r, z), sp, r)

	// cos(r) = 1 + z*(-1/2 + z*(c4 + z*(c6 + z*c8)))
	cp := cf(RopeTabCosC8Word)
	for _, w := range []int{RopeTabCosC6Word, RopeTabCosC4Word,
		RopeTabNegHalfWord, RopeTabOneWord} {
		cp = b.Fma(z, cp, cf(w))
	}

	// Bit 0 of n swaps the two polynomials; bit 1 of n negates sin and bit 1 of
	// n+1 negates cos (see EmitRopeTable). The swap is a Select and the
	// negations an XOR, amd64's spelling, so zeroing RopeTabSignVecWord or
	// RopeTabOneVecWord removes exactly one step here as on the host tiers.
	swap := b.Lt(ir.U32, b.Const(ir.U32, 0), b.And(ir.U32, nb, b.Const(ir.U32, 1)))
	sv, cv := b.Select(ir.F32, swap, cp, sp), b.Select(ir.F32, swap, sp, cp)

	sign := cw(RopeTabSignVecWord)
	neg := func(x, m ir.Value) ir.Value {
		bits := b.And(ir.U32, b.Shl(ir.U32, m, b.Const(ir.U32, 30)), sign)
		return b.Bitcast(ir.F32, b.Xor(ir.U32, b.Bitcast(ir.U32, x), bits))
	}
	sv = neg(sv, nb)
	cv = neg(cv, b.Add(ir.U32, nb, cw(RopeTabOneVecWord)))

	ms := b.Load(ir.F32, pPlanes, b.Const(ir.U32, 0), 8*stride)
	out := b.Mul(ir.U32, flat, b.Const(ir.U32, 2))
	b.Store(pOut, out, b.Mul(ir.F32, cv, ms), 0)
	b.Store(pOut, out, b.Mul(ir.F32, sv, ms), 1)
	return b.Done(), nil
}
