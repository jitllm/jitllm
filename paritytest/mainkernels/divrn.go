package mainkernels

import "github.com/jitllm/jitllm/jit_old/gpu/ir"

// divRN is num/den rounded to nearest even, the IEEE quotient the host's
// VDIVPS and FDIV return, on every backend. ir.OpDiv is that on PTX (div.rn)
// and on Metal's IEEE math, and not on SPIR-V: OpFDiv is allowed 2.5 ulp, and
// NVIDIA's and Intel's Vulkan drivers return another last bit for 4.5% of
// x/127 quotients and 27-28% of 127/x ones. Nor can an FMA correct it there:
// SPIR-V lowers ir.OpFma as a multiply and an add. quantScales builds its two
// quotients from roundQuotient directly; divRN is the general form, and its
// gate (TestDivRNIsTheIEEEQuotient) is roundQuotient's against the backend's
// own estimate.
//
// So the backend's quotient y is only the estimate, and the rounding is
// decided in integers. The correctly rounded quotient lies within divRNReach
// floats of y, so the midpoints between consecutive floats around y are
// counted against the true quotient x: x > m exactly when
//
//	D = ma*2^s - (2*Mc+1)*md > 0
//
// for the midpoint m = (2*Mc+1) * 2^(ec-24) just above candidate c (mantissa
// Mc, exponent ec), with num = ma*2^ea and den = md*2^ed. D is a difference of
// two products near 2^48, but it is small -- |x - m| is a few ulp, which puts
// |D| under 2^28 -- so its low 32 bits, which u32 arithmetic computes exactly,
// are D itself. Every midpoint is independent of the others, so this is a
// few dozen integer operations side by side rather than a chain of them.
//
// Both operands must be positive. Where either is zero, subnormal, infinite
// or NaN, or the quotient is not a normal float, it is the backend's own
// division -- no activation scale reaches that range.
func divRN(b *ir.Builder, num, den ir.Value) ir.Value {
	return roundQuotient(b, num, den, b.Div(ir.F32, num, den), divRNReach)
}

// div127 is x/127 correctly rounded, for x positive: the quantizer's scale,
// in integers and without the backend's division. The mantissa ma times 256
// over 127 is a 25- or 26-bit integer quotient q, and its bits below the 24
// kept decide the rounding alone: a quotient of a float by 127 is never
// exactly halfway between two floats (q odd with no remainder would make 127
// divide ma*256, and then q is a multiple of 256), so the remainder never
// breaks a tie. A zero, subnormal or huge x is the backend's own x/127.
func div127(b *ir.Builder, x ir.Value) ir.Value {
	u := func(v int64) ir.Value { return b.Const(ir.U32, v) }
	nx := b.Bitcast(ir.U32, x)
	ex := b.Shr(ir.U32, nx, u(23))
	ma := b.Add(ir.U32, b.And(ir.U32, nx, u(0x7FFFFF)), u(0x800000))
	q := b.Div(ir.U32, b.Shl(ir.U32, ma, u(8)), u(127)) // [2^24, 2^26)
	sh := b.Add(ir.U32, b.Shr(ir.U32, q, u(25)), u(1))  // bits below the 24 kept
	half := b.Shl(ir.U32, u(1), b.Sub(ir.U32, sh, u(1)))
	up := b.Select(ir.U32, b.Lt(ir.U32, b.And(ir.U32, q, b.Sub(ir.U32, b.Shl(ir.U32, half, u(1)), u(1))), half),
		u(0), u(1))
	m := b.Add(ir.U32, b.Shr(ir.U32, q, sh), up)
	carry := b.Shr(ir.U32, m, u(24)) // rounded up to 2^24
	// The biased exponent: ma*2^(ex-150)/127 = (q/2^sh) * 2^(ex-158+sh).
	e := b.Add(ir.U32, b.Sub(ir.U32, b.Add(ir.U32, ex, sh), u(8)), carry)
	bits := b.Add(ir.U32, b.Shl(ir.U32, e, u(23)), b.And(ir.U32, b.Shr(ir.U32, m, carry), u(0x7FFFFF)))
	return b.Select(ir.F32, b.Lt(ir.U32, b.Sub(ir.U32, ex, u(8)), u(247)), b.Bitcast(ir.F32, bits),
		b.Div(ir.F32, x, b.ConstF32(127)))
}

// roundQuotient is divRN from an estimate est of num/den that is within
// reach-1/2 floats of it, and est itself outside the normal range.
func roundQuotient(b *ir.Builder, num, den, est ir.Value, reach int64) ir.Value {
	u := func(v int64) ir.Value { return b.Const(ir.U32, v) }
	mant := func(bits ir.Value) ir.Value { return b.Add(ir.U32, b.And(ir.U32, bits, u(0x7FFFFF)), u(0x800000)) }
	na, nd := b.Bitcast(ir.U32, num), b.Bitcast(ir.U32, den)
	ea, ed := b.Shr(ir.U32, na, u(23)), b.Shr(ir.U32, nd, u(23))
	ma, md := mant(na), mant(nd)
	hw := est
	ny := b.Bitcast(ir.U32, hw)
	// The candidates are the floats est-reach .. est+reach (bit
	// patterns of positive floats are ordered as the floats are, across
	// binades). The result is the lowest one plus how many midpoints lie
	// below x.
	base := b.Sub(ir.U32, ny, u(reach))
	pick := base
	for j := int64(0); j < 2*reach; j++ {
		c := b.Add(ir.U32, base, u(j))
		ec := b.Shr(ir.U32, c, u(23))
		// s = (ea-150) - (ec-151) - (ed-150): ma*2^s against (2Mc+1)*md.
		s := b.Sub(ir.U32, b.Add(ir.U32, ea, u(151)), b.Add(ir.U32, ec, ed))
		mid := b.Add(ir.U32, b.Shl(ir.U32, mant(c), u(1)), u(1))
		d := b.Sub(ir.U32, b.Shl(ir.U32, ma, s), b.Mul(ir.U32, mid, md))
		// D > 0 as a signed value. D == 0 cannot happen: the quotient of two
		// floats is never exactly halfway between two floats, so there is no
		// tie to break.
		above := b.Lt(ir.U32, b.Sub(ir.U32, d, u(1)), u(0x7FFFFFFF))
		pick = b.Add(ir.U32, pick, b.Select(ir.U32, above, u(1), u(0)))
	}
	// Normal operands (biased exponent 1..254: e-1 wraps past 253 for e == 0)
	// and an estimate a binade inside that range, so that every candidate is
	// a normal float and the shift is defined.
	in := func(e ir.Value, lo, n int64) ir.Value { return b.Lt(ir.U32, b.Sub(ir.U32, e, u(lo)), u(n)) }
	ok := func(p, then ir.Value) ir.Value { return b.Select(ir.F32, p, then, hw) }
	return ok(in(ea, 1, 254), ok(in(ed, 1, 254), ok(in(b.Shr(ir.U32, ny, u(23)), 2, 252), b.Bitcast(ir.F32, pick))))
}

// divRNReach is how many floats either side of the backend's quotient divRN
// considers: SPIR-V allows OpFDiv 2.5 ulp, so the correctly rounded quotient
// is at most three floats away.
const divRNReach = 3
