package jit

import "github.com/jitllm/jitllm/jit/ir"

// divReach is how many floats either side of the device's quotient div
// considers: SPIR-V allows OpFDiv 2.5 ulp (the Vulkan specification's
// precision table), so the correctly rounded quotient is at most three
// floats away.
const divReach = 3

// div is a/b rounded to nearest even, the IEEE quotient, on a device whose
// own division is not (Device.ExactDiv): its quotient is the estimate, and
// the rounding is decided in integers. Vulkan states OpFDiv to 2.5 ulp and
// its fma as a multiply followed by an add, so neither a division nor a
// Newton step through a fused residual is exact there; integers are.
//
// The candidates are the floats within divReach of the estimate y of
// |a|/|b|, and the result the lowest plus how many of the midpoints between
// consecutive candidates lie below the true quotient x: x > m exactly when
//
//	D = ma*2^s - (2*Mc+1)*md > 0
//
// for the midpoint m = (2*Mc+1) * 2^(ec-24) above candidate c (mantissa Mc,
// exponent ec), |a| = ma*2^ea and |b| = md*2^ed. |D| is a few ulp of a
// product near 2^48, under 2^28, so its low 32 bits -- what u32 arithmetic
// computes exactly -- are D. A quotient of two floats is never halfway
// between two floats, so D is never zero. The sign is a's xor b's. Where an
// operand is zero, subnormal, infinite or NaN, or the quotient is not a
// normal float, it is the device's own quotient.
func (x *exprs) div(a, b int) int {
	U, F := ir.U32, ir.F32
	ma0, mb0 := x.abs(a), x.abs(b)
	est := x.op(OpDiv, F, ma0, mb0)
	mant := func(bits int) int { return x.op(OpAdd, U, x.op(OpAnd, U, bits, x.u(0x7fffff)), x.u(0x800000)) }
	na, nd := x.un(OpBitcast, U, ma0), x.un(OpBitcast, U, mb0)
	ea, ed := x.op(OpShr, U, na, x.u(23)), x.op(OpShr, U, nd, x.u(23))
	ma, md := mant(na), mant(nd)
	ny := x.un(OpBitcast, U, est)
	base := x.op(OpSub, U, ny, x.u(divReach))
	pick := base
	for j := uint32(0); j < 2*divReach; j++ {
		c := x.op(OpAdd, U, base, x.u(j))
		ec := x.op(OpShr, U, c, x.u(23))
		s := x.op(OpSub, U, x.op(OpAdd, U, ea, x.u(151)), x.op(OpAdd, U, ec, ed))
		mid := x.op(OpAdd, U, x.op(OpShl, U, mant(c), x.u(1)), x.u(1))
		d := x.op(OpSub, U, x.op(OpShl, U, ma, s), x.op(OpMul, U, mid, md))
		above := x.op(OpLess, ir.Pred, x.op(OpSub, U, d, x.u(1)), x.u(0x7fffffff))
		pick = x.op(OpAdd, U, pick, x.sel(U, above, x.u(1), x.u(0)))
	}
	in := func(e int, lo, n uint32) int {
		return x.op(OpLess, ir.Pred, x.op(OpSub, U, e, x.u(lo)), x.u(n))
	}
	q := x.sel(F, in(x.op(OpShr, U, ny, x.u(23)), 2, 252), x.un(OpBitcast, F, pick), est)
	q = x.sel(F, in(ed, 1, 254), q, est)
	q = x.sel(F, in(ea, 1, 254), q, est)
	sign := x.op(OpAnd, U, x.op(OpXor, U, x.un(OpBitcast, U, a), x.un(OpBitcast, U, b)), x.u(0x80000000))
	return x.un(OpBitcast, F, x.op(OpOr, U, x.un(OpBitcast, U, q), sign))
}
