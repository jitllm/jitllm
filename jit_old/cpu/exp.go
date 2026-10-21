//go:build amd64

package cpu

// The vectorised exp every generated softmax and activation is built on.
//
// Range-reduce, Horner, then build 2^n in the exponent field: exp(x) =
// 2^n * exp(r) with n = round(x*log2e) and r = x - n*ln2, so the only hard part
// is a degree-5 polynomial on |r| <= ln2/2 and an integer shift. No table, no
// branch, no call.
//
// ln2 is split in two for accuracy, not speed: ln2 rounded to one f32 loses
// the low bits of n*ln2, which at |x| ~ 80 exceeds the polynomial's error. hi
// holds the leading bits exactly (n*hi is exact) and lo the remainder.
//
// Degree 5 is set by the reduction bound: the truncation is about r^6/720 =
// 2.4e-6, under an f32 ulp at 1.0, and the outputs are quantized to int8
// downstream. The constants live in elem_const.go, shared by both
// architectures.

// loadExpConsts fills Y7..Y15 with the constants emitExpPS needs, once, outside
// any loop. RBX must already point at expConstsShared().
//
// The register assignment is fixed and documented here rather than passed
// around, because exp uses nine of sixteen YMM and the callers have to know
// which seven are left.
func loadExpConsts(a *Buf) {
	a.VBROADCASTSS(Y9, At(RBX, 0))   // log2e
	a.VBROADCASTSS(Y10, At(RBX, 4))  // -ln2hi
	a.VBROADCASTSS(Y11, At(RBX, 8))  // -ln2lo
	a.VBROADCASTSS(Y12, At(RBX, 12)) // c5
	a.VBROADCASTSS(Y13, At(RBX, 16)) // c4
	a.VBROADCASTSS(Y14, At(RBX, 20)) // c3
	a.VBROADCASTSS(Y15, At(RBX, 24)) // c2
	a.VBROADCASTSS(Y8, At(RBX, 28))  // 1.0
	a.VPBROADCASTD(Y7, At(RBX, 36))  // 127, as an integer
}

// emitExpPS computes exp(dst) into dst, clobbering t0 and t1.
//
// dst may be any of Y0..Y6; t0 and t1 must be two others. Y7..Y15 are the
// constants loadExpConsts put there.
func emitExpPS(a *Buf, dst, t0, t1 Reg) {
	// Clamp from below so the exponent build cannot go denormal. exp of
	// anything under -87.3 is zero in f32, and a softmax feeds this x - max,
	// which is <= 0 and can be very negative on a long row.
	a.VBROADCASTSS(t0, At(RBX, 32))
	a.VMAXPS(dst, dst, t0)
	a.VBROADCASTSS(t0, At(RBX, 40))
	a.VMINPS(dst, dst, t0)

	// n = round(x * log2e), kept both as int (for the exponent) and float
	// (to subtract).
	a.VMULPS(t0, dst, Y9)
	a.VCVTPS2DQ(t1, t0) // n, integer
	a.VCVTDQ2PS(t0, t1) // n, float

	// r = x - n*ln2, in two parts.
	a.VFMADD231PS(dst, t0, Y10)
	a.VFMADD231PS(dst, t0, Y11)

	// exp(r) by Horner: ((((c5*r + c4)*r + c3)*r + c2)*r + 1)*r + 1.
	//
	// VFMADD213PS is dst = src1*dst + src2, which is exactly a Horner step,
	// where the 231 form used above accumulates instead.
	a.VMOVAPSReg(t0, Y12) // p = c5
	a.VFMADD213PS(t0, dst, Y13)
	a.VFMADD213PS(t0, dst, Y14)
	a.VFMADD213PS(t0, dst, Y15)
	a.VFMADD213PS(t0, dst, Y8)
	a.VFMADD213PS(t0, dst, Y8)

	// scale = 2^n, built straight in the exponent field: (n + 127) << 23.
	a.VPADDD(t1, t1, Y7)
	a.VPSLLD(t1, t1, 23)
	a.VMULPS(dst, t0, t1)
}
