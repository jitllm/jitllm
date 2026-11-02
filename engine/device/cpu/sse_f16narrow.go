package cpu

// EmitNarrowF16SSE is EmitNarrowF16 on a host with no F16C: the same Args, the
// same answer bit for bit (quant.EncodeHalf's), rounded in software four lanes
// at a time with integer and one float add:
//
//	s = x & 0x80000000; x ^= s                      the sign, then |x|
//	n = (x + (x>>13 & 1) + 0xC8000FFF) >> 13         normal: rebias, round to even
//	d = bits(float(x) + 0.5) - bits(0.5)             subnormal: the add rounds at 2^-24
//	o = x < 2^-14 ? d : n
//	o = x > 0x477FFFFF ? (x > Inf ? 0x7E00 : 0x7C00) : o
//	h = o | s>>16
//
// The subnormal add is the one float operation, and rounds to nearest even
// under the MXCSR Go runs with. A carry out of n's mantissa lands in the
// exponent, which is what makes a value just under 65520 round to Inf.
//
// Registers: RCX dst, RDX src, R10 units, R11 singles; XMM7..XMM15 constants,
// XMM0..XMM5 the lanes and temporaries.
func EmitNarrowF16SSE() ([]byte, error) {
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))  // Out: binary16
	a.MOVLoad(RDX, At(RDI, 8))  // W:   float32
	a.MOVLoad(R10, At(RDI, 40)) // K
	a.MOVLoad(R11, At(RDI, 32)) // Rows
	for _, c := range []struct {
		x    Reg
		bits uint32
	}{
		{XMM7, 0x200},       // a NaN's quiet bit
		{XMM8, 0x80000000},  // sign
		{XMM9, 0x7F800000},  // Inf
		{XMM10, 0x477FFFFF}, // the largest x that does not overflow at once
		{XMM11, 0x3F000000}, // 0.5: the subnormal rounding magic
		{XMM12, 0x38800000}, // 2^-14, binary16's smallest normal
		{XMM13, 0xC8000FFF}, // the rebias and half an ulp less one
		{XMM14, 1},          // the rounding's odd bit
		{XMM15, 0x7C00},     // binary16 Inf
	} {
		bcastImm32(&a, c.x, RAX, c.bits)
	}
	conv := func(x Reg) {
		t0, t1, t2, t3 := XMM2, XMM3, XMM4, XMM5
		a.PAND(t0, x, XMM8)
		a.PXOR(x, x, t0)
		a.PSRLD(t1, x, 13)
		a.PAND(t1, t1, XMM14)
		a.PADDD(t1, t1, x)
		a.PADDD(t1, t1, XMM13)
		a.PSRLD(t1, t1, 13)
		a.ADDPS(t2, x, XMM11)
		a.PSUBD(t2, t2, XMM11)
		a.PCMPGTD(t3, XMM12, x)
		a.PAND(t2, t2, t3)
		a.PANDN(t3, t3, t1)
		a.POR(t1, t2, t3)
		a.PCMPGTD(t2, x, XMM9)
		a.PAND(t2, t2, XMM7)
		a.POR(t2, t2, XMM15)
		a.PCMPGTD(t3, x, XMM10)
		a.PAND(t2, t2, t3)
		a.PANDN(t3, t3, t1)
		a.POR(x, t2, t3)
		a.PSRLD(t0, t0, 16)
		a.POR(x, x, t0)
	}
	vec := func() {
		a.MOVUPSLoad(XMM0, At(RDX, 0))
		a.MOVUPSLoad(XMM1, At(RDX, 16))
		conv(XMM0)
		conv(XMM1)
		a.PACKUSDW(XMM0, XMM0, XMM1)
		a.MOVDQUStore(At(RCX, 0), XMM0)
	}
	one := func() {
		a.MOVSSLoad(XMM0, At(RDX, 0))
		conv(XMM0)
		a.PEXTRWStore(At(RCX, 0), XMM0, 0)
	}
	for _, l := range []struct {
		n          Reg
		body       func()
		dst, srcSt int32
	}{{R10, vec, 16, 32}, {R11, one, 2, 4}} {
		skip := a.Label()
		a.TESTQ(l.n, l.n)
		a.JZ(skip)
		lp := a.Label()
		a.Bind(lp)
		l.body()
		a.ADDimm(RCX, l.dst)
		a.ADDimm(RDX, l.srcSt)
		a.DEC(l.n)
		a.JNZ(lp)
		a.Bind(skip)
	}
	a.RET()
	return a.Bytes(), nil
}
