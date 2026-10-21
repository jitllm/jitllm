package cpu

// The IQ4_NL row decode: K blocks of quant.IQ4_NL (an f16 scale, then 16
// bytes whose low nibbles are elements 0..15 and high nibbles 16..31, each an
// index into the non-linear table) widened into 32*K floats.
//
//	Out      32*K float32, written
//	W        the blocks, 18 bytes each, no alignment
//	K        how many blocks
//	Scr      IQ4NLConsts: the 16-entry int8 table, then sixteen 0x0F bytes
//	Scratch  32 bytes of the kernel's own
//
// A scale times an int8 is exact in f32, so the decode is bit-identical to
// quant.Dequant32 (TestRowIQ4NLMatchesTheDecoder).

// IQ4NLConsts is the row decode's constant block for table vals.
func IQ4NLConsts(vals [16]int8) []byte {
	b := make([]byte, 32)
	for i, v := range vals {
		b[i] = byte(v)
		b[16+i] = 0x0F
	}
	return b
}

// EmitRowIQ4NLAVX2 is the decode on the AVX2 tier: the scale through
// VCVTPH2PS, the nibbles through VPSHUFB against the table, eight at a time
// sign-extended and converted, times the scale.
//
// Registers: RCX out, RDX the blocks, R10 the count, RBX the constants, R9
// the scratch. Y7 the table, Y6 the nibble mask, Y0 the scale, Y1..Y4
// temporaries.
func EmitRowIQ4NLAVX2() []byte {
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(RDX, At(RDI, 8))  // W
	a.MOVLoad(R10, At(RDI, 40)) // K
	a.MOVLoad(RBX, At(RDI, 56)) // Scr
	a.MOVLoad(R9, At(RDI, 72))  // Scratch
	a.VBROADCASTI128(Y7, At(RBX, 0))
	a.VBROADCASTI128(Y6, At(RBX, 16))
	done := a.Label()
	a.TESTQ(R10, R10)
	a.JZ(done)
	lp := a.Label()
	a.Bind(lp)
	a.VPXOR(Y0, Y0, Y0)
	a.VPINSRWLoad(Y0, Y0, At(RDX, 0), 0)
	a.VCVTPH2PSReg(Y0, Y0)
	a.VBROADCASTSSReg(Y0, Y0)
	a.VMOVDQULoadx(Y1, At(RDX, 2))
	a.VPANDx(Y2, Y1, Y6) // the low nibbles: elements 0..15
	a.VPSRLW(Y3, Y1, 4)
	a.VPANDx(Y3, Y3, Y6) // the high: 16..31
	a.VPSHUFB(Y2, Y7, Y2)
	a.VPSHUFB(Y3, Y7, Y3)
	a.VMOVDQUStorex(At(R9, 0), Y2)
	a.VMOVDQUStorex(At(R9, 16), Y3)
	for g := int32(0); g < 4; g++ {
		a.VPMOVSXBD(Y4, At(R9, 8*g))
		a.VCVTDQ2PS(Y4, Y4)
		a.VMULPS(Y4, Y4, Y0)
		a.VMOVDQUStore(At(RCX, 32*g), Y4)
	}
	a.ADDimm(RCX, 128)
	a.ADDimm(RDX, 18)
	a.DEC(R10)
	a.JNZ(lp)
	a.Bind(done)
	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// EmitRowIQ4NLSSE is the decode on the SSE tier: the scale through the
// software half conversion, the nibbles through PSHUFB (SSSE3), four at a time
// sign-extended (SSE4.1) and converted.
//
// Registers: RCX out, RDX the blocks, R10 the count, RBX the constants, R9
// the scratch, RAX the magic's temporary. XMM13 the table, XMM14 the mask,
// XMM12 the half magic, XMM0 the scale.
func EmitRowIQ4NLSSE() ([]byte, error) {
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(RDX, At(RDI, 8))  // W
	a.MOVLoad(R10, At(RDI, 40)) // K
	a.MOVLoad(RBX, At(RDI, 56)) // Scr
	a.MOVLoad(R9, At(RDI, 72))  // Scratch
	a.MOVDQULoad(XMM13, At(RBX, 0))
	a.MOVDQULoad(XMM14, At(RBX, 16))
	loadHalfMagicSSE(&a, XMM12, RAX)
	done := a.Label()
	a.TESTQ(R10, R10)
	a.JZ(done)
	lp := a.Label()
	a.Bind(lp)
	a.PXOR(XMM1, XMM1, XMM1)
	a.PINSRWLoad(XMM1, XMM1, At(RDX, 0), 0)
	halfToFloatSSE(&a, XMM0, XMM1, XMM5, XMM6, XMM12)
	a.SHUFPS(XMM0, XMM0, XMM0, 0)
	a.MOVDQULoad(XMM1, At(RDX, 2))
	a.MOVAPS(XMM2, XMM1)
	a.PAND(XMM2, XMM2, XMM14)
	a.PSRLW(XMM3, XMM1, 4)
	a.PAND(XMM3, XMM3, XMM14)
	a.PSHUFB(XMM7, XMM13, XMM2)
	a.PSHUFB(XMM8, XMM13, XMM3)
	a.MOVDQUStore(At(R9, 0), XMM7)
	a.MOVDQUStore(At(R9, 16), XMM8)
	for g := int32(0); g < 8; g++ {
		a.PMOVSXBDLoad(XMM4, At(R9, 4*g))
		a.CVTDQ2PS(XMM4, XMM4)
		a.MULPS(XMM4, XMM4, XMM0)
		a.MOVUPSStore(At(RCX, 16*g), XMM4)
	}
	a.ADDimm(RCX, 128)
	a.ADDimm(RDX, 18)
	a.DEC(R10)
	a.JNZ(lp)
	a.Bind(done)
	a.RET()
	return a.Bytes(), nil
}
