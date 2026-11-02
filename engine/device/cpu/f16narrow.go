//go:build amd64

package cpu

// EmitNarrowF16 generates dst[i] = binary16(src[i]) at a runtime length, the
// f16 KV cache's store: Args.K whole vectors of eight and Args.Rows single
// elements, Out the binary16 dst and W the float32 src. It is EmitWiden's
// inverse, and rounds as quant.EncodeHalf does, bit for bit: VCVTPS2PH at
// round-to-nearest-even (imm 0, whatever MXCSR says) handles overflow,
// subnormals and the signed zeros, and a NaN keeps its sign and loses its
// payload (EncodeHalf's sign|0x7E00) -- F16C keeps the payload's top bits, so
// they are cleared after the conversion:
//
//	h &^= (h&0x7FFF > 0x7C00) & 0x01FF
//
// Registers: RCX dst, RDX src, R10 vectors, R11 singles; X13..X15 the
// constants 0x7FFF, 0x7C00 and 0x01FF in every word.
func EmitNarrowF16() []byte {
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out: binary16
	a.MOVLoad(RDX, At(RDI, 8))  // W:   float32
	a.MOVLoad(R10, At(RDI, 40)) // K
	a.MOVLoad(R11, At(RDI, 32)) // Rows
	a.MOVimm32(RAX, 0x7FFF7FFF)
	a.MOVDToX(XMM13, RAX)
	a.PSHUFD(XMM13, XMM13, 0)
	a.MOVimm32(RAX, 0x7C007C00)
	a.MOVDToX(XMM14, RAX)
	a.PSHUFD(XMM14, XMM14, 0)
	a.MOVimm32(RAX, 0x01FF01FF)
	a.MOVDToX(XMM15, RAX)
	a.PSHUFD(XMM15, XMM15, 0)
	fix := func(h Reg) {
		a.VPANDx(Y2, h, Y13)
		a.VPCMPGTWx(Y2, Y2, Y14)
		a.VPANDx(Y2, Y2, Y15)
		a.VPANDNx(h, Y2, h)
	}
	vec := func() {
		a.VMOVDQULoad(Y0, At(RDX, 0))
		a.VCVTPS2PH(Y1, Y0, true, 0)
		fix(Y1)
		a.VMOVDQUStorex(At(RCX, 0), Y1)
	}
	one := func() {
		a.VMOVSSLoad(Y0, At(RDX, 0))
		a.VCVTPS2PH(Y1, Y0, false, 0)
		fix(Y1)
		a.VPEXTRWStore(At(RCX, 0), Y1, 0)
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
	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}
