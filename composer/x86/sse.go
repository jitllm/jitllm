package x86

var ssePrefix = [4]byte{0, 0x66, 0xF3, 0xF2}

const (
	XMM0 Reg = iota
	XMM1
	XMM2
	XMM3
	XMM4
	XMM5
	XMM6
	XMM7
	XMM8
	XMM9
	XMM10
	XMM11
	XMM12
	XMM13
	XMM14
	XMM15
)

func (a *Buf) sseOp(op byte, pp, mm, w byte, dst, src Reg) {
	a.regs.vec(dst)
	if p := ssePrefix[pp]; p != 0 {
		a.u8(p)
	}
	if r := rexFor(dst, src, w); r != 0 {
		a.u8(r)
	}
	a.sseEsc(op, mm)
	a.u8(0xC0 | byte(dst&7)<<3 | byte(src&7))
}

func rexFor(dst, src Reg, w byte) byte {
	var r byte
	if w != 0 {
		r |= 0x08
	}
	if dst.hi() != 0 {
		r |= 0x04
	}
	if src.hi() != 0 {
		r |= 0x01
	}
	if r == 0 {
		return 0
	}
	return 0x40 | r
}

func (a *Buf) sseEsc(op, mm byte) {
	a.u8(0x0F)
	switch mm {
	case m0F38:
		a.u8(0x38)
	case m0F3A:
		a.u8(0x3A)
	}
	a.u8(op)
}

func (a *Buf) sseOpImm(op byte, pp, mm, w byte, dst, src Reg, imm byte) {
	a.sseOp(op, pp, mm, w, dst, src)
	a.u8(imm)
}

func (a *Buf) sseOpMem(op byte, pp, mm, w byte, reg Reg, m Mem) {
	if p := ssePrefix[pp]; p != 0 {
		a.u8(p)
	}
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.rex(w != 0, reg, idx, m.Base)
	a.sseEsc(op, mm)
	a.modrmMem(reg, m)
}

func (a *Buf) sseOpMemImm(op byte, pp, mm, w byte, reg Reg, m Mem, imm byte) {
	a.sseOpMem(op, pp, mm, w, reg, m)
	a.u8(imm)
}

func (a *Buf) sse3Imm(op byte, pp, mm, w byte, commutes bool, dst, src1, src2 Reg, imm byte) {
	switch {
	case dst == src1:
	case dst == src2 && commutes:
		src2 = src1
	case dst == src2:
		a.fail("x86: sse3: dst aliases src2 on a non-commutative op; " +
			"this needs a scratch register the encoder does not own")
	default:
		a.MOVAPS(dst, src1)
	}
	a.sseOpImm(op, pp, mm, w, dst, src2, imm)
}

func (a *Buf) MOVAPS(dst, src Reg) {
	if dst == src {
		return
	}
	a.useISA(ISASSE2, "movaps")
	a.sseOp(0x28, 0, m0F, 0, dst, src)
}

func (a *Buf) sse3(op byte, pp, mm, w byte, commutes bool, dst, src1, src2 Reg) {
	switch {
	case dst == src1:
		a.sseOp(op, pp, mm, w, dst, src2)
	case dst == src2 && commutes:
		a.sseOp(op, pp, mm, w, dst, src1)
	case dst == src2:
		a.fail("x86: sse3: dst aliases src2 on a non-commutative op; " +
			"this needs a scratch register the encoder does not own")
	default:
		a.MOVAPS(dst, src1)
		a.sseOp(op, pp, mm, w, dst, src2)
	}
}
