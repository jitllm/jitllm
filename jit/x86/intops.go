package x86

func (a *Buf) XORQ(dst, src Reg) {
	a.rex(true, src, noIndex, dst)
	a.u8(0x31)
	a.modrmReg(src, dst)
}

func (a *Buf) DIVQ(src Reg) {
	a.rex(true, Y0, noIndex, src)
	a.u8(0xF7)
	a.modrmReg(Reg(6), src)
}

func (a *Buf) MOVLLoad(dst Reg, m Mem) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.rex(false, dst, idx, m.Base)
	a.u8(0x8B)
	a.modrmMem(dst, m)
}

func (a *Buf) MOVLStore(m Mem, src Reg) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.rex(false, src, idx, m.Base)
	a.u8(0x89)
	a.modrmMem(src, m)
}

func (a *Buf) ANDQ(dst, src Reg) {
	a.rex(true, src, noIndex, dst)
	a.u8(0x21)
	a.modrmReg(src, dst)
}

func (a *Buf) ORQ(dst, src Reg) {
	a.rex(true, src, noIndex, dst)
	a.u8(0x09)
	a.modrmReg(src, dst)
}

// ANDimm32 is dst &= imm, the immediate sign-extended to 64 bits.
func (a *Buf) ANDimm32(dst Reg, imm int32) {
	a.rex(true, Y0, noIndex, dst)
	a.u8(0x81)
	a.modrmReg(Reg(4), dst)
	a.u32(uint32(imm))
}

// IMULimm is dst = src * imm, the immediate sign-extended to 64 bits.
func (a *Buf) IMULimm(dst, src Reg, imm int32) {
	a.rex(true, dst, noIndex, src)
	a.u8(0x69)
	a.modrmReg(dst, src)
	a.u32(uint32(imm))
}

// SHRX is dst = src >> cnt, logical, on 64 bits (BMI2).
func (a *Buf) SHRX(dst, src, cnt Reg) {
	a.useISA(ISABMI2, "shrx")
	a.vex(dst, cnt, src, noIndex, false, pF2, m0F38, 1)
	a.u8(0xF7)
	a.modrmReg(dst, src)
}

// SHLX is dst = src << cnt on 64 bits (BMI2).
func (a *Buf) SHLX(dst, src, cnt Reg) {
	a.useISA(ISABMI2, "shlx")
	a.vex(dst, cnt, src, noIndex, false, p66, m0F38, 1)
	a.u8(0xF7)
	a.modrmReg(dst, src)
}

// MOVL is dst = src's low 32 bits, zero-extended.
func (a *Buf) MOVL(dst, src Reg) {
	a.rex(false, src, noIndex, dst)
	a.u8(0x89)
	a.modrmReg(src, dst)
}

// MOVSXD is dst = src's low 32 bits, sign-extended.
func (a *Buf) MOVSXD(dst, src Reg) {
	a.rex(true, dst, noIndex, src)
	a.u8(0x63)
	a.modrmReg(dst, src)
}

func (a *Buf) NEGQ(dst Reg) {
	a.rex(true, Y0, noIndex, dst)
	a.u8(0xF7)
	a.modrmReg(Reg(3), dst)
}

// XORimm32 is dst ^= imm, the immediate sign-extended to 64 bits.
func (a *Buf) XORimm32(dst Reg, imm int32) {
	a.rex(true, Y0, noIndex, dst)
	a.u8(0x81)
	a.modrmReg(Reg(6), dst)
	a.u32(uint32(imm))
}

// ORimm32 is dst |= imm, the immediate sign-extended to 64 bits.
func (a *Buf) ORimm32(dst Reg, imm int32) {
	a.rex(true, Y0, noIndex, dst)
	a.u8(0x81)
	a.modrmReg(Reg(1), dst)
	a.u32(uint32(imm))
}

// CMPL compares x and y as 32-bit values.
func (a *Buf) CMPL(x, y Reg) {
	a.rex(false, y, noIndex, x)
	a.u8(0x39)
	a.modrmReg(y, x)
}

// CMPLimm compares x with imm as 32-bit values.
func (a *Buf) CMPLimm(x Reg, imm int32) {
	a.rex(false, Y0, noIndex, x)
	a.u8(0x81)
	a.modrmReg(Reg(7), x)
	a.u32(uint32(imm))
}

// JBE, JAE, JLE and JGE jump on unsigned below-or-equal, unsigned
// above-or-equal, signed less-or-equal and signed greater-or-equal.
func (a *Buf) JBE(l int) { a.jcc(0x76, 0x86, l) }
func (a *Buf) JAE(l int) { a.jcc(0x73, 0x83, l) }
func (a *Buf) JLE(l int) { a.jcc(0x7E, 0x8E, l) }
func (a *Buf) JGE(l int) { a.jcc(0x7D, 0x8D, l) }

// MOVLimm is dst = imm, zero-extended from 32 bits.
func (a *Buf) MOVLimm(dst Reg, imm uint32) {
	a.rex(false, Y0, noIndex, dst)
	a.u8(0xB8 + dst.lo())
	a.u32(imm)
}

// MOVMemImm is the 64-bit word at m = imm, sign-extended from 32 bits.
func (a *Buf) MOVMemImm(m Mem, imm int32) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.rex(true, Y0, idx, m.Base)
	a.u8(0xC7)
	a.modrmMem(Y0, m)
	a.u32(uint32(imm))
}

// ADDMemImm8 adds imm to the 64-bit word at m.
func (a *Buf) ADDMemImm8(m Mem, imm int8) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.rex(true, Y0, idx, m.Base)
	a.u8(0x83)
	a.modrmMem(Y0, m)
	a.u8(byte(imm))
}

// ADDMemImm32 adds imm to the 64-bit word at m.
func (a *Buf) ADDMemImm32(m Mem, imm int32) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.rex(true, Y0, idx, m.Base)
	a.u8(0x81)
	a.modrmMem(Y0, m)
	a.u32(uint32(imm))
}
