package cpu

// The 64-bit integer instructions the n-gram hash needs (ngram.go,
// ngram_a64.go): the only kernel whose arithmetic is on integers.

// XORQ is xor r64, r64.
func (a *Buf) XORQ(dst, src Reg) {
	a.rex(true, src, noIndex, dst)
	a.u8(0x31)
	a.modrmReg(src, dst)
}

// DIVQ is div r64: the unsigned 128-bit RDX:RAX divided by src, the quotient
// in RAX and the remainder in RDX.
func (a *Buf) DIVQ(src Reg) {
	a.rex(true, Y0, noIndex, src)
	a.u8(0xF7)
	a.modrmReg(Reg(6), src) // /6 = DIV
}

// MOVLLoad is mov r32, [mem]: a 32-bit load, zero-extended into the register.
func (a *Buf) MOVLLoad(dst Reg, m Mem) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.rex(false, dst, idx, m.Base)
	a.u8(0x8B)
	a.modrmMem(dst, m)
}

// MOVLStore is mov [mem], r32: the register's low 32 bits.
func (a *Buf) MOVLStore(m Mem, src Reg) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.rex(false, src, idx, m.Base)
	a.u8(0x89)
	a.modrmMem(src, m)
}

// LDRw is ldr wt, [xn, #off]: a 32-bit load, zero-extended into xt.
func (a *A64) LDRw(rt, rn XReg, off int32) {
	a.w(0xB9400000 | scaled(off, 4, "ldr w") | uint32(rn)<<5 | uint32(rt))
}

// STRw is str wt, [xn, #off]: the register's low 32 bits.
func (a *A64) STRw(rt, rn XReg, off int32) {
	a.w(0xB9000000 | scaled(off, 4, "str w") | uint32(rn)<<5 | uint32(rt))
}

// MULx is mul xd, xn, xm (madd with xzr): the low 64 bits of the product.
func (a *A64) MULx(rd, rn, rm XReg) {
	a.w(0x9B007C00 | uint32(rm)<<16 | uint32(rn)<<5 | uint32(rd))
}

// EORx is eor xd, xn, xm.
func (a *A64) EORx(rd, rn, rm XReg) {
	a.w(0xCA000000 | uint32(rm)<<16 | uint32(rn)<<5 | uint32(rd))
}

// UDIVx is udiv xd, xn, xm: the unsigned quotient.
func (a *A64) UDIVx(rd, rn, rm XReg) {
	a.w(0x9AC00800 | uint32(rm)<<16 | uint32(rn)<<5 | uint32(rd))
}

// MSUBx is msub xd, xn, xm, xa: xa - xn*xm, the remainder after UDIVx.
func (a *A64) MSUBx(rd, rn, rm, ra XReg) {
	a.w(0x9B008000 | uint32(rm)<<16 | uint32(ra)<<10 | uint32(rn)<<5 | uint32(rd))
}
