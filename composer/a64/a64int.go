package a64

// The 64-bit integer instructions the n-gram hash needs: the only kernel
// whose arithmetic is on integers.

// LDRw is ldr wt, [xn, #off]: a 32-bit load, zero-extended into xt.
func (a *A64) LDRw(rt, rn XReg, off int32) {
	a.w(0xB9400000 | a.scaled(off, 4, "ldr w") | uint32(rn)<<5 | uint32(rt))
}

// STRw is str wt, [xn, #off]: the register's low 32 bits.
func (a *A64) STRw(rt, rn XReg, off int32) {
	a.w(0xB9000000 | a.scaled(off, 4, "str w") | uint32(rn)<<5 | uint32(rt))
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
