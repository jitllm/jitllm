//go:build amd64

package cpu

// EmitLn generates x = ln(x) in place over a runtime length, for x positive
// and finite: the log-softmax's ln (emitLnSum), every lane. Scr is
// ActConsts(). A log-mel feature is its one caller (d1-omni's audio front end,
// whose 2^-24 floor keeps every input normal).
func EmitLn() []byte {
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out -> x, in place
	a.MOVLoad(RBX, At(RDI, 56)) // Scr -> consts
	a.MOVLoad(R10, At(RDI, 40)) // K -> whole vectors
	a.MOVLoad(R11, At(RDI, 32)) // Rows -> tail elements
	elemLoop(&a, R10, R11, []Reg{RCX}, func(ld func(Reg, Reg), st func(Reg, Reg)) {
		ld(Y4, RCX)
		a.VPXOR(Y3, Y3, Y3)
		emitLnSum(&a)
		st(RCX, Y3)
	})
	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}
