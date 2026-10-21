//go:build arm64

package cpu

// EmitLn is the NEON twin of the amd64 EmitLn: x = ln(x) in place, Scr
// ActConsts().
func EmitLn() []byte {
	var a A64
	a.LDRx(X1, X0, 0)  // Out -> x, in place
	a.LDRx(X5, X0, 56) // Scr -> consts
	a.LDRx(X4, X0, 40) // K -> whole vectors
	a.LDRx(X6, X0, 32) // Rows -> tail elements
	loadA64ExpConsts(&a)
	a64ElemLoop(&a, X4, X6, []XReg{X1}, func(ld func(VReg, XReg), st func(XReg, VReg)) {
		ld(4, X1)
		a.MOVIzero(3)
		emitA64LnSum(&a)
		st(X1, 3)
	})
	a.RET()
	return a.Bytes()
}
