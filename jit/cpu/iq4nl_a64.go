//go:build arm64

package cpu

// EmitRowIQ4NL is the IQ4_NL row decode in NEON (iq4nl_x86.go has the
// contract): the scale widened by FCVTL and broadcast, the nibbles masked and
// shifted, looked up by TBL, sign-extended twice and converted, four at a time
// times the scale.
//
// Registers: x1 out, x2 the blocks, x4 the count, x5 the constants. v7 the
// table, v6 the nibble mask, v0 the scale, v1..v5 and v16..v19 temporaries.
func EmitRowIQ4NL() []byte {
	var a A64
	a.LDRx(X1, X0, 0)  // Out
	a.LDRx(X2, X0, 8)  // W
	a.LDRx(X4, X0, 40) // K
	a.LDRx(X5, X0, 56) // Scr
	a.LDRq(7, X5, 0)
	a.LDRq(6, X5, 16)
	done := a.Label()
	a.CBZ(X4, done)
	lp := a.Label()
	a.Bind(lp)
	a.LDRh(0, X2, 0)
	a.FCVTL(0, 0)
	a.DUPs4lane(0, 0, 0)
	a.LDURq(1, X2, 2)
	a.AND16b(2, 1, 6)  // the low nibbles: elements 0..15
	a.USHR16b(3, 1, 4) // the high: 16..31
	a.TBL(2, 7, 2)
	a.TBL(3, 7, 3)
	off := int32(0)
	for _, v := range []VReg{2, 3} {
		a.SXTL8h(4, v)
		a.SXTL2_8h(5, v)
		for _, h := range []VReg{4, 5} {
			a.SXTL4s(16, h)
			a.SXTL2_4s(17, h)
			for _, w := range []VReg{16, 17} {
				a.SCVTF4s(w, w)
				a.FMUL4s(w, w, 0)
				a.STRq(w, X1, off)
				off += 16
			}
		}
	}
	a.ADDimm(X1, X1, 128)
	a.ADDimm(X2, X2, 18)
	a.SUBimm(X4, X4, 1)
	a.CBNZ(X4, lp)
	a.Bind(done)
	a.RET()
	return a.Bytes()
}
