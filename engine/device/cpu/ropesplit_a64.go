//go:build arm64

package cpu

import "fmt"

// EmitRoPESplit is the NEON twin of the amd64 XD-RoPE kernel; see rope.go for
// the contract: NEOX pairs whose halves take tables of their own, A then B,
// nrot floats each. LD2 deinterleaves each table's {cos, sin} as EmitRoPE's
// does, and the arithmetic is its NEOX rotation's with B's angle on the
// second half -- to the bit when A equals B.
func EmitRoPESplit(hd, nrot int) ([]byte, error) {
	if nrot <= 0 || nrot%2 != 0 || nrot > hd {
		return nil, fmt.Errorf("jit: EmitRoPESplit: nrot=%d must be even and in (0, hd=%d]", nrot, hd)
	}
	half := nrot / 2
	g := half / 4
	var a A64
	a.LDRx(X1, X0, 0)  // Out: head 0
	a.LDRx(X2, X0, 24) // AScale: the tables
	a.LDRx(X4, X0, 32) // Rows: heads
	done := a.Label()
	a.CBZ(X4, done)

	// rot writes (x0*cA - x1*sA, x0*sB + x1*cB) into v6, v7 from x0 v4, x1
	// v5, cA v2, sA v3, cB v16, sB v17.
	rot := func() {
		a.FMUL4s(6, 4, 2)
		a.FMLS4s(6, 5, 3)
		a.FMUL4s(7, 4, 17)
		a.FMLA4s(7, 5, 16)
	}
	head := a.Label()
	a.Bind(head)
	a.MOVreg(X5, X2) // table A's cursor
	a.MOVreg(X10, X2)
	a.addBig(X10, int32(nrot*4), X9) // table B's
	a.MOVreg(X6, X1)                 // head cursor
	a.MOVreg(X7, X1)                 // the second half's cursor
	a.addBig(X7, int32(half*4), X9)
	for i := 0; i < g; i++ {
		a.LD2Post(2, X5)
		a.LD2Post(16, X10)
		a.LDRq(4, X6, 0)
		a.LDRq(5, X7, 0)
		rot()
		a.STRq(6, X6, 0)
		a.STRq(7, X7, 0)
		a.ADDimm(X6, X6, 16)
		a.ADDimm(X7, X7, 16)
	}
	for i := int32(0); i < int32(half%4); i++ {
		a.LDRs(2, X5, 8*i)
		a.LDRs(3, X5, 8*i+4)
		a.LDRs(16, X10, 8*i)
		a.LDRs(17, X10, 8*i+4)
		a.LDRs(4, X6, 4*i)
		a.LDRs(5, X7, 4*i)
		rot()
		a.STRs(6, X6, 4*i)
		a.STRs(7, X7, 4*i)
	}
	a.addBig(X1, int32(hd*4), X9)
	a.SUBimm(X4, X4, 1)
	a.CBNZ(X4, head)

	a.Bind(done)
	a.RET()
	return a.Bytes(), nil
}
