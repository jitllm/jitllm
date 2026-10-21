//go:build arm64

package cpu

import "fmt"

// EmitRoPE is the NEON twin of the amd64 rotary kernel; see rope.go for the
// contract. LD2 does in one instruction what amd64 needs VSHUFPS and VPERMQ
// for: it deinterleaves the {cos, sin} table, and for the NORM layout it
// splits the head's adjacent pairs as well, so both layouts are the same four
// fused operations on separated operands. The last nrot/2 % 4 pairs are baked
// scalar code.
func EmitRoPE(hd, nrot int, neox bool) ([]byte, error) {
	if nrot <= 0 || nrot%2 != 0 || nrot > hd {
		return nil, fmt.Errorf("jit: EmitRoPE: nrot=%d must be even and in (0, hd=%d]", nrot, hd)
	}
	half := nrot / 2
	g := half / 4
	var a A64
	a.LDRx(X1, X0, 0)  // Out: head 0
	a.LDRx(X2, X0, 24) // AScale: the table
	a.LDRx(X4, X0, 32) // Rows: heads
	done := a.Label()
	a.CBZ(X4, done)

	// rot writes (x0*c - x1*s, x0*s + x1*c) into v6, v7 from x0 v4, x1 v5,
	// c v2, s v3.
	rot := func() {
		a.FMUL4s(6, 4, 2)
		a.FMLS4s(6, 5, 3)
		a.FMUL4s(7, 4, 3)
		a.FMLA4s(7, 5, 2)
	}
	head := a.Label()
	a.Bind(head)
	a.MOVreg(X5, X2) // table cursor
	a.MOVreg(X6, X1) // head cursor
	if neox {
		a.MOVreg(X7, X1) // the second half's cursor
		a.addBig(X7, int32(half*4), X9)
		for i := 0; i < g; i++ {
			a.LD2Post(2, X5)
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
			a.LDRs(4, X6, 4*i)
			a.LDRs(5, X7, 4*i)
			rot()
			a.STRs(6, X6, 4*i)
			a.STRs(7, X7, 4*i)
		}
	} else {
		for i := 0; i < g; i++ {
			a.LD2Post(2, X5)
			a.LD2(4, X6)
			rot()
			a.ST2Post(6, X6)
		}
		for i := int32(0); i < int32(half%4); i++ {
			a.LDRs(2, X5, 8*i)
			a.LDRs(3, X5, 8*i+4)
			a.LDRs(4, X6, 8*i)
			a.LDRs(5, X6, 8*i+4)
			rot()
			a.STRs(6, X6, 8*i)
			a.STRs(7, X6, 8*i+4)
		}
	}
	a.addBig(X1, int32(hd*4), X9)
	a.SUBimm(X4, X4, 1)
	a.CBNZ(X4, head)

	a.Bind(done)
	a.RET()
	return a.Bytes(), nil
}
