package cpu

import "fmt"

// EmitRoPESplitSSE is EmitRoPESplit (rope.go) on legacy SSE: XD-RoPE's
// NEOX rotation whose pair halves take tables of their own, A then B, nrot
// floats each, with EmitRoPESSE's arithmetic (no FMA) on both halves -- its
// NEOX rotation to the bit when A equals B.
func EmitRoPESplitSSE(hd, nrot int) ([]byte, error) {
	if nrot <= 0 || nrot%2 != 0 || nrot > hd {
		return nil, fmt.Errorf("jit: EmitRoPESplitSSE: nrot=%d must be even and in (0, hd=%d]", nrot, hd)
	}
	half := nrot / 2
	tb := int32(4 * nrot) // table B, after A
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))  // Out: head 0
	a.MOVLoad(RDX, At(RDI, 24)) // AScale: the tables
	a.MOVLoad(RAX, At(RDI, 32)) // Rows: heads
	done := a.Label()
	a.TESTQ(RAX, RAX)
	a.JZ(done)

	// rot writes (x0*cA - x1*sA, x0*sB + x1*cB) into XMM6, XMM7 from x0
	// XMM4, x1 XMM5, cA XMM2, sA XMM3, cB XMM9, sB XMM10, with XMM8 as the
	// product temporary.
	rot := func() {
		a.MULPS(XMM6, XMM4, XMM2)
		a.MULPS(XMM8, XMM5, XMM3)
		a.SUBPS(XMM6, XMM6, XMM8)
		a.MULPS(XMM7, XMM4, XMM10)
		a.MULPS(XMM8, XMM5, XMM9)
		a.ADDPS(XMM7, XMM7, XMM8)
	}
	head := a.Label()
	a.Bind(head)
	hb := int32(4 * half) // x[p+half] is hb bytes after x[p]
	g := half / 4
	for i := 0; i < g; i++ {
		p := int32(4 * i)
		for _, t := range []struct {
			off  int32
			c, s Reg
		}{{0, XMM2, XMM3}, {tb, XMM9, XMM10}} {
			a.MOVUPSLoad(XMM0, At(RDX, t.off+8*p))
			a.MOVUPSLoad(XMM1, At(RDX, t.off+8*p+16))
			a.SHUFPS(t.c, XMM0, XMM1, 0x88)
			a.SHUFPS(t.s, XMM0, XMM1, 0xDD)
		}
		a.MOVUPSLoad(XMM4, At(RCX, 4*p))
		a.MOVUPSLoad(XMM5, At(RCX, 4*p+hb))
		rot()
		a.MOVUPSStore(At(RCX, 4*p), XMM6)
		a.MOVUPSStore(At(RCX, 4*p+hb), XMM7)
	}
	for p := int32(4 * g); p < int32(half); p++ {
		a.MOVSSLoad(XMM2, At(RDX, 8*p))
		a.MOVSSLoad(XMM3, At(RDX, 8*p+4))
		a.MOVSSLoad(XMM9, At(RDX, tb+8*p))
		a.MOVSSLoad(XMM10, At(RDX, tb+8*p+4))
		a.MOVSSLoad(XMM4, At(RCX, 4*p))
		a.MOVSSLoad(XMM5, At(RCX, 4*p+hb))
		rot()
		a.MOVSSStore(At(RCX, 4*p), XMM6)
		a.MOVSSStore(At(RCX, 4*p+hb), XMM7)
	}
	a.ADDimm(RCX, int32(hd*4))
	a.DEC(RAX)
	a.JNZ(head)

	a.Bind(done)
	a.RET()
	return a.Bytes(), nil
}
