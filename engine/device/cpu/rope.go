//go:build amd64

package cpu

import "fmt"

// EmitRoPE generates the rotary rotation of a run of heads by one position's
// {cos, sin} table.
//
// The table stays a host table; only its application is generated. See
// nn.Rope.Table for why the host owns the angles.
//
// The two layouts are two kernels, baked:
//
//	NEOX  pair p is (x[p], x[p+nrot/2]). The table is deinterleaved in
//	      registers -- VSHUFPS gathers the even and odd lanes of two loads, and
//	      VPERMQ undoes VSHUFPS's per-128-bit-half order.
//	NORM  pair p is (x[2p], x[2p+1]). VMOVSLDUP/VMOVSHDUP spread cos and sin
//	      over each pair, VPERMILPS swaps the pair, and VADDSUBPS supplies the
//	      rotation's sign pattern: subtract in the even lane, add in the odd.
//
// hd and nrot are baked, so a pair count that is not a whole vector finishes
// in unrolled scalar code, and the dimensions past nrot -- partial rotary --
// are never touched.
//
//	Out     the first head, read and written
//	AScale  the table: nrot floats, {cos, sin} per pair
//	Rows    how many heads, each hd floats after the last
func EmitRoPE(hd, nrot int, neox bool) ([]byte, error) {
	if nrot <= 0 || nrot%2 != 0 || nrot > hd {
		return nil, fmt.Errorf("jit: EmitRoPE: nrot=%d must be even and in (0, hd=%d]", nrot, hd)
	}
	half := nrot / 2
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out: head 0
	a.MOVLoad(RDX, At(RDI, 24)) // AScale: the table
	a.MOVLoad(RAX, At(RDI, 32)) // Rows: heads
	done := a.Label()
	a.TESTQ(RAX, RAX)
	a.JZ(done)

	// rot writes (x0*c - x1*s, x0*s + x1*c) into Y6, Y7 from x0 Y4, x1 Y5,
	// c Y2, s Y3.
	rot := func() {
		a.VMULPS(Y6, Y4, Y2)
		a.VFNMADD231PS(Y6, Y5, Y3)
		a.VMULPS(Y7, Y4, Y3)
		a.VFMADD231PS(Y7, Y5, Y2)
	}
	head := a.Label()
	a.Bind(head)
	if neox {
		g := half / 8
		for i := 0; i < g; i++ {
			p := int32(8 * i)
			a.VMOVDQULoad(Y0, At(RDX, 8*p))
			a.VMOVDQULoad(Y1, At(RDX, 8*p+32))
			a.VSHUFPS(Y2, Y0, Y1, 0x88) // c0 c1 c4 c5 | c2 c3 c6 c7
			a.VPERMQ(Y2, Y2, 0xD8)      // c0 .. c7
			a.VSHUFPS(Y3, Y0, Y1, 0xDD)
			a.VPERMQ(Y3, Y3, 0xD8) // s0 .. s7
			a.VMOVDQULoad(Y4, At(RCX, 4*p))
			a.VMOVDQULoad(Y5, At(RCX, 4*(p+int32(half))))
			rot()
			a.VMOVDQUStore(At(RCX, 4*p), Y6)
			a.VMOVDQUStore(At(RCX, 4*(p+int32(half))), Y7)
		}
		for p := int32(8 * g); p < int32(half); p++ {
			a.VMOVSSLoad(Y2, At(RDX, 8*p))
			a.VMOVSSLoad(Y3, At(RDX, 8*p+4))
			a.VMOVSSLoad(Y4, At(RCX, 4*p))
			a.VMOVSSLoad(Y5, At(RCX, 4*(p+int32(half))))
			rot()
			a.VMOVSSStore(At(RCX, 4*p), Y6)
			a.VMOVSSStore(At(RCX, 4*(p+int32(half))), Y7)
		}
	} else {
		// Four pairs are eight floats, and the head and the table advance in
		// step: pair p is bytes 8p of both.
		g := half / 4
		for i := 0; i < g; i++ {
			off := int32(32 * i)
			a.VMOVDQULoad(Y0, At(RDX, off))
			a.VMOVSLDUP(Y2, Y0) // c0 c0 c1 c1 ...
			a.VMOVSHDUP(Y3, Y0) // s0 s0 s1 s1 ...
			a.VMOVDQULoad(Y4, At(RCX, off))
			a.VPERMILPS(Y5, Y4, 0xB1) // x1 x0 x3 x2 ...
			a.VMULPS(Y4, Y4, Y2)
			a.VMULPS(Y5, Y5, Y3)
			a.VADDSUBPS(Y4, Y4, Y5)
			a.VMOVDQUStore(At(RCX, off), Y4)
		}
		for p := int32(4 * g); p < int32(half); p++ {
			a.VMOVSSLoad(Y2, At(RDX, 8*p))
			a.VMOVSSLoad(Y3, At(RDX, 8*p+4))
			a.VMOVSSLoad(Y4, At(RCX, 8*p))
			a.VMOVSSLoad(Y5, At(RCX, 8*p+4))
			rot()
			a.VMOVSSStore(At(RCX, 8*p), Y6)
			a.VMOVSSStore(At(RCX, 8*p+4), Y7)
		}
	}
	a.ADDimm(RCX, int32(hd*4))
	a.DEC(RAX)
	a.JNZ(head)

	a.Bind(done)
	a.VZEROUPPER()
	a.RET()
	return a.Bytes(), nil
}

// EmitRoPESplit generates XD-RoPE's rotation (HunyuanVL): NEOX-paired, but
// each pair's two halves turn by angles of their own -- transformers'
// recomposition_frequencies assigns a position axis per ELEMENT of
// cat(freq, freq), so on an image's rows x[p] and x[p+nrot/2] take different
// coordinates. The table is two pair tables, nrot floats each: A for the
// first half and B for the second, so
//
//	x[p]        = x[p]*cA - x[p+h]*sA
//	x[p+h]      = x[p]*sB + x[p+h]*cB
//
// which is EmitRoPE's NEOX rotation, to the bit, when A equals B.
//
//	Out     the first head, read and written
//	AScale  the tables: 2*nrot floats, A's {cos, sin} per pair then B's
//	Rows    how many heads, each hd floats after the last
func EmitRoPESplit(hd, nrot int) ([]byte, error) {
	if nrot <= 0 || nrot%2 != 0 || nrot > hd {
		return nil, fmt.Errorf("jit: EmitRoPESplit: nrot=%d must be even and in (0, hd=%d]", nrot, hd)
	}
	half := nrot / 2
	tb := int32(4 * nrot) // table B, after A
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out: head 0
	a.MOVLoad(RDX, At(RDI, 24)) // AScale: the tables
	a.MOVLoad(RAX, At(RDI, 32)) // Rows: heads
	done := a.Label()
	a.TESTQ(RAX, RAX)
	a.JZ(done)

	// rot writes (x0*cA - x1*sA, x0*sB + x1*cB) into Y6, Y7 from x0 Y4, x1
	// Y5, cA Y2, sA Y3, cB Y8, sB Y9.
	rot := func() {
		a.VMULPS(Y6, Y4, Y2)
		a.VFNMADD231PS(Y6, Y5, Y3)
		a.VMULPS(Y7, Y4, Y9)
		a.VFMADD231PS(Y7, Y5, Y8)
	}
	head := a.Label()
	a.Bind(head)
	g := half / 8
	for i := 0; i < g; i++ {
		p := int32(8 * i)
		for _, t := range []struct {
			off  int32
			c, s Reg
		}{{0, Y2, Y3}, {tb, Y8, Y9}} {
			a.VMOVDQULoad(Y0, At(RDX, t.off+8*p))
			a.VMOVDQULoad(Y1, At(RDX, t.off+8*p+32))
			a.VSHUFPS(t.c, Y0, Y1, 0x88)
			a.VPERMQ(t.c, t.c, 0xD8)
			a.VSHUFPS(t.s, Y0, Y1, 0xDD)
			a.VPERMQ(t.s, t.s, 0xD8)
		}
		a.VMOVDQULoad(Y4, At(RCX, 4*p))
		a.VMOVDQULoad(Y5, At(RCX, 4*(p+int32(half))))
		rot()
		a.VMOVDQUStore(At(RCX, 4*p), Y6)
		a.VMOVDQUStore(At(RCX, 4*(p+int32(half))), Y7)
	}
	for p := int32(8 * g); p < int32(half); p++ {
		a.VMOVSSLoad(Y2, At(RDX, 8*p))
		a.VMOVSSLoad(Y3, At(RDX, 8*p+4))
		a.VMOVSSLoad(Y8, At(RDX, tb+8*p))
		a.VMOVSSLoad(Y9, At(RDX, tb+8*p+4))
		a.VMOVSSLoad(Y4, At(RCX, 4*p))
		a.VMOVSSLoad(Y5, At(RCX, 4*(p+int32(half))))
		rot()
		a.VMOVSSStore(At(RCX, 4*p), Y6)
		a.VMOVSSStore(At(RCX, 4*(p+int32(half))), Y7)
	}
	a.ADDimm(RCX, int32(hd*4))
	a.DEC(RAX)
	a.JNZ(head)

	a.Bind(done)
	a.VZEROUPPER()
	a.RET()
	return a.Bytes(), nil
}
