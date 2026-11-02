//go:build arm64

package cpu

// EmitNarrowF16 is the NEON twin of the amd64 EmitNarrowF16: Args.K whole
// quads (ElemLanes is 4 here) and Args.Rows single elements of float32 (W) narrowed to
// binary16 (Out), rounded as quant.EncodeHalf does, bit for bit. FCVTN rounds
// to nearest even under the FPCR Go runs with and quiets a NaN keeping its
// payload's top bits, which EncodeHalf does not, so they are cleared after:
//
//	h &^= (h&0x7FFF > 0x7C00) & 0x01FF
//
// Registers: X1 dst, X2 src, X4 units, X6 singles; V29..V31 the constants.
func EmitNarrowF16() []byte {
	var a A64
	a.LDRx(X1, X0, 0)  // Out
	a.LDRx(X2, X0, 8)  // W
	a.LDRx(X4, X0, 40) // K
	a.LDRx(X6, X0, 32) // Rows
	for _, c := range []struct {
		v    VReg
		bits int64
	}{{29, 0x7FFF7FFF}, {30, 0x7C007C00}, {31, 0x01FF01FF}} {
		a.MOVimm(X9, c.bits)
		a.DUPgp(c.v, X9)
	}
	fix := func(h VReg) {
		a.AND16b(2, h, 29)
		a.CMGT8h(2, 2, 30)
		a.AND16b(2, 2, 31)
		a.BIC16b(h, h, 2)
	}
	for _, l := range []struct {
		n        XReg
		one      bool
		dst, src int32
	}{{X4, false, 8, 16}, {X6, true, 2, 4}} {
		skip := a.Label()
		a.CBZ(l.n, skip)
		lp := a.Label()
		a.Bind(lp)
		if l.one {
			a.LDRs(0, X2, 0)
			a.FCVTN4h(1, 0)
			fix(1)
			a.STRh(1, X1, 0)
		} else {
			a.LDRq(0, X2, 0)
			a.FCVTN4h(1, 0)
			fix(1)
			a.STRd(1, X1, 0)
		}
		a.ADDimm(X1, X1, l.dst)
		a.ADDimm(X2, X2, l.src)
		a.SUBimm(l.n, l.n, 1)
		a.CBNZ(l.n, lp)
		a.Bind(skip)
	}
	a.RET()
	return a.Bytes()
}
