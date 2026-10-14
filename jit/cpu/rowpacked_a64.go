//go:build arm64

package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// EmitPackedRow is the NEON twin of the amd64 row dequantization; see
// rowpacked.go for the contract. Four elements per vector; the secondary
// plane's per-lane shift is USHL by the NEGATED table, since NEON has only a
// variable left shift and a negative count shifts right.
func EmitPackedRow(q kernels.Quant) ([]byte, error) {
	sub, bits, _, biasArray := kernels.Layout(q)
	hi := kernels.HiPlane(q)
	perSuper, _ := kernels.ScaleLayout(q)
	signed := kernels.SignedPayload(q)
	codes := kernels.Codes(q) != nil
	if sub%8 != 0 || (bits != 2 && bits != 4 && bits != 8) || perSuper < 1 || bits == 2 && (sub != 32 || hi != 0 || codes) {
		return nil, fmt.Errorf("jit: EmitPackedRow: %v has no row kernel shape", q)
	}
	pw, hw := sub*bits/32, sub*hi/32
	perWord, scStride := 4, 1
	if biasArray {
		perWord, scStride = 2, 2
	}

	var a A64
	a.LDRx(X1, X0, 0)         // Out
	a.LDRx(X2, X0, 8)         // W
	a.LDRx(X8, X0, 48)        // RowStr
	a.LDRx(X5, X0, 56)        // Scr
	a.LDRx(X9, X0, 144)       // PD
	a.LDRx(XReg(12), X0, 160) // DStr
	a.LDRx(X10, X0, 152)      // PSC
	a.LDRx(XReg(11), X0, 40)  // K: super-blocks
	a.LDRx(XReg(13), X0, 72)  // Scratch
	for _, c := range []struct {
		v   VReg
		off int32
	}{{8, 16}, {9, 20}, {3, 24}, {4, 28}, {11, 32}} {
		a.LDRs(c.v, X5, c.off)
		a.DUPs4(c.v, c.v)
	}
	if codes {
		a.LDRq(10, X5, 0)
	}
	scr := XReg(13)

	sup := a.Label()
	a.Bind(sup)
	for j := 0; j < perSuper; j++ {
		if E8M0Quant(q) {
			// MXFP4's E8M0 byte shifted left 23 is the f32. LDRb rather
			// than LDRs because this reads exactly one row's scale, and
			// four bytes would run off the end on a tensor's last row.
			a.LDRb(6, X9, 0)
			a.UXTL8h(6, 6)
			a.UXTL4s(6, 6)
			a.SHL4s(6, 6, 23)
		} else if kernels.D32(q) {
			a.LDRs(6, X9, 0) // TQ2_F32: the word is the scale
		} else {
			a.LDRh(6, X9, 0)
			a.FCVTL(6, 6)
		}
		if biasArray {
			a.LDRh(7, X9, 2)
			a.FCVTL(7, 7) // dmin
		}
		if kernels.ScStream(q) {
			// Container v27's SC stream (scstream.go); X3 addresses the next
			// word, RowStr on, and V5 is the straddle's scratch.
			next := func(r VReg) {
				a.ADDreg(X3, X10, X8)
				a.LDRs(r, X3, 0)
			}
			a.LDRs(1, X10, 0)
			scField(a64SC(&a), 2, 1, 5, j, false, next)
			a.SCVTF4s(2, 2)
			a.FADD4s(2, 2, 3) // + scOff
			a.FMUL4s(6, 6, 2)
			scField(a64SC(&a), 1, 1, 5, j, true, next)
			a.SCVTF4s(1, 1)
			a.FMUL4s(7, 7, 1) // bias = dmin * min
			if scStep(j) {
				a.ADDreg(X10, X10, X8)
			}
		} else if perSuper > 1 {
			a.LDRs(1, X10, 0)
			if sh := (j % perWord) * scStride * 8; sh > 0 {
				a.USHR4s(1, 1, uint8(sh))
			}
			a.AND16b(2, 1, 9)
			a.SCVTF4s(2, 2)
			a.FADD4s(2, 2, 3) // + scOff
			a.FMUL4s(6, 6, 2)
			if biasArray {
				a.USHR4s(1, 1, 8)
				a.AND16b(1, 1, 9)
				a.SCVTF4s(1, 1)
				a.FMUL4s(7, 7, 1) // bias = dmin * min
			}
			if j%perWord == perWord-1 {
				a.ADDreg(X10, X10, X8)
			}
		}
		if !biasArray {
			a.FMUL4s(7, 6, 4) // bias = scale * biasK
		}
		a.DUPs4(6, 6)
		a.DUPs4(7, 7)

		a.MOVreg(XReg(14), X2)
		for w := 0; w < pw+hw; w++ {
			a.LDRs(0, XReg(14), 0)
			a.STRs(0, scr, int32(4*w))
			a.ADDreg(XReg(14), XReg(14), X8)
		}
		a.MOVreg(X2, XReg(14))
		base := int32(0)
		if bits == 2 {
			// TQ2: the sub-block's eight bytes hold four fields each (packSub);
			// field f of byte p is element 8f+p, so the four fields go to
			// scratch as four eight-byte runs, element order.
			a.LDRd(0, scr, 0)
			a.MOVI16b(3, 0x03)
			for fi := 0; fi < 4; fi++ {
				if fi == 0 {
					a.AND16b(1, 0, 3)
				} else {
					a.USHR16b(1, 0, byte(2*fi))
					a.AND16b(1, 1, 3)
				}
				a.STRd(1, scr, int32(32+8*fi))
			}
			base = 32
		}
		if bits == 4 {
			a.LDRq(0, scr, 0)
			a.AND16b(1, 0, 8)
			a.USHR16b(2, 0, 4)
			if codes {
				a.TBL(1, 10, 1)
				a.TBL(2, 10, 2)
			}
			if sub == 32 {
				a.STRq(1, scr, 32)
				a.STRq(2, scr, 48)
			} else {
				a.STRd(1, scr, 32)
				a.STRd(2, scr, int32(32+sub/2))
			}
			base = 32
		}
		for g := 0; g < sub/4; g++ {
			a.LDRs(1, scr, base+int32(4*g))
			if signed {
				a.SXTL8h(1, 1)
				a.SXTL4s(1, 1)
			} else {
				a.UXTL8h(1, 1)
				a.UXTL4s(1, 1)
			}
			if hi > 0 {
				lanes := 8 / hi
				wd := pw + g/lanes // g is l/4 for this group's elements
				a.LDRs(2, scr, int32(4*wd))
				a.DUPs4(2, 2)
				a.LDRq(5, X5, int32(64+4*sub+16*g)) // negated shifts
				a.USHL4s(2, 2, 5)
				a.AND16b(2, 2, 11)
				a.SHL4s(2, 2, 4)
				a.ORR16b(1, 1, 2)
			}
			a.SCVTF4s(1, 1)
			a.FMUL4s(1, 1, 6)
			a.FSUB4s(1, 1, 7)
			a.STRq(1, X1, int32(16*(g+j*sub/4)))
		}
	}
	a.addBig(X1, int32(4*sub*perSuper), XReg(15))
	a.ADDreg(X9, X9, XReg(12))
	a.SUBimm(XReg(11), XReg(11), 1)
	a.CBNZ(XReg(11), sup)
	a.RET()
	return a.Bytes(), nil
}

// EmitWiden is the NEON twin of the amd64 widen: Args.K whole quads and
// Args.Rows single elements of F16 or BF16 into float32.
func EmitWiden(bf16 bool) []byte {
	var a A64
	a.LDRx(X1, X0, 0)  // Out
	a.LDRx(X2, X0, 8)  // W
	a.LDRx(X4, X0, 40) // K
	a.LDRx(X6, X0, 32) // Rows
	widen := func(v VReg) {
		if bf16 {
			a.SHLL(v, v)
		} else {
			a.FCVTL(v, v)
		}
	}
	for _, l := range []struct {
		n        XReg
		one      bool
		dst, src int32
	}{{X4, false, 16, 8}, {X6, true, 4, 2}} {
		skip := a.Label()
		a.CBZ(l.n, skip)
		lp := a.Label()
		a.Bind(lp)
		if l.one {
			a.LDRh(0, X2, 0)
			widen(0)
			a.STRs(0, X1, 0)
		} else {
			a.LDRd(0, X2, 0)
			widen(0)
			a.STRq(0, X1, 0)
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
