//go:build amd64

package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/format/quant"
)

// The scaffolding the amd64 packed matrix emitters share: the argument loads,
// the row-group loop, the super-block scale pass, the scale-field extraction
// and the cursor steps. Each piece emits exactly the instructions its callers
// wrote out before, in their order; the register plan stays with the caller,
// which passes the registers each piece may use.
//
// The GPR plan the fused and tiled kernels share:
//
//	RDI *Args   RCX Out   R11 A   R12 AScale   R13 RowStr   R10 AHalf or Q32
//	RDX QS      R9  SC (a narrow format: the d plane's row offset)
//	RAX super-blocks left   R8 D   RBX the row counter   R15 the row offset
//	RSI the payload cursor, then the scratch base

// packedArgs loads the pointers every row group reads: Out, A, AScale,
// RowStr, Scr and, in R10, the minimum scratch for a format with one and
// AHalf otherwise. A format with a minimum array has 32-element sub-blocks
// and no per-16 sums, and a 16-wide format is the reverse.
func packedArgs(a *Buf, f *packedFormat) {
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(R11, At(RDI, 16)) // A
	a.MOVLoad(R12, At(RDI, 24)) // AScale
	a.MOVLoad(R13, At(RDI, 48)) // RowStr
	a.MOVLoad(RBX, At(RDI, 56)) // Scr, for the constants
	if f.biasArray {
		a.MOVLoad(R10, At(RDI, 112)) // Q32 -> the minimum scratch
	} else {
		a.MOVLoad(R10, At(RDI, 64)) // AHalf
	}
}

// packedConsts loads the constants a format holds for the whole kernel from
// the scratch at RBX: the scale offset, the secondary plane's pre-shifted
// mask and the code table, each into its register when the format has it.
func packedConsts(a *Buf, f *packedFormat, scOffV, hiMask, lut Reg) {
	if f.scOff != 0 {
		a.VMOVDQULoad(scOffV, At(RBX, 224+int32(scOffSlot(f.scOff))))
	}
	if f.hi != 0 {
		a.VMOVDQULoad(hiMask, At(RBX, hiMaskSlot(f.hi)))
	}
	if f.codes {
		a.VMOVDQULoad(lut, At(RBX, codesSlot))
	}
}

// packedRowLoop runs body once per row group of Args.Rows, R15 the group's
// byte offset into a row-indexed plane and RBX the counter.
func packedRowLoop(a *Buf, group int, body func()) {
	a.MOVLoad(RBX, At(RDI, 32))
	a.SUBQ(R15, R15)
	l := a.Label()
	a.Bind(l)
	body()
	a.ADDimm(R15, int32(group*4))
	a.DEC(RBX)
	a.JNZ(l)
}

// packedCursors loads the plane cursors the super-block loop walks: QS, SC
// (a narrow format has no SC plane, and R9 carries the d plane's row offset
// instead), the super-block count and D, held for the whole kernel.
func packedCursors(a *Buf, f *packedFormat) {
	a.MOVLoad(RDX, At(RDI, 8)) // QS
	if !f.narrow {
		a.MOVLoad(R9, At(RDI, 152)) // SC
	}
	a.MOVLoad(RAX, At(RDI, 40)) // K, in super-blocks
	a.MOVLoad(R8, At(RDI, 144)) // D
}

// packedScalePass converts this super-block's row scales to f32 into Scratch,
// and a format's minimums into the scratch at R10, once per super-block. pay
// holds a wide plane's words and tmp the scale; minT is the minimum's
// temporary. Both shuffle masks are loaded when bothShuf is set, the second
// otherwise only for a format with a minimum. D advances one super-block after.
func packedScalePass(a *Buf, f *packedFormat, t quant.Type, group, grp int, shufLo, shufHi, pay, tmp, minT Reg, bothShuf bool) {
	// RBX is reloaded because packedRowLoop owns it as the row counter: after
	// the first loop it is zero. R8 is not reloaded: it is the d cursor for
	// the whole kernel and advances one super-block per outer iteration.
	a.MOVLoad(RBX, At(RDI, 56))
	a.MOVLoad(RSI, At(RDI, 72)) // Scratch
	a.VMOVDQULoad(shufLo, At(RBX, 64))
	if bothShuf || f.biasArray {
		a.VMOVDQULoad(shufHi, At(RBX, 96))
	}
	if f.narrow {
		a.SUBQ(R9, R9)
	}
	packedRowLoop(a, group, func() {
		for g := 0; g < grp; g++ {
			off := int32(g * 32)
			if f.e8 {
				// MXFP4's super-scale is a biased E8M0 exponent, so the byte
				// shifted left 23 is the f32, exactly (exp.go's 2^x trick).
				a.VPMOVZXBD(tmp, Idx(R8, R9, 1, off/4))
				a.VPSLLD(tmp, tmp, 23)
			} else if f.narrow {
				// Eight f16 are sixteen bytes and VCVTPH2PS widens them in one
				// instruction, where the wide plane needs a load, a VPSHUFB and
				// a VPERMQ to gather eight halves out of eight words.
				a.VCVTPH2PS(tmp, Idx(R8, R9, 1, off/2))
			} else if f.d32 {
				// TQ2_F32: the word is the scale.
				a.VMOVDQULoad(tmp, Idx(R8, R15, 1, off))
			} else {
				a.VMOVDQULoad(pay, Idx(R8, R15, 1, off))
				a.VPSHUFB(tmp, pay, shufLo)
				a.VPERMQ(tmp, tmp, 0xD8)
				a.VCVTPH2PSReg(tmp, tmp)
			}
			a.VMOVDQUStore(Idx(RSI, R15, 1, off), tmp)
			if f.biasArray {
				a.VPSHUFB(minT, pay, shufHi)
				a.VPERMQ(minT, minT, 0xD8)
				a.VCVTPH2PSReg(minT, minT)
				a.VMOVDQUStore(Idx(R10, R15, 1, off), minT)
			}
		}
		if f.narrow {
			a.ADDimm(R9, int32(group*DRowBytes(t)))
		}
	})
	if f.narrow {
		a.ADDQmem(R8, At(RDI, 160)) // half the payload's stride; see DSuperBytes
	} else {
		a.ADDQ(R8, R13) // D advances one super-block
	}
}

// hiField extracts the secondary-plane field at bit s of each byte of src
// into dst, positioned at bit 4: the mask already carries the <<4, so the
// shift is relative to bit 4 and can go either way.
func hiField(a *Buf, dst, src, mask Reg, s int) {
	switch {
	case s < 4:
		a.VPSLLD(dst, src, byte(4-s))
		a.VPAND(dst, dst, mask)
	case s == 4:
		a.VPAND(dst, src, mask)
	default:
		a.VPSRLD(dst, src, byte(s-4))
		a.VPAND(dst, dst, mask)
	}
}

// scByte extracts the sub-block scale byte at bit sh of each packed SC word
// lane into dst as an int32.
func scByte(a *Buf, dst, word Reg, sh byte) {
	if 24-sh > 0 {
		a.VPSLLD(dst, word, 24-sh)
	} else {
		a.VMOVAPSReg(dst, word)
	}
	a.VPSRLD(dst, dst, 24)
}

// scScale extracts sub-block sn's scale from the SC word into dst: a field
// of container v27's stream (scField; borrow is its scratch and next loads a
// straddling field's next word), or the packed byte.
func scScale(a *Buf, f *packedFormat, dst, word, borrow Reg, sn int, next func(Reg)) {
	if f.stream {
		scField(avxSC(a), dst, word, borrow, sn, false, next)
	} else {
		scByte(a, dst, word, f.scShift(sn))
	}
}

// scMinByte extracts the minimum byte, eight bits above the scale's.
func scMinByte(a *Buf, dst, word Reg, sh byte) {
	a.VPSLLD(dst, word, 16-sh)
	a.VPSRLD(dst, dst, 24)
}

// packedSubStep steps the payload cursor past sub-block sn's words, and the
// SC cursor when sn was the last sub-block on its word.
func packedSubStep(a *Buf, f *packedFormat, sn int) {
	for w := 0; w < f.words; w++ {
		a.ADDQ(RDX, R13)
	}
	if f.scWordDone(sn) {
		a.ADDQ(R9, R13)
	}
}

// packedSuperStep steps the activation cursors one super-block, and SC for a
// format with one sub-block a super-block when stepSC is set.
func packedSuperStep(a *Buf, f *packedFormat, stepSC bool) {
	a.ADDimm(R11, int32(f.perSuper*f.sub))
	a.ADDimm(R12, int32(f.perSuper*f.sub/Q8Block*8))
	if f.sub16 {
		a.ADDimm(R10, int32(f.perSuper*4))
	}
	if f.perSuper == 1 && stepSC {
		a.ADDQ(R9, R13)
	}
}

// packedCode closes a kernel and holds it to the packed code budget.
func packedCode(a *Buf, who string, t quant.Type) ([]byte, error) {
	a.VZEROUPPER()
	a.RET()
	code := a.Bytes()
	if len(code) > PackedWideBudget {
		return nil, fmt.Errorf("jit: %s: %s is %d bytes, over the %d budget",
			who, t, len(code), PackedWideBudget)
	}
	return code, nil
}
