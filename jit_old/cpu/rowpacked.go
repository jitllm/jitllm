//go:build amd64

package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// EmitPackedRow generates the dequantization of one row of a packed tensor:
// dst[e] = scale*q[e] - bias, the embedding lookup.
//
// One emitter serves every packed format. The row's words sit RowStr bytes
// apart (the layout is [unit][row]), so a sub-block's words are gathered into
// scratch first; from there the formats differ only in facts kernels.Layout
// and friends expose, each one or two vector instructions on eight elements.
//
//	Out      dst, k floats
//	W        the row's first payload word (&QS[row*4])
//	RowStr   the tensor's row count times four: the step between units
//	PD       the row's d for super-block 0 (the right half of a paired word)
//	DStr     the step between super-blocks in the d plane
//	PSC      the row's first scale word (&SC[row*4]), when the format has one
//	K        super-blocks
//	Scr      PackedRowConsts(q)
//	Scratch  64 bytes of the caller's
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
	perWord, scStride := 4, 1 // scale bytes per word, bytes between them
	if biasArray {
		perWord, scStride = 2, 2
	}

	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))      // Out
	a.MOVLoad(RDX, At(RDI, 8))      // W: payload word cursor, one sub-block at a time
	a.MOVLoad(R8, At(RDI, 48))      // RowStr
	a.MOVLoad(RBX, At(RDI, 56))     // Scr
	a.MOVLoad(R9, At(RDI, 144))     // PD
	a.MOVLoad(R12, At(RDI, 160))    // DStr
	a.MOVLoad(R10, At(RDI, 152))    // PSC
	a.MOVLoad(R11, At(RDI, 40))     // K: super-blocks
	a.MOVLoad(RAX, At(RDI, 72))     // Scratch
	a.VPBROADCASTD(Y8, At(RBX, 16)) // 0x0F0F0F0F
	a.VPBROADCASTD(Y9, At(RBX, 20)) // 0xFF
	if codes {
		a.VBROADCASTI128(Y10, At(RBX, 0))
	}
	if hi > 0 {
		a.VPBROADCASTD(Y11, At(RBX, 32)) // (1<<hi)-1
	}

	sup := a.Label()
	a.Bind(sup)
	for j := 0; j < perSuper; j++ {
		// The scale. d is the super-block's f16, and a k-quant multiplies it by
		// this sub-block's byte plus the format's offset.
		a.VPXOR(Y6, Y6, Y6)
		if E8M0Quant(q) {
			// MXFP4's E8M0 byte shifted left 23 is the f32. VPINSRB rather
			// than VPMOVZXBD because this reads exactly one row's scale, and
			// eight bytes would run off the end on a tensor's last row.
			a.VPINSRBLoad(Y6, Y6, At(R9, 0), 0)
			a.VPSLLD(Y6, Y6, 23)
		} else if kernels.D32(q) {
			a.VMOVSSLoad(Y6, At(R9, 0)) // TQ2_F32: the word is the scale
		} else {
			a.VPINSRWLoad(Y6, Y6, At(R9, 0), 0)
			a.VCVTPH2PSReg(Y6, Y6)
		}
		if biasArray {
			a.VPXOR(Y7, Y7, Y7)
			a.VPINSRWLoad(Y7, Y7, At(R9, 2), 0)
			a.VCVTPH2PSReg(Y7, Y7) // dmin
		}
		if kernels.ScStream(q) {
			// Container v27's stream (scstream.go); the next word is RowStr on.
			next := func(r Reg) { a.VMOVSSLoad(r, Idx(R10, R8, 1, 0)) }
			a.VMOVSSLoad(Y1, At(R10, 0))
			scField(avxSC(&a), Y2, Y1, Y4, j, false, next)
			a.VCVTDQ2PS(Y2, Y2)
			a.VBROADCASTSS(Y3, At(RBX, 24)) // scOff
			a.VADDPS(Y2, Y2, Y3)
			a.VMULPS(Y6, Y6, Y2)
			scField(avxSC(&a), Y1, Y1, Y4, j, true, next)
			a.VCVTDQ2PS(Y1, Y1)
			a.VMULPS(Y7, Y7, Y1) // bias = dmin * min
			if scStep(j) {
				a.ADDQ(R10, R8)
			}
		} else if perSuper > 1 {
			a.VMOVSSLoad(Y1, At(R10, 0))
			if sh := (j % perWord) * scStride * 8; sh > 0 {
				a.VPSRLD(Y1, Y1, byte(sh))
			}
			a.VPAND(Y2, Y1, Y9)
			a.VCVTDQ2PS(Y2, Y2)
			a.VBROADCASTSS(Y3, At(RBX, 24)) // scOff
			a.VADDPS(Y2, Y2, Y3)
			a.VMULPS(Y6, Y6, Y2)
			if biasArray {
				a.VPSRLD(Y1, Y1, 8)
				a.VPAND(Y1, Y1, Y9)
				a.VCVTDQ2PS(Y1, Y1)
				a.VMULPS(Y7, Y7, Y1) // bias = dmin * min
			}
			if j%perWord == perWord-1 {
				a.ADDQ(R10, R8)
			}
		}
		if !biasArray {
			a.VBROADCASTSS(Y3, At(RBX, 28)) // biasK
			a.VMULPS(Y7, Y6, Y3)
		}
		a.VBROADCASTSSReg(Y6, Y6)
		a.VBROADCASTSSReg(Y7, Y7)

		// Gather the sub-block's words: pw payload, then hw secondary.
		a.MOVQ(RSI, RDX)
		for w := 0; w < pw+hw; w++ {
			a.VMOVSSLoad(Y0, At(RSI, 0))
			a.VMOVSSStore(At(RAX, int32(4*w)), Y0)
			a.ADDQ(RSI, R8)
		}
		a.MOVQ(RDX, RSI)
		src := RAX
		if bits == 4 {
			// Byte b of word w holds element 4w+b low and 4w+b+sub/2 high.
			a.VMOVDQULoad(Y0, At(RAX, 0))
			a.VPAND(Y1, Y0, Y8)
			a.VPSRLW(Y2, Y0, 4)
			a.VPAND(Y2, Y2, Y8)
			if codes {
				a.VPSHUFB(Y1, Y10, Y1)
				a.VPSHUFB(Y2, Y10, Y2)
			}
			a.VMOVDQUStore(At(RAX, 32), Y1)
			a.VMOVDQUStore(At(RAX, int32(32+sub/2)), Y2)
		}
		base := int32(0)
		if bits == 4 {
			base = 32
		}
		if bits == 2 {
			// TQ2: a 32-wide sub-block is two words, and its eight-element
			// group g is field g of all eight bytes (packSub), so each group
			// widens the same eight bytes and shifts its own field down.
			a.VPSRLD(Y5, Y9, 6) // 3 in every dword
		}
		for g := 0; g < sub/8; g++ {
			switch {
			case bits == 2:
				a.VPMOVZXBD(Y1, At(src, 0))
				if g > 0 {
					a.VPSRLD(Y1, Y1, byte(2*g))
				}
				a.VPAND(Y1, Y1, Y5)
			case signed:
				a.VPMOVSXBD(Y1, At(src, base+int32(8*g)))
			default:
				a.VPMOVZXBD(Y1, At(src, base+int32(8*g)))
			}
			if hi > 0 {
				// Element l's secondary bits: word pw + (l/4)/(8/hi), shifted
				// by 8*(l%4) + hi*((l/4)%(8/hi)) -- a per-lane shift, which
				// VPSRLVD takes from the table in Scr.
				lanes := 8 / hi
				wd := pw + (8*g/4)/lanes
				a.VPBROADCASTD(Y2, At(RAX, int32(4*wd)))
				a.VMOVDQULoad(Y3, At(RBX, int32(64+32*g)))
				a.VPSRLVD(Y2, Y2, Y3)
				a.VPAND(Y2, Y2, Y11)
				a.VPSLLD(Y2, Y2, 4)
				a.VPOR(Y1, Y1, Y2)
			}
			a.VCVTDQ2PS(Y1, Y1)
			a.VMULPS(Y1, Y1, Y6)
			a.VSUBPS(Y1, Y1, Y7)
			a.VMOVDQUStore(At(RCX, int32(32*(g+j*sub/8))), Y1)
		}
	}
	a.ADDimm(RCX, int32(4*sub*perSuper))
	a.ADDQ(R9, R12)
	a.DEC(R11)
	a.JNZ(sup)

	a.VZEROUPPER()
	a.RET()
	return a.Bytes(), nil
}

// EmitWiden generates dst[i] = float32(src[i]) for binary16 (F16) or bfloat16
// (BF16) src at a runtime length: Args.K whole vectors of eight and Args.Rows
// single elements. It is the embedding lookup for a row-major half-precision
// table -- the one kind of embedding that is not packed.
//
//	Out  dst    W  src
func EmitWiden(bf16 bool) []byte {
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(RDX, At(RDI, 8))  // W
	a.MOVLoad(R10, At(RDI, 40)) // K
	a.MOVLoad(R11, At(RDI, 32)) // Rows
	vec := func() {
		if bf16 {
			a.VPMOVZXWD(Y0, At(RDX, 0))
			a.VPSLLD(Y0, Y0, 16)
		} else {
			a.VCVTPH2PS(Y0, At(RDX, 0))
		}
		a.VMOVDQUStore(At(RCX, 0), Y0)
	}
	one := func() {
		a.VPXOR(Y0, Y0, Y0)
		a.VPINSRWLoad(Y0, Y0, At(RDX, 0), 0)
		if bf16 {
			a.VPSLLD(Y0, Y0, 16)
		} else {
			a.VCVTPH2PSReg(Y0, Y0)
		}
		a.VMOVSSStore(At(RCX, 0), Y0)
	}
	for _, l := range []struct {
		n          Reg
		body       func()
		dst, srcSt int32
	}{{R10, vec, 32, 16}, {R11, one, 4, 2}} {
		skip := a.Label()
		a.TESTQ(l.n, l.n)
		a.JZ(skip)
		lp := a.Label()
		a.Bind(lp)
		l.body()
		a.ADDimm(RCX, l.dst)
		a.ADDimm(RDX, l.srcSt)
		a.DEC(l.n)
		a.JNZ(lp)
		a.Bind(skip)
	}
	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}
