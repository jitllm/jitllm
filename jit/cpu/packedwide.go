//go:build amd64

package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// PackedWideGroup is how many rows one iteration of the wide kernel's row loop
// covers: four YMM of output.
const PackedWideGroup = 32

// PackedWideBudget caps the wide kernel's code size. The sub-block loop is
// unrolled (the scale byte's shift is an immediate), so some formats grow
// large against a 32 KB L1i the row-major kernels also live in.
const PackedWideBudget = 16 << 10

// EmitPackedMatVecWide is the packed matvec with the row loop innermost.
//
// The layout is [sub-block][word][row], so a kernel with register
// accumulators serves at most 64 rows and then jumps nrows*4 to the next
// k-position. Holding the integer accumulators in memory instead lets one
// call serve a worker's whole row range, so each stride step reads kilobytes
// contiguously (cmd/stridebench: throughput rises steeply with chunk length
// and approaches the unstrided rate by about 1 KiB). The cost is an accumulator read-modify-write per row per word,
// L1 traffic against DRAM traffic saved.
//
// Args:
//
//	Out     [rows] f32, ACCUMULATED into -- the caller zeroes it
//	W/PD/PSC  the three spans, at this call's first row
//	A/AScale/AHalf  activations, from element 0
//	Rows    how many PackedWideGroup-row groups this call serves
//	RowStr  the layout's row stride in bytes: the tensor's FULL row count x4
//	K       outer steps
//	Scratch [rows] int32, this worker's private integer accumulators
func EmitPackedMatVecWide(t quant.Type, dk DotKind) ([]byte, error) {
	q, ok := kernels.QuantOf(t)
	if !ok {
		return nil, fmt.Errorf("jit: EmitPackedMatVecWide: %s has no device layout", t)
	}
	// The pre-VNNI path refuses it: this kernel has no call site, so a
	// VPMADDUBSW port would be code no gate runs. TestWideAndTiledAreRefusedPreVNNI.
	if dk == DotVEX {
		return nil, fmt.Errorf("jit: EmitPackedMatVecWide: %s has no pre-VNNI form "+
			"(the fused kernel serves every format, so this one has no call site)", t)
	}
	// It refuses the narrow formats rather than reading their two-rows-per-word
	// d plane wrong. It is unreachable: MatVecPacked prefers the fused
	// kernel at nrows%16 == 0, and every format with a wide kernel has a fused
	// one.
	if NarrowD(t) {
		return nil, fmt.Errorf("jit: EmitPackedMatVecWide: %s packs two rows to a d word; use the fused kernel", t)
	}
	if _, bits, _, _ := kernels.Layout(q); bits == 2 {
		// Unreachable for the same reason; its two activation registers hold
		// a nibble pair, and a 2-bit plane has four fields a byte.
		return nil, fmt.Errorf("jit: EmitPackedMatVecWide: %s has a 2-bit plane; use the fused kernel", t)
	}
	sub, bits, biasK, biasArray := kernels.Layout(q)
	perSuper, scOff := kernels.ScaleLayout(q)
	signed := kernels.SignedPayload(q)
	// pw primary words then hw secondary ones; see qinfo.hi and packSub.
	hiW := kernels.HiPlane(q)
	pw, hw := sub*bits/32, sub*hiW/32
	halfSub := sub / 2
	const vec = 32
	const grp = PackedWideGroup / 8 // YMM per row group

	// One d word per super-block for every format, so there is no parity
	// to unroll: see kernels.PackedWords.
	const supers = 1
	perWord, stride := 4, 8
	if biasArray {
		perWord, stride = 2, 16
	}
	stream := kernels.ScStream(q)

	// Held for the whole kernel; the rest are working registers.
	const (
		mask   = Y0
		corr   = Y1
		shufLo = Y2
		shufHi = Y3
		scOffV = Y4
		pay    = Y5
		tmp    = Y6
		iaccT  = Y7
		actL   = Y8
		actH   = Y9
		wa     = Y10
		wb     = Y11
		wc     = Y12
		wd     = Y13
		// Hoisted per sub-block, not per row group: the activation scale and
		// the bias scalar are invariant across every row of a sub-block.
		da     = Y14
		bscale = Y15
	)

	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(RDX, At(RDI, 8))  // QS
	a.MOVLoad(R11, At(RDI, 16)) // A
	a.MOVLoad(R12, At(RDI, 24)) // AScale
	a.MOVLoad(RAX, At(RDI, 40)) // K
	a.MOVLoad(R13, At(RDI, 48)) // RowStr
	a.MOVLoad(RBX, At(RDI, 56)) // Scr
	a.MOVLoad(R10, At(RDI, 64)) // AHalf
	a.MOVLoad(RSI, At(RDI, 72)) // Scratch -> iacc
	a.MOVLoad(R8, At(RDI, 144)) // D
	a.MOVLoad(R9, At(RDI, 152)) // SC
	if bits == 4 {
		a.VMOVDQULoad(mask, At(RBX, 0))
	} else if signed {
		a.VMOVDQULoad(mask, At(RBX, 192))
	}
	switch {
	case biasArray:
		a.VMOVDQULoad(corr, At(RBX, 128))
	case sub == Q8Block/2:
		a.VMOVDQULoad(corr, At(RBX, 288))
	default:
		a.VMOVDQULoad(corr, At(RBX, 160))
	}
	a.VMOVDQULoad(shufLo, At(RBX, 64))
	a.VMOVDQULoad(shufHi, At(RBX, 96))
	if scOff != 0 {
		a.VMOVDQULoad(scOffV, At(RBX, 224+int32(scOffSlot(scOff))))
	}
	// RBX is the row-group counter from here; the constants are in registers.

	// rowLoop emits a counted loop over this call's row groups. R15 is the byte
	// offset shared by the payload, the scales, the accumulators and the
	// output -- they are all indexed by the same row.
	rowLoop := func(body func()) {
		a.MOVLoad(RBX, At(RDI, 32)) // Rows, in groups
		a.SUBQ(R15, R15)
		l := a.Label()
		a.Bind(l)
		body()
		a.ADDimm(R15, PackedWideGroup*4)
		a.DEC(RBX)
		a.JNZ(l)
	}
	// f16x8 gathers eight strided f16 into eight f32.
	f16x8 := func(dst, src, shuf Reg) {
		a.VPSHUFB(dst, src, shuf)
		a.VPERMQ(dst, dst, 0xD8)
		a.VCVTPH2PSReg(dst, dst)
	}

	outer := a.Label()
	a.Bind(outer)

	for sp := 0; sp < supers; sp++ {
		shuf := shufLo
		_ = sp
		for sn := 0; sn < perSuper; sn++ {
			for w := 0; w < pw; w++ {
				if bits == 4 {
					a.VPBROADCASTD(actL, At(R11, int32(sub*sn+4*w)))
					a.VPBROADCASTD(actH, At(R11, int32(sub*sn+halfSub+4*w)))
				} else {
					a.VPBROADCASTD(actL, At(R11, int32(sub*sn+4*w)))
				}
				first := w == 0
				rowLoop(func() {
					for g := 0; g < grp; g++ {
						off := int32(g * vec)
						a.VMOVDQULoad(pay, Idx(RDX, R15, 1, off))
						if first {
							a.VPXOR(iaccT, iaccT, iaccT)
						} else {
							a.VMOVDQULoad(iaccT, Idx(RSI, R15, 1, off))
						}
						if bits == 4 {
							a.VPAND(tmp, pay, mask)
							a.VPDPBUSD(iaccT, tmp, actL)
							a.VPSRLD(tmp, pay, 4)
							a.VPAND(tmp, tmp, mask)
							a.VPDPBUSD(iaccT, tmp, actH)
						} else if signed {
							a.VPXOR(tmp, pay, mask)
							a.VPDPBUSD(iaccT, tmp, actL)
						} else {
							a.VPDPBUSD(iaccT, pay, actL)
						}
						a.VMOVDQUStore(Idx(RSI, R15, 1, off), iaccT)
					}
				})
				a.ADDQ(RDX, R13)
			}
			// The secondary plane accumulates into the same memory accumulator
			// (its mask is pre-shifted by 4). wa is a scale-pass register and is
			// dead here, so this costs no register.
			if hw > 0 {
				lanes := 8 / hiW
				for hwi := 0; hwi < hw; hwi++ {
					a.VMOVDQULoad(wa, At(RBX, hiMaskSlot(hiW)))
					rowLoop(func() {
						for g := 0; g < grp; g++ {
							off := int32(g * vec)
							a.VMOVDQULoad(pay, Idx(RDX, R15, 1, off))
							a.VMOVDQULoad(iaccT, Idx(RSI, R15, 1, off))
							for half := 0; half < 2; half++ {
								for w := 0; w < pw; w++ {
									gi := w + half*pw
									if gi/lanes != hwi {
										continue
									}
									a.VPBROADCASTD(actL, At(R11, int32(sub*sn+half*halfSub+4*w)))
									switch sh := hiW * (gi % lanes); {
									case sh < 4:
										a.VPSLLD(tmp, pay, byte(4-sh))
										a.VPAND(tmp, tmp, wa)
									case sh == 4:
										a.VPAND(tmp, pay, wa)
									default:
										a.VPSRLD(tmp, pay, byte(sh-4))
										a.VPAND(tmp, tmp, wa)
									}
									a.VPDPBUSD(iaccT, tmp, actL)
								}
							}
							a.VMOVDQUStore(Idx(RSI, R15, 1, off), iaccT)
						}
					})
					a.ADDQ(RDX, R13)
				}
			}

			pairOff := int32(sub * sn / Q8Block * 8)
			sh := byte((sn % perWord) * stride)
			a.VBROADCASTSS(da, At(R12, pairOff)) // d_a
			switch {
			case biasArray:
				// -sum(a)/8 * -8 == sum(a); the sub below applies the sign.
				a.VBROADCASTSS(bscale, At(R12, pairOff+4))
				a.VMULPS(bscale, bscale, corr)
				a.VMULPS(bscale, bscale, da)
			case sub == Q8Block/2:
				a.VBROADCASTSS(bscale, At(R10, int32(4*sn)))
				a.VMULPS(bscale, bscale, corr)
			case biasK != 0 || signed:
				a.VBROADCASTSS(bscale, At(R12, pairOff+4))
				a.VMULPS(bscale, bscale, corr)
			}
			rowLoop(func() {
				for g := 0; g < grp; g++ {
					off := int32(g * vec)
					if kernels.D32(q) {
						a.VMOVDQULoad(wb, Idx(R8, R15, 1, off)) // the d word is the scale
					} else {
						a.VMOVDQULoad(wa, Idx(R8, R15, 1, off)) // the d word
						f16x8(wb, wa, shuf)                     // d
					}
					// Container v27's SC stream (scstream.go): a straddling
					// field's next word is a plane stride on, reached by
					// stepping the cursor there and back.
					nextSC := func(r Reg) {
						a.ADDQ(R9, R13)
						a.VMOVDQULoad(r, Idx(R9, R15, 1, off))
						a.SUBQ(R9, R13)
					}
					if stream {
						// wd is free until Out is read below.
						a.VMOVDQULoad(wc, Idx(R9, R15, 1, off))
						scField(avxSC(&a), tmp, wc, wd, sn, false, nextSC)
						if scOff != 0 {
							a.VPSUBD(tmp, tmp, scOffV)
						}
						a.VCVTDQ2PS(tmp, tmp)
						a.VMULPS(wb, wb, tmp)
					} else if perSuper > 1 {
						a.VMOVDQULoad(wc, Idx(R9, R15, 1, off)) // the scale word
						if 24-sh > 0 {
							a.VPSLLD(tmp, wc, 24-sh)
						} else {
							a.VMOVAPSReg(tmp, wc)
						}
						a.VPSRLD(tmp, tmp, 24)
						if scOff != 0 {
							a.VPSUBD(tmp, tmp, scOffV)
						}
						a.VCVTDQ2PS(tmp, tmp)
						a.VMULPS(wb, wb, tmp)
					}
					a.VMULPS(wb, wb, da) // scale * d_a
					a.VMOVDQULoad(iaccT, Idx(RSI, R15, 1, off))
					a.VCVTDQ2PS(iaccT, iaccT)
					a.VMOVDQULoad(wd, Idx(RCX, R15, 1, off))
					a.VFMADD231PS(wd, iaccT, wb)
					switch {
					case biasArray:
						f16x8(tmp, wa, shufHi) // dmin
						if stream {
							// wc is dead once read: the straddle's scratch.
							scField(avxSC(&a), wa, wc, wc, sn, true, nextSC)
							a.VCVTDQ2PS(wa, wa)
							a.VMULPS(tmp, tmp, wa) // dmin * m
						} else if !kernels.MinInD(q) { // Q5_1's m is one
							a.VPSLLD(wa, wc, 16-sh)
							a.VPSRLD(wa, wa, 24)
							a.VCVTDQ2PS(wa, wa)
							a.VMULPS(tmp, tmp, wa) // dmin * m
						}
						a.VMULPS(tmp, tmp, bscale) // * d_a * -sum(a)/8 * -8
						a.VSUBPS(wd, wd, tmp)
					default:
						// bias = BiasC*scale for a constant-bias format, and the
						// 16-wide ones take their sum per sixteen elements; both
						// fold into bscale, which is hoisted per sub-block.
						a.VMULPS(tmp, wb, bscale)
						a.VADDPS(wd, wd, tmp)
					}
					a.VMOVDQUStore(Idx(RCX, R15, 1, off), wd)
				}
			})
			if stream && scStep(sn) || !stream && perSuper > 1 && sn%perWord == perWord-1 {
				a.ADDQ(R9, R13)
			}
		}
		a.ADDimm(R11, int32(perSuper*sub))
		a.ADDimm(R12, int32(perSuper*sub/Q8Block*8))
		if sub == Q8Block/2 {
			a.ADDimm(R10, int32(perSuper*4))
		}
		a.ADDQ(R8, R13)
		if perSuper == 1 {
			a.ADDQ(R9, R13)
		}
	}
	a.DEC(RAX)
	a.JNZ(outer)

	a.VZEROUPPER()
	a.RET()
	code := a.Bytes()
	if len(code) > PackedWideBudget {
		return nil, fmt.Errorf("jit: EmitPackedMatVecWide: %s is %d bytes, over the %d budget",
			t, len(code), PackedWideBudget)
	}
	return code, nil
}
