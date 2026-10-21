//go:build arm64

package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/format/quant"
)

// EmitGEMMWindow generates the arm64 prefill kernel: out[rows][8*nr] += W . A,
// register-tiled, with SDOT doing the arithmetic. It dispatches on the weight
// format: the k-quants have their own emitter (gemmk_a64.go).
//
// It reads the same packed activations ([kgroup][token][4]) and Args as the
// x86 kernel; arm64 consumes four tokens per 16-byte operand, so a tile of
// 8*nr tokens is 2*nr vectors here.
//
// Three x86 mechanisms do not exist here, as in the decode kernel:
//
//  1. No bias correction: SDOT is signed x signed, so Q8_0 weights are used as
//     loaded, Q4_0 needs one SUB #8, and Args.ASum is never read.
//  2. No broadcast: SDOT's by-element form selects the weight k-group by
//     immediate index, so sixteen bytes loaded once serve four k-groups.
//  3. No spill: with 32 V registers both accumulator sets stay resident and
//     Args.Scratch is unused.
//
// window is how many activation elements share one scale. The kernel cannot
// infer it: WideActWindow drops to 32 as soon as the model contains a
// narrow-block tensor, so the same kernel shape sees both, and hoisting the
// scale at the wrong width would silently apply the wrong scale.
func EmitGEMMWindow(t quant.Type, mr, nr, nb, window int) ([]byte, error) {
	switch t {
	case quant.Q4_K, quant.Q5_K, quant.Q6_K, quant.Q3_K:
		return emitGEMMA64K(t, mr, nr, nb, window)
	}
	return emitGEMMA64(t, mr, nr, nb)
}

func emitGEMMA64(t quant.Type, mr, nr, nb int) ([]byte, error) {
	if t != quant.Q8_0 && t != quant.Q4_0 {
		return nil, fmt.Errorf("jit: EmitGEMM: %s unsupported here", t)
	}
	if mr < 1 || nr < 1 || nb < 1 {
		return nil, fmt.Errorf("jit: EmitGEMM: tile %dx%d nb=%d out of range", mr, nr, nb)
	}
	q4 := t == quant.Q4_0
	// Each tile element needs an int32 accumulator and a float one, and a tile
	// of 8*nr tokens is 2*nr vectors wide. Six spare registers cover the weight
	// staging, the nibble constants and the scale temporaries.
	vecs := mr * nr * 2
	if 2*vecs+6 > 32 {
		return nil, fmt.Errorf("jit: EmitGEMM: tile %dx%d needs %d vectors, have 32", mr, nr, 2*vecs+6)
	}
	bb := int32(t.BlockBytes())
	rowStr := int32(nb) * bb
	// The row stride has to live in a register: ADD's immediate is 12 bits and
	// LDUR's is a signed 9, while a stride reaches 17408 bytes. MOVimm caps the
	// register form at 16 bits, which is a row of two million elements.
	if rowStr > 0xFFFF {
		return nil, fmt.Errorf("jit: EmitGEMM: row stride %d exceeds 65535", rowStr)
	}
	// One weight cursor per row of the tile: a GGUF block is 18 or 34 bytes, so
	// weights need the unscaled LDUR, whose reach is only +-256. Activations,
	// scales and output are float32 arrays at multiples of 16, so those keep
	// the scaled form and its 65520-byte reach.
	rowPtr := []XReg{X7, X11, X12, X13, X14, X15, X16, X17}
	if mr > len(rowPtr) {
		return nil, fmt.Errorf("jit: EmitGEMM: tile height %d exceeds %d row cursors", mr, len(rowPtr))
	}
	tok := int32(8 * nr)
	aKG := tok * 4  // activation bytes per k-group
	aBlk := aKG * 8 // ... per 32-element block
	sBlk := tok * 4 // scale bytes per block
	outRow := tok * 4

	iacc := func(m, j, h int) VReg { return VReg(((m*nr+j)*2 + h)) }
	facc := func(m, j, h int) VReg { return VReg(vecs + ((m*nr+j)*2 + h)) }
	const (
		vW0    = VReg(26) // sixteen weight bytes, k-groups 0..3 of a half-block
		vW1    = VReg(27) // ... and the nibble-expanded second half for Q4_0
		vMask  = VReg(28) // 0x0F
		vEight = VReg(29) // 8, the Q4_0 zero point
		vDw    = VReg(30) // d_w for this (row, block), lane 0
		vTmp   = VReg(31)
	)

	var a A64
	// The accumulators end at v(4*mr*nr-1), at most v23 (2*vecs+6 <= 32 and
	// 2*vecs is a multiple of four), so v24 and v25 are always free for a
	// widened SDOT (sdotemu.go).
	a.DotScratch(V24, V25)
	a.LDRx(X1, X0, 0)  // Out
	a.LDRx(X2, X0, 8)  // W
	a.LDRx(X3, X0, 16) // A, packed [kgroup][token][4]
	a.LDRx(X4, X0, 24) // AScale [block][token]
	a.LDRx(X5, X0, 32) // row groups
	if q4 {
		a.MOVI16b(vMask, 0x0F)
		a.MOVI16b(vEight, 8)
	}

	a.MOVimm(X8, int64(rowStr))

	row := a.Label()
	a.Bind(row)
	a.MOVreg(X9, X3)  // activation cursor for this row group
	a.MOVreg(X10, X4) // scale cursor
	a.MOVimm(X6, int64(nb))
	a.MOVreg(rowPtr[0], X2)
	for m := 1; m < mr; m++ {
		a.ADDreg(rowPtr[m], rowPtr[m-1], X8)
	}
	for m := 0; m < mr; m++ {
		for j := 0; j < nr; j++ {
			for h := 0; h < 2; h++ {
				a.MOVIzero(facc(m, j, h))
			}
		}
	}

	blk := a.Label()
	a.Bind(blk)
	for m := 0; m < mr; m++ {
		for j := 0; j < nr; j++ {
			for h := 0; h < 2; h++ {
				a.MOVIzero(iacc(m, j, h))
			}
		}
	}
	// Eight k-groups per 32-element block, taken sixteen weight bytes at a
	// time: one LDR feeds four SDOTs through the by-element index.
	for half := 0; half < 2; half++ {
		for m := 0; m < mr; m++ {
			if q4 {
				// Q4_0 keeps elements 0..15 in the low nibbles of bytes 0..15
				// and 16..31 in the high nibbles of the same bytes, so one
				// 16-byte load serves both halves: mask for the first, shift
				// for the second. Then SUB 8 makes them signed for SDOT.
				a.LDURq(vW0, rowPtr[m], 2)
				if half == 0 {
					a.AND16b(vW1, vW0, vMask)
				} else {
					a.USHR16b(vW1, vW0, 4)
				}
				a.SUB16b(vW1, vW1, vEight)
			} else {
				a.LDURq(vW1, rowPtr[m], 2+int32(half)*16)
			}
			for g := 0; g < 4; g++ {
				kg := int32(half*4 + g)
				for j := 0; j < nr; j++ {
					for h := 0; h < 2; h++ {
						a.LDRq(vTmp, X9, kg*aKG+int32(j)*32+int32(h)*16)
						a.SDOTelem(iacc(m, j, h), vTmp, vW1, uint8(g))
					}
				}
			}
		}
	}
	// Fold: convert, scale by d_w and the four per-token d_a, accumulate.
	for m := 0; m < mr; m++ {
		a.LDRh(vDw, rowPtr[m], 0)
		a.FCVTsh(vDw, vDw)
		for j := 0; j < nr; j++ {
			for h := 0; h < 2; h++ {
				a.SCVTF4s(iacc(m, j, h), iacc(m, j, h))
				a.LDRq(vTmp, X10, int32(j)*32+int32(h)*16) // four tokens' d_a
				a.FMUL4s(vTmp, vTmp, iacc(m, j, h))
				a.FMLAelem(facc(m, j, h), vTmp, vDw, 0)
			}
		}
	}
	for m := 0; m < mr; m++ {
		a.ADDimm(rowPtr[m], rowPtr[m], bb)
	}
	a.ADDimm(X9, X9, aBlk)
	a.ADDimm(X10, X10, sBlk)
	a.SUBimm(X6, X6, 1)
	a.CBNZ(X6, blk)

	for m := 0; m < mr; m++ {
		for j := 0; j < nr; j++ {
			for h := 0; h < 2; h++ {
				a.STRq(facc(m, j, h), X1, int32(m)*outRow+int32(j)*32+int32(h)*16)
			}
		}
	}
	for m := 0; m < mr; m++ {
		a.ADDreg(X2, X2, X8) // next row group
	}
	a.ADDimm(X1, X1, int32(mr)*outRow)
	a.SUBimm(X5, X5, 1)
	a.CBNZ(X5, row)

	a.RET()
	return a.Bytes(), nil
}
