//go:build amd64

package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/format/quant"
)

// EmitGEMMWindow generates the prefill kernel: out[rows][8*nr] += W[rows][k] . A[8*nr][k],
// register-tiled, with AVX-VNNI doing the arithmetic.
//
// Decode is memory-bound; prefill reads the same weights for a batch of
// tokens, so arithmetic intensity rises and the kernel's tiling decides the
// rate (a tile that spills loses badly on AVX2's 16 registers).
//
// The lanes are tokens: VPDPBUSD accumulates eight dwords, so the weight dword
// (four consecutive k of one row) is broadcast and the activations are the
// memory operand, packed [kgroup][token][4] so eight tokens' four k values are
// one contiguous 32-byte load. Weights stay in their GGUF layout.
//
// VPDPBUSD wants its first source unsigned, so the broadcast weight dword is
// pushed into unsigned range (XOR 0x80 for Q8_0, a nibble mask for Q4_0 whose
// zero point is already 8), and the bias is removed in the fold using
// BiasC(t)*sum(activations) per (block, token), precomputed at pack time.
//
// Per k-group the mix is mr broadcasts + mr XORs + mr*nr VNNI, so widening in
// tokens amortizes the broadcast where widening in rows does not. mr and nr are
// arguments because the best tile is a measurement.
//
// window is how many activation elements share one scale. All four k-quants exploit it: Q3_K's whole fold becomes integer (see
// emitGEMMQ3K), and the other three at least hoist d_a out of the sub-block
// loop, which is the half of the same idea that does not need the int16 bound.
func EmitGEMMWindow(t quant.Type, mr, nr, nb, window int) ([]byte, error) {
	return emitGEMMAny(t, mr, nr, nb, window)
}

func emitGEMMAny(t quant.Type, mr, nr, nb, window int) ([]byte, error) {
	if t == quant.Q4_K || t == quant.Q5_K {
		return emitGEMMQ4K(t, mr, nr, nb, window)
	}
	if t == quant.Q6_K {
		return emitGEMMQ6K(mr, nr, nb, window)
	}
	if t == quant.Q3_K {
		return emitGEMMQ3K(mr, nr, nb, window)
	}
	if t != quant.Q8_0 && t != quant.Q4_0 {
		return nil, fmt.Errorf("jit: EmitGEMM: %s unsupported (Q4_0, Q8_0, Q4_K)", t)
	}
	if mr < 1 || nr < 1 || mr*nr > 12 || mr > 8 {
		return nil, fmt.Errorf("jit: EmitGEMM: tile %dx%d out of range", mr, nr)
	}
	if nb < 1 {
		return nil, fmt.Errorf("jit: EmitGEMM: nb=%d", nb)
	}
	q4 := t == quant.Q4_0
	bb := int32(t.BlockBytes()) // 34 for Q8_0, 18 for Q4_0; both lead with fp16 d
	rowStr := int32(nb) * bb
	tok := int32(8 * nr) // tokens per tile
	aKG := tok * 4       // activation bytes per k-group
	aBlk := aKG * 8      // ... per 32-element block
	sBlk := tok * 4      // scale/sum bytes per block
	outRow := tok * 4    // output bytes per row

	// Accumulators. The int32 set is live across the eight k-groups of one
	// block; the float set is live across every block of the row. Both fit in
	// registers only when 2*mr*nr+2 <= 16 -- otherwise the float set goes to
	// per-worker Scratch, which costs a load/FMA/store per tile element per
	// block but buys back the register room for a wider tile.
	n := mr * nr
	inReg := 2*n+2 <= 16
	iacc := make([]Reg, n)
	for i := range iacc {
		iacc[i] = Reg(i)
	}
	var facc []Reg
	if inReg {
		facc = make([]Reg, n)
		for i := range facc {
			facc[i] = Reg(n + i)
		}
	}
	const (
		konst = Y14 // 0x80, to make the broadcast weight dword unsigned
		wb    = Y15 // broadcast weight dword
		tmpA  = Y12
		tmpB  = Y13
	)
	if inReg && n+n > 12 {
		return nil, fmt.Errorf("jit: EmitGEMM: tile %dx%d collides with scratch registers", mr, nr)
	}

	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(RDX, At(RDI, 8))  // W
	a.MOVLoad(RSI, At(RDI, 16)) // A, packed [kgroup][token][4]
	a.MOVLoad(R8, At(RDI, 24))  // AScale [block][token]
	a.MOVLoad(RAX, At(RDI, 32)) // row groups
	a.MOVLoad(RBX, At(RDI, 56)) // Scr
	a.MOVLoad(R9, At(RDI, 96))  // ASum [block][token], already 128*sum
	if !inReg {
		a.MOVLoad(R15, At(RDI, 72)) // Scratch: the float accumulator tile
	}
	a.VBROADCASTI128(konst, At(RBX, 0))

	row := a.Label()
	a.Bind(row)
	// Fresh cursors for this row group; the activations are re-read per group,
	// which is an L1/L2 hit because mr rows of weights are what stream.
	a.MOVQ(R12, RSI)
	a.MOVQ(R13, R8)
	a.MOVQ(RDI, R9)
	a.MOVimm(R11, int64(nb))
	if inReg {
		for _, r := range facc {
			a.VPXOR(r, r, r)
		}
	} else {
		a.VPXOR(tmpA, tmpA, tmpA)
		for i := 0; i < n; i++ {
			a.VMOVDQUStore(At(R15, int32(i)*32), tmpA)
		}
	}

	blk := a.Label()
	a.Bind(blk)
	for _, r := range iacc {
		a.VPXOR(r, r, r)
	}
	for g := int32(0); g < 8; g++ {
		for m := 0; m < mr; m++ {
			if q4 {
				// Q4_0 is not interleaved: elements 0..15 are the low nibbles of
				// bytes 0..15 and 16..31 the high nibbles of the same bytes. So
				// k-group g reads bytes 4*(g%4) and takes the low nibble for
				// g<4, the high nibble for g>=4.
				//
				// VPSRLW shifts 16-bit lanes, but AND 0x0F afterwards leaves each
				// byte's own high nibble: the neighbour's bits land above bit 3
				// and are masked off.
				a.VPBROADCASTD(wb, At(RDX, int32(m)*rowStr+2+(g%4)*4))
				if g >= 4 {
					a.VPSRLW(wb, wb, 4)
				}
				a.VPAND(wb, wb, konst) // 0x0F: nibbles arrive UNSIGNED, 0..15
			} else {
				// Four consecutive k of row m, straight out of the GGUF block.
				a.VPBROADCASTD(wb, At(RDX, int32(m)*rowStr+2+g*4))
				a.VPXOR(wb, wb, konst) // 0x80: int8 read as unsigned, +128
			}
			for j := 0; j < nr; j++ {
				a.VPDPBUSDMem(iacc[m*nr+j], wb, At(R12, g*aKG+int32(j)*32))
			}
		}
	}
	// Fold: (iacc - 128*sum(a)) * d_w * d_a, accumulated in float.
	for m := 0; m < mr; m++ {
		a.VCVTPH2PSx(tmpA, At(RDX, int32(m)*rowStr))
		a.VBROADCASTSSReg(tmpA, tmpA) // d_w for this row and block
		for j := 0; j < nr; j++ {
			i := m*nr + j
			// Bias removed in int32, before the conversion: both terms are
			// exact integers and the accumulator is dominated by the bias, so
			// doing this in float32 would cancel away significant digits.
			a.VPSUBDMem(iacc[i], iacc[i], At(RDI, int32(j)*32))
			a.VCVTDQ2PS(iacc[i], iacc[i])
			a.VMULPSMem(tmpB, tmpA, At(R13, int32(j)*32)) // d_w * d_a[token]
			if inReg {
				a.VFMADD231PS(facc[i], iacc[i], tmpB)
			} else {
				a.VMULPS(iacc[i], iacc[i], tmpB)
				a.VADDPSMem(iacc[i], iacc[i], At(R15, int32(i)*32))
				a.VMOVDQUStore(At(R15, int32(i)*32), iacc[i])
			}
		}
	}
	a.ADDimm(RDX, bb)
	a.ADDimm(R12, aBlk)
	a.ADDimm(R13, sBlk)
	a.ADDimm(RDI, sBlk)
	a.DEC(R11)
	a.JNZ(blk)

	for m := 0; m < mr; m++ {
		for j := 0; j < nr; j++ {
			i := m*nr + j
			if inReg {
				a.VMOVDQUStore(At(RCX, int32(m)*outRow+int32(j)*32), facc[i])
			} else {
				a.VMOVDQULoad(tmpA, At(R15, int32(i)*32))
				a.VMOVDQUStore(At(RCX, int32(m)*outRow+int32(j)*32), tmpA)
			}
		}
	}
	// RDX has walked one row's worth of blocks; step it to the next row group.
	a.ADDimm(RDX, (int32(mr)-1)*rowStr)
	a.ADDimm(RCX, int32(mr)*outRow)
	a.DEC(RAX)
	a.JNZ(row)

	a.VZEROUPPER()
	a.RET()
	return a.Bytes(), nil
}
