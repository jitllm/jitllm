//go:build amd64

package cpu

// QuantActNarrowBlocks is how many blocks EmitQuantActNarrow does per
// iteration, and it is eight because that is a vector of scales.
const QuantActNarrowBlocks = 8

// QuantActNarrowScratch is the per-worker scratch the narrow kernel needs, in
// float32: eight amaxes, then the eight d and eight inv it derives from them.
const QuantActNarrowScratch = 24

// EmitQuantActNarrow is the activation quantizer for a window of one block --
// Q4_0 and Q8_0, whose accuracy depends on a scale every 32 elements.
//
// The wide kernel driven one block at a time was slower than the Go loop: a
// 32-element block cannot amortise an eight-lane fold, a broadcast and two
// dependent divisions. This kernel folds eight blocks' maxima into one vector,
// divides once, spills d and inv to scratch, and broadcasts lane j back for
// block j. The same two roundings in the same order keep it bit-identical to
// QuantizeQ8Window.
//
//	Q32     the first block's first element        (f32)
//	Cols    how many GROUPS of eight blocks
//	W       &dst[b0*32]
//	Out     &pairs[2*b0]
//	AHalfSum &half[2*b0], read only when half is baked true
//	Scratch QuantActNarrowScratch float32 of per-worker space
//	Scr     QuantActConsts()
func EmitQuantActNarrow(half bool) []byte {
	var a Buf
	a.MOVLoad(RBX, At(RDI, 56))  // Scr     -> constants
	a.MOVLoad(RCX, At(RDI, 112)) // Q32     -> the elements
	a.MOVLoad(RDX, At(RDI, 8))   // W       -> dst
	a.MOVLoad(R8, At(RDI, 0))    // Out     -> pairs
	a.MOVLoad(R12, At(RDI, 72))  // Scratch -> the eight scales
	a.MOVLoad(R10, At(RDI, 80))  // Cols    -> groups of eight blocks
	if half {
		a.MOVLoad(R11, At(RDI, 104)) // AHalfSum
	}
	// The payload's two constants are loop invariant and phase 1 does not
	// touch their registers.
	a.VBROADCASTSS(Y12, At(RBX, 8))  // 0.5
	a.VBROADCASTSS(Y11, At(RBX, 16)) // 0x80000000
	a.VBROADCASTSS(Y15, At(RBX, 12)) // 0x7FFFFFFF, the absolute-value mask

	grp := a.Label()
	a.Bind(grp)

	// ---- phase 1: eight blocks' amax, folded into scratch ----
	//
	// Each block is four vectors; |v| is a mask, not a branch, exactly as the
	// Go loop's Float32bits &^ (1<<31) is.
	for j := 0; j < QuantActNarrowBlocks; j++ {
		base := int32(j) * 128
		a.VMOVDQULoad(Y0, At(RCX, base))
		a.VPAND(Y0, Y0, Y15)
		for w := 1; w < 4; w++ {
			a.VMOVDQULoad(Y14, At(RCX, base+int32(w)*32))
			a.VPAND(Y14, Y14, Y15)
			a.VMAXPS(Y0, Y0, Y14)
		}
		// The same three-fold ladder the wide kernel uses: VPERMQ swaps
		// 128-bit halves and would repeat the fold VEXTRACTF128 just did.
		a.VEXTRACTF128(Y14, Y0, 1)
		a.VMAXPS(Y0, Y0, Y14)
		a.VSHUFPS(Y14, Y0, Y0, 0x4E)
		a.VMAXPS(Y0, Y0, Y14)
		a.VSHUFPS(Y14, Y0, Y0, 0xB1)
		a.VMAXPS(Y0, Y0, Y14)
		a.VMOVSSStore(At(R12, int32(j)*4), Y0)
	}

	// ---- one pair of divisions for all eight ----
	a.VMOVDQULoad(Y0, At(R12, 0))  // the eight amaxes
	a.VBROADCASTSS(Y2, At(RBX, 0)) // 127.0
	a.VDIVPS(Y3, Y0, Y2)           // d
	a.VBROADCASTSS(Y4, At(RBX, 4)) // 1.0
	a.VDIVPS(Y5, Y4, Y3)           // 1/d
	a.VPXOR(Y7, Y7, Y7)
	a.VCMPPS(Y7, Y3, Y7, 4) // d != 0
	a.VPAND(Y5, Y5, Y7)     // inv, zeroed where d is zero
	a.VMOVDQUStore(At(R12, 0), Y3)
	a.VMOVDQUStore(At(R12, 32), Y5)

	// ---- phase 2: the blocks, each with its own lane broadcast back ----
	for j := 0; j < QuantActNarrowBlocks; j++ {
		a.VBROADCASTSS(Y3, At(R12, int32(j)*4))    // d
		a.VBROADCASTSS(Y5, At(R12, 32+int32(j)*4)) // inv
		emitQuantBlock(&a, 1, half, true)
	}

	a.DEC(R10)
	a.JNZ(grp)
	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}
