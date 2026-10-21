//go:build amd64

package cpu

import "math"

// ArgmaxConsts is the constant block EmitArgmax reads:
//
//	0..31   the initial index vector, 0..7
//	32      8, the per-iteration index step
//	36      0x7FFFFFFF, the index a lane that did not tie contributes
//	40      -Inf, the initial maximum
//
// The tail's index step of one is lane 1 of the initial index vector, so it
// needs no word of its own. argmax_other.go keeps a twin of this layout.
func ArgmaxConsts() []float32 {
	c := make([]float32, 11)
	for i := 0; i < 8; i++ {
		c[i] = math.Float32frombits(uint32(i))
	}
	c[8] = math.Float32frombits(8)
	c[9] = math.Float32frombits(0x7FFFFFFF)
	c[10] = float32(math.Inf(-1))
	return c
}

// ArgmaxLanes is the element count EmitArgmax's vector loop consumes per
// iteration. It is not a divisibility requirement: the kernel finishes the
// last n%8 elements one at a time.
const ArgmaxLanes = 8

// EmitArgmax generates the greedy sampler's argmax: the index of the largest
// element of Q32[:K], with the lowest index winning a tie.
//
//	Q32  the elements                     (f32)
//	K    how many elements (not vectors)
//	Out  the maximum value                (f32, written)
//	ASum the index of that maximum        (int32, written)
//	Scr  ArgmaxConsts()
//
// The kernel splits K itself into K>>3 whole vectors and K&7 tail elements.
//
// The tail is scalar, one element at a time (elemLoops' reduction form). An
// overlapping final vector cannot serve n < 8, and a masked one would load up
// to seven floats past the logits, which faults when the vocabulary ends on a
// page boundary.
//
// The tie rule is the contract: model.Greedy is `if v > best`, keeping the
// first of equal logits, and the greedy gate compares token ids against
// llama.cpp. Strict > per lane keeps the first within a lane; the fold keeps
// the lowest lane index across them; and the tail is merged last and only on
// a strict >, because every tail index is higher than every vector index.
//
// It is an index-carrying reduction: the comparison's mask selects the index
// alongside the value, via and/andnot/or (this assembler has no blend).
//
// VMAXPS takes the new value as src1 and the running maximum as src2, the one
// order that agrees with the Go scan: x86 returns src2 when either operand is
// NaN and when both are zeros, so a NaN logit is never taken and -0 against +0
// keeps the earlier. The other order lets a NaN into the running maximum and
// returns index 0x7FFFFFFF (argmaxtail_test.go).
func EmitArgmax() []byte {
	var a Buf
	a.MOVLoad(RSI, At(RDI, 112)) // Q32 -> the elements
	a.MOVLoad(RBX, At(RDI, 56))  // Scr -> constants
	a.MOVLoad(R10, At(RDI, 40))  // K   -> ELEMENTS
	a.MOVQ(R11, R10)
	a.ANDimm8(R11, 7) // the tail: 0..7 elements
	a.SHRimm(R10, 3)  // whole vectors of eight

	a.VBROADCASTSS(Y0, At(RBX, 40)) // vmax = -Inf
	a.VMOVDQULoad(Y1, At(RBX, 0))   // best index per lane = 0..7
	a.VMOVDQULoad(Y2, At(RBX, 0))   // the current index vector
	a.VPBROADCASTD(Y3, At(RBX, 32)) // +8 per iteration

	// The tail's accumulators are separate and lane 0 only: a scalar load
	// zeroes the other seven lanes, and zero is not neutral for a maximum.
	a.VBROADCASTSS(Y9, At(RBX, 40)) // the tail's maximum = -Inf
	a.VPXOR(Y10, Y10, Y10)          // the tail's best index
	a.VPBROADCASTD(Y12, At(RBX, 4)) // +1 per tail element

	// The tail's index cursor is the vector loop's own, Y2, which after k
	// iterations holds 8k in lane 0 -- the index of the first tail element,
	// including when the vector loop never ran.
	elemLoops(&a, R10, R11, []Reg{RSI}, func() {
		a.VMOVDQULoad(Y4, At(RSI, 0))
		// 14 is _CMP_GT_OS: false for a NaN either side, as `v > best` is, so a
		// NaN never wins.
		a.VCMPPS(Y5, Y4, Y0, 14)
		a.VMAXPS(Y0, Y4, Y0)
		a.VPAND(Y6, Y2, Y5)  // the new index where the lane improved
		a.VPANDN(Y7, Y5, Y1) // the old index where it did not
		a.VPOR(Y1, Y6, Y7)
		a.VPADDD(Y2, Y2, Y3)
	}, func() {
		a.VMOVSSLoad(Y4, At(RSI, 0)) // one element, lanes 1..7 zeroed
		a.VCMPPS(Y5, Y4, Y9, 14)
		a.VMAXPS(Y9, Y4, Y9)
		a.VPAND(Y6, Y2, Y5)
		a.VPANDN(Y7, Y5, Y10)
		a.VPOR(Y10, Y6, Y7)
		a.VPADDD(Y2, Y2, Y12)
	})

	// The per-lane maxima have to survive the fold, because the index of the
	// winner is found by asking which lanes EQUAL the overall maximum.
	a.VMAXPS(Y8, Y0, Y0) // a move
	a.VEXTRACTF128(Y4, Y0, 1)
	a.VMAXPS(Y0, Y0, Y4)
	a.VSHUFPS(Y4, Y0, Y0, 0x4E)
	a.VMAXPS(Y0, Y0, Y4)
	a.VSHUFPS(Y4, Y0, Y0, 0xB1)
	a.VMAXPS(Y0, Y0, Y4)
	a.VBROADCASTSSReg(Y0, Y0) // the maximum, in every lane

	// Lanes that hold it keep their index; the rest contribute INT_MAX, so the
	// minimum over the lanes is the lowest index that ties.
	a.VCMPPS(Y5, Y8, Y0, 0) // _CMP_EQ_OQ
	a.VPBROADCASTD(Y6, At(RBX, 36))
	a.VPAND(Y7, Y1, Y5)
	a.VPANDN(Y6, Y5, Y6)
	a.VPOR(Y1, Y7, Y6)
	a.VEXTRACTI128(Y4, Y1, 1)
	a.VPMINSD(Y1, Y1, Y4)
	a.VSHUFPS(Y4, Y1, Y1, 0x4E)
	a.VPMINSD(Y1, Y1, Y4)
	a.VSHUFPS(Y4, Y1, Y1, 0xB1)
	a.VPMINSD(Y1, Y1, Y4)

	// The tail is merged last and on a strict >: every tail index is higher
	// than every vector index, so an equal maximum keeps the vector's. Lanes
	// 1..7 of the mask are meaningless here and nothing reads them: both
	// stores below are lane 0.
	a.VCMPPS(Y5, Y9, Y0, 14)
	a.VPAND(Y6, Y10, Y5)
	a.VPANDN(Y7, Y5, Y1)
	a.VPOR(Y1, Y6, Y7)
	a.VMAXPS(Y0, Y9, Y0)

	a.MOVLoad(R8, At(RDI, 96)) // ASum -> the index
	a.VMOVSSStore(At(R8, 0), Y1)
	a.MOVLoad(R9, At(RDI, 0)) // Out -> the value
	a.VMOVSSStore(At(R9, 0), Y0)
	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}
