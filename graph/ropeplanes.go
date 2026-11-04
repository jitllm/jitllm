package graph

import "math"

// The rotary table's planes (OpRopeTable's), as the model folds a
// configuration's frequencies into them once and every device's table reads
// them: a position's base-128 digits, each a head plane that multiplies the
// digit exactly and a residual plane, four of each a pair, then the table's
// scale.
const (
	// RopeDigits is how many base-128 digits of a position the table folds:
	// four cover positions below 2^28.
	RopeDigits = 4
	// RopeDigitBits is one digit's width: the digit's bits and the head
	// plane's stay within float32's 24.
	RopeDigitBits = 7
	// RopeGrid is the head plane's grid, as its reciprocal's denominator: the
	// largest for which the digit products and their sum stay exact in
	// float32.
	RopeGrid = 1 << 12
	// ropeLanes is the pairs a plane is padded to.
	ropeLanes = 8
)

// RopeStride is the words from one plane of npairs pairs to the next.
func RopeStride(npairs int) int { return (npairs + ropeLanes - 1) &^ (ropeLanes - 1) }

// RopeBlock is the words the planes of npairs pairs and their scale hold.
func RopeBlock(npairs int) int { return 8*RopeStride(npairs) + 1 }

// RopePlaneOff is plane p's first word (0..3 the heads, 4..7 the residuals).
func RopePlaneOff(p, npairs int) int { return p * RopeStride(npairs) }

// RopeScaleOff is the scale's word.
func RopeScaleOff(npairs int) int { return 8 * RopeStride(npairs) }

// RopeHeadSplit splits one quarter-turn constant, folded into [0,4], into the
// head that multiplies a digit exactly and the residual that carries the rest.
func RopeHeadSplit(u float64) (h, l float32) {
	h = float32(math.Round(u*RopeGrid) / RopeGrid)
	return h, float32(u - float64(h))
}
