package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// The rotary table's constant block and its layout, re-exported.
//
// The layout lives in jit_old/gpu/kernels because the device kernel
// (jit_old/gpu/kernels/ropetab.go) needs it and jit_old/cpu already imports kernels,
// so keeping it here would close an import cycle; a second hand-written copy
// would drift. This file is names.
//
// The plane data is not shared: the folded H_i/L_i planes are computed once
// per model on the host (nn.buildRopeTab) and handed to either a host kernel
// (Args.AScale) or a device buffer.
const (
	RopeTabDigits     = kernels.RopeTabDigits
	RopeTabDigitBits  = kernels.RopeTabDigitBits
	RopeTabMaxPos     = kernels.RopeTabMaxPos
	RopeTabConstWords = kernels.RopeTabConstWords
)

// RopeTabStride is the word stride between two planes of a per-model block.
func RopeTabStride(npairs int) int { return kernels.RopeTabStride(npairs) }

// RopeTabBlock is how many float32 words a per-model block for npairs rotary
// pairs occupies: eight planes and the scale.
func RopeTabBlock(npairs int) int { return kernels.RopeTabBlock(npairs) }

// RopeTabPlaneOff is the word offset of plane p (0..3 the heads H, 4..7 the
// residuals L) inside a per-model block.
func RopeTabPlaneOff(p, npairs int) int { return kernels.RopeTabPlaneOff(p, npairs) }

// RopeTabScaleOff is the word offset of mscale inside a per-model block.
func RopeTabScaleOff(npairs int) int { return kernels.RopeTabScaleOff(npairs) }

// RopeTabHeadSplit splits one folded quarter-turn constant into the head that
// multiplies a digit exactly and the residual that carries the rest.
func RopeTabHeadSplit(u float64) (h, l float32) { return kernels.RopeTabHeadSplit(u) }

// RopeTabConsts is the shared constant block every rotary-table kernel reads
// through Args.Scr, and which the device tier uploads as a buffer.
func RopeTabConsts() []float32 { return kernels.RopeTabConsts() }

// ropeTabShapeErr is what every tier's emitter answers for a pair count outside
// the contract, which is only a count of none.
//
// There is no width floor: the tail stores part of a whole vector rather than
// re-running the last whole one, so a pair count smaller than the vector
// (stories260K's four) is served.
func ropeTabShapeErr(npairs int) error {
	return fmt.Errorf("jit: rope table: %d pair(s), want at least one", npairs)
}

// The word indices a gate needs: the violations that prove this kernel's
// structural steps are injected by zeroing one word of the block, which
// removes exactly one step on all four tiers. See
// engine/nn/ropetabviolation_test.go.
const (
	RopeTabMagicWord   = kernels.RopeTabMagicWord
	RopeTabSignVecWord = kernels.RopeTabSignVecWord
	RopeTabOneVecWord  = kernels.RopeTabOneVecWord
	RopeTabTwoVecWord  = kernels.RopeTabTwoVecWord
)

// The BYTE offsets into that block, which is the form an emitter's memory
// operand wants. Derived from the word indices rather than written twice.
const (
	rtMagicOff   = 4 * kernels.RopeTabMagicWord
	rtPio2HiOff  = 4 * kernels.RopeTabPio2HiWord
	rtPio2LoOff  = 4 * kernels.RopeTabPio2LoWord
	rtSinC7Off   = 4 * kernels.RopeTabSinC7Word
	rtSinC5Off   = 4 * kernels.RopeTabSinC5Word
	rtSinC3Off   = 4 * kernels.RopeTabSinC3Word
	rtCosC8Off   = 4 * kernels.RopeTabCosC8Word
	rtCosC6Off   = 4 * kernels.RopeTabCosC6Word
	rtCosC4Off   = 4 * kernels.RopeTabCosC4Word
	rtNegHalfOff = 4 * kernels.RopeTabNegHalfWord
	rtOneOff     = 4 * kernels.RopeTabOneWord
	rtDigitOff   = 4 * kernels.RopeTabDigitWord
	rtSignVecOff = 4 * kernels.RopeTabSignVecWord
	rtOneVecOff  = 4 * kernels.RopeTabOneVecWord
	rtTwoVecOff  = 4 * kernels.RopeTabTwoVecWord
)
