// The device formats' vocabulary (jit/gpu/kernels) under the names the old
// generators use: an alias for each type and constant, a call for each function.

package mainkernels

import (
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
	vocab "github.com/jitllm/jitllm/jit/gpu/kernels"
)

type ActKind = vocab.ActKind
type Arena = vocab.Arena
type ArenaStats = vocab.ArenaStats
type Center = vocab.Center
type Packed = vocab.Packed
type QInfo = vocab.QInfo
type Quant = vocab.Quant

const ActGELUErf = vocab.ActGELUErf
const ActGELU = vocab.ActGELU
const ActIdentity = vocab.ActIdentity
const ActQuickGELU = vocab.ActQuickGELU
const ActReLU2 = vocab.ActReLU2
const ActReLU = vocab.ActReLU
const ActSiLU = vocab.ActSiLU
const ActSigmoid = vocab.ActSigmoid
const ActSitu = vocab.ActSitu
const ActSoftplus = vocab.ActSoftplus
const ActSqrtSigmoid = vocab.ActSqrtSigmoid
const ActSqrtSoftplus = vocab.ActSqrtSoftplus
const ActSwiGLUClamp = vocab.ActSwiGLUClamp
const ActSwiGLUOAI = vocab.ActSwiGLUOAI
const ActTanh = vocab.ActTanh
const ActXIELU = vocab.ActXIELU
const BFloat16 = vocab.BFloat16
const Float16 = vocab.Float16
const Float32 = vocab.Float32

const GELUErfP = vocab.GELUErfP

const MXFP4Bias = vocab.MXFP4Bias
const MXFP4 = vocab.MXFP4
const Q3_K = vocab.Q3_K
const Q4_0 = vocab.Q4_0
const Q4_K = vocab.Q4_K
const Q5_0 = vocab.Q5_0
const Q5_1 = vocab.Q5_1
const Q5_K = vocab.Q5_K
const Q6_K = vocab.Q6_K
const Q8_0 = vocab.Q8_0
const RopeTabConstWords = vocab.RopeTabConstWords
const RopeTabCosC4Word = vocab.RopeTabCosC4Word
const RopeTabCosC6Word = vocab.RopeTabCosC6Word
const RopeTabCosC8Word = vocab.RopeTabCosC8Word
const RopeTabDigitBits = vocab.RopeTabDigitBits
const RopeTabDigitWord = vocab.RopeTabDigitWord
const RopeTabDigits = vocab.RopeTabDigits
const RopeTabGrid = vocab.RopeTabGrid
const RopeTabHalfWord = vocab.RopeTabHalfWord
const RopeTabMagicWord = vocab.RopeTabMagicWord
const RopeTabMaxPos = vocab.RopeTabMaxPos
const RopeTabMod4MaskWord = vocab.RopeTabMod4MaskWord
const RopeTabNegHalfWord = vocab.RopeTabNegHalfWord
const RopeTabOneVecWord = vocab.RopeTabOneVecWord
const RopeTabOneWord = vocab.RopeTabOneWord
const RopeTabPio2HiWord = vocab.RopeTabPio2HiWord
const RopeTabPio2LoWord = vocab.RopeTabPio2LoWord
const RopeTabSignVecWord = vocab.RopeTabSignVecWord
const RopeTabSinC3Word = vocab.RopeTabSinC3Word
const RopeTabSinC5Word = vocab.RopeTabSinC5Word
const RopeTabSinC7Word = vocab.RopeTabSinC7Word
const RopeTabTwoVecWord = vocab.RopeTabTwoVecWord
const ScStreamWords = vocab.ScStreamWords
const SituBeta = vocab.SituBeta
const SituLinearBeta = vocab.SituLinearBeta
const SqrtSigmoidFloor = vocab.SqrtSigmoidFloor
const TQ2F = vocab.TQ2F
const TQ2 = vocab.TQ2
const TanhClamp = vocab.TanhClamp

func Addr(v []uint32) uintptr  { return vocab.Addr(v) }
func Codes(q Quant) *[16]uint8 { return vocab.Codes(q) }
func D32(q Quant) bool         { return vocab.D32(q) }
func DIndex(q Quant, super int, nrows int, r int) (idx int, slot int) {
	return vocab.DIndex(q, super, nrows, r)
}
func DSlots(q Quant) int                                                { return vocab.DSlots(q) }
func DeviceWhyNot(q Quant) string                                       { return vocab.DeviceWhyNot(q) }
func F16(u uint16) float32                                              { return vocab.F16(u) }
func FieldsOf(q Quant) int                                              { return vocab.FieldsOf(q) }
func FloatOf(t quant.Type) (Quant, bool)                                { return vocab.FloatOf(t) }
func HiPlane(q Quant) int                                               { return vocab.HiPlane(q) }
func HostAlign() uint64                                                 { return vocab.HostAlign() }
func IsFloat(q Quant) bool                                              { return vocab.IsFloat(q) }
func Layout(q Quant) (sub int, bits int, biasK float32, biasArray bool) { return vocab.Layout(q) }
func MapOffHeap(n int) (b []byte, mapped bool, err error)               { return vocab.MapOffHeap(n) }
func MinInD(q Quant) bool                                               { return vocab.MinInD(q) }
func NarrowScales(q Quant) bool                                         { return vocab.NarrowScales(q) }
func NewArena(retain uint64) *Arena                                     { return vocab.NewArena(retain) }
func PackActivations(x []float32) (a []uint32, as []float32, sum16 []float32, err error) {
	return vocab.PackActivations(x)
}
func PackActivationsInto(a []uint32, as []float32, sum []float32, x []float32, window int) error {
	return vocab.PackActivationsInto(a, as, sum, x, window)
}
func PackWeights(t Quant, src []byte, nrows int, k int) (qs []uint32, d []uint32, sc []uint32, err error) {
	return vocab.PackWeights(t, src, nrows, k)
}
func PackWeightsInto(t Quant, src []byte, nrows int, k int, qs []uint32, d []uint32, sc []uint32) error {
	return vocab.PackWeightsInto(t, src, nrows, k, qs, d, sc)
}
func PackedWords(t Quant, nrows int, k int) (qs int, d int, sc int, err error) {
	return vocab.PackedWords(t, nrows, k)
}
func PaddedBytes(n int) uint64                          { return vocab.PaddedBytes(n) }
func Ptr(v []uint32) unsafe.Pointer                     { return vocab.Ptr(v) }
func QuantOf(t quant.Type) (Quant, bool)                { return vocab.QuantOf(t) }
func Quants() int                                       { return vocab.Quants() }
func RopeTabBlock(npairs int) int                       { return vocab.RopeTabBlock(npairs) }
func RopeTabConsts() []float32                          { return vocab.RopeTabConsts() }
func RopeTabHeadSplit(u float64) (h float32, l float32) { return vocab.RopeTabHeadSplit(u) }
func RopeTabPlaneOff(p int, npairs int) int             { return vocab.RopeTabPlaneOff(p, npairs) }
func RopeTabScaleOff(npairs int) int                    { return vocab.RopeTabScaleOff(npairs) }
func RopeTabStride(npairs int) int                      { return vocab.RopeTabStride(npairs) }
func RowWindow(t Quant, qs []byte, d []byte, sc []byte, nrows int, k int, lo int, n int, to int, dqs []byte, dd []byte, dsc []byte) error {
	return vocab.RowWindow(t, qs, d, sc, nrows, k, lo, n, to, dqs, dd, dsc)
}
func ScField(sub int, min bool) (w int, b int, hi int) { return vocab.ScField(sub, min) }
func ScStream(q Quant) bool                            { return vocab.ScStream(q) }
func ScValue(words [ScStreamWords]uint32, sub int, min bool) uint8 {
	return vocab.ScValue(words, sub, min)
}
func ScaleLayout(q Quant) (perSuper int, scOff int) { return vocab.ScaleLayout(q) }
func SignedPayload(q Quant) bool                    { return vocab.SignedPayload(q) }
func UnmapOffHeap(b []byte)                         { vocab.UnmapOffHeap(b) }

// qinfo is a format's layout (vocab.QInfo) under the names the old
// generators read it by.
type qinfo struct {
	bits, hi, sub   int
	biasK           float32
	biasArray       bool
	blockB, blockE  int
	signedQ         bool
	perSuper, scOff int
	codes           *[16]uint8
	float           int
	bf16, e8m0      bool
	intAct, d32     bool
}

func infoOf(q Quant) qinfo {
	i := q.Info()
	return qinfo{bits: i.Bits, hi: i.Hi, sub: i.Sub, biasK: i.BiasK, biasArray: i.BiasArray, blockB: i.BlockB,
		blockE: i.BlockE, signedQ: i.SignedQ, perSuper: i.PerSuper, scOff: i.ScOff, codes: i.Codes, float: i.Float,
		bf16: i.BF16, e8m0: i.E8M0, intAct: i.IntAct, d32: i.D32}
}

// The activation sets a gated or an ungated kernel may bake (vocab.Gated,
// vocab.Ungated).
var (
	Gated   = vocab.Gated
	Ungated = vocab.Ungated
)
