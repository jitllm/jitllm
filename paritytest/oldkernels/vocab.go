// The device formats' vocabulary (jit/gpu/kernels) under the names the old
// generators use: an alias for each type and constant, a call for each function.

package oldkernels

import (
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

type ActKind = kernels.ActKind
type Arena = kernels.Arena
type ArenaStats = kernels.ArenaStats
type Center = kernels.Center
type Packed = kernels.Packed
type QInfo = kernels.QInfo
type Quant = kernels.Quant

const ActGELUErf = kernels.ActGELUErf
const ActGELU = kernels.ActGELU
const ActIdentity = kernels.ActIdentity
const ActQuickGELU = kernels.ActQuickGELU
const ActReLU2 = kernels.ActReLU2
const ActReLU = kernels.ActReLU
const ActSiLU = kernels.ActSiLU
const ActSigmoid = kernels.ActSigmoid
const ActSitu = kernels.ActSitu
const ActSoftplus = kernels.ActSoftplus
const ActSqrtSigmoid = kernels.ActSqrtSigmoid
const ActSqrtSoftplus = kernels.ActSqrtSoftplus
const ActSwiGLUClamp = kernels.ActSwiGLUClamp
const ActSwiGLUOAI = kernels.ActSwiGLUOAI
const ActTanh = kernels.ActTanh
const ActXIELU = kernels.ActXIELU
const BFloat16 = kernels.BFloat16
const Float16 = kernels.Float16
const Float32 = kernels.Float32

const GELUErfP = kernels.GELUErfP

const MXFP4Bias = kernels.MXFP4Bias
const MXFP4 = kernels.MXFP4
const Q3_K = kernels.Q3_K
const Q4_0 = kernels.Q4_0
const Q4_K = kernels.Q4_K
const Q5_0 = kernels.Q5_0
const Q5_1 = kernels.Q5_1
const Q5_K = kernels.Q5_K
const Q6_K = kernels.Q6_K
const Q8_0 = kernels.Q8_0
const RopeTabConstWords = kernels.RopeTabConstWords
const RopeTabCosC4Word = kernels.RopeTabCosC4Word
const RopeTabCosC6Word = kernels.RopeTabCosC6Word
const RopeTabCosC8Word = kernels.RopeTabCosC8Word
const RopeTabDigitBits = kernels.RopeTabDigitBits
const RopeTabDigitWord = kernels.RopeTabDigitWord
const RopeTabDigits = kernels.RopeTabDigits
const RopeTabGrid = kernels.RopeTabGrid
const RopeTabHalfWord = kernels.RopeTabHalfWord
const RopeTabMagicWord = kernels.RopeTabMagicWord
const RopeTabMaxPos = kernels.RopeTabMaxPos
const RopeTabMod4MaskWord = kernels.RopeTabMod4MaskWord
const RopeTabNegHalfWord = kernels.RopeTabNegHalfWord
const RopeTabOneVecWord = kernels.RopeTabOneVecWord
const RopeTabOneWord = kernels.RopeTabOneWord
const RopeTabPio2HiWord = kernels.RopeTabPio2HiWord
const RopeTabPio2LoWord = kernels.RopeTabPio2LoWord
const RopeTabSignVecWord = kernels.RopeTabSignVecWord
const RopeTabSinC3Word = kernels.RopeTabSinC3Word
const RopeTabSinC5Word = kernels.RopeTabSinC5Word
const RopeTabSinC7Word = kernels.RopeTabSinC7Word
const RopeTabTwoVecWord = kernels.RopeTabTwoVecWord
const ScStreamWords = kernels.ScStreamWords
const SituBeta = kernels.SituBeta
const SituLinearBeta = kernels.SituLinearBeta
const SqrtSigmoidFloor = kernels.SqrtSigmoidFloor
const TQ2F = kernels.TQ2F
const TQ2 = kernels.TQ2
const TanhClamp = kernels.TanhClamp

func Addr(v []uint32) uintptr  { return kernels.Addr(v) }
func Codes(q Quant) *[16]uint8 { return kernels.Codes(q) }
func D32(q Quant) bool         { return kernels.D32(q) }
func DIndex(q Quant, super int, nrows int, r int) (idx int, slot int) {
	return kernels.DIndex(q, super, nrows, r)
}
func DSlots(q Quant) int                                                { return kernels.DSlots(q) }
func DeviceWhyNot(q Quant) string                                       { return kernels.DeviceWhyNot(q) }
func F16(u uint16) float32                                              { return kernels.F16(u) }
func FieldsOf(q Quant) int                                              { return kernels.FieldsOf(q) }
func FloatOf(t quant.Type) (Quant, bool)                                { return kernels.FloatOf(t) }
func HiPlane(q Quant) int                                               { return kernels.HiPlane(q) }
func HostAlign() uint64                                                 { return kernels.HostAlign() }
func IsFloat(q Quant) bool                                              { return kernels.IsFloat(q) }
func Layout(q Quant) (sub int, bits int, biasK float32, biasArray bool) { return kernels.Layout(q) }
func MapOffHeap(n int) (b []byte, mapped bool, err error)               { return kernels.MapOffHeap(n) }
func MinInD(q Quant) bool                                               { return kernels.MinInD(q) }
func NarrowScales(q Quant) bool                                         { return kernels.NarrowScales(q) }
func NewArena(retain uint64) *Arena                                     { return kernels.NewArena(retain) }
func PackActivations(x []float32) (a []uint32, as []float32, sum16 []float32, err error) {
	return kernels.PackActivations(x)
}
func PackActivationsInto(a []uint32, as []float32, sum []float32, x []float32, window int) error {
	return kernels.PackActivationsInto(a, as, sum, x, window)
}
func PackWeights(t Quant, src []byte, nrows int, k int) (qs []uint32, d []uint32, sc []uint32, err error) {
	return kernels.PackWeights(t, src, nrows, k)
}
func PackWeightsInto(t Quant, src []byte, nrows int, k int, qs []uint32, d []uint32, sc []uint32) error {
	return kernels.PackWeightsInto(t, src, nrows, k, qs, d, sc)
}
func PackedWords(t Quant, nrows int, k int) (qs int, d int, sc int, err error) {
	return kernels.PackedWords(t, nrows, k)
}
func PaddedBytes(n int) uint64                          { return kernels.PaddedBytes(n) }
func Ptr(v []uint32) unsafe.Pointer                     { return kernels.Ptr(v) }
func QuantOf(t quant.Type) (Quant, bool)                { return kernels.QuantOf(t) }
func Quants() int                                       { return kernels.Quants() }
func RopeTabBlock(npairs int) int                       { return kernels.RopeTabBlock(npairs) }
func RopeTabConsts() []float32                          { return kernels.RopeTabConsts() }
func RopeTabHeadSplit(u float64) (h float32, l float32) { return kernels.RopeTabHeadSplit(u) }
func RopeTabPlaneOff(p int, npairs int) int             { return kernels.RopeTabPlaneOff(p, npairs) }
func RopeTabScaleOff(npairs int) int                    { return kernels.RopeTabScaleOff(npairs) }
func RopeTabStride(npairs int) int                      { return kernels.RopeTabStride(npairs) }
func RowWindow(t Quant, qs []byte, d []byte, sc []byte, nrows int, k int, lo int, n int, to int, dqs []byte, dd []byte, dsc []byte) error {
	return kernels.RowWindow(t, qs, d, sc, nrows, k, lo, n, to, dqs, dd, dsc)
}
func ScField(sub int, min bool) (w int, b int, hi int) { return kernels.ScField(sub, min) }
func ScStream(q Quant) bool                            { return kernels.ScStream(q) }
func ScValue(words [ScStreamWords]uint32, sub int, min bool) uint8 {
	return kernels.ScValue(words, sub, min)
}
func ScaleLayout(q Quant) (perSuper int, scOff int) { return kernels.ScaleLayout(q) }
func SignedPayload(q Quant) bool                    { return kernels.SignedPayload(q) }
func UnmapOffHeap(b []byte)                         { kernels.UnmapOffHeap(b) }

// qinfo is a format's layout (kernels.QInfo) under the names the old
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
