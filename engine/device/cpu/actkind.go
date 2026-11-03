package cpu

import (
	"github.com/jitllm/jitllm/engine/graph"
)

// ActKind is the activation a kernel bakes in. It is the kernels package's
// type, so the host emitters and the device kernels name the same set with the
// same codes -- see kernels.ActKind for what each one is.
type ActKind = graph.ActKind

const (
	ActSiLU      = graph.ActSiLU
	ActGELU      = graph.ActGELU
	ActQuickGELU = graph.ActQuickGELU
	ActReLU2     = graph.ActReLU2
	ActReLU      = graph.ActReLU
	ActIdentity  = graph.ActIdentity
	actSigmoid   = graph.ActSigmoid
	ActSwiGLUOAI = graph.ActSwiGLUOAI
	ActXIELU     = graph.ActXIELU
	// ActSwiGLUClamp and ActSqrtSoftplus are DeepSeek V4's: the clamped
	// SwiGLU of its experts and the sqrt-softplus of its router.
	ActSwiGLUClamp  = graph.ActSwiGLUClamp
	ActSqrtSoftplus = graph.ActSqrtSoftplus
	// ActSitu is Kimi-K3's: both operands bounded by a scaled tanh.
	ActSitu = graph.ActSitu
	// ActGELUErf is GELU with erf, transformers' and torch's default.
	ActGELUErf = graph.ActGELUErf
	// ActSqrtSigmoid is Qwen3.8-Flash-Next's n-gram gate.
	ActSqrtSigmoid = graph.ActSqrtSigmoid
)

// Ungated and Gated are kernels.Ungated and kernels.Gated.
var (
	Ungated = graph.Ungated()
	Gated   = graph.Gated()
)

// GELUErfP and GELUErfA are ActGELUErf's erf: t = 1/(1 + P*z), and
// 1 - erf(z) = t*(A[0] + t*(A[1] + t*(A[2] + t*(A[3] + t*A[4])))) * exp(-z*z).
const GELUErfP = float32(0.3275911)

var GELUErfA = [5]float32{0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429}

// ActSitu's tanh is a rational in x, x*P(x^2)/Q(x^2) on x clamped to
// +-TanhClamp (Eigen's generic_fast_tanh_float), on every tier. The exp form
// GELU and the softcap use, 1 - 2/(exp(2x)+1), loses its relative precision
// as x goes to zero, and situ multiplies the result back up by its bound (25
// on up), which takes that loss past the 1e-5 every elementwise kernel here
// holds; the rational keeps ~3e-7 relative everywhere. TanhP is the
// numerator's odd coefficients from x^13 down to x, TanhQ the denominator's
// even ones from x^6 down to 1.
const TanhClamp = float32(7.90531110763549805)

var (
	TanhP = [7]float32{-2.76076847742355e-16, 2.00018790482477e-13, -8.60467152213735e-11,
		5.12229709037114e-08, 1.48572235717979e-05, 6.37261928875436e-04, 4.89352455891786e-03}
	TanhQ = [4]float32{1.19825839466702e-06, 1.18534705686654e-04, 2.26843463243900e-03,
		4.89352518554385e-03}
)
