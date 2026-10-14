package cpu

import "github.com/jitllm/jitllm/jit/gpu/kernels"

// ActKind is the activation a kernel bakes in. It is the kernels package's
// type, so the host emitters and the device kernels name the same set with the
// same codes -- see kernels.ActKind for what each one is.
type ActKind = kernels.ActKind

const (
	ActSiLU      = kernels.ActSiLU
	ActGELU      = kernels.ActGELU
	ActQuickGELU = kernels.ActQuickGELU
	ActReLU2     = kernels.ActReLU2
	ActReLU      = kernels.ActReLU
	ActIdentity  = kernels.ActIdentity
	actSigmoid   = kernels.ActSigmoid
	ActSwiGLUOAI = kernels.ActSwiGLUOAI
	ActXIELU     = kernels.ActXIELU
	// ActSwiGLUClamp and ActSqrtSoftplus are DeepSeek V4's: the clamped
	// SwiGLU of its experts and the sqrt-softplus of its router.
	ActSwiGLUClamp  = kernels.ActSwiGLUClamp
	ActSqrtSoftplus = kernels.ActSqrtSoftplus
	// ActSitu is Kimi-K3's: both operands bounded by a scaled tanh.
	ActSitu = kernels.ActSitu
	// ActGELUErf is GELU with erf, transformers' and torch's default.
	ActGELUErf = kernels.ActGELUErf
	// ActSqrtSigmoid is Qwen3.8-Flash-Next's n-gram gate.
	ActSqrtSigmoid = kernels.ActSqrtSigmoid
)

// Ungated and Gated are kernels.Ungated and kernels.Gated.
var (
	Ungated = kernels.Ungated
	Gated   = kernels.Gated
)
