package kernels

import "strconv"

// The activation a kernel bakes in.
//
// It has its own file because both architectures' emitters bake the same set,
// and a duplicated enum is how two backends come to disagree about which
// activation code 2 is. It is an enum rather than a bool because there are
// more than two ungated activations (CLIP's tower is quick-GELU).
type ActKind uint8

const (
	ActSiLU ActKind = iota
	ActGELU
	// ActQuickGELU is x*sigma(1.702x), the activation OpenAI's CLIP trains
	// with and the one llama.cpp selects when an mmproj sets neither
	// clip.use_gelu nor clip.use_silu.
	//
	// It is neither GELU nor SiLU and either substitution is silent: GELU-tanh
	// parts from it by about 0.02 around |x| = 2, and SiLU is it with 1.702
	// set to 1.
	ActQuickGELU
	// ActSigmoid is the activation alone: sigma(x), not x*sigma(x). A gated
	// attention output is out*sigma(g) with out and g different vectors, so
	// a SiLU kernel would compute g*sigma(g)*out, wrong by a factor of g.
	ActSigmoid
	// ActSwiGLUOAI is gpt-oss's gated activation, and it exists only gated:
	//
	//	x = min(gate, 7)   y = clamp(up, -7, 7)   out = x*sigma(1.702x) * (y+1)
	//
	// Three differences from SwiGLU, each silent alone: the gate is clamped
	// from above only, the up projection from both sides, and (y+1)
	// multiplies rather than y. The constants are the architecture's
	// (llama.cpp hardcodes alpha 1.702 and limit 7; the GGUF carries neither).
	ActSwiGLUOAI
	// ActReLU2 is max(x, 0)^2, the squared ReLU Nemotron's ungated FFN runs
	// (llama.cpp's LLM_FFN_RELU_SQR). It is the one activation here with no
	// exp, and like the others it is baked rather than branched.
	ActReLU2
	// ActReLU is max(x, 0): DeepSeek V3.2's lightning indexer applies it to
	// each head's score before the heads' weighted sum.
	ActReLU
	// ActIdentity is no activation: gated, dst = dst * up, an elementwise
	// product. Gemma 3n scales the active AltUp stream by a learned vector
	// before its per-layer input gate.
	ActIdentity
	// ActXIELU is Apertus's xIELU, ungated, with four numbers per block
	// (alpha_p, alpha_n, beta, eps). It is in neither set below: the numbers
	// are a buffer, so its kernels are XIELU here and cpu's EmitXIELU rather
	// than Act's baked forms.
	ActXIELU
	// ActSwiGLUClamp is DeepSeek V4's gated activation, SwiGLU with both
	// operands clamped and nothing else changed:
	//
	//	x = min(gate, 10)   y = clamp(up, -10, 10)   out = x*sigma(x) * y
	//
	// It is swiglu-oai's clamp without that kind's alpha (1.702) and offset
	// (y+1), at another limit; substituting either kind is silent. The limit
	// is the architecture's (jlm.Flag2SwiGLUClamp).
	ActSwiGLUClamp
	// ActSqrtSoftplus is sqrt(softplus(x)), DeepSeek V4's router gate over
	// each expert's logit. Ungated. softplus is the |x| form,
	// max(x,0) + log(1 + exp(-|x|)), so nothing overflows and no branch is
	// needed (the delta rule's gate builds it the same way).
	ActSqrtSoftplus
	// ActSitu is Kimi-K3's gated activation (Moonshot's SituAndMul), which
	// bounds both operands with a scaled tanh and gates the first by its
	// sigmoid:
	//
	//	out = 4*tanh(gate/4)*sigma(gate) * 25*tanh(up/25)
	//
	// The bounds are the architecture's (activation_situ_beta and
	// activation_situ_linear_beta, 4 and 25 in the published config); the
	// converter refuses another pair. Substituting SwiGLU is silent below the
	// bounds and wrong past them.
	ActSitu
	// ActGELUErf is GELU as transformers' GELUActivation and torch's default
	// compute it, 0.5x(1 + erf(x/sqrt 2)), where ActGELU is the tanh
	// approximation. erf is Abramowitz and Stegun 7.1.26 (absolute error
	// under 1.5e-7) on |x|, and the product is written
	// 0.5*((x + |x|) - |x|*q) with q = 1 - erf(|x|/sqrt 2), so a negative x
	// is the small q times |x| and nothing cancels.
	ActGELUErf
	// ActSqrtSigmoid is Qwen3.8-Flash-Next's n-gram gate,
	//
	//	sigmoid(sign(x) * sqrt(max(|x|, 1e-6)))
	//
	// a sigmoid of the signed square root, the floor on the magnitude the
	// reference's own (clamp_min(1e-6)). Ungated.
	ActSqrtSigmoid
	// ActTanh is tanh alone, the rational form on x clamped to +-TanhClamp
	// (tanhRational): an element a composed kernel bounds a value with
	// (b*tanh(x/b)), never an FFN's activation.
	ActTanh
	// ActSoftplus is log(1 + exp(x)) alone, softplus's series (softplus): an
	// element a composed kernel scores with, never an FFN's activation.
	ActSoftplus
)

// SqrtSigmoidFloor is ActSqrtSigmoid's floor on |x|.
const SqrtSigmoidFloor = float32(1e-6)

// GELUErfP and GELUErfA are ActGELUErf's erf: t = 1/(1 + P*z), and
// 1 - erf(z) = t*(A[0] + t*(A[1] + t*(A[2] + t*(A[3] + t*A[4])))) * exp(-z*z).
const GELUErfP = float32(0.3275911)

var GELUErfA = [5]float32{0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429}

// SituBeta and SituLinearBeta are the bounds ActSitu bakes.
const (
	SituBeta       = 4
	SituLinearBeta = 25
)

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

// Ungated is the set an ungated kernel may bake, in code order. It exists so a
// gate can enumerate the kinds rather than restate them.
var Ungated = [...]ActKind{ActSiLU, ActGELU, ActQuickGELU, ActReLU2, ActReLU, ActSqrtSoftplus, ActGELUErf,
	ActSqrtSigmoid}

// Gated is the set a gated kernel (dst = act(dst) combined with up) may bake,
// for the same reason.
var Gated = [...]ActKind{ActSiLU, ActGELU, ActSwiGLUOAI, ActIdentity, ActSwiGLUClamp, ActSitu, ActGELUErf, ActReLU2}

func (k ActKind) String() string {
	switch k {
	case ActSiLU:
		return "silu"
	case ActGELU:
		return "gelu"
	case ActQuickGELU:
		return "quick-gelu"
	case ActSigmoid:
		return "sigmoid"
	case ActSwiGLUOAI:
		return "swiglu-oai"
	case ActReLU2:
		return "relu2"
	case ActReLU:
		return "relu"
	case ActIdentity:
		return "identity"
	case ActXIELU:
		return "xielu"
	case ActSwiGLUClamp:
		return "swiglu-clamp"
	case ActSqrtSoftplus:
		return "sqrt-softplus"
	case ActSitu:
		return "situ"
	case ActGELUErf:
		return "gelu-erf"
	case ActSqrtSigmoid:
		return "sqrt-sigmoid"
	case ActTanh:
		return "tanh"
	case ActSoftplus:
		return "softplus"
	}
	return "act(" + strconv.Itoa(int(k)) + ")"
}
