package cpu

import (
	"math"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

const (
	expLog2e  = float32(math.Log2E)
	expLn2Hi  = float32(0.693359375)    // exact in f32: 0.693359375 = 1421/2048
	expLn2Lo  = float32(-2.12194440e-4) // ln2 - hi
	expClampL = float32(-87.3)          // exp underflows f32 below this
	// And an upper clamp, which the softmax never needs (it exps x-max <= 0)
	// but GELU does: its tanh feeds exp 2z, which on a real FFN gate can run
	// the exponent build off the end of the f32 field. Clamped, exp returns
	// ~1.6e38 and tanh saturates to 1.
	expClampH = float32(88.0)
)

// expConsts is the constant block emitExpPS reads, in the order it loads them.
// Element 9 holds the bits 127 rather than the value: the exponent bias is an
// integer and the block is float32-typed only because everything else in it is.
func expConstsShared() []float32 {
	return []float32{
		expLog2e,                  // 0
		-expLn2Hi,                 // 1   r += n * (-hi)
		-expLn2Lo,                 // 2   r += n * (-lo)
		float32(1.0 / 120),        // 3   c5
		float32(1.0 / 24),         // 4   c4
		float32(1.0 / 6),          // 5   c3
		float32(0.5),              // 6   c2
		1,                         // 7   one
		expClampL,                 // 8   clamp low
		math.Float32frombits(127), // 9   the exponent bias, as bits
		expClampH,                 // 10  clamp high
	}
}

// expConsts is the amd64 spelling; the block is identical on both.
func expConsts() []float32 { return expConstsShared() }

// ActConsts is the block every elementwise kernel reads, exported for nn.
func ActConsts() []float32 { return actConsts() }

// actConsts extends expConsts with what the two activations need.
//
// Appended rather than interleaved for the same reason cpu.Args is append-only:
// emitExpPS addresses its block by byte offset, so anything inserted before
// offset 40 silently repoints every load in it.
func actConsts() []float32 {
	c := expConsts()
	return append(c,
		0.044715,   // 44
		0.79788456, // 48  sqrt(2/pi)
		2,          // 52
		0.5,        // 56
		1.702,      // 60  quick-GELU's exponent scale (CLIP), and swiglu-oai's alpha
		7,          // 64  swiglu-oai's limit
		-7,         // 68  and its negation, for the clamp on up
		10,         // 72  DeepSeek V4's SwiGLU clamp (ActSwiGLUClamp)
		-10,        // 76  and its negation
		// 80..100: sqrt-softplus's log1p, deltaGateConsts' atanh series (see
		// there for why five terms on u in (0,1]), and the |x| mask.
		2,                                // 80
		float32(1.0/9),                   // 84
		float32(1.0/7),                   // 88
		float32(1.0/5),                   // 92
		float32(1.0/3),                   // 96
		math.Float32frombits(0x7FFFFFFF), // 100
		// 104..116: Kimi-K3's situ (ActSitu), c*tanh(x/c) at its two bounds
		// as 1/c then c.
		float32(1.0/kernels.SituBeta),       // 104
		kernels.SituBeta,                    // 108
		float32(1.0/kernels.SituLinearBeta), // 112
		kernels.SituLinearBeta,              // 116
		// 120..168: situ's tanh, kernels.TanhRational's clamp and its odd
		// numerator (x^13 down to x) and even denominator (x^6 down to 1).
		kernels.TanhClamp,                                                      // 120
		-kernels.TanhClamp,                                                     // 124
		kernels.TanhP[0], kernels.TanhP[1], kernels.TanhP[2], kernels.TanhP[3], // 128..140
		kernels.TanhP[4], kernels.TanhP[5], kernels.TanhP[6], // 144..152
		kernels.TanhQ[0], kernels.TanhQ[1], kernels.TanhQ[2], kernels.TanhQ[3], // 156..168
		// 172..176: the log-softmax's ln of its sum (emitLnSum): the f32
		// mantissa mask, as bits, and ln 2.
		math.Float32frombits(0x007FFFFF), // 172
		float32(math.Ln2),                // 176
		// 180..204: ActGELUErf's 1/sqrt 2 and its erf (kernels.GELUErfP,
		// then GELUErfA from the first coefficient to the fifth).
		float32(1/math.Sqrt2),                    // 180
		kernels.GELUErfP,                         // 184
		kernels.GELUErfA[0], kernels.GELUErfA[1], // 188, 192
		kernels.GELUErfA[2], kernels.GELUErfA[3], // 196, 200
		kernels.GELUErfA[4], // 204
		// 208..212: ActSqrtSigmoid's floor on |x| and the sign bit.
		kernels.SqrtSigmoidFloor,         // 208
		math.Float32frombits(0x80000000), // 212
	)
}

// DeltaGateConsts is the block EmitDeltaGate reads, exported for nn.
func DeltaGateConsts() []float32 { return deltaGateConsts() }

// deltaGateConsts is its own block rather than an extension of actConsts,
// because emitExpPS holds actConsts in a register the whole time and this
// kernel needs a second cursor anyway.
//
// The log1p series is atanh's: with s = u/(2+u), log(1+u) = 2*atanh(s) =
// 2s*(1 + s^2/3 + s^4/5 + ...), and u in (0,1] maps to s in (0,1/3], so five
// terms leave ~1.5e-6 relative error (a plain power series in u would need
// dozens). u is exp(-|z|), which is what bounds it: the caller must present
// softplus in the |z| form (see EmitDeltaGate).
func deltaGateConsts() []float32 {
	c := expConsts()
	return append(c,
		2,                                // 44
		float32(1.0/9),                   // 48  the highest atanh term
		float32(1.0/7),                   // 52
		float32(1.0/5),                   // 56
		float32(1.0/3),                   // 60
		math.Float32frombits(0x7FFFFFFF), // 64  the |x| mask
	)
}

// SoftmaxPad is the length the caller must present.
//
// The padding must contain -FLT_MAX, not the exp clamp (-87.3): softmax
// subtracts the row's maximum first, so padding at the clamp is inert only
// while every real score is above it, and real attention scores can be far
// below (a single-element softmax then returned 1.7e-39 instead of 1).
// -FLT_MAX is below every finite score, so it is never the maximum, and its exp
// argument clamps to zero. The kernels seed their max from the row itself.
func SoftmaxPad(n int) int { return (n + elemLanes - 1) &^ (elemLanes - 1) }

// ElemLanes is how many floats one vector of the elementwise kernels covers:
// eight on amd64's 256-bit YMM, four on arm64's 128-bit NEON. The callers
// express their trip counts in these, so the same nn code drives both.
const ElemLanes = elemLanes
