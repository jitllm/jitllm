// Package oracle is the engine's arithmetic written in plain Go: the reference
// every generated kernel is gated against, and nothing else.
//
// Only tests may import it (TestNoProductionCodeImportsTheOracle): a fallback
// to Go arithmetic hides a missing kernel, which is a kernel to generate.
package oracle

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/format/quant"
)

// MatVec computes out[r] = dot(row r of w, x), for r in [0, nrows).
//
// w is packed GGUF bytes for nrows consecutive rows of k columns. Rows are
// dequantized one at a time into a scratch buffer; nothing expands a whole
// tensor.
//
// A GGUF weight matrix has ne0 = input features (the contiguous, fastest-varying
// axis) and ne1 = output features, so nrows is ne1 and k is ne0.
func MatVec(out []float32, wt quant.Type, w []byte, x []float32, nrows, k int) error {
	if len(out) < nrows {
		return fmt.Errorf("oracle: MatVec: out has %d elements, need %d", len(out), nrows)
	}
	if len(x) != k {
		return fmt.Errorf("oracle: MatVec: x has %d elements, need %d", len(x), k)
	}
	if uint64(k)%wt.BlockElems() != 0 {
		return fmt.Errorf("oracle: MatVec: k=%d is not a multiple of %s's %d elements per block", k, wt, wt.BlockElems())
	}
	rowBytes := uint64(k) / wt.BlockElems() * wt.BlockBytes()
	row := make([]float32, k) // one allocation per call, reused across rows
	for r := 0; r < nrows; r++ {
		lo := uint64(r) * rowBytes
		if lo+rowBytes > uint64(len(w)) {
			return fmt.Errorf("oracle: MatVec: row %d needs bytes [%d,%d) of %d", r, lo, lo+rowBytes, len(w))
		}
		if err := quant.Dequant32(wt, w[lo:lo+rowBytes], row); err != nil {
			return err
		}
		// Neumaier-compensated: ~2*eps independent of k, where a naive
		// sequential f32 sum drifts ~sqrt(k)*eps and would force the NMSE bound
		// open wide enough to hide real kernel bugs. It also keeps the oracle a
		// different method (compensated scalar) from the kernel (vector FMA).
		var sum, comp float32
		for i, v := range row {
			p := v * x[i]
			t := sum + p
			if abs32(sum) >= abs32(p) {
				comp += (sum - t) + p // sum is larger: p's low bits are lost
			} else {
				comp += (p - t) + sum
			}
			sum = t
		}
		out[r] = sum + comp
	}
	return nil
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// Row dequantizes row r of a weight matrix, e.g. one token's embedding.
func Row(dst []float32, wt quant.Type, w []byte, r, k int) error {
	rowBytes := uint64(k) / wt.BlockElems() * wt.BlockBytes()
	lo := uint64(r) * rowBytes
	if lo+rowBytes > uint64(len(w)) {
		return fmt.Errorf("oracle: Row: row %d out of range", r)
	}
	return quant.Dequant32(wt, w[lo:lo+rowBytes], dst[:k])
}

// RMSNorm computes y = x / sqrt(mean(x^2) + eps) * w.
//
// The mean is over the whole vector and the reciprocal square root is applied
// before the weight, matching ggml_rms_norm followed by ggml_mul. Note that some
// architectures bake a +1 into the stored weight at conversion time, so this
// never adds one itself — see model.Config.
func RMSNorm(y, x, w []float64, eps float64) {
	var sum float64
	for _, v := range x {
		sum += v * v
	}
	scale := 1 / math.Sqrt(sum/float64(len(x))+eps)
	for i, v := range x {
		y[i] = v * scale * w[i]
	}
}

// LayerNorm is the mean-and-variance normalisation (vision towers, BERT): it
// subtracts the mean, divides by sqrt(variance), and adds an optional bias b.
//
// The variance is computed from the already-centred values rather than as
// E[x^2] - E[x]^2, which cancels catastrophically at residual-stream
// magnitudes and gives a subtly wrong norm.
func LayerNorm(y, x, w, b []float64, eps float64) {
	n := float64(len(x))
	var mean float64
	for _, v := range x {
		mean += v
	}
	mean /= n
	var varsum float64
	for _, v := range x {
		d := v - mean
		varsum += d * d
	}
	scale := 1 / math.Sqrt(varsum/n+eps)
	if b == nil {
		for i, v := range x {
			y[i] = (v - mean) * scale * w[i]
		}
		return
	}
	for i, v := range x {
		y[i] = (v-mean)*scale*w[i] + b[i]
	}
}

// SiLU is x * sigmoid(x), the SwiGLU gate activation.
func SiLU(x float64) float64 { return x / (1 + math.Exp(-x)) }

// SwiGLUOAI is gpt-oss's gated activation of one (gate, up) pair: the gate
// clamped from above at 7, up clamped to [-7, 7], and (up+1) multiplying
// gate*sigma(1.702*gate). See cpu.ActSwiGLUOAI.
func SwiGLUOAI(gate, up float64) float64 {
	x := math.Min(gate, 7)
	y := math.Max(math.Min(up, 7), -7)
	return x / (1 + math.Exp(-1.702*x)) * (y + 1)
}

// SwiGLUClamp is DeepSeek V4's gated activation of one (gate, up) pair: the
// gate clamped from above at 10, up clamped to [-10, 10], and nothing else
// changed. See cpu.ActSwiGLUClamp.
func SwiGLUClamp(gate, up float64) float64 {
	x := math.Min(gate, 10)
	return x / (1 + math.Exp(-x)) * math.Max(math.Min(up, 10), -10)
}

// Situ is Kimi-K3's gated activation of one (gate, up) pair, Moonshot's
// SituAndMul at the published bounds: 4*tanh(gate/4)*sigma(gate) times
// 25*tanh(up/25). See cpu.ActSitu.
func Situ(gate, up float64) float64 {
	return 4 * math.Tanh(gate/4) / (1 + math.Exp(-gate)) * 25 * math.Tanh(up/25)
}

// SqrtSoftplus is DeepSeek V4's router gate, sqrt(log(1 + exp(x))), with
// torch's softplus threshold (x past 20 is x). See cpu.ActSqrtSoftplus.
func SqrtSoftplus(x float64) float64 {
	if x > 20 {
		return math.Sqrt(x)
	}
	return math.Sqrt(math.Log1p(math.Exp(x)))
}

// SqrtSigmoid is Qwen3.8-Flash-Next's n-gram gate,
// sigmoid(sign(x)*sqrt(max(|x|, 1e-6))), as its reference computes it.
func SqrtSigmoid(x float64) float64 {
	m := math.Sqrt(math.Max(math.Abs(x), 1e-6))
	if x < 0 {
		m = -m
	}
	return 1 / (1 + math.Exp(-m))
}

// GELUErf is GELU as transformers' GELUActivation and torch compute it,
// 0.5x(1 + erf(x/sqrt 2)), with Go's erf.
func GELUErf(x float64) float64 { return 0.5 * x * (1 + math.Erf(x/math.Sqrt2)) }

// GELUTanh is the tanh approximation of GELU, which is what gemma uses and what
// ggml's GGML_UNARY_OP_GELU computes.
func GELUTanh(x float64) float64 {
	const c = 0.7978845608028654 // sqrt(2/pi)
	return 0.5 * x * (1 + math.Tanh(c*(x+0.044715*x*x*x)))
}

// XIELU is Apertus's xIELU of one element, its four numbers in their
// effective form (alpha_p and alpha_n already through the softplus, alpha_n
// with beta added): transformers' XIELUActivation._xielu_python. A float64
// expm1, so it is an independent method from the kernels' polynomial.
func XIELU(x, alphaP, alphaN, beta, eps float64) float64 {
	if x > 0 {
		return alphaP*x*x + beta*x
	}
	return (math.Expm1(math.Min(x, eps))-x)*alphaN + beta*x
}

// Dot32 and AxpyF32 are the two float32 primitives attention is made of. Both
// reslice the source to the destination's length so the loop's bounds check
// is eliminated.

// Dot32 returns the dot product of a and the first len(a) elements of b.
func Dot32(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float32
	i := 0
	// Four accumulators: a single chain is latency-bound on the FP add.
	for ; i+4 <= len(a); i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

// AxpyF32 computes dst += w * src, over len(dst) elements.
func AxpyF32(dst []float32, w float32, src []float32) {
	src = src[:len(dst)]
	for i := range dst {
		dst[i] += w * src[i]
	}
}

// The f32 elementwise ops, matching the engine's f32 residual. The accumulators
// stay float64: a sum of squares and a softmax denominator span a range where
// f32 loses the tail, and each is one scalar against a vector of work.

// RMSNorm32 is RMSNorm over an f32 residual, accumulating in f64.
func RMSNorm32(y, x, w []float32, eps float64) {
	var sum float64
	for _, v := range x {
		sum += float64(v) * float64(v)
	}
	scale := 1 / math.Sqrt(sum/float64(len(x))+eps)
	for i, v := range x {
		y[i] = float32(float64(v) * scale * float64(w[i]))
	}
}

// Softmax32 is the three-pass softmax over an f32 row, accumulating in f64.
//
// Three-pass rather than online, for the reason Softmax records.
func Softmax32(x []float32) {
	if len(x) == 0 {
		return
	}
	mx := x[0]
	for _, v := range x[1:] {
		if v > mx {
			mx = v
		}
	}
	var sum float64
	m := float64(mx)
	for i, v := range x {
		e := math.Exp(float64(v) - m)
		x[i] = float32(e)
		sum += e
	}
	inv := 1 / sum
	for i, v := range x {
		x[i] = float32(float64(v) * inv)
	}
}

// LogSoftmax32 is the log-softmax of an f32 row, x - max - ln(sum exp(x -
// max)), accumulated in f64 and returned in f64 so a gate can hold an f32
// kernel to it without the reference's own rounding.
func LogSoftmax32(x []float32) []float64 {
	out := make([]float64, len(x))
	if len(x) == 0 {
		return out
	}
	mx := x[0]
	for _, v := range x[1:] {
		if v > mx {
			mx = v
		}
	}
	var sum float64
	m := float64(mx)
	for _, v := range x {
		sum += math.Exp(float64(v) - m)
	}
	lse := m + math.Log(sum)
	for i, v := range x {
		out[i] = float64(v) - lse
	}
	return out
}

// Widen and Narrow convert between the oracle's float64 and the engine's f32.
func Widen(dst []float64, src []float32) {
	for i, v := range src {
		dst[i] = float64(v)
	}
}

// Narrow is Widen's inverse.
func Narrow(dst []float32, src []float64) {
	for i, v := range src {
		dst[i] = float32(v)
	}
}

// LayerNorm32 is LayerNorm over f32, for the vision tower. b may be nil.
//
// The variance is computed from the centred values, for the reason LayerNorm
// records; the hazard is worse in f32.
func LayerNorm32(y, x, w, b []float32, eps float64) {
	var mean float64
	for _, v := range x {
		mean += float64(v)
	}
	mean /= float64(len(x))
	var varr float64
	for _, v := range x {
		d := float64(v) - mean
		varr += d * d
	}
	varr /= float64(len(x))
	inv := 1 / math.Sqrt(varr+eps)
	for i, v := range x {
		z := (float64(v) - mean) * inv * float64(w[i])
		if b != nil {
			z += float64(b[i])
		}
		y[i] = float32(z)
	}
}

// fastExp is exp(x) for the softmax path, valid only for x <= 0: softmax
// subtracts the maximum first, which removes overflow and the checks a general
// exp needs.
//
// exp(x) = 2^n * exp(r) with n = round(x*log2e) and r = x - n*ln2, so
// |r| <= ln2/2 and a short polynomial covers it; 2^n is placed directly in a
// float64's exponent field. Accuracy is asserted by a test.
const (
	log2e  = 1.4426950408889634
	ln2hi  = 0.6931471803691238
	ln2lo  = 1.9082149292705877e-10
	expMin = -745.0 // below this the result underflows to zero
)

func fastExp(x float64) float64 {
	if x < expMin {
		return 0
	}
	// n = round(x*log2e), computed without a branch: x is non-positive, so
	// subtracting 0.5 and truncating toward zero rounds to nearest.
	n := int(x*log2e - 0.5)
	fn := float64(n)
	// Two-part ln2 keeps the reduced argument accurate for large |x|.
	r := x - fn*ln2hi - fn*ln2lo
	// exp(r) on |r| <= ln2/2, degree 7 by Horner.
	//
	// Degree 7 gives ~5e-9 (truncation r^8/40320); degree 5 measured 3.2e-6.
	p := 1 + r*(1+r*(0.5+r*(1.0/6+r*(1.0/24+r*(1.0/120+r*(1.0/720+r*(1.0/5040)))))))
	return p * math.Float64frombits(uint64(n+1023)<<52)
}

// Softmax normalises x in place, exponentiating relative to the maximum.
//
// Three passes rather than the one-pass "online" form: the two are the same
// speed on unsorted rows, but the online form's rescale makes it degrade to
// O(n^2) on ascending input (111x at n=4096, bench.TestSoftmaxOnline) and
// compounds fastExp's error. Three passes have no data-dependent cost.
func Softmax(x []float64) {
	if len(x) == 0 {
		return
	}
	mx := x[0]
	for _, v := range x {
		if v > mx {
			mx = v
		}
	}
	var sum float64
	for i, v := range x {
		e := fastExp(v - mx)
		x[i] = e
		sum += e
	}
	inv := 1 / sum
	for i := range x {
		x[i] *= inv
	}
}

// SoftmaxOnline is the fused-max-and-sum "online" softmax, the B arm of the
// experiment in bench/. It rescales everything accumulated whenever a larger
// element arrives: O(n) expected work on random input, O(n^2) on ascending.
func SoftmaxOnline(x []float64) {
	if len(x) == 0 {
		return
	}
	mx := x[0]
	var sum float64
	for i, v := range x {
		if v > mx {
			s := fastExp(mx - v)
			sum *= s
			for j := 0; j < i; j++ {
				x[j] *= s
			}
			mx = v
		}
		e := fastExp(v - mx)
		x[i] = e
		sum += e
	}
	inv := 1 / sum
	for i := range x {
		x[i] *= inv
	}
}

// RopeApply rotates one head by a table nn.Rope.Table filled: pairs (p,
// p+len(cs)/2) for NEOX, (2p, 2p+1) otherwise. It is the reference the
// generated rotary kernels are held to.
func RopeApply(head []float32, cs []float32, neox bool) {
	half := len(cs) / 2
	if half > len(head)/2 {
		half = len(head) / 2
	}
	for p := 0; p < half; p++ {
		c, s := float64(cs[2*p]), float64(cs[2*p+1])
		i0, i1 := 2*p, 2*p+1
		if neox {
			i0, i1 = p, p+half
		}
		x0, x1 := float64(head[i0]), float64(head[i1])
		head[i0] = float32(x0*c - x1*s)
		head[i1] = float32(x0*s + x1*c)
	}
}
