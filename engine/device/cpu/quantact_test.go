//go:build amd64 || arm64

package cpu

import (
	"github.com/jitllm/jitllm/internal/oracle"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
)

// The activation quantizer's gate. The kernel replaces QuantizeQ8Window and
// QuantizeHalfSums on the decode path, so the bar is bit equality: a scale that
// differs in the last bit moves a quantized weight at the boundary, which can
// flip an argmax many layers later.

// quantActGo is what the kernel has to reproduce, straight from the shipping
// Go functions rather than from a re-derivation of them.
func quantActGo(t quant.Type, x []float32, k, window int) ([]int8, []float32, []float32) {
	nb := k / Q8Block
	dst, pairs := make([]int8, k), make([]float32, 2*nb)
	oracle.QuantizeQ8Window(dst, pairs, x, BiasC(t), 0, nb, window)
	var half []float32
	if NeedsHalfSums(t) {
		half = make([]float32, k/16)
		oracle.QuantizeHalfSums(half, dst, BiasC(t))
	}
	return dst, pairs, half
}

// quantActRun drives the kernels over splits, which is how the pool calls
// them, with a scratch per split so a shared one cannot hide a race.
func quantActRun(t *testing.T, kern QuantActKernels, q quant.Type, x []float32, k, window int, splits [][2]int) ([]int8, []float32, []float32) {
	t.Helper()
	nb := k / Q8Block
	dst, pairs := make([]int8, k), make([]float32, 2*nb)
	var half []float32
	if NeedsHalfSums(q) {
		half = make([]float32, k/16)
	}
	for _, s := range splits {
		scr := make([]float32, QuantActNarrowScratch)
		kern.Run(q, dst, pairs, half, scr, x, k, s[0], s[1], window)
	}
	return dst, pairs, half
}

// quantActKernels builds both shapes for a format, or reports which are owed.
func quantActKernels(t *testing.T, q quant.Type) (QuantActKernels, func()) {
	t.Helper()
	em := Native()
	kern := QuantActKernels{Konst: QuantActConsts(q)}
	var codes []*Code
	if b, err := em.QuantAct(NeedsHalfSums(q)); err == nil {
		c, err := MapNamed(b, "quantact_gate")
		if err != nil {
			t.Fatalf("%v: mapping the wide kernel: %v", q, err)
		}
		kern.Wide, codes = c, append(codes, c)
	}
	if b, err := em.QuantActNarrow(NeedsHalfSums(q)); err == nil {
		c, err := MapNamed(b, "quantact_narrow_gate")
		if err != nil {
			t.Fatalf("%v: mapping the narrow kernel: %v", q, err)
		}
		kern.Narrow, codes = c, append(codes, c)
	}
	return kern, func() {
		for _, c := range codes {
			c.Close()
		}
	}
}

// quantActSplits cuts [0,nb) into ragged ranges, including cuts inside an amax
// window: the scale is a property of the window, so a worker owning half of one
// must still scan all of it, and a split that only ever landed on a window
// boundary would never test that.
func quantActSplits(nb int) [][2]int {
	if nb < 4 {
		return [][2]int{{0, nb}}
	}
	return [][2]int{{0, 1}, {1, nb / 2}, {nb / 2, nb - 1}, {nb - 1, nb}}
}

func quantActInput(k int, seed int64) []float32 {
	r := rand.New(rand.NewSource(seed))
	x := make([]float32, k)
	for i := range x {
		switch {
		// An all-zero window arrives as tile padding on ragged prompts; the
		// kernel masks inv (1/0 is +Inf, 0*Inf NaN) and the Go loop branches,
		// and the two must agree.
		case i >= 2*Q8Block && i < 4*Q8Block:
			x[i] = 0
		case i%97 == 0:
			x[i] = -float32(r.Float64()) * 30
		default:
			x[i] = float32(r.NormFloat64())
		}
	}
	return x
}

func TestQuantActMatchesTheGoLoopExactly(t *testing.T) {
	em := Native()
	ran, wide, narrow := 0, 0, 0
	for _, q := range quant.PackedTypes {
		kern, done := quantActKernels(t, q)
		if kern.Wide == nil && kern.Narrow == nil {
			// A tier whose emitters are owed keeps the Go loop; say so, since
			// an absent kernel must not read as a passing gate.
			t.Logf("%v: no activation-quantize kernel on tier %v", q, em.Tier)
			continue
		}
		for _, sh := range []struct{ k, window int }{
			// The narrow shapes exercise whole groups of eight blocks, a
			// group plus a remainder (2048/32 = 64 blocks split raggedly), and
			// a k with fewer than eight blocks in total.
			{64, 32}, {256, 32}, {2048, 32}, {512, 256}, {2048, 256}, {2304, 256},
		} {
			// Ask which kernel the driver takes, so a declined shape cannot
			// pass by comparing the Go loop with itself.
			applies := kern.Applies(sh.window)
			x := quantActInput(sh.k, int64(sh.k+sh.window))
			wq, wp, wh := quantActGo(q, x, sh.k, sh.window)
			gq, gp, gh := quantActRun(t, kern, q, x, sh.k, sh.window, quantActSplits(sh.k/Q8Block))
			for i := range wq {
				if gq[i] != wq[i] {
					t.Fatalf("%v k=%d window=%d: int8 %d differs: kernel %d, Go %d",
						q, sh.k, sh.window, i, gq[i], wq[i])
				}
			}
			for i := range wp {
				if math.Float32bits(gp[i]) != math.Float32bits(wp[i]) {
					t.Fatalf("%v k=%d window=%d: pair %d differs: kernel %v (%#08x), Go %v (%#08x)",
						q, sh.k, sh.window, i, gp[i], math.Float32bits(gp[i]),
						wp[i], math.Float32bits(wp[i]))
				}
			}
			for i := range wh {
				if math.Float32bits(gh[i]) != math.Float32bits(wh[i]) {
					t.Fatalf("%v k=%d window=%d: half %d differs: kernel %v, Go %v",
						q, sh.k, sh.window, i, gh[i], wh[i])
				}
			}
			if applies {
				ran++
				if sh.window > Q8Block {
					wide++
				} else {
					narrow++
				}
			}
		}
		done()
	}
	if Native().Tier == TierAVX2 && (wide == 0 || narrow == 0) {
		t.Fatalf("the AVX2 tier ran %d wide and %d narrow shapes -- both kernels must be "+
			"exercised or this gate covers only half the window range", wide, narrow)
	}
	t.Logf("%d (format, shape) pairs ran a KERNEL (%d wide, %d narrow) and are bit-identical "+
		"to the Go loop on tier %v", ran, wide, narrow, Native().Tier)
}

// TestQuantActGateSeesAWrongConstant runs the gate above against a violation.
// The constant block is where the two layouts differ, so feeding the kernel the
// GEMM's integer bias in place of -biasC/8 is the smallest real defect: every
// int8 is still right and every pair's correction is wrong.
func TestQuantActGateSeesAWrongConstant(t *testing.T) {
	q := quant.Q4_K
	kern, done := quantActKernels(t, q)
	defer done()

	// Both shapes, because they have different epilogues and a violation the
	// wide kernel fails says nothing about the narrow one.
	for _, window := range []int{256, Q8Block} {
		if !kern.Applies(window) {
			t.Skipf("no activation-quantize kernel for window %d on tier %v", window, Native().Tier)
		}
		const k = 512
		nb := k / Q8Block
		x := quantActInput(k, 7)
		_, want, _ := quantActGo(q, x, k, window)

		bad := kern
		bad.Konst = QuantActConsts(q)
		bad.Konst[6] = -bad.Konst[6] // the sign the correction carries
		dst, got := make([]int8, k), make([]float32, 2*nb)
		scr := make([]float32, QuantActNarrowScratch)
		bad.Run(q, dst, got, nil, scr, x, k, 0, nb, window)

		diff := 0
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				diff++
			}
		}
		if diff == 0 {
			t.Fatalf("window %d: the kernel agreed with the Go loop through a flipped bias "+
				"constant -- this comparison cannot fail and certifies nothing", window)
		}
		t.Logf("window %d: a flipped -biasC/8 moves %d of %d pair words", window, diff, len(want))
	}
}

// TestQuantActConstantsAreExact is the arithmetic claim the kernel rests on,
// checked by exhaustion rather than argued from mantissas.
//
// The Go loop computes float32(-float64(sum)*biasC/8) -- an f64 product
// narrowed once. The kernel converts the integer sum to f32 and multiplies by
// a single f32 constant. Those agree only if neither product rounds, so this
// sweeps every sum either can see: a block is 32 elements and each quantizes to
// at most 127 in magnitude, and a half-block is 16.
func TestQuantActConstantsAreExact(t *testing.T) {
	for _, q := range quant.PackedTypes {
		b := BiasC(q)
		for _, c := range []struct {
			name  string
			mul   float64
			limit int
		}{{"-biasC/8", -b / 8, 32 * 127}, {"-biasC/4", -b / 4, 16 * 127}} {
			f32 := float32(c.mul)
			for s := -c.limit; s <= c.limit; s++ {
				want := float32(float64(s) * c.mul)
				got := float32(s) * f32
				if math.Float32bits(got) != math.Float32bits(want) {
					t.Fatalf("%v %s=%v: sum %d gives %v in f32 and %v through f64 -- "+
						"the kernel's single-precision multiply is NOT exact for this "+
						"format, so its epilogue must do the f64 product the Go loop does",
						q, c.name, c.mul, s, got, want)
				}
			}
		}
	}
}

// TestEveryA64KernelTypeHasAConstantBlock is the invariant the arm64 harnesses
// rest on: they walk a64KernelTypes and pass &KernelConst(t)[0], so a type with
// no case in that switch is a panic rather than a missing optimisation. It is
// built on amd64 too, since the tests that index that slice run only on
// darwin/arm64.
func TestEveryA64KernelTypeHasAConstantBlock(t *testing.T) {
	for _, q := range a64KernelTypes {
		if len(KernelConst(q)) == 0 {
			t.Errorf("%v is in a64KernelTypes and KernelConst gives it no bytes -- "+
				"every harness that walks that list takes &konst[0] and panics", q)
		}
	}
}
