//go:build amd64 && jitllmtest

package cpu

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
)

// The SSE tier's activation-quantizer gate. The kernels replace
// QuantizeQ8Window and QuantizeHalfSums, so the bar is bit equality: a
// last-bit scale difference moves a quantized value at the boundary.
//
// It runs on the forced SSE tier, and asserts the SSE tier mapped kernels and
// no AVX2 kernel was mapped, so the tested configuration is the one selected.
// The kernels are reached through QuantActKernels.Run because the driver
// (which shape serves a window, where a group of eight begins, what a partial
// worker must scan) is half of what is gated.

// underSSEQuant runs f with the SSE tier forced and fails if the force did not
// reach the probe, if f mapped no SSE kernel, or if it mapped an AVX2 one.
func underSSEQuant(t *testing.T, f func()) {
	t.Helper()
	old := ForceTierForTest(TierSSE)
	defer ForceTierForTest(old)
	if HostTier() != TierSSE {
		t.Fatalf("the force did not reach the probe: host tier %v", HostTier())
	}
	before := MappedByTier()
	f()
	after := MappedByTier()
	if d := after[TierAVX2] - before[TierAVX2]; d != 0 {
		t.Errorf("%d AVX2 kernel(s) were mapped under the forced SSE tier", d)
	}
	if after[TierSSE] == before[TierSSE] {
		t.Errorf("no SSE-tier kernel was mapped -- this arm did not run the tier it names")
	}
}

// sseQuantActKernels maps both SSE shapes for a format and checks the bytes
// declare the SSE tier and contain no VEX prefix before any of them executes.
func sseQuantActKernels(t *testing.T, q quant.Type) (QuantActKernels, func()) {
	t.Helper()
	half := NeedsHalfSums(q)
	kern := QuantActKernels{Konst: QuantActConsts(q)}
	var codes []*Code
	for _, k := range []struct {
		name string
		body []byte
		dst  **Code
	}{
		{"ssequantact", emitQuantActSSEBody(half), &kern.Wide},
		{"ssequantactnarrow", emitQuantActNarrowSSEBody(half), &kern.Narrow},
	} {
		if got := KernelTier(k.body); got != TierSSE {
			t.Fatalf("%v: %s was generated for tier %v, not SSE -- a VEX instruction "+
				"would fault on the host this kernel exists for", q, k.name, got)
		}
		requireSSEKernel(t, k.name, k.body)
		c, err := MapNamed(k.body, k.name+"_gate")
		if err != nil {
			t.Fatalf("%v: mapping %s: %v", q, k.name, err)
		}
		*k.dst, codes = c, append(codes, c)
	}
	return kern, func() {
		for _, c := range codes {
			c.Close()
		}
	}
}

// sseQuantActZeroWindow is an input whose second amax window is entirely zero,
// with negatives carrying the largest magnitude everywhere else.
//
// quantActInput's zero blocks sit inside a nonzero 256-element window, so the
// masked reciprocal needs this input (a ragged tile's zero padding would give
// 1/0 = +Inf and 0*Inf = NaN). The negatives are largest so a signed-maximum
// scan (a wrong absolute-value mask) reads a different amax.
func sseQuantActZeroWindow(k, window int, seed int64) []float32 {
	w := max(window, Q8Block)
	r := rand.New(rand.NewSource(seed))
	x := make([]float32, k)
	for i := range x {
		switch {
		case k >= 2*w && i >= w && i < 2*w:
			x[i] = 0
		case i%13 == 0:
			x[i] = -float32(r.Float64())*40 - 1
		default:
			x[i] = float32(r.NormFloat64())
		}
	}
	return x
}

// sseQuantActTies is an input made of exact rounding ties, positive and
// negative. Each window's first element is 127, so d = inv = 1 and every other
// element is a tie at x.5. math.Round (the Go loop) rounds half away from
// zero, while CVTPS2DQ under the default MXCSR rounds half to even, so a naive
// port differs only on this input.
func sseQuantActTies(k, window int) []float32 {
	w := max(window, Q8Block)
	x := make([]float32, k)
	for i := range x {
		if i%w == 0 {
			x[i] = 127 // the window's amax, so inv is exactly 1
			continue
		}
		h := float32(i%40) + 0.5
		if i%2 == 1 {
			h = -h
		}
		x[i] = h
	}
	return x
}

// sseQuantActShapes is what the gate sweeps: both window shapes, a k with fewer
// than eight blocks in total, whole groups of eight, and a k whose blocks do not
// divide into groups (2304/32 = 72 = 9 groups; 576/32 = 18 leaves two blocks
// over after two groups).
var sseQuantActShapes = []struct{ k, window int }{
	{64, 32}, {256, 32}, {576, 32}, {2048, 32},
	{512, 256}, {2048, 256}, {2304, 256},
}

// sseQuantActInputs names the three inputs every shape is driven with.
func sseQuantActInputs(k, window int) []struct {
	name string
	x    []float32
} {
	return []struct {
		name string
		x    []float32
	}{
		{"random", quantActInput(k, int64(k+window))},
		{"zerowindow", sseQuantActZeroWindow(k, window, int64(k*7+window))},
		{"ties", sseQuantActTies(k, window)},
	}
}

// sseQuantActDiff compares a kernel run against the Go loop and returns how many
// int8, pair and half words differ.
func sseQuantActDiff(gq, wq []int8, gp, wp, gh, wh []float32) (ints, pairs, halves int) {
	for i := range wq {
		if gq[i] != wq[i] {
			ints++
		}
	}
	for i := range wp {
		if math.Float32bits(gp[i]) != math.Float32bits(wp[i]) {
			pairs++
		}
	}
	for i := range wh {
		if math.Float32bits(gh[i]) != math.Float32bits(wh[i]) {
			halves++
		}
	}
	return
}

// TestSSEQuantActMatchesTheGoLoopExactly is the gate: every packed format, both
// window shapes, three inputs, and ragged block splits that cut inside a window.
func TestSSEQuantActMatchesTheGoLoopExactly(t *testing.T) {
	underSSEQuant(t, func() {
		ran, wide, narrow := 0, 0, 0
		for _, q := range quant.PackedTypes {
			kern, done := sseQuantActKernels(t, q)
			for _, sh := range sseQuantActShapes {
				// Ask the driver which kernel it takes, so a declined shape
				// cannot pass by comparing the Go loop with itself.
				if !kern.Applies(sh.window) {
					t.Fatalf("%v k=%d window=%d: the driver takes no SSE kernel here, "+
						"so this arm would compare the Go loop with itself",
						q, sh.k, sh.window)
				}
				for _, in := range sseQuantActInputs(sh.k, sh.window) {
					wq, wp, wh := quantActGo(q, in.x, sh.k, sh.window)
					ResetQuantActStats()
					gq, gp, gh := quantActRun(t, kern, q, in.x, sh.k, sh.window,
						quantActSplits(sh.k/Q8Block))
					w, n, gl := QuantActStats()
					if gl != 0 {
						t.Fatalf("%v k=%d window=%d %s: %d range(s) fell back to the Go loop "+
							"-- the kernels did not serve this shape", q, sh.k, sh.window, in.name, gl)
					}
					if w+n == 0 {
						t.Fatalf("%v k=%d window=%d %s: no kernel call was counted",
							q, sh.k, sh.window, in.name)
					}
					ints, pairs, halves := sseQuantActDiff(gq, wq, gp, wp, gh, wh)
					if ints+pairs+halves != 0 {
						for i := range wq {
							if gq[i] != wq[i] {
								t.Fatalf("%v k=%d window=%d %s: int8 %d differs: kernel %d, Go %d "+
									"(%d int8, %d pair, %d half words in all)",
									q, sh.k, sh.window, in.name, i, gq[i], wq[i], ints, pairs, halves)
							}
						}
						for i := range wp {
							if math.Float32bits(gp[i]) != math.Float32bits(wp[i]) {
								t.Fatalf("%v k=%d window=%d %s: pair %d differs: kernel %v (%#08x), "+
									"Go %v (%#08x)", q, sh.k, sh.window, in.name, i,
									gp[i], math.Float32bits(gp[i]), wp[i], math.Float32bits(wp[i]))
							}
						}
						for i := range wh {
							if math.Float32bits(gh[i]) != math.Float32bits(wh[i]) {
								t.Fatalf("%v k=%d window=%d %s: half %d differs: kernel %v (%#08x), "+
									"Go %v (%#08x)", q, sh.k, sh.window, in.name, i,
									gh[i], math.Float32bits(gh[i]), wh[i], math.Float32bits(wh[i]))
							}
						}
					}
					ran++
					wide += int(w)
					narrow += int(n)
				}
			}
			done()
		}
		if wide == 0 || narrow == 0 {
			t.Fatalf("the sweep made %d wide and %d narrow kernel calls -- both shapes must "+
				"run or this gate covers half the window range", wide, narrow)
		}
		t.Logf("%d (format, shape, input) arms are bit-identical to QuantizeQ8Window + "+
			"QuantizeHalfSums on the forced SSE tier (%d wide calls, %d narrow)",
			ran, wide, narrow)
	})
}

// TestSSEQuantActGateSeesItsViolations runs the gate above against a defect in
// each part of the kernel. Each is delivered through the constant block, so it
// breaks the shipping kernel itself: a wrong absolute-value mask, rounding
// magnitude, sign mask or correction breaks the scan, the round, the round's
// direction and the epilogue in turn.
func TestSSEQuantActGateSeesItsViolations(t *testing.T) {
	underSSEQuant(t, func() {
		const q = quant.Q6_K // half sums too, so the epilogue's third store is covered
		kern, done := sseQuantActKernels(t, q)
		defer done()

		for _, v := range []struct {
			name  string
			slot  int
			to    float32
			what  string
			input string
		}{
			{"the absolute-value mask", 3, math.Float32frombits(0xFFFFFFFF),
				"the amax scan takes a SIGNED maximum", "zerowindow"},
			{"the rounding magnitude", 2, 0,
				"the payload TRUNCATES instead of rounding half away from zero", "ties"},
			{"the sign mask", 4, math.Float32frombits(0),
				"copysign(0.5, x) becomes +0.5, so negatives round the wrong way", "ties"},
			{"the -biasC/8 correction", 6, -float32(-BiasC(q) / 8),
				"every pair's correction carries the wrong sign", "zerowindow"},
		} {
			for _, window := range []int{256, Q8Block} {
				const k = 2048
				nb := k / Q8Block
				var x []float32
				for _, in := range sseQuantActInputs(k, window) {
					if in.name == v.input {
						x = in.x
					}
				}
				wq, wp, wh := quantActGo(q, x, k, window)

				bad := kern
				bad.Konst = QuantActConsts(q)
				bad.Konst[v.slot] = v.to
				dst, pairs := make([]int8, k), make([]float32, 2*nb)
				half := make([]float32, k/16)
				scr := make([]float32, QuantActNarrowScratch)
				bad.Run(q, dst, pairs, half, scr, x, k, 0, nb, window)

				ints, prs, hlv := sseQuantActDiff(dst, wq, pairs, wp, half, wh)
				if ints+prs+hlv == 0 {
					t.Fatalf("window %d: %s was corrupted so that %s, and the kernel still "+
						"agreed with the Go loop in every word -- this comparison cannot fail "+
						"and certifies nothing", window, v.name, v.what)
				}
				t.Logf("window %3d: %-26s wrong (%s) -> %d of %d int8, %d of %d pair and "+
					"%d of %d half words differ", window, v.name, v.what,
					ints, len(wq), prs, len(wp), hlv, len(wh))
			}
		}
	})
}
