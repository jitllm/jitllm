//go:build jitllmbench && amd64

package cpu

import (
	"math/rand"
	"sort"
	"testing"
	"time"

	"github.com/jitllm/jitllm/format/quant"
)

// TestQuantActRate prices the generated activation quantizer against the Go
// loop it replaces, per format, over a fixed buffer.
//
// It measures at the kernel because whole-token and phase-bracketing harnesses
// are dominated by the pool's dispatch noise around a small phase. The two arms
// run the same buffer through the same driver, alternated, with only the
// kernel pointers differing. The result is the phase's speed-up, not a token
// claim; a token gains that times the phase's share (JITLLM_PROFILE).
func TestQuantActRate(t *testing.T) {
	const iters = 400
	rounds := 9

	type row struct {
		q      quant.Type
		k      int
		window int
	}
	rows := []row{
		{quant.Q4_K, 2048, 256}, // the k-quant decode window, and the common case
		{quant.Q4_K, 8192, 256}, // an FFN width
		{quant.Q6_K, 2048, 256}, // two-plane, and it needs the per-16 sums
		{quant.Q4_0, 2048, Q8Block},
		{quant.Q8_0, 8192, Q8Block}, // gemma-2b's two formats, the narrow window
	}

	for _, r := range rows {
		kern, done := quantActKernels(t, r.q)
		if !kern.Applies(r.window) {
			t.Logf("%v window %d: no kernel on tier %v", r.q, r.window, Native().Tier)
			done()
			continue
		}
		goOnly := QuantActKernels{Konst: kern.Konst} // both codes nil: the Go loop

		nb := r.k / Q8Block
		x := make([]float32, r.k)
		rng := rand.New(rand.NewSource(int64(r.k)))
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		dst := make([]int8, r.k)
		pairs := make([]float32, 2*nb)
		var half []float32
		if NeedsHalfSums(r.q) {
			half = make([]float32, r.k/16)
		}
		scr := make([]float32, QuantActNarrowScratch)

		run := func(kk QuantActKernels) time.Duration {
			t0 := time.Now()
			for i := 0; i < iters; i++ {
				kk.Run(r.q, dst, pairs, half, scr, x, r.k, 0, nb, r.window)
			}
			return time.Since(t0)
		}
		run(kern) // warm the buffers and the code pages
		run(goOnly)

		var ratios []float64
		var ka, gb time.Duration
		for i := 0; i < rounds; i++ {
			var tk, tg time.Duration
			if i%2 == 0 {
				tk, tg = run(kern), run(goOnly)
			} else {
				tg, tk = run(goOnly), run(kern)
			}
			ka, gb = ka+tk, gb+tg
			ratios = append(ratios, float64(tg)/float64(tk))
		}
		sort.Float64s(ratios)
		med := ratios[len(ratios)/2]
		iqr := (ratios[(3*len(ratios))/4] - ratios[len(ratios)/4]) / med
		gbps := func(d time.Duration) float64 {
			return float64(iters*rounds) * float64(r.k*4) / d.Seconds() / 1e9
		}
		t.Logf("%-6v k=%-5d window=%-3d  kernel/go %.3fx  IQR/median %.3f  "+
			"kernel %.2f GB/s  go %.2f GB/s", r.q, r.k, r.window, med, iqr, gbps(ka), gbps(gb))
		done()
	}
}

// TestQuantActRateControl is the A/A: the same kernel in both arms, on the
// harness above, so the ratio can be read against the harness's own bias.
func TestQuantActRateControl(t *testing.T) {
	const iters = 400
	kern, done := quantActKernels(t, quant.Q4_K)
	defer done()
	if kern.Wide == nil {
		t.Skipf("no kernel on tier %v", Native().Tier)
	}
	const k, window = 2048, 256
	nb := k / Q8Block
	x := make([]float32, k)
	rng := rand.New(rand.NewSource(1))
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	dst := make([]int8, k)
	pairs := make([]float32, 2*nb)
	scr := make([]float32, QuantActNarrowScratch)
	run := func() time.Duration {
		t0 := time.Now()
		for i := 0; i < iters; i++ {
			kern.Run(quant.Q4_K, dst, pairs, nil, scr, x, k, 0, nb, window)
		}
		return time.Since(t0)
	}
	run()
	var ratios []float64
	for i := 0; i < 9; i++ {
		a, b := run(), run()
		ratios = append(ratios, float64(b)/float64(a))
	}
	sort.Float64s(ratios)
	med := ratios[len(ratios)/2]
	t.Logf("A/A self-control: median %.4f  IQR/median %.3f  n=%d", med,
		(ratios[(3*len(ratios))/4]-ratios[len(ratios)/4])/med, len(ratios))
}
