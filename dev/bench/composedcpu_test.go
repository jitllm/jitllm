package bench

import (
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/engine/probe"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestComposedCPUAB times a whole decode token with every block on the CPU
// as the composer forms it (model.WithComposedCPU) against main's CPU
// kernels, each arm a model and a state of its own at the shipping tuning,
// interleaved in one process (AB), the same-binary A/A control first on the
// same harness. The model is JITLLM_COMPOSED_MODEL (default Llama-3.2-1B
// Q4_K_M). Each arm's rate is stated as bytes a token and a share of the
// CPU's read rate, measured in the same process by the estimator's probe
// once both arms are closed: a rate above it is a harness bug.
func TestComposedCPUAB(t *testing.T) {
	if os.Getenv("JITLLM_COMPOSED_AB") == "" {
		t.Skip("a measurement: set JITLLM_COMPOSED_AB=1, on an idle host")
	}
	const rounds, iters, warm = 20, 16, 64
	path := os.Getenv("JITLLM_COMPOSED_MODEL")
	if path == "" {
		path = testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.jlm")
	}
	path = testmodels.Resolve(path)
	if _, err := os.Stat(path); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS) -- this measurement proved nothing", err)
	}
	var bpt float64
	arm := func(name string, composed bool) (Case, func() int, func()) {
		m, err := model.Open(path, model.WithComposedCPU(composed))
		if err != nil {
			t.Fatal(err)
		}
		bpt = float64(m.BytesPerToken())
		st := m.NewState(warm + 2*(rounds+2)*iters*2 + 64)
		l, err := st.Prefill(m.Vocab.Encode("The river rose every spring until", true))
		if err != nil {
			t.Fatal(err)
		}
		tok := int32(argmax(l))
		fn := func(n int) {
			for range n {
				lg, err := st.Forward(tok)
				if err != nil {
					t.Fatal(err)
				}
				tok = int32(argmax(lg))
			}
		}
		fn(warm)
		steps := func() int {
			served, declined := st.CPUBlockSteps()
			for _, n := range declined {
				if n > 0 {
					return -n
				}
			}
			return served
		}
		return Case{Name: name, Fn: fn}, steps, func() { st.Close(); m.Close() }
	}
	a, _, closeA := arm("main", false)
	a2, _, closeA2 := arm("main again", false)
	ctl := AB(a, a2, rounds, iters)
	t.Logf("A/A control: %v", ctl)
	closeA()
	closeA2()
	if !ctl.Stable() || ctl.Median < 0.98 || ctl.Median > 1.02 {
		t.Fatalf("the A/A control reads %.4f (IQR/median %.3f): this harness cannot discriminate here", ctl.Median, ctl.IQR/ctl.Median)
	}
	a3, _, closeA3 := arm("main", false)
	b, served, closeB := arm("composed", true)
	r := AB(a3, b, rounds, iters)
	n := served()
	closeA3()
	closeB()
	t.Logf("composed CPU vs main's CPU kernels, %d-token rounds: %v", iters, r)
	if n <= 0 {
		t.Fatalf("the composed arm's decode served %d block steps (negative: declined): it did not time the composer's blocks", n)
	}
	host, err := probe.Host()
	if err != nil {
		t.Fatal(err)
	}
	wall := host.Bandwidth
	rate := func(d float64) (float64, float64) { tok := float64(iters) / d; return tok, tok * bpt / 1e9 }
	tokA, gbA := rate(r.AMedian.Seconds())
	tokB, gbB := rate(r.BMedian.Seconds())
	t.Logf("%.1f MB a token: main %.1f tok/s = %.1f GB/s = %.0f%% of the %.1f GB/s wall, composed %.1f tok/s = %.1f GB/s = %.0f%%",
		bpt/1e6, tokA, gbA, 100*gbA*1e9/wall, wall/1e9, tokB, gbB, 100*gbB*1e9/wall)
	if gbA*1e9 > wall || gbB*1e9 > wall {
		t.Fatalf("a rate above the CPU's %.1f GB/s wall is a harness bug, not a win", wall/1e9)
	}
	if !r.Stable() {
		t.Fatal("unstable: do not quote this row")
	}
}

// TestComposedCPUPrefillAB times a whole prompt with every block on the CPU
// as the composer forms it (its form of several rows) against main's CPU
// prefill, at each prompt length of JITLLM_COMPOSED_PP (default 33, 100
// and 512), each arm a model and a state of its own at the shipping tuning,
// interleaved in one process (AB), a sample one prompt from position zero,
// the same-binary A/A control first on the same harness. The rates are
// prompt tokens a second.
func TestComposedCPUPrefillAB(t *testing.T) {
	if os.Getenv("JITLLM_COMPOSED_AB") == "" {
		t.Skip("a measurement: set JITLLM_COMPOSED_AB=1, on an idle host")
	}
	const rounds = 20
	path := os.Getenv("JITLLM_COMPOSED_MODEL")
	if path == "" {
		path = testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.jlm")
	}
	path = testmodels.Resolve(path)
	if _, err := os.Stat(path); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS) -- this measurement proved nothing", err)
	}
	pps := []int{33, 100, 512}
	if s := os.Getenv("JITLLM_COMPOSED_PP"); s != "" {
		pps = nil
		for _, f := range strings.Split(s, ",") {
			n, err := strconv.Atoi(f)
			if err != nil {
				t.Fatal(err)
			}
			pps = append(pps, n)
		}
	}
	for _, pp := range pps {
		arm := func(name string, composed bool) (Case, func() int, func()) {
			m, err := model.Open(path, model.WithComposedCPU(composed))
			if err != nil {
				t.Fatal(err)
			}
			ids := m.Vocab.Encode(strings.Repeat("The river rose every spring until the bridges were islands. ", 1+pp/8), true)[:pp]
			st := m.NewState(pp + 1)
			fn := func(n int) {
				for range n {
					st.Reset()
					if _, err := st.Prefill(ids); err != nil {
						t.Fatal(err)
					}
				}
			}
			fn(2)
			steps := func() int {
				served, declined := st.CPUBlockSteps()
				for _, n := range declined {
					if n > 0 {
						return -n
					}
				}
				return served
			}
			return Case{Name: name, Fn: fn}, steps, func() { st.Close(); m.Close() }
		}
		a, _, closeA := arm("main", false)
		a2, _, closeA2 := arm("main again", false)
		ctl := AB(a, a2, rounds, 1)
		t.Logf("pp%d A/A control: %v", pp, ctl)
		closeA()
		closeA2()
		if !ctl.Stable() || ctl.Median < 0.97 || ctl.Median > 1.03 {
			t.Errorf("pp%d: the A/A control reads %.4f (IQR/median %.3f): this harness cannot discriminate here", pp, ctl.Median, ctl.IQR/ctl.Median)
			continue
		}
		a3, _, closeA3 := arm("main", false)
		b, served, closeB := arm("composed", true)
		r := AB(a3, b, rounds, 1)
		n := served()
		closeA3()
		closeB()
		if n <= 0 {
			t.Fatalf("pp%d: the composed arm served %d block steps (negative: declined): it did not time the composer's blocks", pp, n)
		}
		t.Logf("pp%d composed CPU vs main's CPU prefill: %v; main %.1f tok/s, composed %.1f tok/s", pp, r,
			float64(pp)/r.AMedian.Seconds(), float64(pp)/r.BMedian.Seconds())
		if !r.Stable() {
			t.Errorf("pp%d unstable: do not quote this row", pp)
		}
	}
}
