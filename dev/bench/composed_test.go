package bench

import (
	"os"
	"strconv"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/engine/probe"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestComposedPartsAB times a whole decode token with the parts the composer
// serves (JITLLM_COMPOSED_PARTS, a tier.Parts mask; default every block whole
// and the head) against the tier's own kernels, every block placed on
// JITLLM_COMPOSED_DEVICE (default the first device), each arm a tier and a
// state of its own at the shipping tuning, interleaved in one process (AB).
// The same-binary A/A control runs first on the same harness and must read
// inside 2% of 1. The model is JITLLM_COMPOSED_MODEL (default Llama-3.2-1B
// Q4_K_M). Each arm's rate is stated as bytes a token and a share of the
// device's read rate, measured in the same process by the estimator's probe
// once both arms are closed: a rate above it is a harness bug.
func TestComposedPartsAB(t *testing.T) {
	if os.Getenv("JITLLM_COMPOSED_AB") == "" {
		t.Skip("a measurement: set JITLLM_COMPOSED_AB=1, on an idle device")
	}
	const rounds, iters, warm = 20, 32, 128
	path := os.Getenv("JITLLM_COMPOSED_MODEL")
	if path == "" {
		path = testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.jlm")
	}
	path = testmodels.Resolve(path)
	if _, err := os.Stat(path); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS) -- this measurement proved nothing", err)
	}
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	parts := tier.ComposedBlock | tier.ComposedHead
	if v := os.Getenv("JITLLM_COMPOSED_PARTS"); v != "" {
		var n int
		for _, c := range v {
			n = n*10 + int(c-'0')
		}
		parts = tier.Parts(n)
	}
	dev := os.Getenv("JITLLM_COMPOSED_DEVICE")
	arm := func(name string, s tier.Parts) (Case, func() tier.Stats, func()) {
		// The shipping default: the device tuned (TuneAuto), into a cache
		// of the arm's own, so neither arm reads the other's answers.
		opts := []tier.Option{tier.WithTuneCache(t.TempDir()), tier.WithConfig(func(c *tier.Config) { c.Composed = s })}
		if dev != "" {
			opts = append(opts, tier.WithDevices(dev))
		}
		g, err := tier.OpenWith(opts...)
		if err != nil || g == nil {
			t.Skipf("no device: %v", err)
		}
		st := m.NewState(warm + 2*(rounds+2)*iters*2 + 64)
		if err := st.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if st.GPULayers() != m.Blocks() {
			t.Fatalf("%s placed %d of %d blocks: this row would time the CPU device too", name, st.GPULayers(), m.Blocks())
		}
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
		t.Logf("%s arm on %s", name, g.Name())
		// Warmed past the tuners' first answers before any round is timed.
		fn(warm)
		return Case{Name: name, Fn: fn}, g.Stats, func() { st.Close(); g.Close() }
	}
	a, _, closeA := arm("tier", 0)
	a2, _, closeA2 := arm("tier again", 0)
	ctl := AB(a, a2, rounds, iters)
	t.Logf("A/A control: %v", ctl)
	if !ctl.Stable() || ctl.Median < 0.98 || ctl.Median > 1.02 {
		t.Fatalf("the A/A control reads %.4f (IQR/median %.3f): this harness cannot discriminate here", ctl.Median, ctl.IQR/ctl.Median)
	}
	// Both arms of the row fresh: the control's arms are a control's
	// rounds deeper into their history.
	closeA()
	closeA2()
	a3, _, closeA3 := arm("tier", 0)
	b, stats, closeB := arm("composed", parts)
	r := AB(a3, b, rounds, iters)
	served, declined, _ := stats().ServedPart(parts)
	closeA3()
	closeB()
	t.Logf("composed parts %#x vs the tier's, %d-token rounds: %v", uint8(parts), iters, r)
	if served == 0 || declined != 0 {
		t.Fatalf("the composed arm served %d steps and declined %d: it did not time the composer's parts", served, declined)
	}
	wall := deviceWall(t, dev)
	bpt := float64(m.BytesPerToken())
	rate := func(d float64) (float64, float64) { tok := float64(iters) / d; return tok, tok * bpt / 1e9 }
	tokA, gbA := rate(r.AMedian.Seconds())
	tokB, gbB := rate(r.BMedian.Seconds())
	t.Logf("%.1f MB a token: tier %.1f tok/s = %.1f GB/s = %.0f%% of the %.1f GB/s wall, composed %.1f tok/s = %.1f GB/s = %.0f%%; composed parts served in %d emitted steps",
		bpt/1e6, tokA, gbA, 100*gbA*1e9/wall, wall/1e9, tokB, gbB, 100*gbB*1e9/wall, served)
	if gbA*1e9 > wall || gbB*1e9 > wall {
		t.Fatalf("a rate above the device's %.1f GB/s wall is a harness bug, not a win", wall/1e9)
	}
	if !r.Stable() {
		t.Fatal("unstable: do not quote this row")
	}
}

// deviceWall is the read rate of the device spec names (as -devices does;
// the first device when empty), measured by the estimator's probe.
func deviceWall(t *testing.T, spec string) float64 {
	t.Helper()
	api, nth := "", 0
	if spec != "" {
		es, err := tier.ParseDevices(spec)
		if err != nil || len(es) != 1 {
			t.Fatalf("the wall is measured on one device, and %q names %d (%v)", spec, len(es), err)
		}
		api = es[0].API
		if es[0].HasSel {
			if nth, err = strconv.Atoi(es[0].Sel); err != nil {
				t.Fatalf("the wall's device %q is not an ordinal", spec)
			}
		}
	}
	devs := backend.Open()
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	for _, d := range devs {
		if api != "" && d.API() != api {
			continue
		}
		if nth > 0 {
			nth--
			continue
		}
		c, err := probe.GPU(d)
		if err != nil {
			t.Fatalf("measuring the wall on %s: %v", d.Name(), err)
		}
		return c.Bandwidth
	}
	t.Fatalf("no device %q to measure the wall on", spec)
	return 0
}

func argmax(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}
