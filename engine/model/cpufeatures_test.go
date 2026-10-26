package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// dbrx's mixture, its q, k and v clamped, joins the gates of the five
// principles: relocation, paging and a warm step that allocates nothing.
func init() {
	principleMixtures = append(principleMixtures, "synth-dbrx.gguf")
}

// cpuFeatureFixtures are float fixtures of the attention features the front
// end states as ops that the CPU device holds whole -- mistral3's attention
// temperature on every layer, read at each row's position -- each with the
// feature taken out. dbrx's and Llama 4's fixtures are mixtures the CPU
// declines by name (routed experts not held whole, Llama 4's chunked
// layers' windowed history).
func cpuFeatureFixtures() map[string][]c6Violation {
	return map[string][]c6Violation{
		"synth-mistral3": {
			{"no attention temperature", noTemp},
			{"the temperature's floor doubled", func(m *Model) func() {
				return cfgMut(m, func(c *Config) { c.AttnTempFloor *= 2 })
			}},
		},
		"synth-mistral3-af": {
			{"no attention temperature", noTemp},
		},
	}
}

// TestComposedCPUFeaturesAreLoadBearing runs each of cpuFeatureFixtures on
// the CPU device, every block composed for it, teacher-forced against main's
// kernels: within 1e-5 of main's logits at every step (float weights, so
// main at another activation window is no control), and each feature taken
// out of the composed arm alone past 100 times its worst and past 1e-4.
func TestComposedCPUFeaturesAreLoadBearing(t *testing.T) {
	const steps, page, bound = 24, 32, 1e-5
	for name, vs := range cpuFeatureFixtures() {
		t.Run(name, func(t *testing.T) {
			path := jlmOf(t, testmodels.Path(name+".gguf"))
			ref, err := Open(path, noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer ref.Close()
			ref.setKVPageForTest(page)
			toks := cpuTokens(t, ref, steps)
			want := cpuLogits(t, ref, toks, cpuFaultNone, false)
			m, err := Open(path, noTune, WithComposedCPU(true))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			m.setKVPageForTest(page)
			if c := coveredBlocks(m); c != m.Cfg.NLayer {
				t.Fatalf("the front end states %d of %d blocks: %s", c, m.Cfg.NLayer, m.Uncovered(0))
			}
			worstOf := func(got [][]float32) float64 {
				w := 0.0
				for i := range want {
					e := logitNMSE(got[i], want[i])
					if math.IsNaN(e) {
						return math.Inf(1)
					}
					w = max(w, e)
				}
				return w
			}
			worst := worstOf(cpuLogits(t, m, toks, cpuFaultNone, true))
			t.Logf("the composed CPU's worst NMSE from main over %d steps: %.3g", steps, worst)
			if !(worst < bound) {
				t.Fatalf("the composed CPU is %.3g from main, bound %g", worst, bound)
			}
			for _, v := range vs {
				undo := v.mut(m)
				bad := worstOf(cpuLogits(t, m, toks, cpuFaultNone, true))
				undo()
				t.Logf("violation %q: worst NMSE %.3g", v.name, bad)
				if !(bad > 100*worst) || !(bad > 10*bound) {
					t.Errorf("%s: the composed CPU cannot see it -- worst NMSE %.3g against a clean %.3g", v.name, bad, worst)
				}
			}
		})
	}
}
