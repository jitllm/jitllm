package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// dbrx's mixture, its q, k and v clamped, joins the gates of the five
// principles: relocation, paging and a warm step that allocates nothing.
func init() {
}

// cpuFeatureFixtures are float fixtures of the attention features the front
// end states as ops that the CPU device holds whole -- mistral3's attention
// temperature on every layer, read at each row's position; cohere2's and
// EXAONE 4's sliding layers, a window of four positions, their history the
// window's; the head's stages after its projection, cohere2's logit scale
// and phi2's output bias -- each with the feature taken out. dbrx's and Llama 4's
// fixtures are mixtures the CPU declines by name (routed experts not held
// whole).
func cpuFeatureFixtures() map[string][]c6Violation {
	window := func(w int) func(m *Model) func() {
		return func(m *Model) func() { return cfgMut(m, func(c *Config) { c.SWAWindow = w }) }
	}
	return map[string][]c6Violation{
		"synth-cohere2": {
			{"a window of eight", window(8)},
			{"no window", window(0)},
			{"no logit scale", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.LogitScale = 1 }) }},
		},
		"synth-phi2": {
			{"the output bias doubled", func(m *Model) func() {
				was := m.outB
				m.outB = make([]float32, len(was))
				for i, v := range was {
					m.outB[i] = 2 * v
				}
				return func() { m.outB = was }
			}},
		},
		"synth-exaone4": {
			{"a window of eight", window(8)},
			{"no window", window(0)},
		},
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
// the CPU device, every block composed for it, teacher-forced against the
// CPU's float forms over the same history (withCPUFloat): within 1e-5 of
// their logits at every step (float weights, so the two part only by their
// reduction order), and each feature taken out of the composed arm alone
// past 100 times its worst and past 1e-4.
func TestComposedCPUFeaturesAreLoadBearing(t *testing.T) {
	const steps, page, bound = 24, 32, 1e-5
	for name, vs := range cpuFeatureFixtures() {
		t.Run(name, func(t *testing.T) {
			path := jlmOf(t, testmodels.Path(name+".gguf"))
			ref, err := Open(path, noTune(), withCPUFloat())
			if err != nil {
				t.Fatal(err)
			}
			defer ref.Close()
			ref.setKVPageForTest(page)
			toks := cpuTokens(t, ref, steps)
			want := cpuLogits(t, ref, toks, cpuFaultNone, true)
			m, err := Open(path, noTune())
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
			t.Logf("the composed CPU's worst NMSE from the float forms over %d steps: %.3g", steps, worst)
			if !(worst < bound) {
				t.Fatalf("the composed CPU is %.3g from the float forms, bound %g", worst, bound)
			}
			// A prompt through the rows' form, its decode after it.
			pw := cpuPrompt(t, ref, toks, steps-4, 4, cpuFaultNone, true)
			pg := cpuPrompt(t, m, toks, steps-4, 4, cpuFaultNone, true)
			pworst := 0.0
			for i := range pw {
				pworst = max(pworst, logitNMSE(pg[i], pw[i]))
			}
			t.Logf("a prompt of %d and its decode: worst NMSE %.3g from the float forms", steps-4, pworst)
			if !(pworst < bound) {
				t.Fatalf("the composed CPU's prompt is %.3g from the float forms, bound %g", pworst, bound)
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
