package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// classicFixtures are the dense classic-transformer fixtures (LayerNorm, a
// parallel attention and FFN, biases, ungated MLPs): synth-dbrx, a mixture, is
// refused by its name (TestFrontStatesTheModelAsOps).
var classicFixtures = []string{"synth-phi2", "synth-starcoder", "synth-stablelm", "synth-stablelm-par",
	"synth-starcoder2", "synth-falcon", "synth-falcon40", "synth-nemotron", "synth-commandr", "synth-cohere2"}

// classicDecline is the one reason the tier leaves a block the front end
// states to its own kernels on these fixtures: a window shorter than the
// session (synth-cohere2's sliding layers).
const classicDecline = "a window"

// coveredBlocks is how many of m's blocks the front end states as ops.
func coveredBlocks(m *Model) int {
	n := 0
	for li := range m.Cfg.NLayer {
		if m.Uncovered(li) == Covered {
			n++
		}
	}
	return n
}

// windowedBlocks is how many of m's blocks attend inside a window shorter than
// a session of sessionLen positions, which the tier does not serve whole.
func windowedBlocks(m *Model, sessionLen int) int {
	n := 0
	for li := range m.Cfg.NLayer {
		if m.Cfg.SWA(li) && m.Cfg.SWAWindow < sessionLen {
			n++
		}
	}
	return n
}

// TestC6ServesComposed holds the composer to each classic-transformer
// fixture's blocks on every device present: every block the front end states
// served whole at every decode step -- served block steps equal steps times
// the blocks the front end covers, less the windowed blocks, which the tier
// declines by the one name classicDecline and counts exactly -- the logits
// within the device band of the host's, and each feature taken out of the
// device session alone, as the composer reads it too, moving them: the composed
// arm of TestC6OnEveryDevice, asked for in any build.
func TestC6ServesComposed(t *testing.T) {
	// The prompt's rows come from the device's batched prefill, whose
	// attention tiles take Q K^T through binary16 where the device has them,
	// and every decoded step reads them: the band a prefilled prompt gets in
	// fixtureOnEveryDevice (1e-5), where a decode from the first token reads
	// 1e-12. The violations below land at 1e-4 and up.
	const bound, promptLen = 1e-5, 3
	ran := 0
	for _, name := range classicFixtures {
		t.Run(name, func(t *testing.T) {
			m, g := openC6(t, name)
			defer m.Close()
			host := m.NewState(len(g.IDs) + 1)
			var want [][]float32
			for _, id := range g.IDs {
				lg, err := host.Forward(id)
				if err != nil {
					t.Fatal(err)
				}
				want = append(want, append([]float32(nil), lg...))
			}
			host.Close()
			// The prompt takes the batched path, which the composer is not asked
			// for; each token after it is a one-row step it is.
			prompt, toks := g.IDs[:promptLen], g.IDs[promptLen:]
			run := func(t *testing.T, dev string) (worst float64, flips int) {
				t.Helper()
				covered := coveredBlocks(m)
				if covered != m.Cfg.NLayer {
					t.Fatalf("the front end states %d of %d blocks: %s %v", covered, m.Cfg.NLayer, m.Uncovered(0), m.UncoveredErr(0))
				}
				// The blocks alone are asked for: the head's graph is the head's own work.
				got, st, _, _ := composedDecode(t, m, dev, prompt, toks, tier.ComposedBlock, true, false)
				served, declined, _ := st.ServedPart(tier.ComposedBlock)
				windowed := windowedBlocks(m, len(g.IDs))
				wantServed, wantDeclined := len(toks)*(covered-windowed), len(toks)*windowed
				if served != wantServed || declined != wantDeclined || served == 0 {
					t.Fatalf("%d block steps served and %d declined, where %d steps of %d covered blocks, %d of them windowed, are %d and %d",
						served, declined, len(toks), covered, windowed, wantServed, wantDeclined)
				}
				for i, k := range st.BlockDeclines {
					if k > 0 && (tier.DeclineName(i) != classicDecline || k != wantDeclined) {
						t.Fatalf("%d block step(s) declined because %s, where %d are windowed", k, tier.DeclineName(i), wantDeclined)
					}
				}
				if st.FormedAtStep != 0 {
					t.Fatalf("%d composed block(s) formed at a step, not at the placement", st.FormedAtStep)
				}
				t.Logf("%s: %d of %d block steps served, %d declined by %q; the head is the tier's (front end: %s)",
					dev, served, len(toks)*covered, declined, classicDecline, m.Uncovered(-1))
				for i := range got {
					p := promptLen - 1 + i
					e := logitNMSE(got[i], want[p])
					if math.IsNaN(e) {
						t.Fatalf("pos %d: the logits are not finite", p)
					}
					worst = max(worst, e)
					if Greedy(got[i]) != Greedy(want[p]) {
						flips++
					}
				}
				return worst, flips
			}
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					worst, flips := run(t, dev)
					if !(worst < bound) || flips > 0 {
						t.Fatalf("worst logit NMSE %.3e, %d argmax flips, against the host", worst, flips)
					}
					t.Logf("composed blocks on %s: worst logit NMSE %.3e", dev, worst)
					for _, v := range c6DevViolations[name] {
						undo := v.mut(m)
						bad, _ := run(t, dev)
						undo()
						if !(bad > 100*worst) || !(bad > 1e-4) {
							t.Errorf("%s: the composed arm cannot see it -- worst NMSE %.3e against a clean %.3e", v.name, bad, worst)
						}
						t.Logf("violation %q: worst NMSE %.3e", v.name, bad)
					}
					ran++
				})
			}
		})
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}
