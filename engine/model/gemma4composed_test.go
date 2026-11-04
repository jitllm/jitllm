package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/device/gpu"
)

// gemma4Composed are the Gemma 4 fixtures the front end states whole: the
// global layers at another head width, kv head count and rotary width than
// the sliding ones, v k's own on the global layers of two, normed with no
// weight, and every block's output scaled by its stored scalar; and the
// E-models' blocks, each reading its slice of the row's per-layer inputs,
// the last two reading an earlier block's history.
func gemma4Composed() []string {
	return []string{"synth-gemma4", "synth-gemma4-kv", "synth-gemma4-wide", "synth-gemma4-e", "synth-gemma4-e-ragged", "synth-gemma4-moe"}
}

// TestGemma4ServesComposed holds the composer to each of gemma4Composed's
// blocks and head on every device present, as TestC6ServesComposed holds
// the classic fixtures: every block step and head served, the logits within
// the device band of the host's, and each of Gemma 4's features taken out
// (gemma4Violations) moving them. The whole step on a card, its tables
// served in each geometry's scratch set (the global set the global table,
// the sliding set the local one) at the prompt's rows and every step's,
// within the bound of the host's logits and of main's tier, and every row
// told twice its position and 16 past far from them. The input rows of the
// E-models are declined by name (open).
func TestGemma4ServesComposed(t *testing.T) {
	const bound = 1e-5
	ran := 0
	for _, name := range gemma4Composed() {
		t.Run(name, func(t *testing.T) {
			m, g := openGemma4(t, name)
			defer m.Close()
			if !gemma4Selects(t, m, name) {
				t.Fatalf("the container does not carry the fixture's graph: %+v", m.Cfg)
			}
			ran += servesComposed(t, m, g.IDs, bound, gemma4ComposedViolations(name), false)
			const promptLen = 3
			host := hostLogits(t, m, g.IDs)
			for _, dev := range stepDevices() {
				t.Run(dev+"/step", func(t *testing.T) {
					prompt, toks := g.IDs[:promptLen], g.IDs[promptLen:]
					from := func(got, want [][]float32, off int) float64 {
						w := 0.0
						for i := range got {
							e := logitNMSE(got[i], want[off+i])
							if math.IsNaN(e) {
								return math.Inf(1)
							}
							w = max(w, e)
						}
						return w
					}
					got, st, _ := composedDecode(t, m, dev, prompt, toks, gpu.ComposedStep, true, false)
					served, td := st.ServedPart(gpu.ComposedTable)
					served += st.ReplayedPart(gpu.ComposedTable)
					nb, db := st.ServedPart(gpu.ComposedBlock)
					np, dp := st.ServedPart(gpu.ComposedPrompt)
					t.Logf("%s: the tables served %d times, declined %d; block steps served %d, declined %d; the prompt's %d served, %d declined (%s)",
						dev, served, td, nb, db, np, dp, declinesOf(st))
					if want := (len(toks) + 1) * composedTables(m); served < want || td != 0 || st.StepDeclines[composer.DeclineNoTable] != 0 {
						t.Fatalf("the tables served %d times (want at least %d, a table a geometry's run of each step and the prompt) and declined %d", served, want, td)
					}
					if n := st.FormedPart(gpu.ComposedTable); n != 0 {
						t.Fatalf("%d table(s) formed at a step, not at the placement (%s)", n, st.LastErr)
					}
					hostW := from(got, host, promptLen-1)
					t.Logf("%s: the composed step %.3e from the host", dev, hostW)
					if !(hostW < bound) {
						t.Fatalf("the composed step is %.3e from the host, bound %g", hostW, bound)
					}
					bad, _, _ := composedDecodeWith(t, m, dev, prompt, toks, gpu.ComposedStep, true, false, nil, gpu.WithRopeFault(2, 16))
					badW := from(bad, host, promptLen-1)
					t.Logf("%s: every row's table at twice its position and 16 past: %.3e from the host", dev, badW)
					if !(badW > 100*hostW) || !(badW > 10*bound) {
						t.Fatalf("every row's table at twice its position and 16 past reads %.3e against a clean %.3e: the served tables cannot see it", badW, hostW)
					}
					// The per-layer inputs' tables are the card's: the input
					// rows that read them run there and write the inputs.
					if m.Cfg.PLEDim != 0 {
						ne, de := st.ServedPart(gpu.ComposedEmbed)
						dw := st.StepDeclines[composer.DeclineEmbedWeight]
						t.Logf("%s: the input rows served %d times, declined %d (%d for the per-layer tables)", dev, ne, de, dw)
						if ne == 0 || de != 0 || dw != 0 {
							t.Fatalf("the input rows served %d times and declined %d (%d for the per-layer tables): the card's input rows did not write the per-layer inputs", ne, de, dw)
						}
					}
				})
			}
		})
	}
	if ran == 0 {
		t.Fatal("no device on this host")
	}
}

// gemma4ComposedViolations is gemma4Violations[name] as the composed gates
// hold them, past 10 times their bound: the final softcap of 30 over a
// random fixture's small logits moves them 1.8e-05 when taken out, so the
// cap is moved to 1 instead, where it bends every logit. A model read
// without its per-layer embeddings leaves their tensors unbound, so the
// front end states no block of it (UncoveredUnbound) and no composed arm
// runs: the composed gates leave the per-layer inputs' norm unweighted
// instead, which every block reads through its second input. Each block
// given the next block's gate places no block on CUDA (the tier took 0 of
// 6), so it is no violation the card arms can run.
func gemma4ComposedViolations(name string) []c6Violation {
	var vs []c6Violation
	for _, v := range gemma4Violations()[name] {
		switch v.name {
		case "no final softcap":
			v = c6Violation{"a final softcap of 1", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.FinalSoftcap = 1 }) }}
		case "no per-layer embeddings":
			v = c6Violation{"the per-layer inputs' norm unweighted", func(m *Model) func() {
				was := m.pleNorm
				ones := make([]float32, len(was))
				for i := range ones {
					ones[i] = 1
				}
				m.pleNorm = ones
				return func() { m.pleNorm = was }
			}}
		}
		vs = append(vs, v)
	}
	return vs
}

// gemma4Selects is gemma4Fixtures' selection for name.
func gemma4Selects(t *testing.T, m *Model, name string) bool {
	t.Helper()
	for _, fx := range gemma4Fixtures() {
		if fx.name == name {
			return fx.sel(m)
		}
	}
	t.Fatalf("%s is not a Gemma 4 fixture", name)
	return false
}

// worstFrom is the worst logit NMSE of got's rows from want's, infinite for
// a row that is not finite.
func worstFrom(want, got [][]float32) float64 {
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
