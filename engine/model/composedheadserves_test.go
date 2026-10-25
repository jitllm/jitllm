package model

import (
	"math"
	"os"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// headServedModels are checkpoints whose heads the composer serves where their
// blocks are not yet served whole (formedModels): rnj-1, whose final softcap
// of 30 was a head graph the tier declined by name at every step.
var headServedModels = []string{"rnj/EssentialAI_rnj-1-instruct-Q4_K_M.gguf"}

// TestComposedHeadServesWhereTheBlocksDoNot decodes each of headServedModels
// with the composer asked for the head alone, every block on the tier's own
// kernels, on each device: the head is served at every step, formed at its
// placement and declined by no name but the prompt's own (composedLater), and
// the tier's own cap is not launched
// over the capped logits it writes. The logits are held to the tier's own head
// and cap, tenfold nearer than the same logits capped twice, or not at all,
// which the head that did either would be.
func TestComposedHeadServesWhereTheBlocksDoNot(t *testing.T) {
	const gen = 24
	ran := 0
	for _, file := range headServedModels {
		t.Run(file, func(t *testing.T) {
			path := testmodels.Path(file)
			if _, err := os.Stat(path); err != nil {
				testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS)", path)
			}
			m, err := Open(jlmOf(t, path), noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if why := m.Uncovered(-1); why != Covered {
				t.Fatalf("the head is refused by %q (%v)", why, m.UncoveredErr(-1))
			}
			c := float64(m.Cfg.FinalSoftcap)
			if c == 0 || m.Cfg.AttnSoftcap != 0 {
				t.Fatalf("%s states a final softcap of %v and an attention softcap of %v: the gate counts the head's cap among the tier's launches, which the attention's would join",
					file, c, m.Cfg.AttnSoftcap)
			}
			ids := m.Vocab.Encode("The capital of France is", true)
			toks := make([]int32, gen)
			for i := range toks {
				toks[i] = int32((int(ids[i%len(ids)])*7 + i) % m.Vocab.Size())
			}
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					want, _, _, _ := composedDecode(t, m, dev, ids, toks, 0, true, false)
					got, st, headOn, census := composedDecode(t, m, dev, ids, toks, tier.ComposedHead, true, false)
					if !headOn {
						t.Skipf("the head does not fit beside the blocks on %s", dev)
					}
					served, declined, _ := st.ServedPart(tier.ComposedHead)
					// The prompt's heads are the tier's, declined by their
					// path's name (composedLater).
					for i, c := range st.HeadDeclines {
						if slices.Contains(composedLater, tier.DeclineName(i)) {
							declined -= c
						}
					}
					if served != len(toks) || declined != 0 || st.FormedAtStep != 0 {
						t.Fatalf("the composed head served %d of %d steps, was declined %d (%s) and %d formed at a step",
							served, len(toks), declined, declinesOf(st), st.FormedAtStep)
					}
					if n := census["softcap"]; n != 0 {
						t.Fatalf("the tier launched its own cap %d times over the capped logits the composed head wrote (%v)", n, census)
					}
					// The logits the tier's head capped, capped again and uncapped.
					worst, twice, none := 0.0, 0.0, 0.0
					for i := 1; i < len(want); i++ {
						again := make([]float32, len(want[i]))
						undone := make([]float32, len(want[i]))
						for j, v := range want[i] {
							r := float64(v) / c
							again[j], undone[j] = float32(c*math.Tanh(r)), v
							// A logit the cap saturated has no inverse.
							if math.Abs(r) < 1-1e-6 {
								undone[j] = float32(c * math.Atanh(r))
							}
						}
						worst = max(worst, logitNMSE(got[i], want[i]))
						twice = max(twice, logitNMSE(again, want[i]))
						none = max(none, logitNMSE(undone, want[i]))
					}
					t.Logf("the composed head served %d steps: worst NMSE %.3e from the tier's own head and cap, where capped twice it would read %.3e and uncapped %.3e",
						served, worst, twice, none)
					if !(worst*10 < min(twice, none)) {
						t.Fatalf("the composed head parts from the tier's by %.3e, where capped twice it would part by %.3e and uncapped by %.3e",
							worst, twice, none)
					}
					ran++
				})
			}
		})
	}
	if ran == 0 {
		t.Fatal("no arm ran")
	}
}
