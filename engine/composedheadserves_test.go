package engine

import (
	"math"
	"os"
	"testing"

	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// headServedModels are checkpoints whose heads the composer serves where their
// blocks are not yet served whole (formedModels): rnj-1, whose final softcap
// of 30 was a head graph the tier declined by name at every step.
func headServedModels() []string { return []string{"rnj/EssentialAI_rnj-1-instruct-Q4_K_M.gguf"} }

// TestComposedHeadServesWhereTheBlocksDoNot decodes each of headServedModels
// with the composer asked for the head alone, every block on the tier's own
// kernels, on each device: the head is served at every step, formed at its
// placement and declined by no name, the prompt's head not asked, and
// the tier's own cap is not launched
// over the capped logits it writes. The logits are held to the CPU's, tenfold
// nearer than the same logits capped twice, or not at all, which the head
// that did either would be: both arms on their float forms, whose band is
// the reduction order's alone, where the shipping forms' int8 activations
// part a whole model's logits by more than a second cap moves them.
func TestComposedHeadServesWhereTheBlocksDoNot(t *testing.T) {
	const gen = 24
	ran := 0
	for _, file := range headServedModels() {
		t.Run(file, func(t *testing.T) {
			path := testmodels.Path(file)
			if _, err := os.Stat(path); err != nil {
				testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS)", path)
			}
			instOpts := []Option{withCPUFloat()}
			m, err := model.Open(jlmOf(t, path))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if why := m.Uncovered(-1); why != model.Covered {
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
					want := cpuDecode(t, m, ids, toks, instOpts...)
					got, st, headOn := composedDecode(t, m, instOpts, dev, ids, toks, nn.ComposedHead, true, true)
					if !headOn {
						t.Skipf("the head does not fit beside the blocks on %s", dev)
					}
					served, declined := st.ServedPart(nn.ComposedHead)
					// Every decode step's head, and the prompt's last row's.
					if served != len(toks)+1 || declined != 0 || st.UnformedAtStep != 0 {
						t.Fatalf("the composed head served %d of %d steps and the prompt's last row, was declined %d (%s) and %d formed at a step",
							served, len(toks), declined, declinesOf(st), st.UnformedAtStep)
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
					t.Logf("the composed head served %d steps: worst NMSE %.3e from the CPU's, where capped twice it would read %.3e and uncapped %.3e",
						served, worst, twice, none)
					if !(worst*10 < min(twice, none)) {
						t.Fatalf("the composed head parts from the CPU's by %.3e, where capped twice it would part by %.3e and uncapped by %.3e",
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
