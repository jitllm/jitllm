package model

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestStepAcrossSessionsLatentMatchesEachAlone is the step gate on multi-head
// latent attention: three sessions decode together through Step, every row's
// latent history its own session's pages, against each session alone,
// teacher-forced. A latent block's history is one row-major row a position --
// the latent and its rotary key -- read by every head, so a row that reads
// another row's pages reads finite, plausible values and stays fluent; only
// the logits can tell.
//
// It walks the DeepSeek fixtures (a mixture with the V3 router, and a dense
// one) and Kimi-Linear's (latent full blocks among Kimi Delta Attention linear
// ones, and NoPE: its rotary channels are copied, never rotated), or the
// models JITLLM_STEP_LATENT names, comma-separated -- a real DeepSeek-V2-Lite
// or Moonlight container, say. Kimi-Linear runs the hybrid step's checks too.
//
// Stats.RowsLatent is the selection check: every latent block of every joint
// step ran in the ragged step, on top of Stats.SessionRows, which says the
// rows went across sessions at all. Each device runs twice, the second time
// with NaN in every fresh buffer.
func TestStepAcrossSessionsLatentMatchesEachAlone(t *testing.T) {
	for _, name := range latentModels() {
		t.Run(filepath.Base(name), func(t *testing.T) {
			m := openLatent(t, name)
			defer m.Close()
			for _, dev := range stepDevices() {
				// The poisoned arm puts NaN in every fresh device buffer: a
				// ragged step pads its rows to the scratch's width, and a
				// padded row's latent, query or partials read where nothing
				// wrote must not reach a real row (RULE 13).
				for _, poison := range []bool{false, true} {
					t.Run(fmt.Sprintf("%s/poisoned=%v", dev, poison), func(t *testing.T) {
						g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff),
							tier.WithPoisonKV(poison), tier.WithPoisonScratch(poison)}, testTierOpts(t)...)...)
						if err != nil || g == nil {
							noDevice(t, dev, err)
						}
						defer g.Close()
						r := latentStepGate(t, m, g)
						checkLinearRowsForm(t, m, g)
						if !(r.worst < mlaDeviceNMSE) {
							t.Fatalf("a latent model's step across sessions disagrees with each session alone: "+
								"NMSE %.3e at row %d", r.worst, r.at)
						}
					})
				}
			}
		})
	}
}

// TestStepRunsAdmitsALatentPrompt is TestStepRunsAdmitsAPromptBesideDecoding
// on the latent models: two sessions decode while a third's prompt rides in
// chunks across sessions, its rows writing their latent rows into the third
// session's pages and attending them in the same step, teacher-forced against
// each session alone. Stats.RowsLatent is the selection check; the violation
// is the late prompt's chunks back to front.
func TestStepRunsAdmitsALatentPrompt(t *testing.T) {
	for _, name := range latentModels() {
		t.Run(filepath.Base(name), func(t *testing.T) {
			m := openLatent(t, name)
			defer m.Close()
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					g := stepTier(t, m, dev, false)
					defer g.Close()
					plan := runsPlan{early: []string{"The capital of France is", "Water boils at a temperature of"},
						late:  "The clerk counted barrels of salt on the upper floor, and",
						chunk: 3, steps: 4}
					want := plan.alone(t, m, g)
					s0 := g.Stats()
					got, rows := plan.together(t, m, g)
					s1 := g.Stats()
					if n := s1.SessionRows - s0.SessionRows; n != rows {
						t.Fatalf("%d rows ran across sessions, want %d: StepRuns went one State after another (%s)",
							n, rows, g.Err())
					}
					if s1.RowsLatent == s0.RowsLatent {
						t.Fatalf("no latent block ran in a step across sessions (%s)", g.Err())
					}
					w, at := worstStep(got, want)
					t.Logf("%d rows, %d latent block-steps: worst NMSE %.3e (row %d) against each session alone",
						rows, s1.RowsLatent-s0.RowsLatent, w, at)
					if !(w < mlaDeviceNMSE) {
						t.Fatalf("a latent step carrying a prompt chunk disagrees with each session alone: NMSE %.3e at row %d",
							w, at)
					}
					stepPos = func(base, j, k int) int { return base + k - 1 - j }
					bad, _ := plan.together(t, m, g)
					stepPos = nil
					if bw, bat := worstStep(bad, want); bw < mlaDeviceNMSE {
						t.Fatalf("chunks at mirrored positions still matched each session alone (NMSE %.3e at row %d)",
							bw, bat)
					} else {
						t.Logf("violation: mirrored chunks read NMSE %.3e at row %d", bw, bat)
					}
				})
			}
		})
	}
}

// latentStepGate is stepGate on a latent model with the selection it needs:
// every latent block of every joint step ran in the ragged step, and on a
// hybrid every linear one across sessions as well.
func latentStepGate(t *testing.T, m *Model, g *tier.GPU) stepResult {
	t.Helper()
	s0 := g.Stats().RowsLatent
	r := stepGate(t, m, g, m.Cfg.Hybrid())
	latent := latentBlocks(m)
	if got := g.Stats().RowsLatent - s0; got != r.steps*latent {
		t.Fatalf("%d latent block(s) ran in ragged steps, want %d (%d steps of %d): the latent "+
			"attention went somewhere else (%s)", got, r.steps*latent, r.steps, latent, g.Err())
	}
	t.Logf("%d latent block(s) of %d ran in each of %d joint steps", latent, m.Cfg.NLayer, r.steps)
	if m.Cfg.Hybrid() {
		hybridStepChecks(t, m, r)
	}
	return r
}

// latentBlocks is how many of m's blocks are latent attention, refusing a
// model with none: every check here would pass with nothing latent in it.
func latentBlocks(m *Model) int {
	n := 0
	if m.Cfg.MLA() {
		for li := 0; li < m.Cfg.NLayer; li++ {
			if !m.Cfg.LayerKind(li).Recurrent() {
				n++
			}
		}
	}
	return n
}

// latentModels is the latent-attention models the step and scheduler gates
// walk: the fixtures, or JITLLM_STEP_LATENT's.
func latentModels() []string {
	if v := os.Getenv("JITLLM_STEP_LATENT"); v != "" {
		return strings.Split(v, ",")
	}
	return []string{"synth-deepseek", "synth-deepseek-dense", "synth-kimilinear"}
}

// openLatent opens a latent-attention model by name: a safetensors fixture
// directory, or a GGUF or container under JITLLM_MODELS. It refuses a model
// with no latent block.
func openLatent(t *testing.T, name string) *Model {
	t.Helper()
	p := testmodels.Resolve(name)
	var src string
	if _, err := os.Stat(filepath.Join(p, "config.json")); err == nil || strings.HasPrefix(filepath.Base(name), "synth-") {
		src = hfContainer(t, name)
	} else {
		if _, err := os.Stat(p); err != nil && !strings.HasSuffix(p, jlm.Ext) {
			t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
		}
		src = jlmOf(t, p)
	}
	m, err := Open(src, noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	if latentBlocks(m) == 0 {
		m.Close()
		t.Fatalf("%s has no latent-attention block -- this gate would prove nothing", name)
	}
	return m
}

// TestSchedulerOnDeviceLatent is TestSchedulerOnDeviceMatchesEachSequenceAlone
// on the latent-attention models: admission and decode as ragged steps whose
// rows each read their own sequence's paged latent history -- a DeepSeek's
// prompts as chunks of consecutive positions beside the decoding rows, a
// Kimi-Linear's a token per step (a hybrid on a device). schedulerOnDeviceModel
// demands latent blocks ran in ragged steps.
func TestSchedulerOnDeviceLatent(t *testing.T) {
	for _, name := range latentModels() {
		t.Run(filepath.Base(name), func(t *testing.T) {
			m := openLatent(t, name)
			defer m.Close()
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					schedulerOnDeviceModel(t, m, func(t *testing.T) *tier.GPU { return stepTier(t, m, dev, false) })
				})
			}
		})
	}
}

// TestSchedulerMixesLatentPromptsIntoDecodeSteps is
// TestSchedulerMixesPromptsIntoDecodeSteps on the latent models, with its
// violation: chunks laid out at mirrored positions must part from the
// sequences alone. A chunk's rows write their latent rows before any row
// attends, so each sees its own sequence's earlier rows of the same step;
// Kimi-Linear's linear blocks take the chunk as a run.
func TestSchedulerMixesLatentPromptsIntoDecodeSteps(t *testing.T) {
	prompts := []string{
		"The capital of France is",
		"Once upon a time, in a small village by the sea, there lived",
		"1, 2, 3,",
		"The quick brown fox jumps over the lazy dog. The quick brown fox",
		"Water boils at",
	}
	gens := []int{12, 20, 8, 16, 10}
	for _, name := range latentModels() {
		t.Run(filepath.Base(name), func(t *testing.T) {
			m := openLatent(t, name)
			defer m.Close()
			ids := make([][]int32, len(prompts))
			for i, pr := range prompts {
				ids[i] = m.Vocab.Encode(pr, true)
			}
			for _, spec := range stepDevices() {
				t.Run(spec, func(t *testing.T) { mixedArm(t, m, ids, gens, spec) })
			}
		})
	}
}

// prefillResult is what latentPrefillArms measured: the batched chunk's final
// residual against the row-by-row arm's, at its 90th percentile and worst
// position, and the logits'.
type prefillResult struct {
	p90, worst, logits float64
	at                 int
}

// TestStepAcrossSessionsNoPELatentGateDiscriminates runs the latent step gate
// on Kimi-Linear with its rotary channels rotated in the ragged step
// (tier.MLAFaultRotateNoPE), as a positional model's would be, and demands the
// gate see it: a NoPE key rotated is exact at position 0 and wrong after, the
// bug AGENTS.md records twice, and a step gate that passed it could not tell
// the ragged path from decode's.
func TestStepAcrossSessionsNoPELatentGateDiscriminates(t *testing.T) {
	m := openLatent(t, "synth-kimilinear")
	defer m.Close()
	if !m.Cfg.NoPosEnc {
		t.Fatal("synth-kimilinear does not set NoPosEnc: the rotation this gate is about runs anyway")
	}
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff),
				tier.WithMLAFault(tier.MLAFaultRotateNoPE)}, testTierOpts(t)...)...)
			if err != nil || g == nil {
				noDevice(t, dev, err)
			}
			defer g.Close()
			r := latentStepGate(t, m, g)
			t.Logf("rotated NoPE channels: worst logit NMSE %.3e (row %d)", r.worst, r.at)
			if r.worst < mlaDeviceNMSE {
				t.Fatalf("the gate passed a ragged step rotating a NoPE key at NMSE %.3e: it cannot see it", r.worst)
			}
		})
	}
}
