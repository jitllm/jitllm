package model

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
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

// TestLatentPrefillBatchesOnTheDevice holds a latent model's batched prompt
// chunk on the device -- the latent and its key written for every row before
// any row attends, the absorbs grouped by head, float banks included, and a
// mixture's experts grouped -- to the same prompt a row at a time on the same
// device (Config.NoBatch), at every position's final residual and at the
// logits. The step gates prefill both their arms alike, so a chunk wrong in
// the same way on both would pass them; this one is the chunk's own gate.
// Stats.MLABatched is the selection: the batched arm ran latent blocks as one
// chunk and the row arm none.
func TestLatentPrefillBatchesOnTheDevice(t *testing.T) {
	for _, name := range latentModels() {
		t.Run(filepath.Base(name), func(t *testing.T) {
			m := openLatent(t, name)
			defer m.Close()
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					r := latentPrefillArms(t, m, dev)
					// TestHybridPrefillBatchesOnTheDevice's bounds, for its
					// reason: the bulk at the band, one position looser for a
					// routing tie a real mixture can break the other way when
					// the arms reduce in different orders.
					if !(r.p90 <= hybridBatchBound) || !(r.worst <= 10*hybridBatchBound) || !(r.logits <= hybridBatchBound) {
						t.Fatalf("the batched chunk parts from the row-by-row device prefill: residual NMSE p90 %.3e, "+
							"worst %.3e at position %d, logits %.3e", r.p90, r.worst, r.at, r.logits)
					}
				})
			}
		})
	}
}

// TestLatentPrefillGateDiscriminates runs the prefill gate on Kimi-Linear with
// its rotary channels rotated in the batched chunk (tier.MLAFaultRotateNoPE)
// and the row-by-row arm left as it ships, and demands it see the difference.
func TestLatentPrefillGateDiscriminates(t *testing.T) {
	m := openLatent(t, "synth-kimilinear")
	defer m.Close()
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			r := latentPrefillArms(t, m, dev, tier.WithMLAFault(tier.MLAFaultRotateNoPE))
			if r.p90 <= hybridBatchBound && r.worst <= 10*hybridBatchBound && r.logits <= hybridBatchBound {
				t.Fatalf("the gate passed a chunk rotating a NoPE key: residual NMSE p90 %.3e, worst %.3e, logits %.3e",
					r.p90, r.worst, r.logits)
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

// latentPrefillArms prefills one prompt on dev twice, as a chunk and a row at
// a time, and compares them; opt joins the tier's options in both arms. It
// fails only when the arms are not the two paths.
func latentPrefillArms(t *testing.T, m *Model, dev string, opt ...tier.Option) prefillResult {
	c := m.Cfg
	ids := m.Vocab.Encode("The clerk counted barrels of salt on the upper floor, and the "+
		"foreman wrote each tally twice, once in ink and once in chalk, so that the night watch "+
		"could check the count against the ledger before the barges left at dawn.", true)
	forced := []int32{5, 9, 3, 7}
	type arm struct {
		rows   []float32
		logits [][]float32
		st     tier.Stats
		err    string
	}
	run := func(noBatch bool) (a arm) {
		g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff),
			tier.WithPoisonScratch(true), tier.WithConfig(func(cf *tier.Config) { cf.NoBatch = noBatch })}, opt...)...)
		if err != nil || g == nil {
			noDevice(t, dev, err)
		}
		defer g.Close()
		st := m.NewState(len(ids) + len(forced) + 1)
		defer st.Close()
		if err := st.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if st.GPULayers() != c.NLayer {
			t.Skipf("CARD TOO SMALL: %d of %d blocks -- this gate proved nothing here (%s)",
				st.GPULayers(), c.NLayer, g.Err())
		}
		// A folded head brings only the logits home, not the chunk's residual
		// rows, which are what this compares.
		st.noHeadFold = true
		a.rows = make([]float32, len(ids)*c.NEmbd)
		st.rowTap = func(base int, rows []float32) { copy(a.rows[base*c.NEmbd:], rows) }
		lg, err := st.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		a.logits = append(a.logits, append([]float32(nil), lg...))
		for _, id := range forced {
			if lg, err = st.Forward(id); err != nil {
				t.Fatal(err)
			}
			a.logits = append(a.logits, append([]float32(nil), lg...))
		}
		a.st, a.err = g.Stats(), g.Err()
		return a
	}
	rows, bat := run(true), run(false)
	if rows.st.MLABatched != 0 || bat.st.MLABatched == 0 {
		t.Fatalf("latent blocks run as a chunk: %d row by row, %d batched -- the arms are not the two paths (%s)",
			rows.st.MLABatched, bat.st.MLABatched, bat.err)
	}
	// MLABatched counts a chunk's emission, so a chunk that failed after its
	// latent blocks and went again a row at a time counts too: the batched arm
	// must have taken one submission for the prompt, one for a head the chunk
	// could not fold, and one a forced token.
	if limit := len(forced) + 2; bat.st.Submits > limit {
		t.Fatalf("the batched arm took %d submissions, past %d: the chunk went a row at a time (%s)",
			bat.st.Submits, limit, bat.err)
	}
	worst, at := 0.0, -1
	per := make([]float64, 0, len(ids))
	for p := range ids {
		r := c.NEmbd
		e, _ := worstOf([][]float32{rows.rows[p*r : (p+1)*r]}, [][]float32{bat.rows[p*r : (p+1)*r]})
		if e > worst {
			worst, at = e, p
		}
		per = append(per, e)
	}
	lg, _ := worstOf(rows.logits, bat.logits)
	sort.Float64s(per)
	p90 := per[len(per)*9/10]
	t.Logf("%d positions, %d latent block-chunks: residual NMSE batched/rows p90 %.3e, worst %.3e at %d, logits %.3e",
		len(ids), bat.st.MLABatched, p90, worst, at, lg)
	return prefillResult{p90: p90, worst: worst, logits: lg, at: at}
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
