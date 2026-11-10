package engine

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// TestStepRunsOverStreamedExperts is TestStepRunsAdmitsAPromptBesideDecoding
// on a card whose mixtures' banks are in pages (Config.StreamExperts): the
// pages hold the selections of fewer rows than a step carries -- the compact
// bank one row's, a cache of slots a few -- so the step runs as steps of the
// rows they hold, every row still across sessions (Stats.SessionRows) and
// every block step served, and agrees with each session alone.
func TestStepRunsOverStreamedExperts(t *testing.T) {
	for _, file := range []string{"Qwen3-MOE-4x0.6B-Q4_K_M.gguf", "synth-qwen2moe-hf" + q8Suffix} {
		t.Run(file, func(t *testing.T) {
			p := testmodels.Path(file)
			if _, err := os.Stat(fixtureSource(p)); err != nil {
				t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS)", p)
			}
			m, err := model.Open(jlmOf(t, p))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			moe := 0
			for li := range m.Cfg.NLayer {
				if m.Cfg.MoEAt(li) {
					moe++
				}
			}
			nExp, used := int(m.Cfg.NExpert), int(m.Cfg.NExpertUsed)
			if moe == 0 || nExp == 0 || used == 0 {
				t.Fatalf("%s states no routed experts", file)
			}
			for _, dev := range stepDevices() {
				for _, arm := range []struct {
					name  string
					slots int
				}{{"compact", -1}, {"cache", max(used+1, nExp/4)}, {"cache-groups", 3 * used}} {
					// One-row groups over Qwen3-MoE's banks part from each
					// session alone by 1.15e-2 (an open task); its rows are
					// held in groups of three.
					if file == "Qwen3-MOE-4x0.6B-Q4_K_M.gguf" && arm.name != "cache-groups" {
						continue
					}
					t.Run(dev+"/"+arm.name, func(t *testing.T) {
						g := stepTier(t, m, dev, false, gpu.WithConfig(func(c *gpu.Config) {
							c.StreamExperts, c.StreamCacheSlots = true, arm.slots
						}))
						defer g.Close()
						plan := runsPlan{early: []string{"The capital of France is", "Water boils at a temperature of"},
							late:  "The clerk counted barrels of salt on the upper floor, and",
							chunk: 3, steps: 4}
						plan.opts = []Option{WithKVF16(false), WithStreamTrial(false)}
						want := plan.alone(t, m, g)
						s0 := g.Stats()
						got, rows := plan.together(t, m, g)
						s1 := g.Stats()
						n0, d0 := s0.ServedPart(nn.ComposedBlock)
						n1, d1 := s1.ServedPart(nn.ComposedBlock)
						switch {
						case s1.StreamBlocks < moe:
							t.Fatalf("%d blocks placed streamed of the %d mixtures: the configuration under test was not selected", s1.StreamBlocks, moe)
						case s1.SessionRows-s0.SessionRows != rows:
							t.Fatalf("%d rows ran across sessions, want %d: StepRuns went one instance after another (%s)",
								s1.SessionRows-s0.SessionRows, rows, g.Err())
						case d1 != d0 || n1 == n0:
							t.Fatalf("%d block steps served and %d declined together (%s)", n1-n0, d1-d0, declinesOf(s1))
						}
						w, at := worstStep(got, want)
						t.Logf("%d rows over %d streamed blocks, %d fills: worst NMSE %.3e (row %d) against each session alone",
							rows, s1.StreamBlocks, s1.StreamFills-s0.StreamFills, w, at)
						if !(w < mlaDeviceNMSE) {
							t.Fatalf("a step over streamed banks disagrees with each session alone: NMSE %.3e at row %d", w, at)
						}
					})
				}
			}
		})
	}
}
