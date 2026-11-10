package engine

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestHeadChunksMatchOneHead runs a ragged step across sessions on a card
// whose per-row logits hold one row (Config.HeadRowBytes): every wanted row
// past the first is headed after the submission, a chunk at a time
// (headChunk), and the step agrees with each session alone as one head over
// every row does. A chunk read at the wrong rows parts it.
func TestHeadChunksMatchOneHead(t *testing.T) {
	m, err := model.Open(jlmOf(t, testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			g := stepTier(t, m, dev, false, gpu.WithConfig(func(c *gpu.Config) {
				c.HeadRowBytes = 4 * m.Cfg.NVocab
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
			if s1.HeadChunks == s0.HeadChunks {
				t.Fatalf("%d rows across sessions ran no head chunk: the configuration under test was not selected (%s)", rows, g.Err())
			}
			w, at := worstStep(got, want)
			t.Logf("%d rows, %d head chunks: worst NMSE %.3e (row %d) against each session alone",
				rows, s1.HeadChunks-s0.HeadChunks, w, at)
			if !(w < mlaDeviceNMSE) {
				t.Fatalf("a step whose head ran in chunks disagrees with each session alone: NMSE %.3e at row %d", w, at)
			}
		})
	}
}
