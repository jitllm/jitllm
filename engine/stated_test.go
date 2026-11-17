package engine

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestStatedSamplerIsTheContainers holds the engine's default draw to the
// container's statement: the qwen3 fixture, converted from its directory,
// carries Qwen3's generation_config.json (do_sample, temperature 0.6, top-k
// 20, top-p 0.95) and opens with that draw, which StatedSampler draws with;
// a draw that does not sample is greedy, its repetition penalty read over
// the whole context.
func TestStatedSamplerIsTheContainers(t *testing.T) {
	p := testmodels.Path("qwen3")
	if _, err := os.Stat(p); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate would prove nothing (RULE 11)", err)
	}
	m, err := model.Open(jlmOf(t, p))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	want := jlm.Generation{DoSample: true, Temperature: 0.6, TopK: 20, TopP: 0.95, TypicalP: 1, RepetitionPenalty: 1}
	if m.Cfg.Generation == nil || *m.Cfg.Generation != want {
		t.Fatalf("the model opened with the draw %+v, its file states %+v", m.Cfg.Generation, want)
	}
	s := StatedSampler(*m.Cfg.Generation, 512)
	if s.Temp != 0.6 || s.TopK != 20 || s.TopP != 0.95 || s.MinP != 0 || s.RepeatPen != 1 || s.RepeatLastN != 0 {
		t.Errorf("the stated draw samples as %+v", *s)
	}
	g := jlm.ReferenceGeneration()
	g.RepetitionPenalty = 1.1
	if s := StatedSampler(g, 512); s.Temp != 0 || s.RepeatPen != 1.1 || s.RepeatLastN != 512 {
		t.Errorf("a draw that does not sample: temperature %v, penalty %v over %d tokens", s.Temp, s.RepeatPen, s.RepeatLastN)
	}
}
