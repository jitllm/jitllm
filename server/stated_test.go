package server

import (
	"testing"

	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/model"
)

// TestASessionWithNoSamplingDrawsTheStatedDraw: a session opened with no
// sampling draws what its model's container states (engine.StatedSampler,
// over the session's context), each session seeded apart; one opened with
// its own sampling keeps it.
func TestASessionWithNoSamplingDrawsTheStatedDraw(t *testing.T) {
	gen := jlm.ReferenceGeneration()
	gen.DoSample, gen.Temperature, gen.TopK, gen.TopP, gen.RepetitionPenalty = true, 0.6, 20, 0.95, 1.05
	lm := &LoadedModel{m: &model.Model{Cfg: &model.Config{Generation: gen}}}
	want := engine.StatedSampler(gen, 64)
	a, b := sessionSampler(nil, lm, 64), sessionSampler(nil, lm, 64)
	if a.Temp != 0.6 || a.TopK != 20 || a.TopP != 0.95 || a.RepeatPen != 1.05 || a.RepeatLastN != 64 ||
		a.Temp != want.Temp || a.TopK != want.TopK || a.TopP != want.TopP || a.MinP != want.MinP || a.TypicalP != want.TypicalP {
		t.Errorf("a session with no sampling draws %+v, the container states %+v", a, *want)
	}
	if a.Seed == b.Seed {
		t.Errorf("two sessions drew one seed, %d", a.Seed)
	}
	own := engine.Sampler{Temp: 1.3, TopK: 5, Seed: 7}
	if got := sessionSampler(&own, lm, 64); got.Temp != 1.3 || got.TopK != 5 || got.TopP != 0 || got.Seed != 7 {
		t.Errorf("a session's own sampling drew %+v, want %+v", got, own)
	}
}
