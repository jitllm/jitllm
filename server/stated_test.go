package server

import (
	"errors"
	"testing"

	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/model"
)

// TestASessionWithNoSamplingDrawsTheStatedDraw: a session opened with no
// sampling draws what its model's container states (engine.StatedSampler,
// over the session's context), each session seeded apart; one opened with
// its own sampling keeps it. A container that states no draw gives a
// session opened with none no draw: a generate asking for none is refused
// as the caller's input, naming the model, and one stating its own runs.
func TestASessionWithNoSamplingDrawsTheStatedDraw(t *testing.T) {
	gen := jlm.ReferenceGeneration()
	gen.DoSample, gen.Temperature, gen.TopK, gen.TopP, gen.RepetitionPenalty = true, 0.6, 20, 0.95, 1.05
	lm := &LoadedModel{id: "m", m: &model.Model{Cfg: &model.Config{Generation: &gen}}}
	want := engine.StatedSampler(gen, 64)
	a, aerr := sessionSampler(nil, lm, 64, "")
	b, berr := sessionSampler(nil, lm, 64, "")
	if aerr != nil || berr != nil {
		t.Fatalf("a container stating its draw refused it: %v, %v", aerr, berr)
	}
	if a.Temp != 0.6 || a.TopK != 20 || a.TopP != 0.95 || a.RepeatPen != 1.05 || a.RepeatLastN != 64 ||
		a.Temp != want.Temp || a.TopK != want.TopK || a.TopP != want.TopP || a.MinP != want.MinP || a.TypicalP != want.TypicalP {
		t.Errorf("a session with no sampling draws %+v, the container states %+v", a, *want)
	}
	if a.Seed == b.Seed {
		t.Errorf("two sessions drew one seed, %d", a.Seed)
	}
	own := engine.Sampler{Temp: 1.3, TopK: 5, Seed: 7}
	if got, err := sessionSampler(&own, lm, 64, ""); err != nil || got.Temp != 1.3 || got.TopK != 5 || got.TopP != 0 || got.Seed != 7 {
		t.Errorf("a session's own sampling drew %+v (%v), want %+v", got, err, own)
	}

	old := &LoadedModel{id: "old", m: &model.Model{Cfg: &model.Config{}}}
	sm, undrawn := sessionSampler(nil, old, 64, "the-model")
	var nd *model.NoStatedDrawError
	if !errors.Is(undrawn, ErrInvalid) || !errors.As(undrawn, &nd) || nd.Name != "the-model" {
		t.Fatalf("a container stating no draw gave a session %+v and %v, want the caller's input naming it", sm, undrawn)
	}
	s := &Session{sampling: sm, undrawn: undrawn}
	if _, err := s.draw(nil); !errors.As(err, &nd) {
		t.Errorf("a generate asking no draw of it ran: %v", err)
	}
	if got, err := s.draw(&own); err != nil || got.Temp != 1.3 || got.Seed != 7 {
		t.Errorf("a generate stating its own draw drew %+v (%v)", got, err)
	}
	if _, undrawn := sessionSampler(&own, old, 64, ""); undrawn != nil {
		t.Errorf("a session opened with its own draw was refused: %v", undrawn)
	}

	// ModelInfo states the draw a session opened with none takes, and the
	// wire's sampling of it draws that.
	p := pbStatedSampling(lm, 64)
	got := pbSampling(p)
	near := func(a, b float64) bool { return a > b-1e-6 && a < b+1e-6 }
	if p == nil || !near(got.Temp, want.Temp) || got.TopK != want.TopK || !near(got.TopP, want.TopP) || !near(got.MinP, want.MinP) ||
		!near(got.TypicalP, want.TypicalP) || !near(got.RepeatPen, want.RepeatPen) || got.RepeatLastN != want.RepeatLastN {
		t.Errorf("ModelInfo states the draw as %+v, the container's is %+v", got, *want)
	}
	if pbStatedSampling(old, 64) != nil {
		t.Error("ModelInfo stated a draw for a container stating none")
	}
}
