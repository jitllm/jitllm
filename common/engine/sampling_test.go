package engine

import (
	"errors"
	"testing"

	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/model"
)

// Every knob must reach the engine's sampler.
//
// Guards against settings that persist to disk but never reach the sampler.
func TestEverySamplingKnobReachesTheSampler(t *testing.T) {
	in := session.Sampling{
		Temp: 0.7, TopK: 41, TopP: 0.93, MinP: 0.04,
		RepeatPen: 1.15, RepeatLastN: 128, Seed: 99,
	}
	ref := jlm.ReferenceGeneration()
	got, err := samplerFor(in, &ref, 4096, "m")
	if err != nil {
		t.Fatal(err)
	}

	for _, c := range []struct {
		name      string
		got, want float64
	}{
		{"Temp", got.Temp, 0.7},
		{"TopP", got.TopP, 0.93},
		{"MinP", got.MinP, 0.04},
		{"RepeatPen", got.RepeatPen, 1.15},
	} {
		if c.got < c.want-1e-6 || c.got > c.want+1e-6 {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}
	if got.TopK != 41 || got.RepeatLastN != 128 || got.Seed != 99 {
		t.Errorf("TopK/RepeatLastN/Seed = %d/%d/%d, want 41/128/99",
			got.TopK, got.RepeatLastN, got.Seed)
	}
}

// Greedy must stay reachable, exactly.
//
// Temp == 0 cannot be approximated: engine.Sampler short-circuits to argmax,
// and every correctness comparison in the engine is of argmax.
func TestGreedyIsReachableExactly(t *testing.T) {
	ref := jlm.ReferenceGeneration()
	got, err := samplerFor(session.Sampling{Temp: 0, TopK: 40, TopP: 0.95, MinP: 0.05}, &ref, 4096, "m")
	if err != nil || got.Temp != 0 {
		t.Errorf("Temp = %v, want exactly 0: anything else is not greedy", got.Temp)
	}
}

// The app's own defaults are the model's: the draw its container states
// reaches the engine, a repetition penalty over the whole window, and a knob
// set makes the settings the caller's.
func TestTheDefaultSettingsAreTheModelsOwn(t *testing.T) {
	d := session.DefaultSampling()
	g := jlm.Generation{DoSample: true, Temperature: 0.7, TopK: 20, TopP: 0.8, TypicalP: 1, RepetitionPenalty: 1.1}
	got, err := samplerFor(d, &g, 4096, "m")
	if err != nil {
		t.Fatal(err)
	}
	if got.Temp != 0.7 || got.TopK != 20 || got.TopP != 0.8 || got.RepeatPen != 1.1 || got.RepeatLastN != 4096 {
		t.Errorf("the defaults reached the engine as %+v, not the model's %+v", *got, g)
	}
	ref := jlm.ReferenceGeneration()
	if greedy, err := samplerFor(d, &ref, 4096, "m"); err != nil || greedy.Temp != 0 {
		t.Errorf("a model that states no sampling drew %+v (%v), not greedily", greedy, err)
	}
	var nd *model.NoStatedDrawError
	if sm, err := samplerFor(d, nil, 4096, "old.jlm"); !errors.As(err, &nd) || nd.Name != "old.jlm" {
		t.Errorf("the model's own settings of a container stating no draw drew %+v (%v), not a refusal naming it", sm, err)
	}
	for _, k := range session.SamplingKnobs() {
		if k.Caption(d) != "the model's own" {
			t.Errorf("%s says %q of the model's own draw", k.Name, k.Caption(d))
		}
	}
	k := session.SamplingKnobs()[0]
	k.Set(&d, 0.5)
	if sm, err := samplerFor(d, nil, 4096, "old.jlm"); d.Stated || err != nil || sm.Temp != 0.5 {
		t.Errorf("the caller's own settings were refused, or a knob left the model's draw: %+v (%v)", sm, err)
	}
	if sm, err := samplerFor(d, &g, 4096, "m"); d.Stated || err != nil || sm.Temp != 0.5 {
		t.Errorf("setting %s left the model's draw in place: %+v", k.Name, d)
	}
	if samplingOf(session.DefaultSampling()) != nil {
		t.Error("the model's own draw was sent to the service as settings of the app's")
	}
}
