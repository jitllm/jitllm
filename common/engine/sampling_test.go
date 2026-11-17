package engine

import (
	"errors"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/model"
)

// Every knob must reach the engine's sampler.
//
// Guards against settings that persist to disk but never reach the sampler.
func TestEverySamplingKnobReachesTheSampler(t *testing.T) {
	in := session.Sampling{
		Temp: 0.7, TopK: 41, TopP: 0.93, MinP: 0.04, TypicalP: 0.9,
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
		{"TypicalP", got.TypicalP, 0.9},
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

// The knobs show the model's own draw and edit from it: over the draw the
// container states (ownSampling) each says its value and that it is the
// model's, and a knob set on what they show changes that value alone. Where
// the draw is not known they say only that it is the model's.
func TestAKnobMovedOffTheModelsOwnKeepsTheRest(t *testing.T) {
	g := jlm.Generation{DoSample: true, Temperature: 0.6, TopK: 20, TopP: 0.95, TypicalP: 0.9, RepetitionPenalty: 1.05}
	own := ownSampling(&g, 4096)
	d := session.DefaultSampling()
	d.Seed = 7
	shown := d.Shown(own)
	for _, k := range session.SamplingKnobs() {
		if c := k.Caption(shown); !strings.HasSuffix(c, ", the model's own") || strings.HasPrefix(c, "the model's own") {
			t.Errorf("%s says %q of a draw it knows", k.Name, c)
		}
		if c := k.Caption(d.Shown(session.Sampling{})); c != "the model's own" {
			t.Errorf("%s says %q of a draw it does not know", k.Name, c)
		}
	}
	for _, k := range session.SamplingKnobs() {
		if k.Name == "Temperature" {
			k.Set(&shown, 0.8)
		}
	}
	got, err := samplerFor(shown, &g, 4096, "m")
	if err != nil {
		t.Fatal(err)
	}
	want := engine.StatedSampler(g, 4096)
	near := func(a, b float64) bool { return a > b-1e-6 && a < b+1e-6 }
	if shown.Stated || !near(got.Temp, 0.8) || got.TopK != want.TopK || !near(got.TopP, want.TopP) ||
		!near(got.TypicalP, want.TypicalP) || !near(got.RepeatPen, want.RepeatPen) || got.RepeatLastN != want.RepeatLastN ||
		got.Seed != 7 {
		t.Errorf("the temperature moved to 0.8 drew %+v, not the model's %+v at that temperature", *got, *want)
	}
	if ownSampling(nil, 4096).Own {
		t.Error("a container stating no draw gave the knobs one to show")
	}
}
