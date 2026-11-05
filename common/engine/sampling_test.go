package engine

import (
	"testing"

	"github.com/jitllm/jitllm/common/session"
)

// Every knob must reach the engine's sampler.
//
// Guards against settings that persist to disk but never reach the sampler.
func TestEverySamplingKnobReachesTheSampler(t *testing.T) {
	in := session.Sampling{
		Temp: 0.7, TopK: 41, TopP: 0.93, MinP: 0.04,
		RepeatPen: 1.15, RepeatLastN: 128, Seed: 99,
	}
	got := samplerFor(in)

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
	got := samplerFor(session.Sampling{Temp: 0, TopK: 40, TopP: 0.95, MinP: 0.05})
	if got.Temp != 0 {
		t.Errorf("Temp = %v, want exactly 0: anything else is not greedy", got.Temp)
	}
}

// The app's own defaults must be the ones that reach the engine.
func TestTheDefaultSettingsAreWhatTheEngineGets(t *testing.T) {
	d := session.DefaultSampling()
	got := samplerFor(d)
	if got.MinP == 0 || got.RepeatPen <= 1 {
		t.Errorf("MinP %v / RepeatPen %v: the defaults that were never read are "+
			"still not read", got.MinP, got.RepeatPen)
	}
}
