package main

import (
	"testing"

	"github.com/jitllm/jitllm/common/session"
)

// TestSavingTheSettingsKeepsTheModelsOwnDraw: saving the settings with the
// knobs untouched leaves the draw the model's own, and a knob changed there
// changes that value alone, the model's other values kept.
func TestSavingTheSettingsKeepsTheModelsOwnDraw(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	t.Setenv("HOME", dir)
	t.Setenv("AppData", dir)
	r := newTestRoot(t)
	r.env.cfg.Sampling = session.DefaultSampling()
	r.env.f.st.Own.Set(session.Sampling{Temp: 0.6, TopK: 20, TopP: 0.95, TypicalP: 0.9, RepeatPen: 1, Own: true})

	p := newSettingsPage(r.env)
	p.apply()
	if s := r.env.cfg.Sampling; !s.Stated {
		t.Fatalf("saving the settings untouched made the draw the caller's: %+v", s)
	}

	p = newSettingsPage(r.env)
	temp := -1
	for i, k := range session.SamplingKnobs() {
		if k.Name == "Temperature" {
			temp = i
		}
	}
	if temp < 0 || p.d.knobs[temp] != "0.6" {
		t.Fatalf("the temperature knob shows %q, not the model's 0.6", p.d.knobs[max(temp, 0)])
	}
	p.d.knobs[temp] = "0.8"
	p.apply()
	if s := r.env.cfg.Sampling; s.Stated || s.Temp != 0.8 || s.TopK != 20 || s.TopP != 0.95 || s.TypicalP != 0.9 || s.RepeatPen != 1 {
		t.Errorf("the temperature changed to 0.8 left %+v, not the model's other values", s)
	}
}
