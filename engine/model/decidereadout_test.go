package model

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/jinja"
)

// readoutPrompt is a decoder readout's prompt as the principle gates run it
// in pieces: start readies the State, the caller prefills ids in as many
// calls as it likes, and finish reads the scores from the last call's logits
// -- a label readout's label logits, or Clef's head over every row the calls
// kept.
type readoutPrompt struct {
	ids    []int32
	start  func()
	finish func(lg []float32) []float32
}

// readoutOf is the first question of the request as a label readout asks
// it, or the whole request as Clef asks it.
func readoutOf(t *testing.T, d *Decider, state jinja.Value, qs []DecisionQuestion) readoutPrompt {
	t.Helper()
	if d.kind == jlm.DecisionClef {
		p, err := d.clefPrompt(state, qs)
		if err != nil {
			t.Fatal(err)
		}
		return readoutPrompt{ids: p.ids, start: func() { d.clefBegin(p) }, finish: func([]float32) []float32 {
			d.clefEnd()
			out := make([]float32, len(p.opts))
			if err := d.clefHead(p, out); err != nil {
				t.Fatal(err)
			}
			return out
		}}
	}
	ids, groups, err := d.prompt(state, &qs[0], 0)
	if err != nil {
		t.Fatal(err)
	}
	return readoutPrompt{ids: ids, start: d.st.Reset, finish: func(lg []float32) []float32 {
		s, err := d.labelScores(lg, groups)
		if err != nil {
			t.Fatal(err)
		}
		return s
	}}
}
