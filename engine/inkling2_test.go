package engine

import (
	"testing"

	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/model"
)

// TestInkling2FeaturesAreLoadBearing holds Inkling 2's statement
// (convert/models/inkling2.json) against transformers' own class: the stated
// graph within its golden's bound, and each feature, removed, outside it --
// the relative bias by backward distance, a row's own sinks, the full
// layers' per-row convolution taps and the second pass's messages.
func TestInkling2FeaturesAreLoadBearing(t *testing.T) {
	const name = "synth-hf-inkling2"
	m, err := model.Open(hfContainer(t, name))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	bound := hfGoldBound()[name]
	if w := hfWorst(t, m, name); !(w <= bound) {
		t.Fatalf("the stated graph: worst NMSE %.3e against transformers (bound %.0e)", w, bound)
	}
	attend := func(edit func(o *graph.Op)) func(*model.Model) func() {
		return func(m *model.Model) func() { return restated(m, editing(opsOfKind(graph.OpAttend), edit)) }
	}
	noBias := attend(func(o *graph.Op) {
		if o.Attr.Extent > 0 {
			o.In = append(o.In[:1:1], o.In[2:]...)
			o.Attr.Extent = 0
		}
	})
	noSinks := attend(func(o *graph.Op) {
		if o.Attr.RowSinks {
			o.In = o.In[:len(o.In)-1]
			o.Attr.RowSinks = false
		}
	})
	noTaps := func(m *model.Model) func() {
		return restated(m, editing(opsOfKind(graph.OpStateConv), func(o *graph.Op) { o.In = o.In[:1] }))
	}
	noMessages := func(m *model.Model) func() {
		ungated := func(o *graph.Op) {
			o.Attr.Expr = []graph.Term{{Fn: graph.MapAdd, Args: [3]graph.Arg{{Kind: graph.ArgIn}, {Kind: graph.ArgConst}}}}
		}
		undo := []func(){}
		for _, n := range []string{"hq.gated", "hk.gated", "hv.gated", "fi.gated"} {
			undo = append(undo, restated(m, editing(opsNamed(n), ungated)))
		}
		return func() {
			for i := len(undo) - 1; i >= 0; i-- {
				undo[i]()
			}
		}
	}
	for _, c := range []struct {
		name  string
		apply func(*model.Model) func()
	}{
		{"no relative bias", noBias},
		{"no sinks", noSinks},
		{"no per-row convolution taps", noTaps},
		{"no messages from the first pass", noMessages},
	} {
		undo := c.apply(m)
		w := hfWorst(t, m, name)
		undo()
		if !(w > bound) {
			t.Errorf("%s: worst NMSE %.3e, inside the band (%.0e): the feature is not load-bearing", c.name, w, bound)
			continue
		}
		t.Logf("%s: worst NMSE %.3e", c.name, w)
	}
}
