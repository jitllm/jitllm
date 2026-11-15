package convert

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestEachRunsItsSteps: an each unrolls its steps once a run, the run's
// index read by its name, and joins the runs' last values in one concat; a
// run named as a fact is refused, and so is the index read outside it.
func TestEachRunsItsSteps(t *testing.T) {
	const ok = `{"block": [
	  {"op": "in", "width": "embd", "out": "x"},
	  {"op": "each", "n": "2 + 1", "as": "j", "name": "parts", "out": "y", "steps": [
	    {"op": "emit", "kind": "slice", "in": "x", "attr": {"Offset": "j", "Width": "1"}, "outs": [{"name": "part", "width": "1"}], "out": "p"}
	  ]},
	  {"op": "out", "in": "y"}]}`
	parse := func(src string) (*mjGraph, error) {
		var g mjGraph
		if err := json.Unmarshal([]byte(src), &g); err != nil {
			return nil, err
		}
		return &g, g.check()
	}
	g, err := parse(ok)
	if err != nil {
		t.Fatalf("the control: %v", err)
	}
	ops, err := g.unrolled(g.Block, (&blockFacts{c: &jlm.Config{}}).vars(), nil)
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, o := range ops {
		got = append(got, fmt.Sprintf("%s:%s:%v", o.Op, o.Out, o.runs["j"]))
	}
	want := []string{"in:x:0", "emit:y#0:0", "emit:y#1:1", "emit:y#2:2", "concat:y:0", "out::0"}
	if !slices.Equal(got, want) {
		t.Errorf("unrolled to %v, want %v", got, want)
	}
	if c := ops[4]; !slices.Equal(c.In, mjNames{"y#0", "y#1", "y#2"}) || c.Name != "parts" {
		t.Errorf("the join reads %v as %q", c.In, c.Name)
	}
	for _, c := range []struct{ name, from, to, err string }{
		{"a run named as a fact", `"as": "j"`, `"as": "embd"`, "a fact"},
		{"the index outside its each", `{"op": "out", "in": "y"}`,
			`{"op": "emit", "kind": "slice", "in": "x", "attr": {"Offset": "j", "Width": "1"}, "outs": [{"name": "q", "width": "1"}], "out": "q"}, {"op": "out", "in": "y"}`,
			"not a fact of a block"},
		{"no count", `"n": "2 + 1", `, ``, "states a count"},
	} {
		src := strings.Replace(ok, c.from, c.to, 1)
		if src == ok {
			t.Fatalf("%s: the mutation changed nothing", c.name)
		}
		if _, err := parse(src); err == nil || !strings.Contains(err.Error(), c.err) {
			t.Errorf("%s: %v, want an error saying %q", c.name, err, c.err)
		}
	}
}
