package graph

import "testing"

// TestDriftIsGenericOps verifies the shadow arm's comparison and holds it to
// what the arm binds: four inputs, the block's input, the two outputs and the
// two-word state, the state alone coming out, and nothing in it a weight, a
// history or a model.
func TestDriftIsGenericOps(t *testing.T) {
	const n = 64
	g := (&Drift{NEmbd: n}).Build()
	if err := Verify(g, Facts{}); err != nil {
		t.Fatalf("the comparison does not verify: %v", err)
	}
	widths := []int{n, n, n, 2}
	if len(g.In) != len(widths) || len(g.Out) != 1 {
		t.Fatalf("%d inputs and %d outputs, want 4 and 1", len(g.In), len(g.Out))
	}
	for i, w := range widths {
		if got := g.Values[g.In[i]].Width; got != w {
			t.Errorf("input %d is %d wide, want %d", i, got, w)
		}
	}
	if got := g.Values[g.Out[0]].Width; got != 2 {
		t.Errorf("the state comes out %d wide, want 2", got)
	}
	kinds := map[OpKind]int{}
	for _, o := range g.Ops {
		kinds[o.Kind]++
		if len(o.Weight) != 0 || len(o.Reads) != 0 || len(o.Writes) != 0 {
			t.Errorf("%s binds a weight or a state: the comparison reads only its inputs", o.Kind)
		}
	}
	for _, k := range []OpKind{OpMap, OpReduce, OpSlice, OpConcat} {
		if kinds[k] == 0 {
			t.Errorf("the comparison has no %s", k)
		}
	}
	if kinds[OpReduce] != 2 {
		t.Errorf("%d reductions, want the two squared distances", kinds[OpReduce])
	}
	for k := range kinds {
		switch k {
		case OpMap, OpReduce, OpSlice, OpConcat:
		default:
			t.Errorf("the comparison uses %s, which is not a primitive", k)
		}
	}
}
