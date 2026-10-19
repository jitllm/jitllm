package graph

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// inventoryOf is the inventory a graph's own bindings state.
func inventoryOf(g *Graph) map[jlm.Role]Shape {
	inv := map[jlm.Role]Shape{}
	for _, o := range g.Ops {
		for _, w := range o.Weight {
			inv[w.Role] = Shape{K: w.K, Rows: w.Rows, Sheets: w.Sheets}
		}
	}
	return inv
}

// TestConformerVerifies builds a FastConformer block, verifies it, and holds
// the verifier to a configuration that disagrees with the graph: the
// convolution's taps, and a dense block's facts in place of the conformer's.
func TestConformerVerifies(t *testing.T) {
	c := Conformer{Block: 3, NEmbd: 64, NFFN: 256, NHead: 4, HeadDim: 16, Taps: 9, Eps: 1e-5}
	g := c.Build()
	inv := inventoryOf(g)
	if err := Verify(g, Facts{Inventory: inv, Conformer: &c}); err != nil {
		t.Fatal(err)
	}
	if u := Unbound(g, Facts{Inventory: inv, Conformer: &c}); len(u) != 0 {
		t.Fatalf("unbound %v", u)
	}
	kinds := map[OpKind]int{}
	for _, o := range g.Ops {
		kinds[o.Kind]++
	}
	if kinds[OpConv] != 1 || kinds[OpAffine] != 1 || kinds[OpRelAttend] != 1 || kinds[OpLayerNorm] != 5 {
		t.Fatalf("ops %v: want one conv, affine and relative attention, and five LayerNorms", kinds)
	}
	wrong := c
	wrong.Taps = 7
	if Verify(g, Facts{Inventory: inv, Conformer: &wrong}) == nil {
		t.Fatal("a graph of 9 taps verified against a configuration of 7")
	}
	if Verify(g, Facts{Inventory: inv}) == nil {
		t.Fatal("a conformer graph verified against a dense block's facts")
	}
}

// TestClampTableVerifies builds a dense block with clipped linears: every
// linear's input and output clamped at its place in the table, and a table
// the configuration does not state refused.
func TestClampTableVerifies(t *testing.T) {
	d := Dense{Block: 0, NEmbd: 32, NHead: 2, NKVHead: 2, HeadDim: 16, NFFN: 64, Eps: 1e-6, Scale: 0.25,
		Act: ActGELUTanh, NonCausal: true, NoRope: true, ClampTable: jlm.RoleVClamp}
	g := d.Build()
	inv := inventoryOf(g)
	if err := Verify(g, Facts{Config: d, Inventory: inv}); err != nil {
		t.Fatal(err)
	}
	clamps := 0
	for _, o := range g.Ops {
		if o.Kind == OpClamp {
			clamps++
		}
	}
	if clamps != 14 {
		t.Fatalf("%d clamps, want 14: an input and an output on each of seven linears", clamps)
	}
	none := d
	none.ClampTable = 0
	if Verify(g, Facts{Config: none, Inventory: inv}) == nil {
		t.Fatal("a clipped graph verified against a configuration with no bounds table")
	}
}
