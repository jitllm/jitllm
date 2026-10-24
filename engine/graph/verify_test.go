package graph

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// block is a small attention-then-MLP block over d and its inventory.
func block(d, ffn int) (*Graph, Facts) {
	b := NewBlock(0)
	x := b.In("x", d)
	n := b.Norm(x, jlm.RoleAttnNorm, 1e-5)
	q := b.MatVec(n, jlm.RoleAttnQ, d)
	k := b.MatVec(n, jlm.RoleAttnK, d)
	v := b.MatVec(n, jlm.RoleAttnV, d)
	hist := b.KVAppend(k, v, 0)
	a := b.Attend(q, hist, Attrs{NHead: 2, NKVHead: 2, HeadDim: d / 2, Scale: 1})
	x2 := b.Add(x, b.MatVec(a, jlm.RoleAttnOut, d))
	m := b.Norm(x2, jlm.RoleFFNNorm, 1e-5)
	g := b.MatVec(m, jlm.RoleFFNGate, ffn)
	u := b.MatVec(m, jlm.RoleFFNUp, ffn)
	b.Out(b.Add(x2, b.MatVec(b.ActMul(ActSiLU, g, u), jlm.RoleFFNDown, d)))
	inv := map[jlm.Role]Shape{
		jlm.RoleAttnNorm: {K: d}, jlm.RoleFFNNorm: {K: d},
		jlm.RoleAttnQ: {K: d, Rows: d}, jlm.RoleAttnK: {K: d, Rows: d}, jlm.RoleAttnV: {K: d, Rows: d}, jlm.RoleAttnOut: {K: d, Rows: d},
		jlm.RoleFFNGate: {K: d, Rows: ffn}, jlm.RoleFFNUp: {K: d, Rows: ffn}, jlm.RoleFFNDown: {K: ffn, Rows: d},
	}
	return b.G, Facts{Inventory: inv}
}

// TestVerifierHoldsTheStructure verifies a correct block, then each
// violation of what the verifier checks, every one of which must be refused
// naming its fault.
func TestVerifierHoldsTheStructure(t *testing.T) {
	g, f := block(8, 16)
	if err := Verify(g, f); err != nil {
		t.Fatalf("a correct block is refused: %v", err)
	}
	if u := Unbound(g, f); len(u) != 0 {
		t.Fatalf("a correct block leaves %v unbound", u)
	}
	for _, c := range []struct {
		name, want string
		break_     func(g *Graph, f Facts)
	}{
		{"another version", "op language", func(g *Graph, f Facts) { g.Version = Version + 1 }},
		{"a read before its write", "before anything writes it", func(g *Graph, f Facts) { g.Ops[0], g.Ops[1] = g.Ops[1], g.Ops[0] }},
		{"two producers", "already has a producer", func(g *Graph, f Facts) { g.Ops[2].Out = g.Ops[1].Out }},
		{"an attention before its append", "in force", func(g *Graph, f Facts) { g.Ops[4].Reads[0].Version = 2 }},
		{"a role the container does not store", "does not store", func(g *Graph, f Facts) { delete(f.Inventory, jlm.RoleAttnQ) }},
		{"a weight bound at another shape", "stored", func(g *Graph, f Facts) { f.Inventory[jlm.RoleFFNDown] = Shape{K: 8, Rows: 16} }},
		{"a norm's scale wider than its row", "a scale of", func(g *Graph, f Facts) { g.Ops[0].Weight[0].K = 16; f.Inventory[jlm.RoleAttnNorm] = Shape{K: 16} }},
		{"an output never written", "never written", func(g *Graph, f Facts) { g.Out = append(g.Out, ValueID(len(g.Values))) }},
	} {
		g, f := block(8, 16)
		c.break_(g, f)
		err := Verify(g, f)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: Verify said %v, want an error naming %q", c.name, err, c.want)
		}
	}
	g, f = block(8, 16)
	f.Inventory[jlm.RoleAttnQBias] = Shape{K: 8}
	if u := Unbound(g, f); len(u) != 1 || u[0] != jlm.RoleAttnQBias {
		t.Errorf("a stored bias no op binds is reported as %v", u)
	}
}
