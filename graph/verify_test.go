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
	b.Out(b.Add(x2, b.MatVec(b.Expr("act", ffn, []Term{
		{Fn: MapSigmoid, Args: [3]Arg{{Kind: ArgIn, Index: 0}}},
		{Fn: MapMul, Args: [3]Arg{{Kind: ArgIn, Index: 0}, {Kind: ArgTerm, Index: 0}}},
		{Fn: MapMul, Args: [3]Arg{{Kind: ArgTerm, Index: 1}, {Kind: ArgIn, Index: 1}}},
	}, nil, g, u), jlm.RoleFFNDown, d)))
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
		{"a value of another grid's rows", "rows", func(g *Graph, f Facts) { g.Values[g.Ops[1].Out[0]].Rows = 3 }},
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

// TestVerifierHoldsALayerNormsBias holds a LayerNorm's optional second weight
// to a vector of the row's width, and a stored bias the op does not bind to
// being reported.
func TestVerifierHoldsALayerNormsBias(t *testing.T) {
	const d = 8
	build := func() (*Graph, Facts) {
		b := NewBlock(0)
		b.Out(b.LayerNorm(b.In("x", d), jlm.RoleAttnNorm, 1e-5, jlm.RoleAttnNormBias))
		return b.G, Facts{Inventory: map[jlm.Role]Shape{jlm.RoleAttnNorm: {K: d}, jlm.RoleAttnNormBias: {K: d}}}
	}
	g, f := build()
	if err := Verify(g, f); err != nil {
		t.Fatalf("a LayerNorm with its bias is refused: %v", err)
	}
	if u := Unbound(g, f); len(u) != 0 {
		t.Fatalf("a LayerNorm with its bias leaves %v unbound", u)
	}
	for _, c := range []struct {
		name   string
		break_ func(g *Graph, f Facts)
	}{
		{"a bias wider than the row", func(g *Graph, f Facts) {
			g.Ops[0].Weight[1].K = 2 * d
			f.Inventory[jlm.RoleAttnNormBias] = Shape{K: 2 * d}
		}},
		{"a bias that is a matrix", func(g *Graph, f Facts) {
			g.Ops[0].Weight[1].Rows = d
			f.Inventory[jlm.RoleAttnNormBias] = Shape{K: d, Rows: d}
		}},
		{"a third weight", func(g *Graph, f Facts) {
			g.Ops[0].Weight = append(g.Ops[0].Weight, Binding{Role: jlm.RoleFFNNormBias, K: d})
			f.Inventory[jlm.RoleFFNNormBias] = Shape{K: d}
		}},
	} {
		g, f := build()
		c.break_(g, f)
		if err := Verify(g, f); err == nil || !strings.Contains(err.Error(), "a scale and a bias") {
			t.Errorf("%s: Verify said %v, want an error naming the scale and the bias", c.name, err)
		}
	}
	g, f = build()
	g.Ops[0].Weight = g.Ops[0].Weight[:1]
	if err := Verify(g, f); err != nil {
		t.Fatalf("a LayerNorm without a bias is refused: %v", err)
	}
	if u := Unbound(g, f); len(u) != 1 || u[0] != jlm.RoleAttnNormBias {
		t.Errorf("a stored norm bias no op binds is reported as %v", u)
	}
}

// TestVerifierHoldsAMapsWindow verifies a map reading one element of a stored
// vector, then each window the vector cannot give, every one refused naming it.
func TestVerifierHoldsAMapsWindow(t *testing.T) {
	build := func(off, width int, bound bool) (*Graph, Facts) {
		b := NewBlock(0)
		x := b.In("x", 8)
		var w []Binding
		if bound {
			w = []Binding{{Role: jlm.RoleXIELU, K: 4}}
		}
		b.Out(b.Emit(OpMap, []ValueID{x}, []Result{{Name: "y", Width: 8}}, w, Attrs{Fn: MapMul, Offset: off, Width: width}, nil, nil)[0])
		return b.G, Facts{Inventory: map[jlm.Role]Shape{jlm.RoleXIELU: {K: 4}}}
	}
	if g, f := build(3, 1, true); Verify(g, f) != nil {
		t.Fatalf("the last element of the vector is refused: %v", Verify(g, f))
	}
	// A table of several rows is its elements in order: a window on its
	// second row is held to the table, past it refused.
	table := func(off int) error {
		b := NewBlock(0)
		x := b.In("x", 8)
		w := []Binding{{Role: jlm.RoleXIELU, K: 4, Rows: 2}}
		b.Out(b.Emit(OpMap, []ValueID{x}, []Result{{Name: "y", Width: 8}}, w, Attrs{Fn: MapMul, Offset: off, Width: 4}, nil, nil)[0])
		return Verify(b.G, Facts{Inventory: map[jlm.Role]Shape{jlm.RoleXIELU: {K: 4, Rows: 2}}})
	}
	if err := table(4); err != nil {
		t.Fatalf("a window on a table's second row is refused: %v", err)
	}
	if err := table(5); err == nil || !strings.Contains(err.Error(), "a window of") {
		t.Errorf("a window past a table of two rows: Verify said %v", err)
	}
	for _, c := range []struct {
		name       string
		off, width int
		bound      bool
	}{
		{"past the vector", 4, 1, true},
		{"wider than the vector", 2, 3, true},
		{"before the vector", -1, 1, true},
		{"a negative width", 0, -1, true},
		{"no vector", 0, 1, false},
	} {
		g, f := build(c.off, c.width, c.bound)
		if err := Verify(g, f); err == nil || !strings.Contains(err.Error(), "a window of") {
			t.Errorf("%s: Verify said %v, want an error naming the window", c.name, err)
		}
	}
}

// TestVerifierHoldsAnEmbeddingsRange verifies an embedding of a range of
// its table's rows, from row 0 and from a later row, and refuses a range
// past the table and a first row on an op that is not an embedding.
func TestVerifierHoldsAnEmbeddingsRange(t *testing.T) {
	inv := map[jlm.Role]Shape{jlm.RoleNGramTable: {K: 4, Rows: 9}}
	embed := func(from, rows int) error {
		b := NewBlock(0)
		ix := b.In("ix", 1)
		b.G.Values[ix].Elem = ElemU32
		b.Out(b.Emit(OpEmbed, []ValueID{ix}, []Result{{Name: "e", Width: 4}},
			[]Binding{{Role: jlm.RoleNGramTable, K: 4, Rows: rows, From: from}}, Attrs{}, nil, nil)[0])
		return Verify(b.G, Facts{Inventory: inv})
	}
	for _, c := range []struct{ from, rows int }{{0, 9}, {0, 4}, {3, 4}, {5, 4}} {
		if err := embed(c.from, c.rows); err != nil {
			t.Errorf("rows %d to %d of a table of 9 are refused: %v", c.from, c.from+c.rows, err)
		}
	}
	for _, c := range []struct{ from, rows int }{{6, 4}, {0, 10}, {-1, 4}} {
		if err := embed(c.from, c.rows); err == nil {
			t.Errorf("rows %d to %d of a table of 9 verify", c.from, c.from+c.rows)
		}
	}
	b := NewBlock(0)
	x := b.In("x", 4)
	b.Out(b.Emit(OpMatVec, []ValueID{x}, []Result{{Name: "y", Width: 4}},
		[]Binding{{Role: jlm.RoleNGramTable, K: 4, Rows: 4, From: 3}}, Attrs{}, nil, nil)[0])
	if err := Verify(b.G, Facts{Inventory: inv}); err == nil || !strings.Contains(err.Error(), "only an embedding") {
		t.Errorf("a matrix-vector product from row 3: Verify said %v", err)
	}
}

// TestVerifierHoldsASelection verifies an attention over a selection of its
// own history's keys, then each violation of a selection's structure.
func TestVerifierHoldsASelection(t *testing.T) {
	build := func(a Attrs, cut func(g *Graph)) error {
		b := NewBlock(0)
		k := b.In("k", 48)
		h := StateRef{State: StateKV, Version: b.KVAppend(k, k, 0)}
		sel := b.Select(b.In("iq", 32), h, a, b.In("iw", 2))
		b.Out(b.AttendSelected(b.In("q", 64), h, sel, Attrs{NHead: 4, NKVHead: 2, HeadDim: 16, Width: 48}))
		if cut != nil {
			cut(b.G)
		}
		return Verify(b.G, Facts{})
	}
	good := Attrs{NHead: 2, NKVHead: 2, HeadDim: 16, Offset: 32, Width: 48, Span: 4, TopK: 2, Local: 1}
	if err := build(good, nil); err != nil {
		t.Fatalf("a correct selection is refused: %v", err)
	}
	for _, c := range []struct {
		name, want string
		a          func(a *Attrs)
		cut        func(g *Graph)
	}{
		{"a key past the row", "a key of", func(a *Attrs) { a.Offset = 40 }, nil},
		{"groups that do not divide the heads", "groups", func(a *Attrs) { a.NKVHead = 3 }, nil},
		{"no block", "blocks of", func(a *Attrs) { a.Span = 0 }, nil},
		{"weights of another count", "weights", func(a *Attrs) {}, func(g *Graph) { g.Values[g.Ops[1].In[1]].Width = 3 }},
		{"a selection over another history", "its own history", func(a *Attrs) {},
			func(g *Graph) { g.Ops[len(g.Ops)-1].Reads[0] = SharedHistory(0) }},
	} {
		a := good
		c.a(&a)
		err := build(a, c.cut)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: Verify said %v, want an error naming %q", c.name, err, c.want)
		}
	}
}

// TestVerifierHoldsAGridsRows verifies a regroup and a row drop that state
// their output's rows, over an input of the step's rows and of its own, and
// refuses one that states none over the step's rows and one whose rows are
// not its input's arithmetic -- a drop stating none takes its input's.
func TestVerifierHoldsAGridsRows(t *testing.T) {
	build := func(inRows, groupRows, dropRows int) error {
		b := NewBlock(-1)
		x := b.In("x", 4)
		b.G.Values[x].Rows = inRows
		g := b.Emit(OpRegroup, []ValueID{x}, []Result{{Name: "g", Width: 16, Rows: groupRows}}, nil, Attrs{Span: 2}, nil, nil)[0]
		b.Out(b.Emit(OpDropRows, []ValueID{g}, []Result{{Name: "d", Width: 16, Rows: dropRows}}, nil, Attrs{Offset: 1}, nil, nil)[0])
		return Verify(b.G, Facts{})
	}
	for _, c := range []struct{ in, group, drop int }{{0, 16, 15}, {64, 16, 15}, {0, 9, 8}} {
		if err := build(c.in, c.group, c.drop); err != nil {
			t.Errorf("%+v: a regroup and a drop stating their rows are refused: %v", c, err)
		}
	}
	for _, c := range []struct {
		in, group, drop int
		want            string
	}{{0, 0, 15, "states no rows"}, {0, 16, 0, "writes 16, not 15"}, {64, 15, 14, "writes 15, not 16"}, {64, 16, 16, "writes 16, not 15"}} {
		if err := build(c.in, c.group, c.drop); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%+v: Verify said %v, want an error naming %q", c, err, c.want)
		}
	}
}
