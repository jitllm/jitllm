package gpu

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// vecsOf is every vector hs binds, in its order, and the ones read beside
// the projection.
func vecsOf(hs headSpec) (vs []graph.Binding, beside []jlm.Role) {
	hs.eachVec(func(b graph.Binding, f32 bool) {
		vs = append(vs, b)
		if f32 {
			beside = append(beside, b.Role)
		}
	})
	return vs, beside
}

// headWant is the vectors a head of 8 rows over 24 ids binds: its norm, the
// norm's bias and the projection's bias where it has them.
func headWant(norm, normBias, bias bool) (vs []graph.Binding, beside []jlm.Role) {
	if norm {
		vs = append(vs, graph.Binding{Role: jlm.RoleOutputNorm, K: 8})
	}
	if normBias {
		vs = append(vs, graph.Binding{Role: jlm.RoleOutputNormBias, K: 8})
	}
	if bias {
		vs = append(vs, graph.Binding{Role: jlm.RoleOutputBias, K: 24})
		beside = []jlm.Role{jlm.RoleOutputBias}
	}
	return vs, beside
}

// TestHeadShapeReadsTheHeadsGraph holds the tier's reading of a head graph to
// what it binds: every norm (or none), bias, softcap, scale and hidden output
// graph.Head states is read back as its projection and the vectors it names,
// by role, and so is a head whose ops are in another order. A graph binding a
// second matrix, or whose output is its input, is refused, and a head whose
// graph names a vector the tier did not upload is not served. Reading the
// graph and asking whether the head serves run every decode token and
// allocate nothing.
func TestHeadShapeReadsTheHeadsGraph(t *testing.T) {
	type norm struct{ none, layer, bias bool }
	for _, n := range []norm{{none: true}, {}, {layer: true}, {layer: true, bias: true}} {
		for _, bias := range []bool{false, true} {
			for _, softcap := range []float64{0, 30} {
				for _, scale := range []float64{0, 8} {
					for _, hidden := range []bool{false, true} {
						h := graph.Head{NEmbd: 8, NVocab: 24, Eps: 1e-5, LayerNorm: n.layer, NormBias: n.bias,
							Output: jlm.RoleTokenEmbd, Bias: bias, Softcap: softcap, Scale: scale, Hidden: hidden, NoNorm: n.none}
						hs, ok := headShape(h.Build(), nil)
						vs, beside := vecsOf(hs)
						wv, wb := headWant(!n.none, n.bias, bias)
						if !ok || hs.out != jlm.RoleTokenEmbd || !slices.Equal(vs, wv) || !slices.Equal(beside, wb) {
							t.Errorf("%+v: read as %v %v %v (%v), want %v %v", h, hs.out, vs, beside, ok, wv, wb)
						}
					}
				}
			}
		}
	}
	full := graph.Head{NEmbd: 8, NVocab: 24, Eps: 1e-5, LayerNorm: true, NormBias: true, Output: jlm.RoleOutput,
		Bias: true, Softcap: 30, Scale: 8}
	reordered := full.Build()
	reordered.Ops[len(reordered.Ops)-1].In[0] = reordered.Ops[1].Out[0]
	hs, ok := headShape(reordered, nil)
	vs, beside := vecsOf(hs)
	wv, wb := headWant(true, true, true)
	if !ok || hs.out != jlm.RoleOutput || !slices.Equal(vs, wv) || !slices.Equal(beside, wb) {
		t.Errorf("a head whose last op reads the projection: read as %v %v %v (%v)", hs.out, vs, beside, ok)
	}
	for _, c := range []struct {
		name string
		make func() *graph.Graph
	}{
		{"no graph", func() *graph.Graph { return nil }},
		{"an output that is the input", func() *graph.Graph {
			h := graph.Head{NEmbd: 8, NVocab: 24, Output: jlm.RoleOutput, NoNorm: true, Hidden: true}
			g := h.Build()
			g.Out[1] = g.In[0]
			return g
		}},
		{"a second matrix", func() *graph.Graph {
			g := full.Build()
			g.Ops[2].Weight = append(g.Ops[2].Weight, graph.Binding{Role: jlm.RoleTokenEmbd, K: 8, Rows: 24})
			return g
		}},
	} {
		if _, ok := headShape(c.make(), nil); ok {
			t.Errorf("%s: read as a head the tier serves", c.name)
		}
	}
	g := full.Build()
	if n := testing.AllocsPerRun(100, func() { headShape(g, nil) }); n != 0 {
		t.Errorf("reading a head's graph allocates %.0f times", n)
	}
	hidden := full
	hidden.Hidden = true
	f := headForm{graph: g, hidden: hidden.Build()}
	var vecs []roleBuf
	for _, r := range []jlm.Role{jlm.RoleOutputNorm, jlm.RoleOutputNormBias, jlm.RoleOutputBias} {
		vecs = append(vecs, roleBuf{role: r, buf: &fakeBuf{}})
	}
	if err := f.serves(vecs); err != nil {
		t.Errorf("a head holding every vector its graph names: %v", err)
	}
	if n := testing.AllocsPerRun(100, func() { f.serves(vecs) }); n != 0 {
		t.Errorf("asking whether the head serves allocates %.0f times", n)
	}
	other := full.Build()
	other.Ops[2].Weight = append(other.Ops[2].Weight, graph.Binding{Role: jlm.RoleAttnNorm, K: 24})
	otherHidden := hidden.Build()
	otherHidden.Ops[2].Weight = append(otherHidden.Ops[2].Weight, graph.Binding{Role: jlm.RoleAttnNorm, K: 24})
	if err := (headForm{graph: other, hidden: otherHidden}).serves(vecs); err == nil {
		t.Error("a head whose graph names a vector the tier did not upload is served")
	}
}
