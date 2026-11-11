package gpu

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestHeadShapeReadsTheHeadsGraph holds the tier's reading of a head graph to
// what it binds: every norm (or none), bias, softcap, scale and hidden output
// graph.Head states is read back as its projection and the vectors it names,
// and so is a head whose ops are in another order. A graph binding what the
// tier holds no head buffer for -- a second matrix, another vector -- or
// whose output is its input is refused. The reading runs every decode token
// and allocates nothing.
func TestHeadShapeReadsTheHeadsGraph(t *testing.T) {
	type norm struct{ none, layer, bias bool }
	for _, n := range []norm{{none: true}, {}, {layer: true}, {layer: true, bias: true}} {
		for _, bias := range []bool{false, true} {
			for _, softcap := range []float64{0, 30} {
				for _, scale := range []float64{0, 8} {
					for _, hidden := range []bool{false, true} {
						h := graph.Head{NEmbd: 8, NVocab: 24, Eps: 1e-5, LayerNorm: n.layer, NormBias: n.bias,
							Output: jlm.RoleTokenEmbd, Bias: bias, Softcap: softcap, Scale: scale, Hidden: hidden, NoNorm: n.none}
						hs, ok := headShape(h.Build())
						want := headSpec{out: jlm.RoleTokenEmbd, hasNorm: !n.none, hasNormBias: n.bias, hasBias: bias}
						if !ok || hs != want {
							t.Errorf("%+v: read as %+v (%v), want %+v", h, hs, ok, want)
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
	if hs, ok := headShape(reordered); !ok || hs != (headSpec{out: jlm.RoleOutput, hasNorm: true, hasNormBias: true, hasBias: true}) {
		t.Errorf("a head whose last op reads the projection: read as %+v (%v)", hs, ok)
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
		{"a vector the tier holds no head buffer for", func() *graph.Graph {
			g := full.Build()
			g.Ops[2].Weight = append(g.Ops[2].Weight, graph.Binding{Role: jlm.RoleAttnNorm, K: 24})
			return g
		}},
	} {
		if _, ok := headShape(c.make()); ok {
			t.Errorf("%s: read as a head the tier serves", c.name)
		}
	}
	g := full.Build()
	if n := testing.AllocsPerRun(100, func() { headShape(g) }); n != 0 {
		t.Errorf("reading a head's graph allocates %.0f times", n)
	}
}
