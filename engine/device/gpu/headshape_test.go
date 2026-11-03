package gpu

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestHeadShapeReadsTheHeadsGraph holds the tier's reading of a head graph to
// graph.Head: every norm (or none), bias, softcap, scale and hidden output it
// states is read back as the roles it binds, and a graph that is not a norm
// or none, a projection and a chain of elementwise ops on its logits is
// refused by name. The reading runs every decode token and allocates nothing.
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
						want := headSpec{out: jlm.RoleTokenEmbd, layerNorm: n.layer, hasNormBias: n.bias, hasBias: bias, noNorm: n.none}
						if !n.none {
							want.norm = jlm.RoleOutputNorm
						}
						if n.bias {
							want.normBias = jlm.RoleOutputNormBias
						}
						if bias {
							want.bias = jlm.RoleOutputBias
						}
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
	for _, c := range []struct {
		name string
		make func() *graph.Graph
	}{
		{"no graph", func() *graph.Graph { return nil }},
		{"a head with no norm whose hidden rows are not its rows' copy", func() *graph.Graph {
			h := graph.Head{NEmbd: 8, NVocab: 24, Output: jlm.RoleOutput, NoNorm: true, Hidden: true}
			g := h.Build()
			g.Out[1] = g.Ops[0].Out[0]
			return g
		}},
		{"a chain cut by an op reading an earlier value", func() *graph.Graph {
			g := full.Build()
			g.Ops[len(g.Ops)-1].In[0] = g.Ops[1].Out[0]
			return g
		}},
		{"an output that is not the chain's end", func() *graph.Graph {
			g := full.Build()
			g.Out[0] = g.Ops[1].Out[0]
			return g
		}},
		{"a norm that is not the first op", func() *graph.Graph {
			g := full.Build()
			g.Ops[0], g.Ops[1] = g.Ops[1], g.Ops[0]
			return g
		}},
		{"a weighted op in the chain", func() *graph.Graph {
			g := full.Build()
			g.Ops[2].Weight = append(g.Ops[2].Weight, graph.Binding{Role: jlm.RoleOutputBias, K: 24})
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
