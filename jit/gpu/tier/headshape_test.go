package tier

import (
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestHeadShapeReadsTheHeadsGraph holds the tier's reading of a head graph to
// graph.Head: every norm, bias, softcap and scale it states is read back as
// the roles it binds, and a graph that is not a norm, a projection and a
// chain of elementwise ops on its logits is refused by name. The reading runs
// every decode token and allocates nothing.
func TestHeadShapeReadsTheHeadsGraph(t *testing.T) {
	for _, layerNorm := range []bool{false, true} {
		for _, normBias := range []bool{false, true} {
			for _, bias := range []bool{false, true} {
				for _, softcap := range []float64{0, 30} {
					for _, scale := range []float64{0, 8} {
						if normBias && !layerNorm {
							continue
						}
						h := graph.Head{NEmbd: 8, NVocab: 24, Eps: 1e-5, LayerNorm: layerNorm, NormBias: normBias,
							Output: jlm.RoleTokenEmbd, Bias: bias, Softcap: softcap, Scale: scale}
						hs, ok := headShape(h.Build())
						want := headSpec{norm: jlm.RoleOutputNorm, out: jlm.RoleTokenEmbd, layerNorm: layerNorm, hasNormBias: normBias, hasBias: bias}
						if normBias {
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
		{"a lone projection", func() *graph.Graph { return graph.MatVec(jlm.RoleOutput, 24, 8) }},
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
