package graph

import "github.com/jitllm/jitllm/format/jlm"

// Before is g with ops ahead of its own: build reads g's input row x and
// returns the row g's ops then read in its place.
func (g *Graph) Before(build func(d *Graph, x ValueID) ValueID) *Graph {
	d := &Graph{Version: g.Version, Block: g.Block, Values: append([]Value(nil), g.Values...),
		In: append([]ValueID(nil), g.In...), Out: append([]ValueID(nil), g.Out...)}
	x := g.In[0]
	y := build(d, x)
	for _, o := range g.Ops {
		o.In = append([]ValueID(nil), o.In...)
		for i, v := range o.In {
			if v == x {
				o.In[i] = y
			}
		}
		d.op(o)
	}
	for i, v := range d.Out {
		if v == x {
			d.Out[i] = y
		}
	}
	return d
}

// Scorer is a decision head's readout (Laya's): the rows at the request's
// markers, each LayerNormed against RoleScorerNorm, through RoleScorer with
// its bias, GELU in erf form, and RoleScorerOut with its bias to one value.
type Scorer struct {
	NEmbd int
	Eps   float64
}

func (s *Scorer) Build() *Graph {
	g := &Graph{Block: -1}
	d := s.NEmbd
	x := g.value("x", d)
	g.In = []ValueID{x}
	m := g.row("markers", d, Op{Kind: OpTakeRows, In: []ValueID{x}})
	n := g.row("sc_normed", d, Op{Kind: OpLayerNorm, In: []ValueID{m},
		Weight: []Binding{{Role: jlm.RoleScorerNorm, K: d}, {Role: jlm.RoleScorerNormBias, K: d}}, Attr: Attrs{Eps: s.Eps}})
	h := g.row("sc_h", d, Op{Kind: OpMatVec, In: []ValueID{n},
		Weight: []Binding{{Role: jlm.RoleScorer, K: d, Rows: d}, {Role: jlm.RoleScorerBias, K: d}}})
	a := g.row("sc_act", d, Op{Kind: OpAct, In: []ValueID{h}, Attr: Attrs{Act: ActGELUErf}})
	g.Out = []ValueID{g.row("score", 1, Op{Kind: OpMatVec, In: []ValueID{a},
		Weight: []Binding{{Role: jlm.RoleScorerOut, K: d, Rows: 1}, {Role: jlm.RoleScorerOutBias, K: 1}}})}
	return g
}

// DecisionPrologue is a decision head's first block's prologue (Laya's):
// the encoder's last row LayerNormed against RoleOutputNorm, no bias, then
// the request's question type's row of RoleTokenTypes added.
func DecisionPrologue(nEmbd, types int, eps float64) func(d *Graph, x ValueID) ValueID {
	return func(g *Graph, x ValueID) ValueID {
		n := g.row("final_norm", nEmbd, Op{Kind: OpLayerNorm, In: []ValueID{x},
			Weight: []Binding{{Role: jlm.RoleOutputNorm, K: nEmbd}}, Attr: Attrs{Eps: eps}})
		t := g.row("type_row", nEmbd, Op{Kind: OpEmbed, Weight: []Binding{{Role: jlm.RoleTokenTypes, K: nEmbd, Rows: types}},
			Attr: Attrs{Request: true}})
		return g.row("x_typed", nEmbd, Op{Kind: OpAdd, In: []ValueID{n, t}})
	}
}

// Input is a new input row of g, n wide.
func (g *Graph) Input(name string, n int) ValueID {
	v := g.value(name, n)
	g.In = append(g.In, v)
	return v
}
