package graph

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
)

// Delta describes a gated delta rule's sublayer (qwen3next's, Kimi Delta
// Attention's, Qwen3.8-Flash-Next's): from its normed input row, the rule
// writes one output per value head, each head is RMS-normalised against one
// shared scale, gated elementwise by a projection of the same input, and
// projected back to the residual's width. The rule is one opaque op
// (OpGatedDelta); what follows it is ops the rewrites match.
//
// The graph is the sublayer's, not the block's: its input is the row the
// rule and the gate projection read (after whatever mixes or normalises the
// residual into it), its output the projection before the residual takes it.
// A block's other sublayers (its FFN or mixture, a hyper-connection's mixes)
// are not in it, and its Facts hold only the sublayer's tensors.
type Delta struct {
	Block         int
	NEmbd         int // the input row's width, and the output's
	VHeads, VDim  int
	Eps           float64
	Gate          Act // ActSiLU or ActSigmoid; 0 for no gate (the norm goes straight to the output projection)
	GateRank      int // the gate's two factors (RoleSSMGA, RoleSSMGB) at this rank; 0 for one matrix (RoleAttnGate)
	Rule          []Binding
	GateA, GateB  Shape // the gate's stored shapes: GateA alone where GateRank is 0
	NormW, OutRow Shape
}

// DeltaRuleRoles are the tensors a gated delta rule reads besides its gate,
// its output norm and its output projection: the q/k/v projection, beta's
// (and alpha's) logits, the convolution, the decay and its bias, and KDA's
// low-rank forget gate.
var DeltaRuleRoles = []jlm.Role{jlm.RoleAttnQKV, jlm.RoleSSMBA, jlm.RoleSSMConv1d, jlm.RoleSSMA,
	jlm.RoleSSMDtBias, jlm.RoleSSMFA, jlm.RoleSSMFB}

// DeltaRoles are every tensor the sublayer graph binds: the rule's, the
// gate's, the output norm's and the output projection's. A Facts inventory
// for a Delta holds the block's tensors of these roles.
var DeltaRoles = append(append([]jlm.Role(nil), DeltaRuleRoles...),
	jlm.RoleAttnGate, jlm.RoleSSMGA, jlm.RoleSSMGB, jlm.RoleSSMNorm, jlm.RoleSSMOut)

// Build is the sublayer's graph.
func (d *Delta) Build() *Graph {
	g := &Graph{Version: Version, Block: d.Block}
	inner := d.VHeads * d.VDim
	x := g.value("x", d.NEmbd)
	g.In = []ValueID{x}
	o := g.value("rule", inner)
	g.op(Op{Kind: OpGatedDelta, In: []ValueID{x}, Out: []ValueID{o}, Weight: d.Rule,
		Reads: []StateRef{{"rec", 0}}, Writes: []StateRef{{"rec", 1}}})
	var z ValueID
	if d.Gate != 0 {
		if d.GateRank > 0 {
			lr := g.value("gate_lr", d.GateRank)
			g.op(Op{Kind: OpMatVec, In: []ValueID{x}, Out: []ValueID{lr},
				Weight: []Binding{{Role: jlm.RoleSSMGA, K: d.GateA.K, Rows: d.GateA.Rows}}})
			z = g.value("z", inner)
			g.op(Op{Kind: OpMatVec, In: []ValueID{lr}, Out: []ValueID{z},
				Weight: []Binding{{Role: jlm.RoleSSMGB, K: d.GateB.K, Rows: d.GateB.Rows}}})
		} else {
			z = g.value("z", inner)
			g.op(Op{Kind: OpMatVec, In: []ValueID{x}, Out: []ValueID{z},
				Weight: []Binding{{Role: jlm.RoleAttnGate, K: d.GateA.K, Rows: d.GateA.Rows}}})
		}
	}
	on := g.value("rule_normed", inner)
	g.op(Op{Kind: OpHeadNorm, In: []ValueID{o}, Out: []ValueID{on},
		Weight: []Binding{{Role: jlm.RoleSSMNorm, K: d.NormW.K, Rows: d.NormW.Rows}},
		Attr:   Attrs{Eps: d.Eps, HeadDim: d.VDim}})
	y := on
	if d.Gate != 0 {
		y = g.value("gated", inner)
		g.op(Op{Kind: OpActMul, In: []ValueID{z, on}, Out: []ValueID{y}, Attr: Attrs{Act: d.Gate}})
	}
	out := g.value("out", d.NEmbd)
	g.op(Op{Kind: OpMatVec, In: []ValueID{y}, Out: []ValueID{out},
		Weight: []Binding{{Role: jlm.RoleSSMOut, K: d.OutRow.K, Rows: d.OutRow.Rows}}})
	g.Out = []ValueID{out}
	return g
}

// NormGate is a head norm whose normalised row only a gating act_mul reads,
// as its second operand: the gate is the norm's epilogue, one launch where
// two were, and the normalised row is never written or read
// (kernels.HeadNormGated on the device). Matched by those edges in any graph
// that has them; a gated delta rule's output norm is the first.
var NormGate = &Rewrite{
	Name: "normgate",
	Match: func(g *Graph) []Region {
		use, _ := users(g)
		var out []Region
		for i, o := range g.Ops {
			if o.Kind != OpHeadNorm || len(o.Out) != 1 {
				continue
			}
			for _, u := range use[o.Out[0]] {
				if a := g.Ops[u]; a.Kind == OpActMul && len(a.In) == 2 && a.In[1] == o.Out[0] {
					out = append(out, Region{Ops: []int{i, u}})
					break
				}
			}
		}
		return out
	},
	Legal: func(g *Graph, r Region) error {
		use, _ := users(g)
		if len(r.Ops) != 2 {
			return fmt.Errorf("the region has %d ops, not a head norm and its gate", len(r.Ops))
		}
		hn, am := g.Ops[r.Ops[0]], g.Ops[r.Ops[1]]
		switch {
		case hn.Kind != OpHeadNorm || am.Kind != OpActMul:
			return fmt.Errorf("the region is %s then %s, not head_norm then act_mul", hn.Kind, am.Kind)
		case am.In[1] != hn.Out[0]:
			return fmt.Errorf("the act_mul does not gate the normalised row")
		case am.In[0] == hn.Out[0]:
			return fmt.Errorf("the gate is the normalised row itself")
		case len(use[hn.Out[0]]) != 1:
			return fmt.Errorf("the normalised row has %d readers; the epilogue writes it nowhere", len(use[hn.Out[0]]))
		case am.Attr.Act != ActSiLU && am.Attr.Act != ActSigmoid:
			return fmt.Errorf("activation %d has no gated-norm epilogue", am.Attr.Act)
		}
		return nil
	},
	Contract: map[string]Contract{
		"ptx":   {Exact: true, Gate: "backend.TestHeadNormGatedIsTheTwoLaunches, model.TestNormGateMatchesTwoLaunches"},
		"spirv": {Exact: true, Gate: "backend.TestHeadNormGatedIsTheTwoLaunches, model.TestNormGateMatchesTwoLaunches"},
		"msl":   {Exact: true, Gate: "backend.TestHeadNormGatedIsTheTwoLaunches, model.TestNormGateMatchesTwoLaunches"},
	},
	// One launch fewer at every row class, and the normalised row's write and
	// read: four bytes each per element.
	Bounds: func(g *Graph, r Region) Resources {
		w := int64(g.Values[g.Ops[r.Ops[0]].Out[0]].Width)
		return Resources{Launches: 1, TrafficBytes: 8 * w}
	},
	Keys: func(g *Graph, r Region) string { return fmt.Sprintf("g%d", g.Ops[r.Ops[1]].Attr.Act) },
}
