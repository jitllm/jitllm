package model

import (
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// deltaGraph is a gated delta block's sublayer graph (graph.Delta): built,
// verified and matched at Open for every block whose mixer is a gated delta
// rule, whatever the rest of the block is, so a device asks the rewrites
// matched on it (nn.LayerPlan.Rewrites) at its own Target. It has no host
// lowering: the host runs the hand-written sublayer, which computes the
// graph's ops in the graph's order (delta.go's linearAttn).
type deltaGraph struct {
	g       *graph.Graph
	offered []graph.Offered
}

// deltaFacts is what the container states about block li's gated delta
// sublayer, or false where the block has none: a recurrent block that is
// neither Mamba's scan (SSD, Mamba-1) nor a short convolution, and does not
// attend beside it.
func (m *Model) deltaFacts(li int) (graph.Facts, bool) {
	c := m.Cfg
	if !c.LayerKind(li).Recurrent() || c.LayerKind(li).Attends() || c.ShortConv() {
		return graph.Facts{}, false
	}
	geo := c.delta()
	if geo.ssd || geo.mamba1 {
		return graph.Facts{}, false
	}
	inv := map[jlm.Role]graph.Shape{}
	want := map[jlm.Role]bool{}
	for _, r := range graph.DeltaRoles {
		want[r] = true
	}
	for i := range m.container.Entries() {
		e := &m.container.Entries()[i]
		if int(e.Block) != li || !want[e.Role] {
			continue
		}
		s := graph.Shape{K: int(e.Dims[0])}
		if e.NDim >= 2 {
			s.Rows = int(e.Dims[1])
		}
		inv[e.Role] = s
	}
	norm, ok1 := inv[jlm.RoleSSMNorm]
	out, ok2 := inv[jlm.RoleSSMOut]
	if !ok1 || !ok2 {
		return graph.Facts{}, false
	}
	d := graph.Delta{Block: li, NEmbd: c.NEmbd, VHeads: geo.vHeads, VDim: geo.vDim, Eps: c.RMSEps,
		NormW: norm, OutRow: out}
	d.Gate = graph.ActSiLU
	if geo.chanDecay || c.q4eSigmoidGate() {
		d.Gate = graph.ActSigmoid
	}
	if a, ok := inv[jlm.RoleSSMGA]; ok {
		d.GateA, d.GateB, d.GateRank = a, inv[jlm.RoleSSMGB], a.Rows
	} else if a, ok := inv[jlm.RoleAttnGate]; ok {
		d.GateA = a
	} else {
		d.Gate = 0
	}
	for _, r := range graph.DeltaRuleRoles {
		if s, ok := inv[r]; ok {
			d.Rule = append(d.Rule, graph.Binding{Role: r, K: s.K, Rows: s.Rows})
		}
	}
	return graph.Facts{Delta: &d, Inventory: inv}, true
}

// buildDeltaGraphs builds, verifies and matches every gated delta block's
// sublayer graph. A graph that does not verify is not offered: the block
// keeps the tier's reference recipes, and the reason is kept for Explain.
func (m *Model) buildDeltaGraphs() {
	m.deltaGraphs = make([]*deltaGraph, m.Cfg.NLayer)
	m.deltaGraphErr = nil
	for li := range m.deltaGraphs {
		f, ok := m.deltaFacts(li)
		if !ok {
			continue
		}
		g := f.Delta.Build()
		if err := graph.Verify(g, f); err != nil {
			m.deltaGraphErr = append(m.deltaGraphErr, err)
			continue
		}
		m.deltaGraphs[li] = &deltaGraph{g: g, offered: graph.Match(g, graph.Rewrites...)}
	}
}

// DeltaGraph is block li's gated delta sublayer graph, or nil where the block
// has none or its graph did not verify (DeltaGraphErrors).
func (m *Model) DeltaGraph(li int) *graph.Graph {
	if li < 0 || li >= len(m.deltaGraphs) || m.deltaGraphs[li] == nil {
		return nil
	}
	return m.deltaGraphs[li].g
}

// DeltaGraphErrors are the verifier's refusals of the sublayer graphs.
func (m *Model) DeltaGraphErrors() []error { return m.deltaGraphErr }
