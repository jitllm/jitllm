package model

import (
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// mixGraph is a routed mixture block's sublayer graph (graph.Mixture): built,
// verified and matched at Open for every block whose FFN is a mixture the op
// language states, so a device asks the rewrites matched on it
// (nn.LayerPlan.Rewrites) at its own Target: the few-row expert form and the
// prompt chunk's expert waves. It has no host lowering: the host runs the
// hand-written mixture (moe.go), which computes the graph's ops in its order.
type mixGraph struct {
	g       *graph.Graph
	offered []graph.Offered
}

// mixtureFacts is what the container states about block li's mixture, or
// false where the block has none the op language states: a dense block or a
// dense lead; DeepSeek V4's per-row selection and hash blocks; Kimi-K3's
// latent mixture; Gemma 4's router norm and dense FFN beside the mixture;
// experts stored one matrix each rather than as banks; ungated experts; and
// an activation the graph has no name for. Such a block's device decides
// from the tier's own facts, as before the graph.
func (m *Model) mixtureFacts(li int) (graph.Facts, bool) {
	f, why := m.mixtureWhy(li)
	return f, why == "" && f.Mixture != nil
}

// mixtureWhy is mixtureFacts with the structural reason a mixture block's
// facts are not stated ("" where they are, and for a block with no mixture).
func (m *Model) mixtureWhy(li int) (graph.Facts, string) {
	c := m.Cfg
	l := &m.layers[li]
	switch {
	case !c.MoEAt(li):
		return graph.Facts{}, ""
	}
	act := graphAct(c.Act)
	if act == 0 {
		return graph.Facts{}, UncoveredUngated
	}
	want := map[jlm.Role]bool{}
	for _, r := range graph.MixtureRoles {
		want[r] = true
	}
	inv := map[jlm.Role]graph.Shape{}
	for i := range m.container.Entries() {
		e := &m.container.Entries()[i]
		if int(e.Block) != li {
			continue
		}
		switch e.Role {
		case jlm.RoleRouterNorm:
			// Gemma 4's router norm is its block's (Dense.DenseMix).
			if !(c.DenseMoE && e.Role == jlm.RoleRouterNorm) {
				return graph.Facts{}, UncoveredMixRouter
			}
		case jlm.RoleExpGate, jlm.RoleExpUp, jlm.RoleExpDown:
			return graph.Facts{}, UncoveredMixExperts
		}
		if !want[e.Role] {
			continue
		}
		s := graph.Shape{K: int(e.Dims[0])}
		if e.NDim >= 2 {
			s.Rows = int(e.Dims[1])
		}
		if e.NDim >= 3 {
			s.Sheets = int(e.Dims[2])
		}
		inv[e.Role] = s
	}
	for _, r := range []jlm.Role{jlm.RoleRouter, jlm.RoleExpGateBank, jlm.RoleExpUpBank, jlm.RoleExpDownBank} {
		if _, ok := inv[r]; !ok && !(l.ungatedExp && r == jlm.RoleExpGateBank) {
			return graph.Facts{}, UncoveredMixExperts
		}
	}
	mx := graph.Mixture{Block: li, NEmbd: c.NEmbd, NExpert: c.NExpert, K: c.NExpertUsed, NFF: c.NFFNExp, Act: act, GLU: c.GLU,
		Router: graph.RouterSoftmax, Norm: !c.NoExpertNorm, Scale: c.ExpertScale,
		Groups: c.NExpertGroup, GroupsUsed: c.NExpertGroupUsed, SharedFF: c.NFFNShExp, WeightIn: c.ExpertWeightIn,
		Ungated: l.ungatedExp, Latent: c.ExpertLatent, Eps: c.RMSEps, Inv: inv}
	switch {
	case c.SparseMixer != 0:
		mx.Router = graph.RouterSparseMixer
	case c.ExpertSigmoid:
		mx.Router = graph.RouterSigmoid
	case c.ExpertSqrtSoftplus:
		mx.Router = graph.RouterSqrtSoftplus
	}
	// Gemma 4's dense MLP is the shared expert's tensors, its width theirs.
	if c.DenseMoE {
		mx.SharedFF = inv[jlm.RoleShExpGate].Rows
		if mx.SharedFF == 0 {
			return graph.Facts{}, UncoveredMixExperts
		}
	}
	if mx.SharedFF > 0 {
		for _, r := range []jlm.Role{jlm.RoleShExpGate, jlm.RoleShExpUp, jlm.RoleShExpDown} {
			if _, ok := inv[r]; !ok && !(mx.Ungated && r == jlm.RoleShExpGate) {
				return graph.Facts{}, UncoveredMixExperts
			}
		}
		_, mx.SharedGated = inv[jlm.RoleShRouter]
	}
	return graph.Facts{Mixture: &mx, Inventory: inv}, ""
}

// buildMixGraphs builds, verifies and matches every mixture block's sublayer
// graph. A graph that does not verify is not offered: the block keeps the
// tier's own decisions, and the reason is kept (MixGraphErrors).
func (m *Model) buildMixGraphs() {
	m.mixGraphs = make([]*mixGraph, m.Cfg.NLayer)
	m.mixGraphErr = nil
	for li := range m.mixGraphs {
		f, ok := m.mixtureFacts(li)
		// Gemma 4's mixture is its block's FFN (Dense.DenseMix), not a
		// mixture sublayer with a shared expert added: the tier keeps its
		// own decisions for it.
		if !ok || m.Cfg.DenseMoE {
			continue
		}
		g := f.Mixture.Build()
		if err := graph.Verify(g, f); err != nil {
			m.mixGraphErr = append(m.mixGraphErr, err)
			continue
		}
		m.mixGraphs[li] = &mixGraph{g: g, offered: graph.Match(g, graph.Rewrites...)}
	}
}

// MixGraph is block li's mixture sublayer graph, or nil where the block has
// none or its graph did not verify (MixGraphErrors).
func (m *Model) MixGraph(li int) *graph.Graph {
	if li < 0 || li >= len(m.mixGraphs) || m.mixGraphs[li] == nil {
		return nil
	}
	return m.mixGraphs[li].g
}

// MixGraphErrors are the verifier's refusals of the mixture graphs.
func (m *Model) MixGraphErrors() []error { return m.mixGraphErr }
