package model

import (
	"math"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// An encoder's decision head (Laya's, d1-omni's) is blocks after the
// encoder's in the one block space: pre-norm blocks over every row, both
// ways, of the head count and MLP width the container states
// (jlm.Config.DecisionHeads, the stored up projection's rows), with no
// positions, biased projections and LayerNorms and an ungated ReLU MLP.

// encHeadBlocks is how many decision head blocks follow the encoder's.
func (m *Model) encHeadBlocks() int {
	if m.enc == nil || m.enc.mb == nil {
		return 0
	}
	return len(m.enc.mb.head)
}

// headBlockGraph is decision head block li's graph (li in the model's block
// space, past the encoder's).
func (m *Model) headBlockGraph(li int) (*graph.Graph, Uncovered, error) {
	c, mb := m.Cfg, m.enc.mb
	f, _ := m.inventory(int32(li))
	has := func(r jlm.Role) bool {
		_, ok := f.Inventory[r]
		return ok
	}
	stored := func(r jlm.Role) []jlm.Role {
		if has(r) {
			return []jlm.Role{r}
		}
		return nil
	}
	d, heads := c.NEmbd, mb.headHeads
	hd, ff := d/heads, f.Inventory[jlm.RoleFFNUp].Rows
	b := graph.NewBlock(li)
	ln := func(v graph.ValueID, r, bias jlm.Role) graph.ValueID {
		return b.LayerNorm(v, r, c.RMSEps, stored(bias)...)
	}
	x := b.In("x", d)
	n := ln(x, jlm.RoleAttnNorm, jlm.RoleAttnNormBias)
	q := b.MatVec(n, jlm.RoleAttnQ, d, stored(jlm.RoleAttnQBias)...)
	k := b.MatVec(n, jlm.RoleAttnK, d, stored(jlm.RoleAttnKBias)...)
	v := b.MatVec(n, jlm.RoleAttnV, d, stored(jlm.RoleAttnVBias)...)
	hist := graph.StateRef{State: graph.StateKV, Version: b.KVAppend(k, v, 0)}
	att := b.AttendOf(q, hist, graph.Attrs{NHead: heads, NKVHead: heads, HeadDim: hd, Scale: 1 / math.Sqrt(float64(hd)), Bidir: true})
	x2 := b.Add(x, b.MatVec(att, jlm.RoleAttnOut, d, stored(jlm.RoleAttnOutBias)...))
	n2 := ln(x2, jlm.RoleFFNNorm, jlm.RoleFFNNormBias)
	up := b.MatVec(n2, jlm.RoleFFNUp, ff, stored(jlm.RoleFFNUpBias)...)
	relu := b.Expr("relu", ff, []graph.Term{{Fn: graph.MapMax, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0}, {Kind: graph.ArgConst, Const: 0}}}}, nil, up)
	b.Out(b.Add(x2, b.MatVec(relu, jlm.RoleFFNDown, d, stored(jlm.RoleFFNDownBias)...)))
	return verified(b, f)
}
