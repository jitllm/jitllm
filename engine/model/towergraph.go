package model

import (
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// A vision tower's block is a dense block, non-causal, stated in the same op
// language as a decoder's so the device compiles it as it compiles one. Its
// roles are what each tensor does in the block, not the container's names: a
// tower's MLP matrices are told apart by shape (SmolVLM's mmproj names its
// expansion ffn_down), so the role a tensor plays is read off the loaded
// block. A block the dense language does not state (a convolution block, a
// gate bias) has no graph, and the device names that as its reason.

// towerRoles is block l's tensors by the role each plays: every matrix and
// vector it holds, with the shape a graph binding checks.
func towerRoles(l *layer) map[jlm.Role]graph.Shape {
	inv := map[jlm.Role]graph.Shape{}
	for _, m := range [...]struct {
		r jlm.Role
		t *tensor
	}{{jlm.RoleAttnQ, &l.wq}, {jlm.RoleAttnK, &l.wk}, {jlm.RoleAttnV, &l.wv}, {jlm.RoleAttnOut, &l.wo},
		{jlm.RoleFFNGate, &l.gate}, {jlm.RoleFFNUp, &l.up}, {jlm.RoleFFNDown, &l.down}} {
		if m.t.e != nil {
			inv[m.r] = graph.Shape{K: m.t.k, Rows: m.t.rows}
		}
	}
	for _, v := range towerVectors(l) {
		if len(v.V) != 0 {
			inv[v.Role] = graph.Shape{K: len(v.V)}
		}
	}
	if l.clamp != nil {
		inv[jlm.RoleVClamp] = graph.Shape{K: len(l.clamp)}
	}
	return inv
}

// towerVectors is block l's vectors by the role each plays.
func towerVectors(l *layer) [14]nn.Vector {
	return [...]nn.Vector{
		{Role: jlm.RoleAttnNorm, V: l.attnNorm}, {Role: jlm.RoleAttnNormBias, V: l.attnNormB},
		{Role: jlm.RoleFFNNorm, V: l.ffnNorm}, {Role: jlm.RoleFFNNormBias, V: l.ffnNormB},
		{Role: jlm.RoleAttnQBias, V: l.bq}, {Role: jlm.RoleAttnKBias, V: l.bk},
		{Role: jlm.RoleAttnVBias, V: l.bv}, {Role: jlm.RoleAttnOutBias, V: l.bo},
		{Role: jlm.RoleFFNUpBias, V: l.upB}, {Role: jlm.RoleFFNDownBias, V: l.downB},
		{Role: jlm.RoleAttnQNorm, V: l.qNorm}, {Role: jlm.RoleAttnKNorm, V: l.kNorm},
		{Role: jlm.RolePostAttnNorm, V: l.postAttnNorm}, {Role: jlm.RolePostFFNNorm, V: l.postFFNNorm},
	}
}

// towerGraph is vision block i's verified graph (block li of the model's
// block space), or nil where the dense language does not state the block.
func towerGraph(sc *Config, l *layer, li int) *graph.Graph {
	// A windowed block's mask is the picture's (SetKeyRuns), not the
	// block's: its graph is the dense block's, and the attention reads the
	// windows the picture staged.
	if l.mn != nil || l.gateB != nil {
		return nil
	}
	d := graph.Dense{Block: li, NEmbd: sc.NEmbd, NHead: sc.NHead, NKVHead: sc.NKVHead, HeadDim: sc.HeadDim,
		NFFN: sc.NFFN, Eps: sc.RMSEps, Scale: sc.AttnScale, Act: graphAct(sc.Act),
		LayerNorm: sc.LayerNorm, NormBias: l.attnNormB != nil, FFNNormBias: l.ffnNormB != nil,
		QKVBias: l.bq != nil, OutBias: l.bo != nil, UpBias: l.upB != nil, DownBias: l.downB != nil,
		Ungated: l.gate.e == nil, NonCausal: true, NoRope: sc.NRot == 0,
		QKNorm: l.qNorm != nil, VNorm: sc.VNorm, PostNorm: l.postAttnNorm != nil}
	if l.clamp != nil {
		d.ClampTable = jlm.RoleVClamp // the clipped linears' bounds
	}
	if sc.NRot != 0 {
		d.NRot, d.Layout = sc.NRot, graph.RopeNorm
		if sc.RopeNeox {
			d.Layout = graph.RopeNeox
		}
	}
	g := d.Build()
	f := graph.Facts{Config: d, Inventory: towerRoles(l)}
	if len(graph.Unbound(g, f)) > 0 || graph.Verify(g, f) != nil {
		return nil
	}
	return g
}
