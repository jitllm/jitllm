package gpu

import (
	"math"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// This helper is untagged so the package compiles on darwin: the untagged
// actdecline_test.go uses it, and keeping it in the linux-only
// visionmem_test.go broke the whole package's build on arm64.

// visionPlan is a ViT block: LayerNorm with biases, biased q/k/v/o, an ungated
// MLP with biases, non-causal attention over the call's rows and no rotary.
// NRot is zero, so initScratch emits no RoPE kernel and the copy path carries
// k into the cache.
func visionPlan() *nn.LayerPlan {
	p := &nn.LayerPlan{
		NonCausal: true, LayerNorm: true, UngatedFFN: true,
		NEmbd: 256, NHead: 4, NKVHead: 4, HeadDim: 64, NRot: 0, NFFN: 1024,
		MaxSeq: 64, Act: nn.ActGELU, RMSEps: 1e-5,
	}
	p.Graph = visionBlockGraph(p)
	return p
}

// visionBlockGraph is p's block as the front end states a ViT's: biased
// LayerNorms and projections, attention both ways over the call's rows, and
// an ungated GELU MLP with its biases. The roles are visionBlocks'.
func visionBlockGraph(p *nn.LayerPlan) *graph.Graph {
	b := graph.NewBlock(0)
	qd := p.NHead * p.HeadDim
	mapc := func(name string, f graph.MapFn, k float64, in ...graph.ValueID) graph.ValueID {
		return b.Emit(graph.OpMap, in, []graph.Result{{Name: name, Width: p.NFFN}}, nil, graph.Attrs{Fn: f, Const: k}, nil, nil)[0]
	}
	x := b.In("x", p.NEmbd)
	n := b.LayerNorm(x, jlm.RoleAttnNorm, p.RMSEps, jlm.RoleAttnNormBias)
	q := b.MatVec(n, jlm.RoleAttnQ, qd, jlm.RoleAttnQBias)
	k := b.MatVec(n, jlm.RoleAttnK, qd, jlm.RoleAttnKBias)
	v := b.MatVec(n, jlm.RoleAttnV, qd, jlm.RoleAttnVBias)
	hist := graph.StateRef{State: graph.StateKV, Version: b.KVAppend(k, v, 0)}
	att := b.AttendOf(q, hist, graph.Attrs{NHead: p.NHead, NKVHead: p.NKVHead, HeadDim: p.HeadDim,
		Scale: 1 / math.Sqrt(float64(p.HeadDim)), Bidir: true})
	x = b.Add(x, b.MatVec(att, jlm.RoleAttnOut, p.NEmbd, jlm.RoleAttnOutBias))
	n = b.LayerNorm(x, jlm.RoleFFNNorm, p.RMSEps, jlm.RoleFFNNormBias)
	up := b.MatVec(n, jlm.RoleFFNUp, p.NFFN, jlm.RoleFFNUpBias)
	// GELU, exactly: x/2 (1 + erf(x/sqrt 2)).
	erf := mapc("act.erf", graph.MapErf, 0, mapc("act.arg", graph.MapMul, 1/math.Sqrt2, up))
	h := mapc("act", graph.MapMul, 0.5, mapc("act.x", graph.MapMul, 0, up, mapc("act.one", graph.MapAdd, 1, erf)))
	b.Out(b.Add(x, b.MatVec(h, jlm.RoleFFNDown, p.NEmbd, jlm.RoleFFNDownBias)))
	return b.G
}
