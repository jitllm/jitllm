package model

import "github.com/jitllm/jitllm/engine/nn"

// Qwen3.8-Flash-Next on a device: the plan and weights a block carries
// (nn.Q4EPlan, nn.HCWeights, nn.NGramWeights).

// q4ePlan fills block li's Qwen3.8-Flash-Next fields of plan.
func (s *Instance) q4ePlan(li int, plan *nn.LayerPlan) {
	c := s.m.Cfg
	if !c.Q4E() {
		return
	}
	q := &nn.Q4EPlan{HCMult: c.HCMult, HCLowRank: c.HCLowRank}
	if c.QSAAt(li) {
		q.QSARatio = c.QSARatio
		if c.NRot > 0 {
			q.QSARope = s.m.rope.TabPlanes(c.NRot / 2)
		}
		plan.IdxHeads, plan.IdxHeadDim, plan.IdxTopK = c.IdxHeads, c.IdxHeadDim, c.IdxTopK
	} else {
		plan.IdxHeads, plan.IdxHeadDim, plan.IdxTopK = 0, 0, 0
	}
	if c.NGramBlock != 0 {
		q.NGramWidth, q.NGramConv, q.NGramDil = c.NGramWidth(), c.NGramConv, c.NGram
		q.NGramHere = c.NGramAt(li)
	}
	if c.LayerKind(li).Recurrent() {
		plan.Recurrent.SigmoidGate = c.q4eSigmoidGate()
		plan.Recurrent.TailLen = c.recTailLen()
	}
	plan.Q4E = q
}

// q4eWeights adds block li's Qwen3.8-Flash-Next tensors to w.
func (s *Instance) q4eWeights(li int, w *nn.LayerWeights) {
	l := &s.m.layers[li]
	d := l.q4e
	if d == nil {
		return
	}
	hc := func(h *q4eHC) nn.HCWeights {
		return nn.HCWeights{Down: wt(h.down), Up: wt(h.up), Inject: wt(h.inj)}
	}
	w.HCAttn, w.HCFFN = hc(&d.attn), hc(&d.ffn)
	if s.m.Cfg.QSAAt(li) {
		w.IdxQ, w.IdxK = wt(l.idxQ), wt(l.idxK)
	}
	if s.m.Cfg.NGramAt(li) {
		w.NGram = nn.NGramWeights{Key: wt(d.ngKey), Value: wt(d.ngVal), Conv: d.ngCv}
	}
}
