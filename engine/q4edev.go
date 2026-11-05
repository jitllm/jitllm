package engine

import (
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/model"
)

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
			q.QSARope = s.m.Rope.TabPlanes(c.NRot / 2)
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
		plan.Recurrent.SigmoidGate = c.Q4eSigmoidGate()
		plan.Recurrent.TailLen = c.RecTailLen()
	}
	plan.Q4E = q
}

// q4eWeights adds block li's Qwen3.8-Flash-Next tensors to w.
func (s *Instance) q4eWeights(li int, w *nn.LayerWeights) {
	l := &s.m.Layers[li]
	d := l.Q4e
	if d == nil {
		return
	}
	hc := func(h *model.Q4eHC) nn.HCWeights {
		return nn.HCWeights{Down: model.Wt(h.Down), Up: model.Wt(h.Up), Inject: model.Wt(h.Inj)}
	}
	w.HCAttn, w.HCFFN = hc(&d.Attn), hc(&d.Ffn)
	if s.m.Cfg.QSAAt(li) {
		w.IdxQ, w.IdxK = model.Wt(l.IdxQ), model.Wt(l.IdxK)
	}
	if s.m.Cfg.NGramAt(li) {
		w.NGram = nn.NGramWeights{Key: model.Wt(d.NgKey), Value: model.Wt(d.NgVal), Conv: d.NgCv}
	}
}
