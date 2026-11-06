package engine

import (
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
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
		plan.Recurrent.SigmoidGate = c.DeltaSigmoidGate
		plan.Recurrent.TailLen = c.RecTailLen()
	}
	plan.Q4E = q
}

// q4eWeights adds the n-gram block's convolution to w.
func (s *Instance) q4eWeights(li int, l *model.Layer, w *nn.LayerWeights) {
	if d := l.Q4e; d != nil && s.m.Cfg.NGramAt(li) {
		w.NGram = nn.NGramWeights{Conv: d.NgCv}
	}
}
