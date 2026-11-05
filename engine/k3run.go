package engine

import (
	"github.com/jitllm/jitllm/nn"
)

// k3Plan adjusts block li's plan for Kimi-K3: the residual attention's mixes
// and checkpoint, the latent mixture's width, KDA's decay bound and the MLA
// latent norms' epsilon (Kimi-Linear's too).
func (s *Instance) k3Plan(li int, plan *nn.LayerPlan) {
	c := s.m.Cfg
	plan.LatentNormEps = c.LatentNormEps
	if c.LayerKind(li).Recurrent() {
		plan.Recurrent.DecayBound = c.KDALowerBound
	}
	if !c.ResAttn() && c.ExpertLatent == 0 {
		return
	}
	k := &nn.K3Plan{Streams: c.Streams()}
	if c.MoEAt(li) {
		k.Latent = c.ExpertLatent
	}
	if c.ResAttn() {
		k.BankAttn, k.BankFFN = c.ResBank(li, false), c.ResBank(li, true)
		k.Push, _ = c.ResCheckpoint(li)
	}
	plan.K3 = k
}
