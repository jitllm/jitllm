package model

import (
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// DeepSeek V4 on a device (ds4.go is the graph, jit_old/gpu/tier/ds4.go its
// device side): the plan the blocks are offered with, the weights, the token
// ids a hash block routes by, and the entries that travel with a block's rows.

// streamHead says the head reads a collapse of several residual streams,
// which the host takes (Gemma 3n's AltUp, DeepSeek V4's hyper-connections,
// Kimi-K3's residual attention): a device runs the blocks and hands every
// stream back, and never folds the head into them.
func (c *Config) streamHead() bool { return c.AltUp != 0 || c.DSV4() || c.ResAttn() || c.Q4E() }

// expandStreams fills every stream of n rows of x from its embedding in
// stream 0.
func (s *State) expandStreams(x []float32, n int) error {
	if s.m.Cfg.DSV4() {
		s.ds4Expand(x, n)
		return nil
	}
	if s.m.Cfg.ResAttn() {
		s.k3Expand(x, n)
		return nil
	}
	if s.m.Cfg.Q4E() {
		s.q4eExpand(x, n)
		return nil
	}
	return s.altExpand(x, n)
}

// collapseStreams writes the head's input into stream 0 of n rows of x.
func (s *State) collapseStreams(x []float32, n int) error {
	if s.m.Cfg.DSV4() {
		return s.ds4Collapse(x, n)
	}
	if s.m.Cfg.ResAttn() {
		return s.k3Collapse(x, n)
	}
	if s.m.Cfg.Q4E() {
		return s.q4eCollapse(x, n)
	}
	return s.altCollapse(x, n)
}

// rowStreams copies row i of a chunk x of n rows, stream-major, into one
// row's streams (s.rowx), which it returns; back copies them into the chunk.
func (s *State) rowStreams(x []float32, i, n int, back bool) []float32 {
	c := s.m.Cfg
	if len(s.rowx) < c.ResidW() {
		s.rowx = make([]float32, c.ResidW())
	}
	for k := 0; k < c.Streams(); k++ {
		one := c.stream(s.rowx, k, 1, 0)
		if back {
			copy(c.stream(x, k, n, i), one)
		} else {
			copy(one, c.stream(x, k, n, i))
		}
	}
	return s.rowx[:c.ResidW()]
}

// ds4PlanOf is the model's DeepSeek V4 geometry as a device takes it, nil on
// every other model.
func ds4PlanOf(c *Config) *nn.DS4Plan {
	if !c.DSV4() {
		return nil
	}
	return &nn.DS4Plan{
		HCMult: c.HCMult, HCIters: c.HCIters, HCEps: float32(c.HCEps),
		OGroups: c.OGroups, OLoraRank: c.OLoraRank,
		RateCSA: c.CompRateCSA, RateHCA: c.CompRateHCA,
		IdxHeads: c.IdxHeads, IdxHeadDim: c.IdxHeadDim, IdxTopK: c.IdxTopK,
		ExpertSqrtSoftplus: c.ExpertSqrtSoftplus, Window: c.SWAWindow, Fault: c.ds4Fault,
	}
}

// ds4Plan adjusts block li's plan for DeepSeek V4: its compression and its
// routing, the query latent, and no other model's indexer (the DS4Plan
// carries this one's).
func (s *State) ds4Plan(li int, plan *nn.LayerPlan) {
	c := s.m.Cfg
	if !c.DSV4() {
		return
	}
	plan.Comp = int(c.CompAt(li))
	plan.HashExperts = s.m.layers[li].ds4 != nil && s.m.layers[li].ds4.hash != nil
	plan.QLoraRank = c.QLoraRank
	plan.IdxHeads, plan.IdxHeadDim, plan.IdxTopK = 0, 0, 0
}

// ds4Weights adds block li's DeepSeek V4 tensors to w.
func (s *State) ds4Weights(li int, w *nn.LayerWeights) {
	l := &s.m.layers[li]
	d := l.ds4
	if d == nil {
		return
	}
	w.Wqa, w.Wqb = wt(l.wqa), wt(l.wqb)
	w.QANorm, w.KVANorm = l.qaNorm, l.kvaNorm
	w.HCAttnFn, w.HCFFNFn = wt(d.attnFn), wt(d.ffnFn)
	w.HCAttnScale, w.HCAttnBase = d.attnScale, d.attnBase
	w.HCFFNScale, w.HCFFNBase = d.ffnScale, d.ffnBase
	w.WoA = wt(d.woA)
	if s.m.Cfg.CompAt(li) != jlm.CompNone {
		w.CompKV, w.CompGate = wt(d.compKV), wt(d.compGate)
		w.CompAPE, w.CompNorm = d.compAPE, d.compNorm
	}
	if s.m.Cfg.CompAt(li) == jlm.CompCSA {
		w.IdxQB, w.IdxProj = wt(l.idxQB), wt(l.idxProj)
		w.IdxCompKV, w.IdxCompGate = wt(d.idxCompKV), wt(d.idxCompGate)
		w.IdxCompAPE, w.IdxCompNorm = d.idxCompAPE, d.idxCompNorm
	}
	w.HashExperts = d.hash
}

// tokenIDs hands the device the rows' token ids for the call about to be
// made (nn.TokenDevice): DeepSeek V4's hash blocks route by them. Nothing
// where no block does.
func (s *State) tokenIDs(ids []int32) {
	if s.ldTokens != nil && s.m.Cfg.NHashLayers > 0 {
		s.ldTokens.SetTokenIDs(ids)
	}
}

// migrateEnt moves a DeepSeek V4 block's entries beside its rows: every
// sequence of the State that has history, row r's pos[r] positions closing
// pos[r]/rate entries. Nothing on a block that keeps none.
func (s *State) migrateEnt(li int, pos []int, base func(r int) int, toDevice bool) bool {
	rate := s.kv.entRate(li)
	if rate == 0 {
		return true
	}
	ed, ok := s.ld.(nn.EntDevice)
	if !ok {
		return false
	}
	e := &s.kv.ent[li]
	l := s.entLayout(li)
	for r, p := range pos {
		n := p / rate
		if n == 0 {
			continue
		}
		buf := make([]float32, l.slots(n*l.kvDim()))
		if toDevice && !e.gather(l, r, n, buf, buf) {
			return false
		}
		if !ed.MigrateEnt(li, base(r), buf, n, toDevice) {
			return false
		}
		if !toDevice && !e.scatter(l, r, n, buf, buf) {
			return false
		}
	}
	return true
}
