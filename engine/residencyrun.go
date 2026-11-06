package engine

import "github.com/jitllm/jitllm/model"

// initResidency sizes the manager. It is built for every mixture of experts,
// whether or not the model fits, and a zero budget means "track but never
// evict" rather than "do nothing".
//
// It is built even when the model fits, because a service may tighten the
// budget after loading (hand memory to another model, park this one); the
// budget is the control. Tracking costs one array write per selected expert.
//
// The budget covers experts only: attention weights, norms and the embedding
// are read every token, so their bytes come off the top.
func (s *Instance) initResidency(budget uint64) {
	c := s.c
	if !c.MoE() || len(s.m.Layers) == 0 {
		return
	}
	l := &s.m.Layers[0]
	if len(l.Experts) == 0 {
		return
	}
	// From the banks' shapes, not from expert 0's views: a view is empty until
	// a token selects that expert and its page is held.
	each := (model.TensorBytes(l.Gate) + model.TensorBytes(l.Up) + model.TensorBytes(l.Down)) / uint64(c.NExpert)
	total := each * uint64(c.NExpert) * uint64(len(s.m.Layers))
	dense := s.m.WeightBytes() - total // attention, norms, embedding, routers

	// A budget that the experts already fit under, or no budget at all, means
	// track and do not evict. The manager still exists so that a later
	// SetMemBudget can tighten it without a reload.
	expert := uint64(0)
	if budget > dense && total > budget-dense {
		expert = budget - dense
	}
	if s.res == nil {
		s.res = &model.Residency{
			Each:     each,
			Used:     make([]int32, c.NExpert*len(s.m.Layers)),
			Resident: make([]bool, c.NExpert*len(s.m.Layers)),
		}
	}
	s.res.SetBudget(expert)
}

// reap drops the coldest experts until the resident set is inside the budget.
// Called once per token, after every layer has routed.
//
// Once per token, not per layer: evicting inside a token would drop experts a
// later layer of the same token is about to use.
func (s *Instance) reap() {
	r := s.res
	if r == nil {
		return
	}
	r.Tok++
	// Zero is "track, never evict".
	if r.Budget == 0 || r.Bytes <= r.Budget {
		return
	}
	// Eviction belongs to the container: every expert is its own page, evicted
	// least recently used by the frames holding it (jlm.Frames), and the
	// instance's page quota arms it (Instance.SetPageBudget). This manager only tracks; r.bytes is not
	// decremented here because nothing here frees memory.
}

// Resident reports the expert bytes jitllm is currently keeping, and the budget it
// keeps them under. Both zero when residency is not managed.
func (s *Instance) Resident() (bytes, budget uint64) {
	if s.res == nil {
		return 0, 0
	}
	return s.res.Bytes, s.res.Budget
}
