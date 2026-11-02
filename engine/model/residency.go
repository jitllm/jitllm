package model

// Expert residency: keeping the experts the router keeps asking for, and
// dropping the ones it does not.
//
// Which experts a token reads changes every token, and a generic LRU cannot
// know what the router is about to do. The experts a token used predict the
// next token's well enough to be worth tracking.
//
// Eviction itself is jlm's (every expert is its own container page since v26);
// this manager only tracks. See docs/engineering-history/placement.md.
type residency struct {
	// budget is the bytes of expert weights that may stay resident. Zero
	// means track but never evict.
	budget uint64
	// used[layer*NExpert+e] is the token at which that expert was last selected,
	// and 0 means never. resident is the same shape.
	used     []int32
	resident []bool
	bytes    uint64 // currently resident expert bytes
	each     uint64 // bytes of one expert, all three roles
	tok      int32
}

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
	if !c.MoE() || len(s.m.layers) == 0 {
		return
	}
	l := &s.m.layers[0]
	if len(l.experts) == 0 {
		return
	}
	// From the banks' shapes, not from expert 0's views: a view is empty until
	// a token selects that expert and its page is held.
	each := (tensorBytes(l.gate) + tensorBytes(l.up) + tensorBytes(l.down)) / uint64(c.NExpert)
	total := each * uint64(c.NExpert) * uint64(len(s.m.layers))
	dense := s.m.WeightBytes() - total // attention, norms, embedding, routers

	// A budget that the experts already fit under, or no budget at all, means
	// track and do not evict. The manager still exists so that a later
	// SetMemBudget can tighten it without a reload.
	expert := uint64(0)
	if budget > dense && total > budget-dense {
		expert = budget - dense
	}
	if s.res == nil {
		s.res = &residency{
			each:     each,
			used:     make([]int32, c.NExpert*len(s.m.layers)),
			resident: make([]bool, c.NExpert*len(s.m.layers)),
		}
	}
	s.res.SetBudget(expert)
}

// SetBudget retargets an existing manager without disturbing what it has
// learned. Zero stops eviction; a smaller number demotes on the next token.
//
// The counts and the resident set survive, so reclaiming memory later does
// not have to rediscover which experts matter.
func (r *residency) SetBudget(b uint64) {
	if r != nil {
		r.budget = b
	}
}

// touch records that layer li selected these experts for the current token.
func (r *residency) touch(li int, nExpert int, sel []int32) {
	if r == nil {
		return
	}
	for _, e := range sel {
		i := li*nExpert + int(e)
		r.used[i] = r.tok
		if !r.resident[i] {
			r.resident[i] = true
			r.bytes += r.each
		}
	}
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
	r.tok++
	// Zero is "track, never evict".
	if r.budget == 0 || r.bytes <= r.budget {
		return
	}
	// Eviction belongs to the container: every expert is its own page, evicted
	// least recently used by jlm's pager (jlm.File.ExpertPage), and
	// Model.SetPageBudget arms it. This manager only tracks; r.bytes is not
	// decremented here because nothing here frees memory.
}

// Resident reports the expert bytes jitllm is currently keeping, and the budget it
// keeps them under. Both zero when residency is not managed.
func (s *Instance) Resident() (bytes, budget uint64) {
	if s.res == nil {
		return 0, 0
	}
	return s.res.bytes, s.res.budget
}
