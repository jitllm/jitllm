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
type Residency struct {
	// budget is the bytes of expert weights that may stay resident. Zero
	// means track but never evict.
	Budget uint64
	// used[layer*NExpert+e] is the token at which that expert was last selected,
	// and 0 means never. resident is the same shape.
	Used     []int32
	Resident []bool
	Bytes    uint64 // currently resident expert bytes
	Each     uint64 // bytes of one expert, all three roles
	Tok      int32
}

// SetBudget retargets an existing manager without disturbing what it has
// learned. Zero stops eviction; a smaller number demotes on the next token.
//
// The counts and the resident set survive, so reclaiming memory later does
// not have to rediscover which experts matter.
func (r *Residency) SetBudget(b uint64) {
	if r != nil {
		r.Budget = b
	}
}

// touch records that layer li selected these experts for the current token.
func (r *Residency) Touch(li int, nExpert int, sel []int32) {
	if r == nil {
		return
	}
	for _, e := range sel {
		i := li*nExpert + int(e)
		r.Used[i] = r.Tok
		if !r.Resident[i] {
			r.Resident[i] = true
			r.Bytes += r.Each
		}
	}
}

// blockHeld reports whether block li's page is held, so a fused run does not
// stay open across the read of one that is not.
func (m *Model) BlockHeld(li int) bool {
	c, cf := m.Container, m.Frames
	return c == nil || li >= c.NPages() || cf.Resident(li)
}
