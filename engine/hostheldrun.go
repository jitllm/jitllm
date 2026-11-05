package engine

import (
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// holdHost makes d the device this instance counts against the host's budgets,
// releasing the one it had, and re-divides the page budget for the change.
func (s *Instance) holdHost(d nn.LayerDevice) {
	if s.hostRes != nil {
		s.m.UnholdOnHost(s.hostRes)
		s.hostRes = nil
	}
	if hr, ok := d.(nn.HostReserver); ok {
		s.hostRes = hr
		s.m.HoldOnHost(hr)
	}
	// Whatever was last seen belonged to the old device.
	s.hostSeen = model.NoHostSeen
	s.m.FollowHostHeld()
}

// followHost re-reads what the attached device holds of the host's memory and,
// when it moved, re-divides the two budgets that pay for it: this instance's
// expert residency and the model's pages. It runs after every placement call
// and at the end of every forward entry point (kvCheck), where a block moved by
// growth, relocation or a demotion has settled. Unchanged, it is one read.
func (s *Instance) followHost() {
	if s.hostRes == nil {
		return
	}
	h := s.hostRes.HostReserved()
	if h == s.hostSeen {
		return
	}
	s.hostSeen = h
	s.applyMemBudget()
	s.m.FollowHostHeld()
}
