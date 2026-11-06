package engine

import (
	"github.com/jitllm/jitllm/nn"
)

// noHostSeen is a holding no device reports, so the next followHost applies.
const noHostSeen = ^uint64(0)

// holdHost makes d the device this instance counts against the host's budgets,
// letting the one it had go, and re-divides the page budget for the change.
func (s *Instance) holdHost(d nn.LayerDevice) {
	s.hostRes = nil
	if hr, ok := d.(nn.HostReserver); ok {
		s.hostRes = hr
	}
	// Whatever was last seen belonged to the old device.
	s.hostSeen = noHostSeen
	s.applyQuota()
}

// applyQuota hands this instance's page quota to its CPU device, with the
// device it attached and what that holds of the host's memory: the CPU
// device budgets the model's frames by every instance holding the model
// there (cpudev.Quota). An instance the root made shares the root's.
func (s *Instance) applyQuota() {
	if s.quota == nil {
		return
	}
	var held uint64
	if s.hostRes != nil {
		held = s.hostRes.HostReserved()
	}
	s.quota.Set(s.pageQuota.Load(), held)
}

// SetPageBudget is this instance's page quota, in bytes: what the model's
// frames on its CPU device may keep for it, and what caps the frames of the
// card it attached, 0 for unlimited. Paging does not turn on or off: a quota
// that holds every block simply never evicts. What the attached device holds
// of the host's memory comes off the CPU device's share, now and whenever
// placement moves it (followHost). It returns the pages the CPU device's
// frames hold.
func (s *Instance) SetPageBudget(bytes uint64) int {
	s.pageQuota.Store(bytes)
	if fb, ok := s.ld.(nn.FrameBudgeter); ok && s.m.Container != nil {
		fb.SetFrameBudget(s.m.Container, bytes)
	}
	s.applyQuota()
	return s.fr.ResidentPages()
}

// PageBudget is this instance's page quota (SetPageBudget).
func (s *Instance) PageBudget() uint64 { return s.pageQuota.Load() }

// holdOn makes o the device route r holds the model on (nn.ModelOwner),
// letting the one it held go: a device keeps what the model placed until the
// last instance holding it lets it go.
func (s *Instance) holdOn(r int, o nn.ModelOwner) {
	if o == s.owners[r] {
		return
	}
	if s.owners[r] != nil {
		s.owners[r].ReleaseModel(s.m.Id)
	}
	s.owners[r] = o
	if o != nil {
		o.HoldModel(s.m.Id)
	}
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
	s.applyQuota()
}
