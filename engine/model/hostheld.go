package model

import (
	"sync"

	"github.com/jitllm/jitllm/engine/nn"
)

// A device whose memory is the host's (an integrated GPU, Apple Silicon)
// spends the bytes the host's pager and expert residency would otherwise
// spend. The host gives up what such a device HOLDS -- its blocks, KV pages
// and scratch, as nn.HostReserver reports them -- and not what it may grow
// into: an iGPU opened by a bare `-devices gpu` that took no block costs the
// host nothing. What it holds moves whenever a block moves, so the instance that
// moved it re-divides both budgets (Instance.followHost).

// pageAsk is the page budget as the caller set it (Model.SetPageBudget, or the
// open-time budget), before what the attached host-memory devices hold comes
// off it.
type pageAsk struct {
	mu sync.Mutex
	// bytes is the caller's budget; zero is unlimited.
	bytes uint64
	// devs is every attached device that reports a host holding, counted by
	// how many instances attached it: one tier serves every session of a model,
	// and it is subtracted once.
	devs map[nn.HostReserver]int
}

// pageNet is the budget the pager is handed: the caller's, less what the
// attached devices hold of the same memory. A holding that covers the whole
// budget leaves one byte -- one page at a time -- because the pager reads zero
// as unlimited. Callers hold m.pages.mu.
func (m *Model) pageNet() uint64 {
	p := &m.pages
	if p.bytes == 0 {
		return 0
	}
	var held uint64
	for d := range p.devs {
		held += d.HostReserved()
	}
	if held >= p.bytes {
		return 1
	}
	return p.bytes - held
}

// applyPageBudget hands the pager a budget and re-points every block at the
// frames it kept. Callers hold m.pages.mu.
func (m *Model) applyPageBudget(net uint64) {
	m.container.SetBudget(net)
	// Whatever was evicted faults back in as it is visited. Dense weights are
	// unaffected -- they never leave. Rebinding writes only what moved
	// (sameSpan), under the lock pageIn binds under.
	m.bind.Lock()
	defer m.bind.Unlock()
	for i := range m.layers {
		// The shapes were validated at load; a failure here would mean the
		// container changed underneath, and pageIn reports the same error.
		bindPacked(m.container, m.Cfg, &m.layers[i])
	}
}

// followHostHeld re-divides the page budget after what the host-memory devices
// hold moved. It does nothing for an unlimited budget or one already right.
func (m *Model) followHostHeld() {
	if m.container == nil {
		return
	}
	m.pages.mu.Lock()
	defer m.pages.mu.Unlock()
	if m.pages.bytes == 0 {
		return
	}
	if net := m.pageNet(); net != m.container.Budget() {
		m.applyPageBudget(net)
	}
}

// holdOnHost counts d against the page budget while an instance has it attached;
// unholdOnHost is its release. Neither re-divides: the instance does, once its
// placement is settled (followHost).
func (m *Model) holdOnHost(d nn.HostReserver) {
	m.pages.mu.Lock()
	defer m.pages.mu.Unlock()
	if m.pages.devs == nil {
		m.pages.devs = map[nn.HostReserver]int{}
	}
	m.pages.devs[d]++
}

func (m *Model) unholdOnHost(d nn.HostReserver) {
	m.pages.mu.Lock()
	defer m.pages.mu.Unlock()
	if m.pages.devs[d]--; m.pages.devs[d] <= 0 {
		delete(m.pages.devs, d)
	}
}

// holdHost makes d the device this instance counts against the host's budgets,
// releasing the one it had, and re-divides the page budget for the change.
func (s *Instance) holdHost(d nn.Device) {
	if s.hostRes != nil {
		s.m.unholdOnHost(s.hostRes)
		s.hostRes = nil
	}
	if hr, ok := d.(nn.HostReserver); ok {
		s.hostRes = hr
		s.m.holdOnHost(hr)
	}
	// Whatever was last seen belonged to the old device.
	s.hostSeen = noHostSeen
	s.m.followHostHeld()
}

// noHostSeen is a holding no device reports, so the next followHost applies.
const noHostSeen = ^uint64(0)

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
	s.m.followHostHeld()
}
