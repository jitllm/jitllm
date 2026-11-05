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
	m.Container.SetBudget(net)
	// Whatever was evicted faults back in as it is visited. Dense weights are
	// unaffected -- they never leave. Rebinding writes only what moved
	// (sameSpan), under the lock pageIn binds under.
	m.Bind.Lock()
	defer m.Bind.Unlock()
	for i := range m.Layers {
		// The shapes were validated at load; a failure here would mean the
		// container changed underneath, and pageIn reports the same error.
		bindPacked(m.Container, m.Cfg, &m.Layers[i])
	}
}

// followHostHeld re-divides the page budget after what the host-memory devices
// hold moved. It does nothing for an unlimited budget or one already right.
func (m *Model) FollowHostHeld() {
	if m.Container == nil {
		return
	}
	m.pages.mu.Lock()
	defer m.pages.mu.Unlock()
	if m.pages.bytes == 0 {
		return
	}
	if net := m.pageNet(); net != m.Container.Budget() {
		m.applyPageBudget(net)
	}
}

// holdOnHost counts d against the page budget while an instance has it attached;
// unholdOnHost is its release. Neither re-divides: the instance does, once its
// placement is settled (followHost).
func (m *Model) HoldOnHost(d nn.HostReserver) {
	m.pages.mu.Lock()
	defer m.pages.mu.Unlock()
	if m.pages.devs == nil {
		m.pages.devs = map[nn.HostReserver]int{}
	}
	m.pages.devs[d]++
}

func (m *Model) UnholdOnHost(d nn.HostReserver) {
	m.pages.mu.Lock()
	defer m.pages.mu.Unlock()
	if m.pages.devs[d]--; m.pages.devs[d] <= 0 {
		delete(m.pages.devs, d)
	}
}

// noHostSeen is a holding no device reports, so the next followHost applies.
const NoHostSeen = ^uint64(0)
