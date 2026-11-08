package server

import (
	"sort"
	"sync"
)

// CPUGateID is the gate every CPU-only session queues on. The CPU is a
// device like any other, so it gets a gate like any other.
const CPUGateID = "cpu"

// gate records which sessions are running on one device, the CPU among them. It
// does not queue them: two sessions on one card are correct at once (each
// holds its own history through gpu.GPU.Attach, and the tier takes its
// scratch one step at a time), and two on the CPU take turns on one shared
// pool a region at a time (sched.Shared). So a generate is never held behind
// another one for its whole length; it interleaves with it token by token.
//
// What is recorded is what GetDeviceQueue and ListDevices report: the
// sessions inside, in arrival order.
type gate struct {
	id string

	mu      sync.Mutex
	holders map[string]bool
	queue   []string
}

// executionNote is what a device's execution mode says about itself.
const executionNote = "sessions run beside each other: wholly on a device, as rows of one " +
	"step; otherwise interleaved a step at a time, the device's scratch and the host's " +
	"shared pool taken per step, never per request"

// ExecutionMode mirrors the proto enum without the server package depending on
// generated code for its own internals.
type ExecutionMode int

const (
	ExecutionUnspecified ExecutionMode = iota
	ExecutionParallel
	ExecutionSerialised
)

// ErrQueueTimeout is returned when a caller declined to keep waiting for a
// row of its model's step loop (batch.go). It is NOT a refusal of the
// session: the session stays open and may try again.
const ErrQueueTimeout serverError = "server: timed out waiting for a row of the step loop"

func newGate(id string) *gate {
	return &gate{id: id, holders: map[string]bool{}}
}

// acquire records the session as running here. It returns how many were
// already running, never a wait: nothing queues on a gate.
func (g *gate) acquire(sessionID string) (depth int32) {
	g.mu.Lock()
	defer g.mu.Unlock()
	depth = int32(len(g.holders))
	g.holders[sessionID] = true
	g.queue = append(g.queue, sessionID)
	return depth
}

func (g *gate) release(sessionID string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	delete(g.holders, sessionID)
	for i, s := range g.queue {
		if s == sessionID {
			g.queue = append(g.queue[:i], g.queue[i+1:]...)
			break
		}
	}
}

// snapshot is what GetDeviceQueue reports: the sessions running here, and
// none waiting.
func (g *gate) snapshot() (queue []string, running int, waiting int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	queue = append(queue, g.queue...)
	return queue, len(g.holders), 0
}

// holds reports whether id is inside the gate now.
func (g *gate) holds(id string) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.holders[id]
}

// gateSet is the gates one generate is recorded on: a session split across
// two cards runs on both, and on the host for the blocks left there.
type gateSet struct {
	gates []*gate
	held  []*gate
	sid   string
}

func (e *Engine) gatesFor(ids []string) *gateSet {
	seen := map[string]bool{}
	sorted := append([]string(nil), ids...)
	sort.Strings(sorted)
	gs := &gateSet{}
	for _, id := range sorted {
		if seen[id] {
			continue
		}
		seen[id] = true
		gs.gates = append(gs.gates, e.gate(id))
	}
	return gs
}

// acquire records the session on every gate, and returns how many sessions
// were already running on the busiest of them.
func (gs *gateSet) acquire(sessionID string) int32 {
	gs.sid = sessionID
	var depth int32
	for _, g := range gs.gates {
		depth = max(depth, g.acquire(sessionID))
		gs.held = append(gs.held, g)
	}
	return depth
}

func (gs *gateSet) release() {
	for i := len(gs.held) - 1; i >= 0; i-- {
		gs.held[i].release(gs.sid)
	}
	gs.held = nil
}
