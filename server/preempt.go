package server

import (
	"fmt"
	"sync/atomic"

	"github.com/jitllm/jitllm/engine/model"
)

// Preemption under memory pressure. A model may carry a KV budget
// (SetKVBudget): the bytes its sessions' histories may hold at once, wherever
// they live (model.State.HistoryBytes). A generate that would take the model
// past it parks other sessions of the model first (model.State.Park: their
// blocks' KV and recurrent state to the CPU from the device, their sealed pages
// into a store) and the parked session resumes, byte for byte, the next time
// it generates. Nothing is refused for it and nothing is recomputed.
//
// The victim is the lowest priority (SetPriority), then the least recently
// stepped. Only an idle session is a victim: one whose lock is free, so it is
// never parked inside a step or a generate. When no idle session is left to
// park, the generate runs over the budget, as it did before there was one;
// the budget orders who gives way, it does not refuse.

// preemptStats counts what preemption did, for Stats and the gates.
type preemptStats struct {
	parks, resumes    atomic.Int64
	pagesOut, pagesIn atomic.Int64
	// blocksHome is the blocks parking moved to the CPU from a device.
	blocksHome         atomic.Int64
	overBudgetAdmitted atomic.Int64
}

// SetKVBudget caps the bytes model id's sessions' histories may hold at once;
// 0, the default, is no cap.
func (e *Engine) SetKVBudget(id string, bytes uint64) error {
	lm, err := e.Model(id)
	if err != nil {
		return err
	}
	lm.mu.Lock()
	lm.kvBudget = bytes
	lm.mu.Unlock()
	return nil
}

// SetPriority orders this session against the others of its model when one
// must be parked: a lower priority is parked first. 0 is the default.
func (s *Session) SetPriority(p int32) { s.priority.Store(p) }

// Parked reports whether the session is parked.
func (s *Session) Parked() bool { return s.parked.Load() }

// PreemptCounts is what preemption has done: sessions parked and resumed, and
// the KV pages (a key and a value page each count) sent out and brought in.
func (e *Engine) PreemptCounts() (parks, resumes, pagesOut, pagesIn int64) {
	p := &e.preempt
	return p.parks.Load(), p.resumes.Load(), p.pagesOut.Load(), p.pagesIn.Load()
}

// admitKV makes room for s to generate grow more positions: it resumes s if it
// is parked, then parks other sessions of its model until the model's
// histories fit its KV budget with s's grown one. The caller holds s.mu.
func (e *Engine) admitKV(s *Session, grow int) error {
	if err := e.resumeLocked(s); err != nil {
		return err
	}
	lm := s.lm
	lm.mu.Lock()
	budget := lm.kvBudget
	lm.mu.Unlock()
	if budget == 0 {
		return nil
	}
	need := s.st.HistoryBytes() + uint64(max(grow, 0))*s.st.PositionBytes()
	// busy is the sessions found stepping: looked past, not waited on.
	var busy map[*Session]bool
	for {
		e.mu.RLock()
		var used uint64
		var victim *Session
		for _, o := range lm.sessions {
			if o == s || o.closed.Load() || o.parked.Load() {
				continue
			}
			used += o.snapHist.Load()
			if victim == nil || o.priority.Load() < victim.priority.Load() ||
				o.priority.Load() == victim.priority.Load() && o.lastUsed.Load() < victim.lastUsed.Load() {
				if !o.running.Load() && !busy[o] {
					victim = o
				}
			}
		}
		e.mu.RUnlock()
		if used+need <= budget {
			return nil
		}
		if victim == nil {
			e.preempt.overBudgetAdmitted.Add(1)
			return nil
		}
		parked, err := e.park(victim)
		if err != nil {
			return err
		}
		if !parked {
			if busy == nil {
				busy = map[*Session]bool{}
			}
			busy[victim] = true
		}
	}
}

// park parks v if it is idle, reporting whether it did. An idle session's
// lock is free; TryLock never waits on one that is stepping, which a later
// admitKV may take once it is idle.
//
// With a memory cache the session parks into it (model.State.ParkInto), through
// a view that pins what it is given until the session resumes: the model's
// one store holds parked histories and cached prompts alike, counted in its
// bytes and so in the model's budget, and no eviction can take a page a parked
// session will fault back. Without one the State parks as it would alone.
func (e *Engine) park(v *Session) (bool, error) {
	if !v.mu.TryLock() {
		return false, nil
	}
	defer v.mu.Unlock()
	return true, e.parkLocked(v)
}

// resumeLocked resumes s if it is parked. The caller holds s.mu, or is the
// step loop that owns s's row.
func (e *Engine) resumeLocked(s *Session) error {
	if !s.st.Parked() {
		return nil
	}
	in, err := s.st.Resume()
	if err != nil {
		return fmt.Errorf("server: session %q could not resume: %w", s.id, err)
	}
	if s.parkView != nil {
		// Every page is back in the session: the memory cache may evict
		// them in their turn from here.
		s.parkView.Release()
		s.parkView = nil
	}
	s.parked.Store(false)
	e.preempt.resumes.Add(1)
	e.preempt.pagesIn.Add(int64(in))
	return nil
}

// parkLocked parks v. The caller holds v.mu, or is the step loop that owns
// v's row (a time slice, batchfair.go).
func (e *Engine) parkLocked(v *Session) error {
	if v.closed.Load() || v.st.Parked() {
		v.snapHist.Store(0)
		return nil
	}
	var ps model.ParkStats
	var err error
	if store := v.lm.ttft.store; store != nil {
		view := store.Pinned()
		if ps, err = v.st.ParkInto(view); err == nil {
			v.parkView = view
		} else {
			view.Release()
		}
	} else {
		ps, err = v.st.Park()
	}
	if err != nil {
		return fmt.Errorf("server: parking session %q: %w", v.id, err)
	}
	v.parked.Store(true)
	v.snapHist.Store(0)
	e.preempt.parks.Add(1)
	e.preempt.pagesOut.Add(int64(ps.PagesOut))
	e.preempt.blocksHome.Add(int64(ps.Blocks))
	v.refresh()
	return nil
}
