package server

import (
	"sync"
	"time"

	"github.com/jitllm/jitllm/engine/nn"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// The orchestrator (docs/design/orchestrator.md): the server moving blocks
// itself. When a session starts a generate, its host blocks are offered to
// its model's devices (promote); when a card declines for lack of room, the
// least deserving idle sessions on the cards -- other models' first, lowest
// priority, then idle longest -- bring their blocks home with their history
// (demote), and the offer is made again. A running session is never a victim,
// a session or model the caller placed is never moved, and nothing is
// refused: a promotion that does not fit runs where it is.

// orchLogSize bounds the move log the stats carry.
const orchLogSize = 256

// orchWhy is why the orchestrator made a move.
type orchWhy uint8

const (
	whyGenerating orchWhy = iota // promote: the session started a generate
	whyRoomMade                  // promote: after demoting another session
	whyMakeRoom                  // demote: room for the session named in For
)

func (w orchWhy) String() string {
	switch w {
	case whyGenerating:
		return "promote: generating"
	case whyRoomMade:
		return "promote: room made"
	case whyMakeRoom:
		return "make room"
	}
	return ""
}

// OrchMove is one orchestrated move.
type OrchMove struct {
	At      time.Time
	Session string
	Model   string
	// Blocks moved, ToHost where they went home (else onto the cards), Why,
	// and For the session room was made for (whyMakeRoom).
	Blocks int
	ToHost bool
	Why    orchWhy
	For    string
	Millis float64
}

// orchestrator is the Engine's record of what it moved: a ring of the last
// orchLogSize moves and the counters, under one lock -- generates write it
// from their own goroutines and the stats stream reads it.
type orchestrator struct {
	mu  sync.Mutex
	log [orchLogSize]OrchMove
	n   int // moves logged, of which the ring holds the last orchLogSize
	// Promotions and demotions made, and the blocks they moved.
	promotions, demotions int64
	blocksUp, blocksHome  int64
}

// note records m and counts it.
func (o *orchestrator) note(m OrchMove) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.log[o.n%orchLogSize] = m
	o.n++
	if m.ToHost {
		o.demotions++
		o.blocksHome += int64(m.Blocks)
	} else {
		o.promotions++
		o.blocksUp += int64(m.Blocks)
	}
}

// OrchMoves is the orchestrator's recent moves, oldest first.
func (e *Engine) OrchMoves() []OrchMove {
	o := &e.orch
	o.mu.Lock()
	defer o.mu.Unlock()
	k := min(o.n, orchLogSize)
	out := make([]OrchMove, k)
	for i := range out {
		out[i] = o.log[(o.n-k+i)%orchLogSize]
	}
	return out
}

// OrchCounts is what the orchestrator has done: promotions, demotions, and
// the blocks they moved up and home.
func (e *Engine) OrchCounts() (promotions, demotions, blocksUp, blocksHome int64) {
	o := &e.orch
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.promotions, o.demotions, o.blocksUp, o.blocksHome
}

// Unpin hands a session's placement back to the orchestrator.
func (s *Session) Unpin() { s.pinned.Store(false) }

// promote offers s's host blocks to its model's devices, demoting idle
// sessions to make room when the cards decline. The caller holds s.mu.
func (e *Engine) promote(s *Session) error {
	lm := s.lm
	if e.cfg.NoOrchestrate || s.pinned.Load() || lm.callerPlaced || lm.device() == nil {
		return nil
	}
	n := lm.m.Cfg.NLayer
	if s.st.GPULayers() >= n {
		return nil
	}
	if err := e.bindSessionLocked(s); err != nil {
		return nil // a session that cannot attach runs where it is
	}
	e.place(s, whyGenerating)
	for s.st.GPULayers() < n {
		v := e.demotionVictim(s)
		if v == nil {
			break
		}
		if !e.demote(v, s.id) {
			continue
		}
		if e.place(s, whyRoomMade) == 0 {
			// The room that victim gave back let no block in: the card's
			// shortfall is elsewhere; stop rather than empty every card.
			break
		}
	}
	s.refresh()
	return nil
}

// place offers s's host blocks to the cards and logs what landed.
func (e *Engine) place(s *Session, why orchWhy) int {
	start := time.Now()
	before := s.st.GPULayers()
	s.st.PlaceBlocks(0, s.lm.m.Cfg.NLayer)
	got := s.st.GPULayers() - before
	if got > 0 {
		e.orch.note(OrchMove{At: start, Session: s.id, Model: s.lm.id, Blocks: got, Why: why,
			Millis: float64(time.Since(start).Microseconds()) / 1000})
	}
	return got
}

// demotionVictim is the least deserving session holding device blocks that
// is not s, not running, not pinned and not caller-placed: other models'
// sessions first, then the lowest priority, then the one idle longest. nil
// when there is none. One pass, no list.
func (e *Engine) demotionVictim(s *Session) *Session {
	e.mu.RLock()
	defer e.mu.RUnlock()
	var best *Session
	for _, v := range e.sessions {
		if v == s || v.closed.Load() || v.running.Load() || v.pinned.Load() || v.lm.callerPlaced ||
			v.snapDevBlocks.Load() == 0 {
			continue
		}
		if best == nil || lessDeserving(v, best, s.lm) {
			best = v
		}
	}
	return best
}

// lessDeserving orders demotion victims for a promotion of a session of lm:
// another model's session first, then the lower priority, then idle longer.
func lessDeserving(a, b *Session, lm *LoadedModel) bool {
	if oa, ob := a.lm != lm, b.lm != lm; oa != ob {
		return oa
	}
	if pa, pb := a.priority.Load(), b.priority.Load(); pa != pb {
		return pa < pb
	}
	return a.lastUsed.Load() < b.lastUsed.Load()
}

// demote brings v's blocks home with their history, for the session named
// for, when v's lock is free (an idle session; a busy one is skipped, never
// waited on). It reports whether it moved anything.
func (e *Engine) demote(v *Session, forID string) bool {
	if !v.mu.TryLock() {
		return false
	}
	defer v.mu.Unlock()
	start := time.Now()
	mv, _ := v.st.MoveBlocks(0, v.lm.m.Cfg.NLayer, nn.CPU)
	n := len(mv.Moved)
	v.refresh()
	if n == 0 {
		return false
	}
	e.orch.note(OrchMove{At: start, Session: v.id, Model: v.lm.id, Blocks: n, ToHost: true, Why: whyMakeRoom, For: forID,
		Millis: float64(time.Since(start).Microseconds()) / 1000})
	return true
}

// pbOrchestrator is the orchestrator's counters and recent moves for the
// stats stream.
func (e *Engine) pbOrchestrator() *v1.OrchestratorStats {
	pro, dem, up, home := e.OrchCounts()
	moves := e.OrchMoves()
	out := &v1.OrchestratorStats{Enabled: !e.cfg.NoOrchestrate, Promotions: pro, Demotions: dem, BlocksUp: up,
		BlocksHome: home, Moves: make([]*v1.OrchestratorMove, len(moves))}
	for i, m := range moves {
		to, why := "device", m.Why.String()
		if m.ToHost {
			to = "host"
		}
		if m.For != "" {
			why += " for " + m.For
		}
		out.Moves[i] = &v1.OrchestratorMove{UnixMillis: m.At.UnixMilli(), SessionId: m.Session, ModelId: m.Model,
			Blocks: int32(m.Blocks), To: to, Why: why, Millis: m.Millis}
	}
	return out
}
