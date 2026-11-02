package server

import (
	"sync"
	"time"

	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/engine/nn"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// The orchestrator (docs/design/orchestrator.md): the server moving blocks
// itself. When a session starts a generate, its host blocks are offered to
// its model's devices (promote); when a card declines for lack of room, the
// least deserving idle sessions on the cards -- other models' first, lowest
// priority, then idle longest -- move their blocks to the CPU with their history
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
	// Blocks moved, ToHost where they went to the CPU (else onto the cards), Why,
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
// the blocks they moved up and to the CPU.
func (e *Engine) OrchCounts() (promotions, demotions, blocksUp, blocksHome int64) {
	o := &e.orch
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.promotions, o.demotions, o.blocksUp, o.blocksHome
}

// Unpin hands a session's placement back to the orchestrator.
func (s *Session) Unpin() { s.pinned.Store(false) }

// promote offers the host blocks of the instance s runs on to its model's
// devices, demoting other models' idle instances to make room when the cards
// decline. The caller holds s.mu.
func (e *Engine) promote(s *Session) error {
	lm := s.lm
	if e.cfg.NoOrchestrate || s.pinned.Load() || lm.callerPlaced || lm.device() == nil {
		return nil
	}
	n := lm.m.Cfg.NLayer
	if instBlocks(lm) >= n {
		return nil
	}
	if err := e.bindSessionLocked(s); err != nil {
		return nil // a session that cannot attach runs where it is
	}
	e.place(s, whyGenerating)
	for instBlocks(lm) < n {
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

// instBlocks is how many blocks lm's instance runs on a device.
func instBlocks(lm *LoadedModel) int {
	var n int
	lm.inst.With(func(in *model.Instance) { n = in.GPULayers() })
	return n
}

// place offers the host blocks of the instance s runs on to the cards and
// logs what landed.
func (e *Engine) place(s *Session, why orchWhy) int {
	start := time.Now()
	var got int
	s.in().Place(func(in *model.Instance) {
		before := in.GPULayers()
		in.PlaceBlocks(0, s.lm.m.Cfg.NLayer)
		got = in.GPULayers() - before
	})
	if got > 0 {
		e.orch.note(OrchMove{At: start, Session: s.id, Model: s.lm.id, Blocks: got, Why: why,
			Millis: float64(time.Since(start).Microseconds()) / 1000})
	}
	return got
}

// demotionVictim is the least deserving session holding device blocks that
// is not s, not running, not pinned, not caller-placed and of a model whose
// devices list the CPU (the one destination demote moves blocks to): other
// models' sessions first, then the lowest priority, then the one idle
// longest. nil when there is none. One pass, no list.
func (e *Engine) demotionVictim(s *Session) *Session {
	e.mu.RLock()
	defer e.mu.RUnlock()
	var best *Session
	for _, v := range e.sessions {
		// One instance a model: demoting v's is demoting s's own model's
		// when they share it.
		if v == s || v.lm == s.lm || v.closed.Load() || v.running.Load() || v.pinned.Load() || v.lm.callerPlaced ||
			!v.lm.cpuListed || v.snapDevBlocks.Load() == 0 || v.lm.anyRunning() {
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

// demote moves v's blocks to the CPU device with their history, for the
// session named for, when v's lock is free (an idle session; a busy one is skipped, never
// waited on). It reports whether it moved anything.
func (e *Engine) demote(v *Session, forID string) bool {
	if !v.mu.TryLock() {
		return false
	}
	defer v.mu.Unlock()
	start := time.Now()
	var mv model.BlockMove
	v.in().Place(func(in *model.Instance) { mv, _ = in.MoveBlocks(0, v.lm.m.Cfg.NLayer, nn.CPU) })
	n := len(mv.Moved)
	// Every idle session of the instance brings its history home too, or its
	// device view would keep the blocks on the card.
	e.settle(v.lm)
	v.refresh()
	if n == 0 {
		return false
	}
	e.orch.note(OrchMove{At: start, Session: v.id, Model: v.lm.id, Blocks: n, ToHost: true, Why: whyMakeRoom, For: forID,
		Millis: float64(time.Since(start).Microseconds()) / 1000})
	return true
}

// settle brings every idle session of lm under its instance's placement
// (model.Instance.Settle).
func (e *Engine) settle(lm *LoadedModel) {
	e.mu.RLock()
	ss := make([]*Session, 0, len(lm.sessions))
	for _, s := range lm.sessions {
		ss = append(ss, s)
	}
	e.mu.RUnlock()
	for _, s := range ss {
		if !s.mu.TryLock() {
			continue
		}
		lm.inst.Settle(s.ms)
		s.mu.Unlock()
	}
}

// anyRunning says some session of lm is generating.
func (lm *LoadedModel) anyRunning() bool {
	for _, s := range lm.sessions {
		if s.running.Load() {
			return true
		}
	}
	return false
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
