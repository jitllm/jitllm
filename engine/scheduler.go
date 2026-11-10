package engine

import (
	"context"
	"fmt"
	"sync"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/nn"
)

// Phase is the part of a session's turn a lease is for.
type Phase uint8

const (
	// PhasePrefill runs a prompt into the session's history.
	PhasePrefill Phase = iota
	// PhaseGenerate decodes the session's next tokens.
	PhaseGenerate
	numPhases
)

// A Scheduler leases a session an instance for a phase and takes it back.
// Many sessions hold leases on one instance at once, and the instance's
// steps batch them (StepRuns, Batcher); a session keeps its lease for the
// whole phase. A model's scheduler is WithScheduler's, or its Pool.
type Scheduler interface {
	// Acquire leases s an instance for phase p, waiting until one has room
	// or ctx ends.
	Acquire(ctx context.Context, s *Session, p Phase) (*Lease, error)
	// Release hands the lease back.
	Release(l *Lease)
}

// Lease is a session's hold on an instance for a phase.
type Lease struct {
	Instance *Instance
	Session  *Session
	Phase    Phase
}

// WithScheduler makes sc the scheduler that leases the sessions an instance
// opens (Instance.NewSession); a Pool's sessions are its own.
func WithScheduler(sc Scheduler) Option { return func(l *loadOpts) { l.opt.sched = sc } }

// Pool is the default Scheduler over a set of instances (Serve): it leases
// the instance for the phase with the fewest leases that has room for the
// session, one served for that phase alone first, and waits when none has.
// A model has one (Model.Serve); NewPool makes another, for a Batcher of
// its own (BatchOver).
type Pool struct {
	mu   sync.Mutex
	list []*served
	wake chan struct{}
}

// served is an instance a Pool hands out: the phases it is for and the
// sessions it batches at once.
type served struct {
	in     *Instance
	phases [numPhases]bool
	width  int
	leases int
}

// NewPool is a Pool serving no instance yet.
func NewPool() *Pool { return &Pool{} }

// Close closes every instance pl serves, which it serves no longer.
func (pl *Pool) Close() {
	for _, in := range pl.take() {
		in.Close()
	}
}

// take is every instance pl serves, which it serves no longer.
func (pl *Pool) take() []*Instance {
	pl.mu.Lock()
	defer pl.mu.Unlock()
	out := make([]*Instance, 0, len(pl.list))
	for _, sv := range pl.list {
		out = append(out, sv.in)
	}
	pl.list = nil
	return out
}

// Serve hands instance in to the pool for phases (every phase when none is
// named), batching up to width sessions at once (MaxStepRows when width <
// 1). The instance was composed when it was made; serving it composes
// nothing. The model's pool's instances close with the model (Model.Close).
func (pl *Pool) Serve(in *Instance, width int, phases ...Phase) error {
	if width < 1 {
		width = MaxStepRows
	}
	// Every instance a session leases reads its history in one format: a
	// later one takes the first's while it holds no history.
	if first := pl.first(); first != nil {
		var err error
		in.With(func(in *Instance) {
			switch {
			case in.kvFmt == first.kvFmt:
			case in.pos == 0:
				in.setKVFmt(first.kvFmt)
			default:
				err = fmt.Errorf("model: the instance's history is %v and the pool's %v", in.kvFmt, first.kvFmt)
			}
		})
		if err != nil {
			return err
		}
	}
	sv := &served{in: in, width: width}
	if len(phases) == 0 {
		for p := range sv.phases {
			sv.phases[p] = true
		}
	}
	for _, p := range phases {
		if p >= numPhases {
			return fmt.Errorf("model: phase %d is not a phase", p)
		}
		sv.phases[p] = true
	}
	pl.mu.Lock()
	pl.list = append(pl.list, sv)
	pl.wakeAll()
	pl.mu.Unlock()
	return nil
}

// Unserve takes in back from the pool once no session leases it.
func (pl *Pool) Unserve(in *Instance) error {
	pl.mu.Lock()
	defer pl.mu.Unlock()
	for i, sv := range pl.list {
		if sv.in != in {
			continue
		}
		if sv.leases > 0 {
			return fmt.Errorf("model: %d session(s) still lease the instance", sv.leases)
		}
		pl.list = append(pl.list[:i], pl.list[i+1:]...)
		return nil
	}
	return nil
}

// first is the pool's first instance, which lays out its sessions'
// history (NewSession), nil while it serves none.
func (pl *Pool) first() *Instance {
	pl.mu.Lock()
	defer pl.mu.Unlock()
	if len(pl.list) == 0 {
		return nil
	}
	return pl.list[0].in
}

// NewSession opens a conversation of up to maxSeq positions leased from
// the pool, its history laid out as the pool's instances' is. It composes
// nothing.
func (pl *Pool) NewSession(maxSeq int) (*Session, error) {
	in := pl.first()
	if in == nil {
		return nil, fmt.Errorf("model: no instance is served (Serve)")
	}
	sess, err := in.NewSession(maxSeq)
	if err != nil {
		return nil, err
	}
	sess.sched = pl
	return sess, nil
}

// wakeAll tells every Acquire waiting that the instances changed. Called
// under mu.
func (pl *Pool) wakeAll() {
	if pl.wake != nil {
		close(pl.wake)
		pl.wake = nil
	}
}

func (pl *Pool) Acquire(ctx context.Context, s *Session, p Phase) (*Lease, error) {
	for {
		pl.mu.Lock()
		if len(pl.list) == 0 {
			pl.mu.Unlock()
			return nil, fmt.Errorf("model: no instance is served (Serve)")
		}
		var best *served
		fits := false
		for _, sv := range pl.list {
			if !sv.phases[p] || !sv.in.holds(s) {
				continue
			}
			fits = true
			if sv.leases >= sv.width {
				continue
			}
			if best == nil || sv.only(p) && !best.only(p) || sv.only(p) == best.only(p) && sv.leases < best.leases {
				best = sv
			}
		}
		if best != nil {
			best.leases++
			pl.mu.Unlock()
			s.held1 = Lease{Instance: best.in, Session: s, Phase: p}
			return &s.held1, nil
		}
		if !fits {
			pl.mu.Unlock()
			return nil, fmt.Errorf("model: no served instance holds a session of %d positions for phase %d", s.limit, p)
		}
		if err := ctx.Err(); err != nil {
			pl.mu.Unlock()
			return nil, err
		}
		if pl.wake == nil {
			pl.wake = make(chan struct{})
		}
		w := pl.wake
		pl.mu.Unlock()
		select {
		case <-w:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}

func (pl *Pool) Release(l *Lease) {
	pl.mu.Lock()
	for _, sv := range pl.list {
		if sv.in == l.Instance && sv.leases > 0 {
			sv.leases--
			break
		}
	}
	pl.wakeAll()
	pl.mu.Unlock()
}

// only says sv serves phase p and no other.
func (sv *served) only(p Phase) bool {
	for q, on := range sv.phases {
		if on != (Phase(q) == p) {
			return false
		}
	}
	return true
}

// holds says s's history fits instance s0: the same blocks and no more
// positions than it was composed for. A history in another format is laid
// out in s0's when s0 takes it (Instance.relayout).
func (s0 *Instance) holds(s *Session) bool {
	return s.model == s0.m && s.limit <= s0.maxSeq && s.span == [2]int{s0.lo, s0.hi}
}

// NewSession is a session of up to maxSeq positions laid out as s's history
// is, leased by s's model's scheduler. It composes nothing.
func (s *Instance) NewSession(maxSeq int) (*Session, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	sess, err := s.newHistory(min(maxSeq, s.m.Cfg.NCtx))
	if err != nil {
		return nil, err
	}
	sess.sched = s.opt.sched
	return sess, nil
}

// newHistory is a session laid out as s's history is, of up to maxSeq
// positions.
func (s *Instance) newHistory(maxSeq int) (*Session, error) {
	sess := &Session{model: s.m, limit: maxSeq, kvFormat: s.kvFmt, span: [2]int{s.lo, s.hi}}
	l := s.kvl
	l.maxSeq = maxSeq
	sess.kv = newKVCacheRange(s.c, s.nextCacheID(), 1, l, nil, s.opt.kvPage, s.lo, s.hi, nil, s.storage().PagePool(), s.entPlanes())
	sess.kv.reserve(maxSeq)
	if err := sess.kv.useArenas(maxSeq, s.kvArenaFor); err != nil {
		return nil, err
	}
	sess.bpos = make([]int, 1)
	sess.logits = make([]float32, s.c.NVocab)
	sess.rmarks = model.RopeMarks{}
	s.makeRec(sess.kv.view, 1)
	return sess, nil
}

// Acquire leases the session an instance for phase p unless it holds one
// for p, the lease it held for another phase handed back first. The lease
// stays held until Release or an Acquire for another phase; StepRuns runs a
// leased session's runs on its instance.
func (s *Session) Acquire(ctx context.Context, p Phase) error {
	if s.lease != nil && s.lease.Phase == p {
		return nil
	}
	s.Release()
	l, err := s.scheduler().Acquire(ctx, s, p)
	if err != nil {
		return err
	}
	s.lease = l
	return nil
}

// scheduler is the session's: the one that leases it, nil for a session no
// scheduler opened.
func (s *Session) scheduler() Scheduler { return s.sched }

// Run is Acquire, then a call of f with the instance running for the
// session. f must not call StepRuns: the instance is held for the call.
func (s *Session) Run(ctx context.Context, p Phase, f func(in *Instance) error) error {
	if err := s.Acquire(ctx, p); err != nil {
		return err
	}
	in := s.lease.Instance
	s.leave(in)
	in.mu.Lock()
	defer in.mu.Unlock()
	if err := in.use(s); err != nil {
		return err
	}
	return f(in)
}

// Hold is Acquire, then the instance locked and running for the session
// until Unhold: for a caller that makes several calls on it in a row. It
// allocates nothing. The caller must not call StepRuns while it holds it.
func (s *Session) Hold(ctx context.Context, p Phase) (*Instance, error) {
	if err := s.Acquire(ctx, p); err != nil {
		return nil, err
	}
	in := s.lease.Instance
	s.leave(in)
	in.mu.Lock()
	if err := in.use(s); err != nil {
		in.mu.Unlock()
		return nil, err
	}
	return in, nil
}

// Unhold lets go of the instance Hold returned.
func (s *Session) Unhold(in *Instance) { in.mu.Unlock() }

// Release hands the session's lease back to the scheduler. Its history
// stays where it is.
func (s *Session) Release() {
	if s.lease == nil {
		return
	}
	s.scheduler().Release(s.lease)
	s.lease = nil
}

// Lease is the lease the session holds, nil between phases.
func (s *Session) Lease() *Lease { return s.lease }

// Close hands the lease back and frees the session's history wherever it
// is: its pages to the model's pool, its device history to the device.
func (s *Session) Close() {
	s.Release()
	if s.on != nil {
		s.on.forget(s)
		s.on = nil
	}
	if d, ok := s.ld.(nn.Session); ok {
		d.Detach()
	}
	s.ld, s.tier, s.held = nil, nil, nil
	if s.kv != nil {
		s.kv.giveBack()
		s.kv = nil
	}
}
