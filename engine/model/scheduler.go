package model

import (
	"context"
	"fmt"
	"sync"

	"github.com/jitllm/jitllm/engine/nn"
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

// A Scheduler leases a session one of its model's instances for a phase and
// takes it back. Many sessions hold leases on one instance at once, and the
// instance's steps batch them (StepRuns); a session keeps its lease for the
// whole phase. A model's scheduler is WithScheduler's, or the default one.
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

// WithScheduler makes sc the model's scheduler, in place of the default one.
func WithScheduler(sc Scheduler) Option { return func(l *loadOpts) { l.opt.sched = sc } }

// served is an instance the model's scheduler hands out (Model.Serve): the
// phases it is for and the sessions it batches at once.
type served struct {
	in     *Instance
	phases [numPhases]bool
	width  int
	leases int
}

// instances is the model's served instances and the default scheduler's
// state over them.
type instances struct {
	mu   sync.Mutex
	list []*served
	wake chan struct{}
}

// Serve hands instance in to the model's scheduler for phases (every phase
// when none is named), batching up to width sessions at once (MaxStepRows
// when width < 1). The instance was composed when it was made; serving it
// composes nothing.
func (m *Model) Serve(in *Instance, width int, phases ...Phase) error {
	if in.m != m {
		return fmt.Errorf("model: the instance is another model's")
	}
	if width < 1 {
		width = MaxStepRows
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
	m.insts.mu.Lock()
	m.insts.list = append(m.insts.list, sv)
	m.insts.wakeAll()
	m.insts.mu.Unlock()
	return nil
}

// Unserve takes in back from the scheduler once no session leases it.
func (m *Model) Unserve(in *Instance) error {
	m.insts.mu.Lock()
	defer m.insts.mu.Unlock()
	for i, sv := range m.insts.list {
		if sv.in != in {
			continue
		}
		if sv.leases > 0 {
			return fmt.Errorf("model: %d session(s) still lease the instance", sv.leases)
		}
		m.insts.list = append(m.insts.list[:i], m.insts.list[i+1:]...)
		return nil
	}
	return nil
}

// wakeAll tells every Acquire waiting that the instances changed. Called
// under mu.
func (is *instances) wakeAll() {
	if is.wake != nil {
		close(is.wake)
		is.wake = nil
	}
}

// scheduler is the model's: WithScheduler's, or the default one.
func (m *Model) scheduler() Scheduler {
	if m.opt.sched != nil {
		return m.opt.sched
	}
	return defaultScheduler{m}
}

// defaultScheduler leases the served instance for the phase with the fewest
// leases that has room for the session, one served for that phase alone
// first, and waits when none has.
type defaultScheduler struct{ m *Model }

func (d defaultScheduler) Acquire(ctx context.Context, s *Session, p Phase) (*Lease, error) {
	is := &d.m.insts
	for {
		is.mu.Lock()
		if len(is.list) == 0 {
			is.mu.Unlock()
			return nil, fmt.Errorf("model: no instance serves this model (Model.Serve)")
		}
		var best *served
		fits := false
		for _, sv := range is.list {
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
			is.mu.Unlock()
			s.held1 = Lease{Instance: best.in, Session: s, Phase: p}
			return &s.held1, nil
		}
		if !fits {
			is.mu.Unlock()
			return nil, fmt.Errorf("model: no served instance holds a session of %d positions for phase %d", s.limit, p)
		}
		if is.wake == nil {
			is.wake = make(chan struct{})
		}
		w := is.wake
		is.mu.Unlock()
		select {
		case <-w:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}

func (d defaultScheduler) Release(l *Lease) {
	is := &d.m.insts
	is.mu.Lock()
	for _, sv := range is.list {
		if sv.in == l.Instance && sv.leases > 0 {
			sv.leases--
			break
		}
	}
	is.wakeAll()
	is.mu.Unlock()
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

// holds says s's history fits instance s0: the same blocks, the same KV
// format, and no more positions than it was composed for.
func (s0 *Instance) holds(s *Session) bool {
	return s.model == s0.m && s.limit <= s0.maxSeq && s.kvFormat == s0.kvFmt && s.span == [2]int{s0.lo, s0.hi}
}

// NewSession opens a conversation of up to maxSeq positions on the model's
// served instances, its history laid out as theirs is. It composes nothing.
func (m *Model) NewSession(maxSeq int) (*Session, error) {
	m.insts.mu.Lock()
	var in *Instance
	if len(m.insts.list) > 0 {
		in = m.insts.list[0].in
	}
	m.insts.mu.Unlock()
	if in == nil {
		return nil, fmt.Errorf("model: no instance serves this model (Model.Serve)")
	}
	maxSeq = min(maxSeq, m.Cfg.NCtx)
	in.mu.Lock()
	defer in.mu.Unlock()
	return in.newHistory(maxSeq)
}

// newHistory is a session laid out as s's history is, of up to maxSeq
// positions.
func (s *Instance) newHistory(maxSeq int) (*Session, error) {
	sess := &Session{model: s.m, limit: maxSeq, kvFormat: s.kvFmt, span: [2]int{s.lo, s.hi}}
	l := s.kvl
	l.maxSeq = maxSeq
	sess.kv = newKVCacheRange(s.c, nextCacheID(), 1, l, nil, s.m.opt.kvPage, s.lo, s.hi)
	sess.kv.usePool(&s.m.kvPool)
	sess.kv.reserve(maxSeq)
	if err := sess.kv.useArenas(maxSeq, s.m.kvArenaFor); err != nil {
		return nil, err
	}
	sess.bpos = make([]int, 1)
	sess.logits = make([]float32, s.c.NVocab)
	sess.rmarks = make([][]ropeMark, 1)
	if s.rconv != nil {
		sess.rconv = make([][]float32, len(s.rconv))
		sess.rstate = make([][]float32, len(s.rstate))
		for li := range s.rconv {
			if s.rconv[li] != nil {
				sess.rconv[li] = make([]float32, s.dg.convStateLen())
				sess.rstate[li] = make([]float32, s.dg.stateStride())
			}
		}
	}
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
	l, err := s.model.scheduler().Acquire(ctx, s, p)
	if err != nil {
		return err
	}
	s.lease = l
	return nil
}

// Run is Acquire, then a call of f with the instance running for the
// session.
func (s *Session) Run(ctx context.Context, p Phase, f func(in *Instance) error) error {
	if err := s.Acquire(ctx, p); err != nil {
		return err
	}
	in := s.lease.Instance
	in.mu.Lock()
	defer in.mu.Unlock()
	if err := in.use(s); err != nil {
		return err
	}
	return f(in)
}

// Release hands the session's lease back to the scheduler. Its history
// stays where it is.
func (s *Session) Release() {
	if s.lease == nil {
		return
	}
	s.model.scheduler().Release(s.lease)
	s.lease = nil
}

// Lease is the lease the session holds, nil between phases.
func (s *Session) Lease() *Lease { return s.lease }

// Close hands the lease back and frees the session's history wherever it
// is: its pages to the model's pool, its device history to the device.
func (s *Session) Close() {
	s.Release()
	if d, ok := s.ld.(nn.Session); ok {
		d.Detach()
	}
	s.ld, s.tier, s.held = nil, nil, nil
	if s.kv != nil {
		s.model.kvPool.put(s.kv)
		s.kv = nil
	}
}
