package engine

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/nn"
)

// Session is one conversation's history: what moves with it from instance
// to instance. Opening one composes nothing.
type Session struct {
	// model is the session's; limit the most positions its history holds,
	// kvFormat its pages' format and span the blocks it has history of, as
	// the instance it was laid out by has (holds). lease is the lease it
	// holds, nil between phases.
	model    *model.Model
	limit    int
	kvFormat nn.KVFmt
	span     [2]int
	lease    *Lease
	// sched is the scheduler that leases the session; on the instance whose
	// calls last ran for it (use), which Close hands its own session back.
	sched Scheduler
	on    *Instance
	// held1 is the lease the default scheduler hands out, kept in the
	// session so acquiring one allocates nothing.
	held1 Lease
	pos   int
	// logits is what the session's last step ended on.
	logits []float32
	// parked is Park's: the session is preempted. ParkedOut and ParkedIn
	// count the KV pages Park sent to the store and Resume brought back
	// (park.go).
	parked    bool
	parkStore *MemStore
	// parkPrev is the store ParkInto swapped out, given back by Resume.
	parkPrev            KVStore
	ParkedOut, ParkedIn int64
	// kv is the paged KV history (see engine/model/kvpage.go). Pages are committed as
	// the sequence reaches them rather than reserved for the whole context.
	kv *kvCache
	// spanTok is every row's token of the mixed prompt being prefilled, -1
	// for a row that has none (tokensOf): what a model with per-layer
	// embeddings, or DeepSeek V4's hash blocks, read where the prefill has no
	// ids. nil outside one.
	spanTok []int32
	// kvSkipped is what the last PrefillCached restored rather than computed.
	kvSkipped int
	// promptRows counts the prompt rows this session ran through its blocks,
	// over its life (PrefillRows): what a benchmark checks to show a prompt
	// was computed, not restored.
	promptRows int
	// forwardPrompts counts the prompts run a token at a time through the
	// decode path (WithPrefillForwardRows).
	forwardPrompts int
	// interrupt, when set, is asked before every prefill chunk
	// (SetPrefillInterrupt).
	interrupt func() bool
	// bpos is the sequence's position as a row of a step, one entry. pos
	// stays the high-water mark: the maxSeq bound and Pos() read it.
	bpos []int
	// rmarks is, per slot, how the rows turn where an image has laid them
	// out off their cache positions (mrope.go). Empty for every sequence
	// that has seen no image, and on every single-axis model.
	rmarks    model.RopeMarks
	retireErr error // a device reset Retire or Reset could not make; see Retire
	// ld is the session's view of its instance's device, which holds its
	// history there; ldRoom to ldQSA are ld's optional faces, nil where it
	// has none, set with ld (setLD). tier is the device ld was attached
	// from, and held the blocks whose history ld holds while the session
	// is not the one its instance runs (use).
	ld nn.LayerDevice
	// routes is the devices the session's blocks run on (route): its card
	// view and its calls on the CPU device (lanes, bound by the instance it
	// leases, bindLanes).
	routes   [2]route
	ldRoom   nn.RoomReporter
	ldSpill  nn.Spiller
	ldStep   nn.SessionStepper
	ldRows   nn.RowsDevice
	ldInputs nn.LayerInputDevice
	ldMarks  nn.RopeMarksDevice
	// markErr is the device's failure to hold a sequence's rotary marks
	// (sendMarks), which the next call returns (marksHeld).
	markErr  error
	ldEmbed  nn.EmbedDevice
	ldPrompt nn.PromptDevice
	tier     nn.LayerDevice
	held     []int8
	// kvRoom is the positions ld holds room for (reserveRuns), 0 when not
	// known.
	kvRoom int
	// away says the bound session's history is not where its instance
	// places the blocks, held says where it is (Park).
	away bool
	// recMark is where the cached prompt in flight parts from the one before
	// it (markParting), a restore point beside the spaced ones; recLast is
	// that one's key.
	recMark int
	recLast []int32
	// recPrefill says a chunked prefill is running, which keeps its restore
	// points itself (recRowPoint).
	recPrefill bool
}

// use makes sess the session s's calls run for. A block s places
// differently from where sess's history is moves that history over, the
// source freed once it has; s's device calls go through sess's view. It
// composes nothing and, with every block where sess left it, allocates
// nothing.
func (s *Instance) use(sess *Session) error {
	if s.Session == sess {
		return s.comeBack()
	}
	if cur := s.Session; cur != nil && !cur.away {
		cur.held = append(cur.held[:0], s.at...)
	}
	s.Session = sess
	sess.on = s
	s.bindLanes(sess)
	if sess.kvFormat != s.kvFmt {
		if err := s.relayout(); err != nil {
			return err
		}
	}
	if sess != s.own && !slices.Contains(s.hist, sess) {
		s.hist = append(s.hist, sess)
	}
	s.headTo(sess)
	if err := s.attachTier(); err != nil {
		return err
	}
	if s.kv.devAt != nil {
		s.kv.devAt = s.devAtFn()
	}
	if s.kv.recAt != nil {
		s.kv.recAt, s.kv.endsAt = s.recFns()
	}
	sess.away = true
	return s.comeBack()
}

// histWritten reports whether a session under the instance holds history:
// the instance keeps the format it was written in (SetDeviceLayers).
func (s *Instance) histWritten() bool {
	for _, o := range s.hist {
		if o.pos > 0 {
			return true
		}
	}
	return false
}

// relayout lays the bound session's empty history out in the instance's
// format: one opened before the instance stepped its cache down for a
// device (SetDeviceLayers). A session with history keeps the format it was
// written in, and is refused here as the pool refuses it.
func (s *Instance) relayout() error {
	sess := s.Session
	if sess.pos != 0 {
		return fmt.Errorf("model: the session's history is %v and the instance's %v", sess.kvFormat, s.kvFmt)
	}
	l := s.kvl
	l.maxSeq = sess.limit
	old := sess.kv
	sess.kv = newKVCacheRange(s.c, old.id, 1, l, old.store, s.opt.kvPage, s.lo, s.hi, old.view, s.eng.cpu.KVPool(s.m.Id))
	sess.kv.reserve(sess.limit)
	sess.kv.ns, sess.kv.budget = old.ns, old.budget
	if err := sess.kv.useArenas(sess.limit, s.kvArenaFor); err != nil {
		return err
	}
	old.giveBack()
	sess.kvFormat = s.kvFmt
	return nil
}

// comeBack brings the bound session's history to where s places each
// block, when it is elsewhere (use, Park).
func (s *Instance) comeBack() error {
	sess := s.Session
	if !sess.away || sess.parked {
		return nil
	}
	for li := s.lo; li < s.hi; li++ {
		want, has := s.devAt(li), li < len(sess.held) && sess.held[li] == routeView
		if want == has {
			continue
		}
		if err := s.moveHistory(li, want); err != nil {
			return err
		}
		if li < len(sess.held) {
			sess.held[li] = s.at[li]
		}
	}
	sess.away = false
	return nil
}

// attachTier gives the bound session a view of s's device: its own, from
// s's device, where it has none or one of another device -- its history
// there coming to the CPU first.
// borrowsJIT reports whether this instance runs on another's JIT, which that
// owner attaches to a device and closes: none does.
func (s *Instance) borrowsJIT() bool { return false }

func (s *Instance) attachTier() error {
	sess := s.Session
	if sess.ld != nil && sess.tier == s.ldCand {
		if !s.borrowsJIT() {
			shareActWindow(s.m, sess.ld)
		}
		return nil
	}
	if sess.ld != nil {
		for li := range sess.held {
			if sess.held[li] != routeView {
				continue
			}
			if !s.migrateKV(li, s.pos, false) || !s.migrateRec(li, false) {
				return fmt.Errorf("model: block %d's history would not leave its device", li)
			}
			sess.ld.ReleaseLayers(li, li+1)
			sess.held[li] = routeLanes
		}
		if d, ok := sess.ld.(nn.Session); ok {
			d.Detach()
		}
		sess.ld, sess.tier = nil, nil
	}
	if s.ldCand == nil {
		return nil
	}
	ld := s.ldCand
	if a, ok := ld.(nn.Attacher); ok {
		ld = a.Attach()
	}
	s.setLD(ld)
	sess.tier, sess.kvRoom = s.ldCand, 0
	if !s.borrowsJIT() {
		shareActWindow(s.m, ld)
	}
	return nil
}

// moveHistory moves block li's history of the bound session to the device
// s runs it on (dev) or to the CPU, and frees the side it left.
func (s *Instance) moveHistory(li int, dev bool) error {
	s.kvRoom = 0
	if !dev {
		if !s.migrateKV(li, s.pos, false) || !s.migrateRec(li, false) {
			return fmt.Errorf("model: block %d's history would not come to the CPU", li)
		}
		s.ld.ReleaseLayers(li, li+1)
		return nil
	}
	w := s.offerWeights(li, false)
	plan := *s.planFor(li)
	if !s.ld.PrepLayer(li, &plan, &w) {
		return fmt.Errorf("model: the device holding block %d has no room for this session's history of it", li)
	}
	if !s.ld.ReserveKV(s.pos) || !s.migrateKV(li, s.pos, true) || !s.migrateRec(li, true) {
		s.ld.ReleaseLayers(li, li+1)
		return fmt.Errorf("model: block %d's history would not move to its device", li)
	}
	if li < len(s.kv.layers) {
		s.kv.layers[li].dropResident()
	}
	return nil
}

// devAtFn and recFns are s's methods a session's cache asks through, made
// once per instance so binding a session allocates nothing.
func (s *Instance) devAtFn() func(int) nn.KVShareDevice {
	if s.fnDevAt == nil {
		s.fnDevAt = s.kvDevAt
	}
	return s.fnDevAt
}

func (s *Instance) recFns() (func(int) bool, func(int) [kvEndSeats]uint8) {
	if s.fnRecAt == nil {
		s.fnRecAt, s.fnEndsAt = s.recPointAt, s.endsAt
	}
	return s.fnRecAt, s.fnEndsAt
}

// Limit is the most positions the session's history holds.
func (s *Session) Limit() int { return s.limit }

// Settle brings sess's history to where s places each block, sess need
// hold no lease: after s moved blocks, so that no session's device view
// keeps a block it no longer runs there.
func (s *Instance) Settle(sess *Session) error {
	var err error
	s.As(sess, func(s *Instance) {
		if sess.ld == nil && sess.tier == nil {
			return // a session that never ran on a device holds nothing there
		}
		err = s.comeBack()
	})
	return err
}

// forget lets go of sess, which is closing or leaving for another
// instance: where its history is (held) is taken first, and the instance's
// calls run for its own session again, its single matvecs through that
// session's view.
func (s *Instance) forget(sess *Session) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if i := slices.Index(s.hist, sess); i >= 0 {
		s.hist = slices.Delete(s.hist, i, i+1)
	}
	if s.Session != sess || s.own == nil {
		return
	}
	s.bindOwn()
}

// bindOwn binds the instance's own session, the bound one's place taken
// first: its calls run for no session's history, its single matvecs
// through its own view.
func (s *Instance) bindOwn() {
	if sess := s.Session; !sess.away {
		sess.held = append(sess.held[:0], s.at...)
		sess.away = true
	}
	s.Session = s.own
	s.own.away = true
	s.headTo(s.own)
	if s.own.ld != nil && !s.borrowsJIT() {
		shareActWindow(s.m, s.own.ld)
	}
}

// headTo points the device's head at sess's logits: a token's projection
// lands in the session the instance runs for.
func (s *Instance) headTo(sess *Session) {
	for _, h := range [2]*nn.Head{s.head, s.headReady} {
		if h != nil && !h.RowLogits {
			h.Logits = sess.logits
		}
	}
}

// settleHist brings the history of every session under s (hist), and the
// instance's own session's view, beside the bound one to where s now places
// each block, and binds the bound one again: a move of the instance's
// blocks moves its sessions' histories with it, and no view keeps a block
// the instance moved off. A session whose history would not move stays away
// and comes back at its next run (use).
func (s *Instance) settleHist() error {
	cur := s.Session
	if s.settling || s.own == nil {
		return nil
	}
	s.settling = true
	defer func() { s.settling = false }()
	var err error
	for _, o := range s.hist {
		if o == cur || o.parked || o.ld == nil && o.tier == nil {
			continue
		}
		if e := s.use(o); e != nil && err == nil {
			err = e
		}
	}
	if o := s.own; o != cur && o.ld != nil {
		if e := s.use(o); e != nil && err == nil {
			err = e
		}
	}
	switch {
	case s.Session == cur:
	case cur == s.own:
		s.bindOwn()
	default:
		if e := s.use(cur); e != nil && err == nil {
			err = e
		}
	}
	return err
}

// roomFor answers a session that could not bring its history to s's
// placement (err) by moving blocks to the CPU, every session's history of
// each with them, until every session's fits (relocateUntil), the own
// session bound for the moves: it holds no history to carry. err where they
// do not.
func (s *Instance) roomFor(err error) error {
	if !s.relocate || !s.listed(routeLanes) || s.own == nil || s.ld == nil {
		return err
	}
	if s.Session != s.own {
		if e := s.use(s.own); e != nil {
			return err
		}
	}
	s.relocateUntil(0, func() bool { return s.settleHist() == nil })
	if e := s.settleHist(); e != nil {
		return err
	}
	return nil
}

// leave lets go of the instance the session last ran on when its next call
// runs on another, before that one is locked: two instances' locks are never
// held at once.
func (s *Session) leave(next *Instance) {
	if s.on != nil && s.on != next {
		s.on.forget(s)
		s.on = nil
	}
}

// Bind calls f with the instance locked and sess bound as a step binds it:
// its view of the instance's device attached and its history where the
// instance places each block. A move prepared for sess (PrepareMove) is
// committed the same way, sess bound.
func (s *Instance) Bind(sess *Session, f func(*Instance)) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.use(sess); err != nil {
		return err
	}
	f(s)
	return nil
}

// Place calls f with the instance locked and its own session bound, its view
// where the instance places each block: a placement (SetDeviceLayers,
// SetGPULayers, PlaceBlocks, MoveBlocks) moves every session's history with
// the blocks (settleHist), so it runs as no session in particular, and a
// session that never ran on the device needs no view of it to move it.
func (s *Instance) Place(f func(*Instance)) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.own != nil && s.Session != s.own {
		if err := s.use(s.own); err != nil {
			return err
		}
	}
	f(s)
	return nil
}

// vacate brings every other session's history of blocks [lo, hi) to the
// CPU, the own session's view among them, and lets their views of the blocks
// go: the tier moves a block between devices with one view holding it, and
// settleHist brings the others after the blocks once they have landed.
func (s *Instance) vacate(lo, hi int) error {
	cur := s.Session
	defer func() { s.Session = cur }()
	release := func(o *Session) error {
		if o == cur || o.ld == nil || o.parked {
			return nil
		}
		s.Session = o
		for li := lo; li < hi && li < len(o.held); li++ {
			if o.held[li] != routeView {
				continue
			}
			if !s.migrateKV(li, s.pos, false) || !s.migrateRec(li, false) {
				return fmt.Errorf("model: block %d's history would not leave its device", li)
			}
			o.ld.ReleaseLayers(li, li+1)
			o.held[li] = routeLanes
		}
		o.away = true
		return nil
	}
	if !cur.away {
		cur.held = append(cur.held[:0], s.at...)
	}
	for _, o := range s.hist {
		if err := release(o); err != nil {
			return err
		}
	}
	if s.own != nil {
		return release(s.own)
	}
	return nil
}

// With calls f with the instance locked: a read of its placement while
// sessions lease it.
func (s *Instance) With(f func(*Instance)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	f(s)
}

// As is With with the instance's calls reading and writing sess's history
// where it is, sess need hold no lease: a read, a Park or a Resume of a
// session that is not running. What f moves of it, its next run (use)
// brings back under the instance's placement.
func (s *Instance) As(sess *Session, f func(*Instance)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if prev := s.Session; prev != sess {
		if !prev.away {
			prev.held = append(prev.held[:0], s.at...)
		}
		sess.away = true
		s.Session = sess
		defer func() { s.Session = prev }()
	}
	f(s)
}
