package model

import (
	"fmt"
	"sync/atomic"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/cpu"
)

// Session is one conversation's history: what moves with it from instance
// to instance. Opening one composes nothing.
type Session struct {
	// model is the session's; limit the most positions its history holds,
	// kvFormat its pages' format and span the blocks it has history of, as
	// the instance it was laid out by has (holds). lease is the lease it
	// holds, nil between phases.
	model    *Model
	limit    int
	kvFormat cpu.KVFmt
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
	// kvErr is the first page fault of the current step that could not be
	// recovered. It is set inside a pool closure, which has no error path, and
	// drained by kvCheck at the entry point.
	kvErr atomic.Pointer[error]
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
	interrupt     func() bool
	rconv, rstate [][]float32
	// bpos is a position per row, which is what makes a batch ragged. It is
	// always allocated, one entry even for a single sequence (NewBatch(1) is
	// legal). pos stays the high-water mark across rows: the maxSeq bound
	// and Pos() want the furthest row.
	bpos []int
	// rmarks is, per slot, where an image has moved the rotary position of
	// the rows after it off their cache position (mrope.go). Empty for every
	// sequence that has seen no image, and on every single-axis model.
	rmarks [][]ropeMark
	// ximg is, per slot, the cache position each picture starts at under
	// XD-RoPE, whose image rows carry the picture's ordinal (xdRope).
	ximg      [][]int
	retireErr error // a device reset Retire or Reset could not make; see Retire
	// ld is the session's view of its instance's device, which holds its
	// history there; ldRoom to ldQSA are ld's optional faces, nil where it
	// has none, set with ld (setLD). tier is the device ld was attached
	// from, and held the blocks whose history ld holds while the session
	// is not the one its instance runs (use).
	ld       nn.LayerDevice
	ldRoom   nn.RoomReporter
	ldSpill  nn.Spiller
	ldStep   nn.SessionStepper
	ldRows   nn.RowsDevice
	ldNamed  nn.NamedDevice
	ldInputs nn.LayerInputDevice
	ldTokens nn.TokenDevice
	ldRope   nn.RopeAxesDevice
	ldEmbed  nn.EmbedDevice
	ldPrompt nn.PromptDevice
	ldNGram  nn.NGramDevice
	ldQSA    nn.QSADevice
	tier     nn.LayerDevice
	held     []bool
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
	// ranAt is when this session last started a decode token, in
	// nanoseconds past servingEpoch plus one (0: never); see serving.go.
	ranAt atomic.Int64
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
		cur.held = append(cur.held[:0], s.onDev...)
	}
	s.Session = sess
	sess.on = s
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

// comeBack brings the bound session's history to where s places each
// block, when it is elsewhere (use, Park).
func (s *Instance) comeBack() error {
	sess := s.Session
	if !sess.away || sess.parked {
		return nil
	}
	for li := s.lo; li < s.hi; li++ {
		want, has := s.devAt(li), li < len(sess.held) && sess.held[li]
		if want == has {
			continue
		}
		if err := s.moveHistory(li, want); err != nil {
			return err
		}
		if li < len(sess.held) {
			sess.held[li] = want
		}
	}
	sess.away = false
	return nil
}

// attachTier gives the bound session a view of s's device: its own, from
// s's device, where it has none or one of another device -- its history
// there coming to the CPU first.
func (s *Instance) attachTier() error {
	sess := s.Session
	if sess.ld != nil && sess.tier == s.ldCand {
		if !s.borrowsJIT() {
			s.jit.SetDevice(sess.ld)
		}
		return nil
	}
	if sess.ld != nil {
		for li := range sess.held {
			if !sess.held[li] {
				continue
			}
			if !s.migrateKV(li, s.pos, false) || !s.migrateRec(li, false) {
				return fmt.Errorf("model: block %d's history would not leave its device", li)
			}
			sess.ld.ReleaseLayers(li, li+1)
			sess.held[li] = false
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
		s.jit.SetDevice(ld)
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
	w := nn.LayerWeights{Ensure: s.ensureLayer(li), EnsureExperts: s.ensureSelected(li),
		PrefetchExperts: s.prefetchSelected(li), SpeculateExperts: s.speculateSelected(li),
		HostExperts:     s.hostExpertsFor(li),
		HostExpertsRows: s.hostExpertsRowsFor(li), ExpertUses: s.hot.layer(li)}
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

// forget lets go of sess, which is closing: the instance's calls run for
// its own session again, its single matvecs through that session's view.
func (s *Instance) forget(sess *Session) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.Session != sess || s.own == nil {
		return
	}
	s.Session = s.own
	s.own.away = true
	s.headTo(s.own)
	if s.own.ld != nil && !s.borrowsJIT() {
		s.jit.SetDevice(s.own.ld)
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
			prev.held = append(prev.held[:0], s.onDev...)
		}
		sess.away = true
		s.Session = sess
		defer func() { s.Session = prev }()
	}
	f(s)
}
