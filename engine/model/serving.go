package model

import (
	"sync"
	"time"

	"github.com/jitllm/jitllm/engine/sched"
)

// Serving split: while one session decodes, another session's prefill runs on
// the E-cores instead of the P-cores the decoder is on.
//
// On a host with both (macOS on Apple Silicon) a prefill beside a decode
// otherwise shares the P-cores' crew, and every region of either waits its
// turn on one lock, so the decoder never fuses and loses most of its rate (the
// decoder measured 11 tok/s beside a prefill against 52 alone). With the split
// the decoder keeps its P-core pool and its fused runs, and the prefill takes
// the E-cores' own pool and lock, its threads at the E-core class (sched.
// LowerToE) so the OS keeps them there. A prefill with no other session
// decoding is not split: it keeps the P and E cores, exactly as before, and so
// does a session alone. See nn.JIT.BeginPrefillE.

// servingGrace is how long after its last token a State still counts as
// decoding: a streaming client's gaps between tokens are short against it, and
// a session that has stopped is not held against the next prefill for long.
const servingGrace = 250 * time.Millisecond

var servingEpoch = time.Now()

// live is the model's open States, for a prefill to ask whether another is
// decoding.
type live struct {
	mu sync.Mutex
	s  []*State
}

func (l *live) add(s *State) {
	l.mu.Lock()
	l.s = append(l.s, s)
	l.mu.Unlock()
}

func (l *live) remove(s *State) {
	l.mu.Lock()
	for i, o := range l.s {
		if o == s {
			l.s = append(l.s[:i], l.s[i+1:]...)
			break
		}
	}
	l.mu.Unlock()
}

// touch records that s is decoding now. One clock read and a store, so a warm
// token still allocates nothing.
func (s *State) touch() { s.ranAt.Store(int64(time.Since(servingEpoch)) + 1) }

// decoding reports whether s ran a token within servingGrace.
func (s *State) decoding() bool {
	t := s.ranAt.Load()
	return t != 0 && time.Since(servingEpoch)-time.Duration(t-1) < servingGrace
}

// othersDecoding reports whether a State of this model other than s is
// decoding.
func (s *State) othersDecoding() bool {
	l := &s.m.live
	l.mu.Lock()
	defer l.mu.Unlock()
	for _, o := range l.s {
		if o != s && o.decoding() {
			return true
		}
	}
	return false
}

// splitWanted is whether s's next prefill chunk goes to the E-cores: the model
// allows it (WithServingSplit) and another session is decoding.
func (s *State) splitWanted() bool {
	return !s.m.opt.noServingSplit && s.othersDecoding()
}

// prefillPlace puts a prefill's chunks where they belong, chunk by chunk: a
// decoder that starts mid-prompt moves the rest of the prompt to the E-cores,
// and one that stops brings it back.
type prefillPlace struct {
	s       *State
	started bool
	onE     bool
	low     sched.Lowered
}

// at is asked before each chunk.
func (p *prefillPlace) at() {
	e := p.s.splitWanted()
	if p.started && e == p.onE {
		return
	}
	p.end()
	if e && p.s.jit.BeginPrefillE() {
		p.onE, p.started = true, true
		p.s.ServedOnE++
		p.low = sched.LowerToE()
		return
	}
	p.s.jit.BeginPrefill()
	p.onE, p.started = false, true
}

// end leaves wherever the prefill is, back on the decode pool.
func (p *prefillPlace) end() {
	if !p.started {
		return
	}
	if p.onE {
		p.low.Raise()
		p.s.jit.EndPrefillE()
	} else {
		p.s.jit.EndPrefill()
	}
	p.started, p.onE = false, false
}
