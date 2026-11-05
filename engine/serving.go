package engine

import (
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

// touch records that s is decoding now, on its seat on the CPU.
func (s *Instance) touch() { s.seat.Touch() }

// othersDecoding reports whether an instance on the CPU other than s is
// decoding.
func (s *Instance) othersDecoding() bool { return s.eng.cpu.OthersDecoding(s.seat) }

// splitWanted is whether s's next prefill chunk goes to the E-cores: the model
// allows it (WithServingSplit) and another session is decoding.
func (s *Instance) splitWanted() bool {
	return !s.opt.noServingSplit && s.othersDecoding()
}

// prefillPlace puts a prefill's chunks where they belong, chunk by chunk: a
// decoder that starts mid-prompt moves the rest of the prompt to the E-cores,
// and one that stops brings it back.
type prefillPlace struct {
	s       *Instance
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
	if e && p.s.cpu.workers.BeginPrefillE() {
		p.onE, p.started = true, true
		p.s.ServedOnE++
		p.low = p.s.cpu.workers.LowerToE()
		return
	}
	p.s.cpu.workers.BeginPrefill()
	p.onE, p.started = false, true
}

// end leaves wherever the prefill is, back on the decode pool.
func (p *prefillPlace) end() {
	if !p.started {
		return
	}
	if p.onE {
		p.low.Raise()
		p.s.cpu.workers.EndPrefillE()
	} else {
		p.s.cpu.workers.EndPrefill()
	}
	p.started, p.onE = false, false
}
