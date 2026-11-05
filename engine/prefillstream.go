package engine

import (
	"fmt"
	"os"

	"github.com/jitllm/jitllm/engine/nn"
)

// A prompt's host blocks, streamed through the device.
//
// A block that does not fit on the card runs on the host, and for decode that
// is the right place: a token is bandwidth-bound, and the host reads the
// block's bytes from its own memory where streaming it would read them and then
// cross the link. A prompt is compute-bound, and one upload serves a whole chunk
// of rows, so the same block may be faster streamed for the prompt. Whether it is
// depends on the link, the card and the host in front of it, so it is measured:
// the first chunk runs as placed, the second with the host blocks streamed (their
// history moved up, outside the timed window), and the faster rate per token
// keeps the rest of the prompt. After the prompt the streamed blocks move to the CPU,
// history and all, once, and decode runs as placed.
//
// It is a placement like any other, so the caller forces it either way
// (WithPrefillStream).

// PrefillStream is whether a prompt's host blocks stream through the device.
type PrefillStream int

const (
	// PrefillStreamAuto measures, on prompts of three chunks or more.
	PrefillStreamAuto PrefillStream = iota
	// PrefillStreamOn streams them from the first chunk, on any prompt.
	PrefillStreamOn
	// PrefillStreamOff keeps them on the host.
	PrefillStreamOff
)

// WithPrefillStream sets whether a prompt's host blocks stream through the
// device for the prompt (PrefillStream).
func WithPrefillStream(m PrefillStream) Option {
	return func(l *loadOpts) { l.opt.prefillStream = m }
}

// WithPrefillStreamDebug traces each prompt's stream-or-keep decision to
// stderr.
func WithPrefillStreamDebug(on bool) Option {
	return func(l *loadOpts) { l.opt.prefillStreamDebug = on }
}

// prefillStreamMargin is how much faster, per token, the streamed chunk must
// run to keep streaming: moving the blocks to the CPU at the end costs a migration
// the chunks after the measured one have to repay.
const prefillStreamMargin = 1.05

// prefillStreamer is one prompt's streaming: which blocks it moved, what it
// measured and what it decided.
type prefillStreamer struct {
	s        *Instance
	nd       nn.NamedDevice
	mode     PrefillStream
	chunk    int   // the chunk about to run
	streamed []int // the blocks moved onto the device for this prompt
	// resident are the device's own blocks, streamed too for the prompt: a
	// card full of them leaves no slot for another block to swap through,
	// and a streamed block evicts only streamed ones. After the prompt they
	// are resident again, paged back in at their next use.
	resident []int
	perTok   [2]float64 // seconds a token: as placed, streamed
	decided  bool
	verbose  bool
}

// PrefillStreams reports how many prompts streamed their host blocks through
// the device and how many measured the host faster and kept them.
func (s *Instance) PrefillStreams() (streamed, kept int) { return s.pfStreamed, s.pfKept }

// newPrefillStreamer is nil where nothing could stream: no device that places
// by name, every block already on it, none on it at all (the card declined
// the model), a batch row, an embedding run, or a prompt too short to measure
// on (auto).
func (s *Instance) newPrefillStreamer(chunks int, bidir bool) *prefillStreamer {
	mode := s.opt.prefillStream
	if mode == PrefillStreamOff || s.ld == nil || bidir || s.emb != nil || s.rows != nil {
		return nil
	}
	if n := s.devCount(); n == 0 || n == s.hi-s.lo {
		return nil
	}
	nd, ok := s.ld.(nn.NamedDevice)
	if !ok || mode == PrefillStreamAuto && chunks < 3 {
		return nil
	}
	return &prefillStreamer{s: s, nd: nd, mode: mode, verbose: s.opt.prefillStreamDebug}
}

// before runs ahead of a chunk that starts at position pos: the first streamed
// chunk moves the host blocks up. Outside the timed window.
func (ps *prefillStreamer) before(pos int) {
	if ps == nil {
		return
	}
	if ps.mode == PrefillStreamOn && ps.chunk == 0 || ps.mode == PrefillStreamAuto && ps.chunk == 1 {
		ps.start(pos)
	}
}

// after records a chunk of n rows that took sec seconds and, on auto, decides
// after the streamed one. end is the position after the chunk.
func (ps *prefillStreamer) after(n int, sec float64, end int) {
	if ps == nil {
		return
	}
	defer func() { ps.chunk++ }()
	if ps.mode != PrefillStreamAuto || ps.decided || n == 0 {
		return
	}
	switch ps.chunk {
	case 0:
		ps.perTok[0] = sec / float64(n)
	case 1:
		ps.perTok[1] = sec / float64(n)
		ps.decided = true
		keep := len(ps.streamed) > 0 && ps.perTok[0] > prefillStreamMargin*ps.perTok[1]
		if ps.verbose {
			fmt.Fprintf(os.Stderr, "prefill stream: %d block(s), %.3f ms a token as placed, %.3f streamed -> %v\n",
				len(ps.streamed), 1e3*ps.perTok[0], 1e3*ps.perTok[1], keep)
		}
		if !keep {
			ps.stop(end)
			ps.s.pfKept++
		}
	}
}

// finish moves the streamed blocks to the CPU after the prompt's last chunk.
func (ps *prefillStreamer) finish(pos int) {
	if ps == nil {
		return
	}
	if len(ps.streamed) > 0 {
		ps.s.pfStreamed++
	}
	ps.stop(pos)
}

// start offers every host block to the device streamed and moves its history
// up. A block the device declines, or whose history does not move, stays on the CPU.
func (ps *prefillStreamer) start(pos int) {
	s := ps.s
	for li := s.lo; li < s.hi; li++ {
		if s.devAt(li) && !s.pinnedOnDev(li) {
			ps.nd.Stream(li, true)
			ps.resident = append(ps.resident, li)
		}
	}
	for li := s.lo; li < s.hi; li++ {
		if s.devAt(li) {
			continue
		}
		// A block the placement keeps on the host stays there.
		if pl, named := s.placeOf(li); named && onCPU(pl.On) {
			continue
		}
		ps.nd.Stream(li, true)
		s.offerRange(li, li+1)
		if !s.devAt(li) {
			ps.nd.Stream(li, false)
			continue
		}
		if !s.migrateKV(li, pos, true) || !s.migrateRec(li, true) {
			s.ld.ReleaseLayers(li, li+1)
			s.unmarkOnDev(li)
			ps.nd.Stream(li, false)
			continue
		}
		ps.streamed = append(ps.streamed, li)
	}
}

// stop moves every streamed block to the CPU, its history first, as relocation
// does, and clears its stream mark. pos is the history's length.
func (ps *prefillStreamer) stop(pos int) {
	s := ps.s
	for _, li := range ps.streamed {
		if !s.devAt(li) {
			// Relocation already took it to the CPU during the prompt, history and
			// all, and marked it for reclaim -- which would place it back
			// RESIDENT at the next room report, and a streamed block was
			// never the placement's. It stays on the CPU, as the placement had it.
			if s.moved != nil {
				s.moved[li] = false
			}
			ps.nd.Stream(li, false)
			continue
		}
		if s.migrateKV(li, pos, false) && s.migrateRec(li, false) {
			s.ld.ReleaseLayers(li, li+1)
			s.unmarkOnDev(li)
		}
		// A block whose history did not move to the CPU stays on the device,
		// streamed: running it there is right, only slower for decode.
		ps.nd.Stream(li, s.devAt(li))
	}
	ps.streamed = nil
	for _, li := range ps.resident {
		ps.nd.Stream(li, false)
	}
	ps.resident = nil
}
