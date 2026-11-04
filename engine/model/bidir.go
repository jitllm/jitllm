package model

import (
	"github.com/jitllm/jitllm/engine/nn"
)

// Bidirectional runs inside a causal prompt: Gemma 3's reference masks an
// image's tokens so they see each other in both directions, while every other
// token stays causal (its token_type_ids blockwise overlay, OR'd with the
// causal mask, AND'd with the sliding window on the local layers). The mask
// here is a per-row key COUNT, as it is everywhere in this engine: a row in a
// run counts keys up to the run's end instead of up to itself. Every row of a
// run must therefore be in the same prefill chunk, since none of their scores
// exist until all their keys are in the cache.

// bidirEnd is the end of the run holding position pos, or 0 when pos is in
// none.
func (s *Instance) bidirEnd(pos int) int {
	for _, r := range s.bidir {
		if pos >= r.Lo && pos < r.Hi {
			return r.Hi
		}
	}
	return 0
}

// bidirStart is the start of the run holding position pos, 0 when pos is
// in none.
func (s *Instance) bidirStart(pos int) int {
	for _, r := range s.bidir {
		if pos >= r.Lo && pos < r.Hi {
			return r.Lo
		}
	}
	return 0
}

// runsAt reports whether block li's attention honours the instance's runs:
// every block's, or the sliding blocks' alone where the model's image rows
// see each other on those (Config.BidirSWA).
func (s *Instance) runsAt(li int) bool {
	return len(s.bidir) > 0 && s.c.LayerKind(li).Attends() && (!s.c.BidirSWA || s.c.SWA(li))
}

// bidirChunk is a chunk's row count n, starting at position p0, moved so no
// run is cut: a run that would straddle the end is left to the next chunk when
// rows come before it, and taken whole (past n) when it starts the chunk.
func (s *Instance) bidirChunk(p0, n int) int {
	for _, r := range s.bidir {
		if r.Lo < p0+n && r.Hi > p0+n {
			if r.Lo > p0 {
				return r.Lo - p0
			}
			return r.Hi - p0
		}
	}
	return n
}

// bidirIn reports whether a run lies in the chunk [p0, p0+n).
func (s *Instance) bidirIn(p0, n int) bool {
	for _, r := range s.bidir {
		if r.Lo < p0+n && r.Hi > p0 {
			return true
		}
	}
	return false
}

// bidirTell hands the runs to the device before a chunk's blocks run there,
// and reports whether it can honour them; a device that cannot must not run a
// chunk holding one, since its causal answer would be fluent and wrong.
func (s *Instance) bidirTell(p0, n int) bool {
	kd, ok := s.ld.(nn.KeyRunDevice)
	if !s.bidirIn(p0, n) {
		if ok {
			kd.SetKeyRuns(nil, nil)
		}
		return true
	}
	return ok && kd.SetKeyRuns(s.bidir, nil)
}

// ncArena is a non-causal segment's history on the CPU device
// (cpuserve.go), a tower's or an encoder's: its keys live one block
// deep, so every block borrows the one page in turn, the pool its form
// addresses (kvArena).
type ncArena struct {
	arena *kvArena
	// page is the arena's page, the segment's rows rounded up to a power of
	// two: the composer's page arithmetic is shifts on the CPU (AVX2 has no
	// packed integer division).
	page int
	// run is the one run of a composed CPU block's rows; causalFault, a
	// gate's violation, leaves the rows causal.
	run         [1]nn.KeyRun
	causalFault bool
}

// nc is the instance's non-causal segment's arena, nil for an instance of none.
func (s *Instance) nc() *ncArena { return nil }

// unmapNC releases the segment's arena.
func (s *Instance) unmapNC() {
	if a := s.nc(); a != nil && a.arena != nil {
		a.arena.unmap()
		a.arena = nil
	}
}
