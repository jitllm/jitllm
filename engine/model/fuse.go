package model

import (
	"sync/atomic"
)

// A decode token runs each run of consecutive resident host blocks as ONE pool
// region (sched.Pool.Fuse): the workers join once and every op of every block
// in the run -- the norms, the projections, the rotary and cache write,
// attention, the activation, the residual adds -- is a phase of that region
// rather than a region of its own. The kernels, their chunks and their order
// are the per-op path's, so the answer is the same bits
// (TestFusedBlocksMatchPerOp); what goes is the round trip per op.
//
// The run is bridged across blocks for as long as nothing ends it: a device
// run, a block that has to be read from the container first, or the end of
// the token. A relocation, an eviction or a seam move needs no re-link,
// because the run is not compiled: it is the same per-block code, so the next
// token's runs follow the next token's placement and residency as they are.

// fuseRun opens a fused run unless one is open or this instance runs per op.
func (s *Instance) fuseRun() {
	if s.fused || s.fuseOff {
		return
	}
	// A run starts with nothing to look ahead to: a set named in an earlier
	// run is the instance's to refill now, and no worker may be reading it.
	s.jit.SetAhead(nil)
	s.jit.SetLookahead(!s.lookOff)
	fuseTriesTotal.Add(1)
	if s.jit.Fuse() {
		s.fused = true
		s.FusedRuns++
		fusedRunsTotal.Add(1)
	}
}

// unfuse closes the open fused run, if there is one.
func (s *Instance) unfuse() {
	if s.fused {
		s.fused = false
		s.jit.Unfuse()
	}
}

// SetBlockFusion turns fused host runs on or off for this instance's decode
// tokens (WithBlockFusion is the model-wide default): the A/B arm and the
// gates' per-op reference.
func (s *Instance) SetBlockFusion(on bool) { s.fuseOff = !on }

// blockResident reports whether block li's page is held, so a fused run does
// not stay open across the read of one that is not.
func (m *Model) blockResident(li int) bool {
	c := m.container
	return c == nil || li >= c.NPages() || c.Resident(li)
}

// fusedRunsTotal counts fused runs across every instance, for the gates that must
// show a fused run happened inside them.
var fusedRunsTotal atomic.Int64

// fuseTriesTotal counts the fused runs asked for, opened or not: a gate whose
// subtests all skipped (no model on a model-free runner) asked for none, and
// has nothing to say about fusion either way.
var fuseTriesTotal atomic.Int64

// aheadOps is how many lookahead sets a block has: what it reads after its
// q/k/v, after its o, after its gate/up and after its down.
const aheadOps = 4

// SetLookahead turns the fused runs' lookahead on or off for this instance's
// decode tokens (WithLookahead is the model-wide default).
func (s *Instance) SetLookahead(on bool) { s.lookOff = !on }

// SetMatVecGroups turns groupOf on or off inside fused runs: off, a fused
// run's matvecs and their activation quantization are a phase each.
func (s *Instance) SetMatVecGroups(on bool) { s.groupOff = !on }
