package model

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
	s.m.fuseTries.Add(1)
	if s.cpu.workers.Fuse() {
		s.fused = true
		s.FusedRuns++
		s.m.fusedRuns.Add(1)
	}
}

// unfuse closes the open fused run, if there is one.
func (s *Instance) unfuse() {
	if s.fused {
		s.fused = false
		s.cpu.workers.Unfuse()
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

// SetMatVecGroups turns groupOf on or off inside fused runs: off, a fused
// run's matvecs and their activation quantization are a phase each.
func (s *Instance) SetMatVecGroups(on bool) { s.groupOff = !on }
