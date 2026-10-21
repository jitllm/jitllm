package tier

import (
	"math/bits"
	"time"
)

// A submission's run time is bounded, because a driver may end one that runs
// too long without saying so. i915 force-signals a fence still pending after
// CONFIG_DRM_I915_FENCE_TIMEOUT (10 s on Ubuntu's kernels): the queue reports
// idle, the read-back returns, and whatever the cut-off blocks would have
// written is missing. On an integrated Intel GPU a SigLIP block over a
// 1035-patch picture takes 0.93 s, so MiniCPM-V's 27 tower blocks as one
// submission read back a residual with NMSE 1.4 against the host, and degraded
// block by block from the 21st on. Windows' TDR resets a device at 2 s.
//
// So submit cuts a range by the time it measured: each submission takes as
// many blocks as defaultSubmitBudget holds at the per-block cost last seen for
// that row count. A row count it has not seen yet, from probeRows rows up, is
// probed with one block first. A decode token (a row) is never probed: its
// blocks cost microseconds to milliseconds and the range stays whole.

// defaultSubmitBudget is a submission's run-time budget when Config leaves it
// zero: half Windows' TDR, a tenth of i915's fence timeout.
const defaultSubmitBudget = time.Second

// probeRows is the narrowest submission whose cost is probed before a range is
// coalesced.
const probeRows = 64

// subClass is the cost class of a submission of rows rows: rows rounded up to
// a power of two, so a 1035-row picture and a 1024-row chunk share an estimate
// and a 1-row token never shares one with a prompt.
func subClass(rows int) int { return bits.Len(uint(max(rows, 1) - 1)) }

// submitSpan is how many blocks one submission of rows rows may take. Callers
// hold g.mu.
func (g *devTier) submitSpan(rows int) int {
	budget := g.SubmitBudget
	if budget == 0 {
		budget = defaultSubmitBudget
	}
	if budget < 0 {
		return int(^uint(0) >> 1)
	}
	per, ok := g.subCost[subClass(rows)]
	switch {
	case ok && per > 0:
		return max(1, int(budget/per))
	case ok, rows < probeRows:
		return int(^uint(0) >> 1)
	}
	return 1
}

// noteSubmit records that n blocks of rows rows took d. An estimate only
// rises at once: one fast submission (a warm cache, a smaller grid in the same
// class) must not license a range that a slow one showed would overrun.
// Callers hold g.mu.
func (g *devTier) noteSubmit(rows, n int, d time.Duration) {
	if n <= 0 {
		return
	}
	if g.subCost == nil {
		g.subCost = map[int]time.Duration{}
	}
	c := subClass(rows)
	per := d / time.Duration(n)
	if old, ok := g.subCost[c]; ok && per < old {
		per = (old + per) / 2
	}
	g.subCost[c] = per
}
