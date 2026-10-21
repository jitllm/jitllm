package tier

import (
	"testing"
	"time"
)

// TestASubmissionIsCutAtItsRunTimeBudget: a range whose measured per-block
// cost overruns the budget goes as several submissions, each within it, and
// gives the same residual as the range in one. The cut is what keeps a slow
// device under its driver's fence timeout (subbudget.go): an integrated Intel
// GPU ran MiniCPM-V's tower as one 19 s submission, i915 signalled the fence at
// 10 s, and the read-back was a residual 21 blocks in.
func TestASubmissionIsCutAtItsRunTimeBudget(t *testing.T) {
	const n = 6
	run := func(budget time.Duration, per time.Duration) ([]float32, int) {
		g, p := pageTier(t, n, n+1, false)
		if got := placedTotal(g); got != n {
			t.Fatalf("%d of %d blocks placed: %s", got, n, g.Err())
		}
		d := g.dev(0)
		d.mu.Lock()
		d.SubmitBudget = budget
		if per > 0 {
			d.subCost = map[int]time.Duration{subClass(1): per}
		}
		d.mu.Unlock()
		x := make([]float32, p.NEmbd)
		cs := make([]float32, p.NRot)
		for i := range x {
			x[i] = float32(i%7) * 0.01
		}
		before := g.Stats().Submits
		if !g.Layers(0, n, 0, 1, x, cs, nil, nil) {
			t.Fatalf("Layers: %s", g.Err())
		}
		return x, g.Stats().Submits - before
	}
	whole, one := run(-1, 0)
	if one != 1 {
		t.Fatalf("an uncut range of %d resident blocks went as %d submissions, want 1", n, one)
	}
	// A block measured at 400 ms under the default 1 s budget: the first
	// piece takes two. The fake device runs it in microseconds, which halves
	// the estimate to 200 ms, so the other four go as one.
	cut, k := run(0, 400*time.Millisecond)
	if want := 2; k != want {
		t.Fatalf("%d blocks at 400 ms each under a 1 s budget went as %d submissions, want %d", n, k, want)
	}
	for i := range whole {
		if whole[i] != cut[i] {
			t.Fatalf("the cut range's residual differs at %d: %v against %v", i, cut[i], whole[i])
		}
	}
	// A block that alone overruns the budget still runs, one a submission.
	if _, k := run(time.Millisecond, time.Second); k != n {
		t.Fatalf("blocks each over the budget went as %d submissions, want %d", k, n)
	}
	// An unmeasured row count from probeRows up is probed with one block;
	// a token is not.
	d := newDevice(&devShared{Config: &Config{}})
	if got := d.submitSpan(probeRows); got != 1 {
		t.Fatalf("an unmeasured %d-row submission may take %d blocks, want a 1-block probe", probeRows, got)
	}
	if got := d.submitSpan(1); got < n {
		t.Fatalf("an unmeasured token may take %d blocks: a decode token was probed", got)
	}
	d.noteSubmit(probeRows, 4, 2*time.Second)
	if got := d.submitSpan(probeRows); got != 2 {
		t.Fatalf("after 4 blocks in 2 s, a 1 s submission may take %d blocks, want 2", got)
	}
	d.noteSubmit(probeRows, 4, 0)
	if got := d.submitSpan(probeRows); got != 4 {
		t.Fatalf("a fast measurement after a slow one gives %d blocks, want the average's 4", got)
	}
}
