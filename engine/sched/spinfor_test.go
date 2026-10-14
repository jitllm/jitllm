package sched

import (
	"testing"
	"time"
)

// TestSetSpinForHoldsUntilSetSpinning: SetSpinFor replaces the budget a
// worker reads at its next wait, SetSpinning(true) restores the pool's own,
// and a region run after either still completes on every worker.
func TestSetSpinForHoldsUntilSetSpinning(t *testing.T) {
	p := New([]int{0, 1, 2}, WithSpin(250*time.Microsecond))
	defer p.Close()
	run := func() {
		var hits [64]int
		p.DoRange(len(hits), 1, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				hits[i]++
			}
		})
		for i, h := range hits {
			if h != 1 {
				t.Fatalf("item %d ran %d times", i, h)
			}
		}
	}
	p.SetSpinFor(2 * time.Millisecond)
	if got := time.Duration(p.w().spin.Load()); got != 2*time.Millisecond {
		t.Fatalf("SetSpinFor(2ms) left a budget of %v", got)
	}
	run()
	p.SetSpinFor(-time.Second)
	if p.Spinning() {
		t.Fatal("a negative SetSpinFor left the workers spinning")
	}
	run()
	p.SetSpinning(true)
	if got := time.Duration(p.w().spin.Load()); got != 250*time.Microsecond {
		t.Fatalf("SetSpinning(true) restored %v, want the pool's 250us", got)
	}
	run()
}
