//go:build linux

package sched

import (
	"testing"
	"time"
)

// TestDoLatency measures what one Pool.Do round trip costs with no work in it.
//
// It matters because decode is dispatch-dense (hundreds of regions a token):
// at 2 us that is noise, at 25 us a third of the token.
// model.TestFusionCeiling measures the in-situ cost per region.
//
// Both spacings are reported. Back-to-back keeps the workers hot and reads low,
// which flatters the number; the gap version is what a real forward pass sees
// between two matvecs separated by an RMSNorm.
func TestDoLatency(t *testing.T) {
	for _, n := range []int{1, 2, 4, 6} {
		cpus := PCores()
		if len(cpus) < n {
			t.Skipf("only %d P-cores", len(cpus))
		}
		p := New(cpus[:n])
		noop := func(worker, lo, hi int) {}

		const iters = 20000
		// total > chunk, or Do short-circuits and runs inline.
		for i := 0; i < 2000; i++ {
			p.Do(n*8, 1, noop)
		}
		start := time.Now()
		for i := 0; i < iters; i++ {
			p.Do(n*8, 1, noop)
		}
		hot := time.Since(start) / iters

		// With a gap, so the workers have actually parked between calls.
		const gaps = 2000
		var spaced time.Duration
		for i := 0; i < gaps; i++ {
			busy(30 * time.Microsecond)
			s := time.Now()
			p.Do(n*8, 1, noop)
			spaced += time.Since(s)
		}
		spaced /= gaps
		// A region with only two chunks: participation is capped by the work,
		// so this should cost what a two-participant pool costs, not what an
		// n-participant one does.
		for i := 0; i < 2000; i++ {
			p.Do(2, 1, noop)
		}
		start = time.Now()
		for i := 0; i < iters; i++ {
			p.Do(2, 1, noop)
		}
		small := time.Since(start) / iters

		p.Close()
		t.Logf("workers=%d  back-to-back %v   after a 30us gap %v   two-chunk region %v",
			n, hot, spaced, small)
	}
}

func busy(d time.Duration) {
	end := time.Now().Add(d)
	for time.Now().Before(end) {
	}
}
