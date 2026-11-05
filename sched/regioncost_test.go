package sched

import (
	"os"
	"sort"
	"testing"
	"time"
)

// TestRegionRoundTrip prices one pool region's control plane with no work in
// it: publish, wake, every worker's acknowledgement, return. A measurement
// (JITLLM_REGION_COST=1); run under taskset to choose the cores, e.g. one
// socket against the same count split across two.
func TestRegionRoundTrip(t *testing.T) {
	if os.Getenv("JITLLM_REGION_COST") == "" {
		t.Skip("a measurement: JITLLM_REGION_COST=1, under taskset")
	}
	p := New(DecodeCores())
	defer p.Close()
	n := p.N()
	sink := make([]int64, 64*n)
	for _, gap := range []time.Duration{0, 20 * time.Microsecond} {
		var d []time.Duration
		for i := 0; i < 20000; i++ {
			if gap > 0 {
				for s := time.Now(); time.Since(s) < gap; {
				}
			}
			t0 := time.Now()
			p.Do(n, 1, func(w, lo, hi int) { sink[w*64]++ })
			d = append(d, time.Since(t0))
		}
		sort.Slice(d, func(i, j int) bool { return d[i] < d[j] })
		t.Logf("%d participants, %v between regions: median %v  p90 %v  p99 %v",
			n, gap, d[len(d)/2], d[len(d)*9/10], d[len(d)*99/100])
	}
}
