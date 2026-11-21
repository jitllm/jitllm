package bench

import (
	"testing"

	"github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/engine/probe"
)

// TestHostProbeOrders holds the host probe's speeds to the order physics
// allows, on a box with nothing else running: a nearer cache no more than a
// fifth slower to read than a farther one (an Apple core streams its L2
// nearly as fast as its L1, and one thread's L1 rate reads below it), memory
// no more than a fifth faster than the farthest cache (one Xeon thread reads
// its L3 and its memory at one rate, 27.1 and 27.0 GB/s), all threads at
// least most of one's rate, and a mix of classes issuing about as fast as
// its float chains alone. Beside other work each of these turned
// (engine/probe's TestHostMeasures holds the rest).
func TestHostProbeOrders(t *testing.T) {
	d, err := probe.Host()
	if err != nil {
		t.Skip(err)
	}
	if f := d.Rates[cost.ClassFloat].Thread; d.Issue.Thread < f*0.9 {
		t.Errorf("a mix issues %.2f G/s, below its float chains' %.2f alone", d.Issue.Thread/1e9, f/1e9)
	}
	prev := 0.0
	for l := len(d.Caches) - 1; l >= 0; l-- {
		c := d.Caches[l]
		if c.Bytes == 0 {
			continue
		}
		if c.Rate.Thread < 0.8*prev || c.Rate.Total < c.Rate.Thread*0.7 {
			t.Errorf("L%d: %.1f GB/s one thread, %.1f all, after %.1f farther out", l+1, c.Rate.Thread/1e9, c.Rate.Total/1e9, prev/1e9)
		}
		prev = c.Rate.Thread
	}
	far := d.Caches[0]
	for _, c := range d.Caches {
		if c.Bytes > 0 {
			far = c
		}
	}
	if far.Rate.Thread < 0.8*d.Thread || d.Bandwidth < d.Thread*0.7 {
		t.Errorf("memory %.1f GB/s one thread, %.1f all, the farthest cache %.1f", d.Thread/1e9, d.Bandwidth/1e9, far.Rate.Thread/1e9)
	}
	t.Logf("mix %.2f G/s, memory %.1f GB/s one thread, %.1f all", d.Issue.Thread/1e9, d.Thread/1e9, d.Bandwidth/1e9)
}
