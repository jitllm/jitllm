package probe

import (
	"testing"

	"github.com/jitllm/jitllm/engine/cost"
)

// TestHostMeasures probes the host and holds what it measures to the
// order physics allows: every level's bandwidth positive, a nearer cache
// nearer than a farther one and than memory and no more than a fifth slower
// to read (an Apple core streams its L2 nearly as fast as its L1, and one
// thread's L1 rate reads below it), all threads at least most of one's rate,
// and an empty call under a microsecond.
func TestHostMeasures(t *testing.T) {
	d, err := Host()
	if err != nil {
		t.Skip(err)
	}
	t.Logf("workers %d, call %.1f ns", d.Slots, d.Launch*1e9)
	for l, c := range d.Caches {
		t.Logf("L%d %8d KiB: %6.1f GB/s one thread, %6.1f all, %5.1f ns a read, %6.1f ns a stream's start, keeps %.2f, %.2f cycling", l+1, c.Bytes>>10, c.Rate.Thread/1e9, c.Rate.Total/1e9, c.Latency*1e9, c.Start*1e9, c.Retain, c.Cycle)
	}
	t.Logf("memory: %6.1f GB/s one thread, %6.1f all, %5.1f ns a read, %6.1f ns a stream's start", d.Thread/1e9, d.Bandwidth/1e9, d.Latency*1e9, d.Start*1e9)
	names := [cost.NumClasses]string{"float", "special", "int", "index", "lanes", "group", "mem"}
	for cl, r := range d.Rates {
		t.Logf("%-8s %7.2f G/s one thread, %7.2f all, %5.2f ns before its result is used", names[cl], r.Thread/1e9, r.Total/1e9, d.Delay[cl]*1e9)
	}
	t.Logf("mixed    %7.2f G/s one thread, %7.2f all", d.Issue.Thread/1e9, d.Issue.Total/1e9)
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
	if d.Thread <= 0 || d.Thread >= far.Rate.Thread || d.Bandwidth < d.Thread*0.7 {
		t.Errorf("memory %.1f GB/s one thread, %.1f all", d.Thread/1e9, d.Bandwidth/1e9)
	}
	wait := d.Latency
	for l := len(d.Caches) - 1; l >= 0; l-- {
		if c := d.Caches[l]; c.Bytes > 0 {
			if c.Latency <= 0 || c.Latency >= wait {
				t.Errorf("L%d waits %.1f ns, the level beyond it %.1f", l+1, c.Latency*1e9, wait*1e9)
			}
			wait = c.Latency
		}
	}
	if d.Launch <= 0 || d.Launch > 1e-6 {
		t.Errorf("an empty call costs %v", d.Launch)
	}
}
