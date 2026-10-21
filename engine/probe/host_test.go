package probe

import (
	"testing"

	"github.com/jitllm/jitllm/jit"
)

// TestHostMeasures probes the host and holds what it measures to the
// order physics allows: every level's bandwidth positive, nearer caches
// faster than farther ones and than memory, all threads at least one's
// rate, and an empty call under a microsecond.
func TestHostMeasures(t *testing.T) {
	d, err := Host()
	if err != nil {
		t.Skip(err)
	}
	t.Logf("workers %d, call %.1f ns", d.Slots, d.Launch*1e9)
	for l, c := range d.Caches {
		t.Logf("L%d %8d KiB: %6.1f GB/s one thread, %6.1f all", l+1, c.Bytes>>10, c.Rate.Thread/1e9, c.Rate.Total/1e9)
	}
	t.Logf("memory: %6.1f GB/s one thread, %6.1f all", d.Thread/1e9, d.Bandwidth/1e9)
	names := [jit.NumClasses]string{"float", "special", "int", "index", "lanes", "group", "mem"}
	for cl, r := range d.Rates {
		t.Logf("%-8s %7.2f G/s one thread, %7.2f all", names[cl], r.Thread/1e9, r.Total/1e9)
	}
	prev := 0.0
	for l := len(d.Caches) - 1; l >= 0; l-- {
		c := d.Caches[l]
		if c.Bytes == 0 {
			continue
		}
		if c.Rate.Thread <= prev || c.Rate.Total < c.Rate.Thread*0.9 {
			t.Errorf("L%d: %.1f GB/s one thread, %.1f all, after %.1f farther out", l+1, c.Rate.Thread/1e9, c.Rate.Total/1e9, prev/1e9)
		}
		prev = c.Rate.Thread
	}
	if d.Thread <= 0 || d.Thread >= d.Caches[len(d.Caches)-1].Rate.Thread || d.Bandwidth < d.Thread*0.9 {
		t.Errorf("memory %.1f GB/s one thread, %.1f all", d.Thread/1e9, d.Bandwidth/1e9)
	}
	if d.Launch <= 0 || d.Launch > 1e-6 {
		t.Errorf("an empty call costs %v", d.Launch)
	}
}
