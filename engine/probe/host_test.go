package probe

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/engine/device/cpu"
)

// TestHostMeasures probes the host and holds what it measures to what other
// work on the box cannot turn: every level's bandwidth and the mix's issue
// rate positive and finite, each cache answering a read sooner than the
// level beyond it, and an empty call under a microsecond. How the speeds
// order is a floor in dev/bench (TestHostProbeOrders): measured beside other
// work a mix issued at 2.85 G/s against its float chains' 4.98 alone, and
// memory read faster than the farthest cache.
func TestHostMeasures(t *testing.T) { hostMeasures(t, cpu.New()) }

// hostMeasures holds the probe of h to what TestHostMeasures states.
func hostMeasures(t *testing.T, h *cpu.Target) {
	d, err := HostOn(h)
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
	measured := func(v float64) bool { return v > 0 && !math.IsInf(v, 0) }
	if !measured(d.Issue.Thread) || !measured(d.Issue.Total) {
		t.Errorf("a mix issues %v one thread, %v all", d.Issue.Thread, d.Issue.Total)
	}
	for l, c := range d.Caches {
		if c.Bytes > 0 && (!measured(c.Rate.Thread) || !measured(c.Rate.Total)) {
			t.Errorf("L%d: %v one thread, %v all", l+1, c.Rate.Thread, c.Rate.Total)
		}
	}
	if !measured(d.Thread) || !measured(d.Bandwidth) {
		t.Errorf("memory %v one thread, %v all", d.Thread, d.Bandwidth)
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
