//go:build linux

package sched

import "testing"

// TestMemBudgetSeesTheCgroup: the violation is reading /proc/meminfo and
// stopping, which under scripts/cap 4G gives a budget several times the cgroup
// limit. It asserts the relationship rather than a value, because the value
// depends on the cap the suite runs under.
func TestMemBudgetSeesTheCgroup(t *testing.T) {
	lim, avail, b := host().cgroupLimit(), host().meminfoAvailable(), MemBudget()
	t.Logf("cgroup %.2f GB, MemAvailable %.2f GB, budget %.2f GB",
		float64(lim)/1e9, float64(avail)/1e9, float64(b)/1e9)
	if avail == 0 {
		t.Skip("no /proc/meminfo")
	}
	if b == 0 {
		t.Fatal("budget is 0 with MemAvailable known")
	}
	if b > avail {
		t.Errorf("budget %d exceeds MemAvailable %d", b, avail)
	}
	if lim > 0 && b > lim {
		t.Errorf("budget %d exceeds the cgroup limit %d -- /proc/meminfo was believed over the cgroup", b, lim)
	}
	// The headroom is not decoration: the KV cache is anonymous and
	// unreclaimable, and a cgroup that reaches MemoryHigh is throttled.
	if lim > 0 && b >= lim {
		t.Errorf("budget %d leaves no headroom under the limit %d", b, lim)
	}
}
