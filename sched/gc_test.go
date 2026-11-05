package sched

import "testing"

// TestGCReserveKeepsTheBudgetOffTheGoal gates the collector spiral (see
// GCPlan), which no engine counter shows. The property is arithmetic:
// whatever a plan's Budget is, the collector must be left a share of its usable
// heap. A nine-tenths budget fails at once. It holds where the weights are on
// the heap; off it (TestOffHeapWeightsTakeNoReserve) there is no goal to keep
// them off.
func TestGCReserveKeepsTheBudgetOffTheGoal(t *testing.T) {
	g := GCPlan{OnHeap: true}
	u := GCUsable()
	if u == 0 {
		t.Skip("no cgroup limit here, so the collector is unconstrained and has no goal to sit on")
	}
	cap := g.Cap()
	if cap == 0 {
		t.Fatalf("a %d-byte usable heap left no budget at all", u)
	}
	if cap >= u {
		t.Fatalf("the budget cap is %d of a %d-byte usable heap: the collector is "+
			"left nothing, which is the spiral this reserve exists to stop", cap, u)
	}
	// The derived reserve is a quarter of the usable heap, up to 8 GiB: a
	// big cgroup does not need a quarter of itself to keep the collector off
	// its goal, so above 32 GiB the cap passes 75% on purpose. A tiny reserve
	// is the regime that spirals the collector.
	if want := min(u/4, 8<<30); u-cap < want {
		t.Fatalf("the budget cap is %d of a %d-byte usable heap (%.0f%%), a reserve of %d against %d: "+
			"measured, 91%% (9.71 of 9.79 GiB) is 458 GC cycles in a two-token run",
			cap, u, 100*float64(cap)/float64(u), u-cap, want)
	}
	if b := g.Budget(); b > cap {
		t.Fatalf("Budget returned %d, above its own cap of %d: the reserve is "+
			"computed and not applied", b, cap)
	}
	t.Logf("usable %d, cap %d (%.0f%%), Budget %d",
		u, cap, 100*float64(cap)/float64(u), g.Budget())
}

// TestGCReserveIsSettable covers the knob itself.
func TestGCReserveIsSettable(t *testing.T) {
	if GCUsable() == 0 {
		t.Skip("no cgroup limit here")
	}
	was := GCPlan{OnHeap: true}.Cap()
	if got := (GCPlan{OnHeap: true, Reserve: GCUsable() / 2}).Cap(); got >= was {
		t.Fatalf("a half-heap reserve gave a cap of %d, against %d derived: the "+
			"reserve did not reach Cap", got, was)
	}
}

// TestOffHeapWeightsTakeNoReserve: weights off the Go heap are not in the
// collector's goal, so the budget is not capped against it; a capped budget
// would hold a model below what the cgroup can hold whole. On the heap the
// same cgroup must still be capped.
func TestOffHeapWeightsTakeNoReserve(t *testing.T) {
	if GCUsable() == 0 {
		t.Skip("no cgroup limit here, so there is no reserve to drop")
	}
	if !offHeapPlatform {
		t.Skip("no anonymous mmap here, so the weights are always on the heap")
	}
	if c := GCBudgetCap(); c != 0 {
		t.Fatalf("weights off the heap and the budget is still capped at %d", c)
	}
	if c := (GCPlan{OnHeap: true}).Cap(); c == 0 {
		t.Fatal("weights on the heap and no cap: the reserve is gone for the arm that needs it")
	}
}
