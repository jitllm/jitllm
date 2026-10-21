package tier

import (
	"testing"
	"time"
)

// TestGroupTunerClimbsToTheKnee feeds the fill tuner a cost curve with its
// minimum at 4 groups: it must climb 1 -> 2 -> 4, refuse 8, and hold 4; a
// stated Config.StreamGroups is never tuned.
func TestGroupTunerClimbsToTheKnee(t *testing.T) {
	cost := map[int]time.Duration{1: 180, 2: 120, 4: 100, 8: 130, 16: 200}
	g := newDevice(&devShared{Config: &Config{}})
	for i := 0; i < 400; i++ {
		n := g.streamGroups(3)
		g.fillDone(cost[n] * time.Millisecond)
	}
	if !g.ftune.groups.settled() || g.ftune.groups.bestValue() != 4 || g.StreamGroupsTuned != 4 {
		t.Fatalf("settled %v on %d groups (reported %d), want 4", g.ftune.groups.settled(), g.ftune.groups.bestValue(), g.StreamGroupsTuned)
	}
	// Then the half: a curve with its knee at 8 MiB, the group count held.
	hcost := map[int]time.Duration{4 << 20: 130, 8 << 20: 100, 12 << 20: 104, 16 << 20: 120, 32 << 20: 150}
	for i := 0; i < 400; i++ {
		g.streamGroups(3)
		g.fillDone(hcost[g.pinHalfBytes()] * time.Millisecond)
	}
	if !g.ftune.half.settled() || g.StreamPinHalfTuned != 12<<20 && g.StreamPinHalfTuned != 8<<20 {
		t.Fatalf("the half settled %v on %d", g.ftune.half.settled(), g.StreamPinHalfTuned)
	}
	f := newDevice(&devShared{Config: &Config{StreamGroups: 2}})
	for i := 0; i < 50; i++ {
		f.fillDone(cost[f.streamGroups(3)] * time.Millisecond)
	}
	if f.streamGroups(3) != 2 {
		t.Fatal("a stated group count was tuned")
	}
}
