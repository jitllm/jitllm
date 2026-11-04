package sched

import (
	"runtime"
	"testing"
)

// TestRaiseProcsOverlapping: two phases overlap and end in either order. The
// widest live request holds while any is live, and the value found before the
// first is back after the last.
func TestRaiseProcsOverlapping(t *testing.T) {
	c := NewCrews()
	base := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(base)
	want := func(step string, n int) {
		t.Helper()
		if got := runtime.GOMAXPROCS(0); got != n {
			t.Fatalf("%s: GOMAXPROCS %d, want %d", step, got, n)
		}
	}
	for _, narrowFirst := range []bool{true, false} {
		narrow := c.RaiseProcs(base + 2)
		want("narrow begins", base+2)
		wide := c.RaiseProcs(base + 4)
		want("wide begins", base+4)
		if narrowFirst {
			narrow()
			want("narrow ends first", base+4)
			narrow()
			want("narrow ends twice", base+4)
			wide()
		} else {
			wide()
			want("wide ends first", base+2)
			narrow()
		}
		want("both ended", base)
	}
	// A request below the current value changes nothing and restores nothing.
	low := c.RaiseProcs(1)
	want("a narrower request", base)
	low()
	want("its release", base)
}
