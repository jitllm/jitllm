//go:build darwin

package metal

import (
	"testing"
	"time"
)

// TestOptsArePerContext: two contexts in one process keep their own compile
// options and spin budget, whichever was opened last.
func TestOptsArePerContext(t *testing.T) {
	for _, fastFirst := range []bool{true, false} {
		fast, strict := Opts{FastMath: true, Spin: -1}, Opts{Spin: 5 * time.Millisecond}
		first, second := fast, strict
		if !fastFirst {
			first, second = strict, fast
		}
		a, err := OpenWith(first)
		if err != nil {
			t.Skipf("no Metal device: %v", err)
		}
		defer a.Close()
		b, err := OpenWith(second)
		if err != nil {
			t.Fatalf("OpenWith: %v", err)
		}
		defer b.Close()
		f, s := a, b
		if !fastFirst {
			f, s = b, a
		}
		if f.opts != 0 || f.spin != 0 {
			t.Errorf("fast first=%v: the fast-math context has options %#x and spin %v", fastFirst, f.opts, f.spin)
		}
		if s.opts == 0 || s.spin != 5*time.Millisecond {
			t.Errorf("fast first=%v: the strict context has options %#x and spin %v", fastFirst, s.opts, s.spin)
		}
	}
}
