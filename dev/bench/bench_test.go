package bench

import (
	"math"
	"runtime"
	"strings"
	"testing"
	"time"
)

// spin is a deterministic ~fixed-cost unit of work.
func spin(iters int) {
	var s float64
	for i := 0; i < iters*20000; i++ {
		s += math.Sqrt(float64(i & 1023))
	}
	runtime.KeepAlive(s)
}

// TestSelfAB is the harness's own calibration: comparing a function against
// itself must come out at 1.00. If this drifts, every ratio the harness reports
// is suspect, so it is the first test in the package and not an afterthought.
func TestSelfAB(t *testing.T) {
	a := Case{Name: "spin-a", Fn: spin}
	b := Case{Name: "spin-b", Fn: spin}
	r := AB(a, b, 20, 5)
	t.Log(r)
	if math.Abs(r.Median-1.0) > 0.05 {
		t.Errorf("a function against itself measured %.4fx, want 1.00 +/- 0.05", r.Median)
	}
	if len(r.Ratios) != 20 {
		t.Errorf("got %d rounds, want 20", len(r.Ratios))
	}
}

// TestPhysicsGuard: a benchmark claiming more bandwidth than the machine has is
// a harness bug. It must be loud.
func TestPhysicsGuard(t *testing.T) {
	if raceEnabled {
		// The guard is deliberately off under -race; see guard_race.go.
		t.Skip("physics guard is disabled under -race")
	}
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("an impossible bandwidth claim did not panic")
		}
		msg, _ := r.(string)
		for _, want := range []string{"implies", "GB/s", "exceeds", "ceiling"} {
			if !strings.Contains(msg, want) {
				t.Errorf("panic message %q is missing %q", msg, want)
			}
		}
		t.Log(msg)
	}()
	// One terabyte per iteration, on a function that does nothing.
	impossible := Case{
		Name:         "impossible",
		BytesPerIter: 1 << 40,
		Threads:      1,
		Fn:           func(int) {},
	}
	AB(impossible, impossible, 2, 1)
}

// TestPhysicsGuardAllowsReal: a claim that fits under the ceiling must pass, or
// the guard is just a tripwire in the way.
func TestPhysicsGuardAllowsReal(t *testing.T) {
	const n = 1 << 20
	buf := make([]float64, n)
	c := Case{
		Name:         "real-read",
		BytesPerIter: n * 8,
		Threads:      1,
		Fn: func(iters int) {
			var s float64
			for i := 0; i < iters; i++ {
				for _, v := range buf {
					s += v
				}
			}
			runtime.KeepAlive(s)
		},
	}
	r := AB(c, c, 4, 2) // must not panic
	t.Log(r)
}

func TestMemWall(t *testing.T) {
	if raceEnabled {
		// The detector instruments MemWall's stream loop, so the rate is the
		// detector's, not the bus's; see guard_race.go.
		t.Skip("the memory wall is not measurable under -race")
	}
	one := MemWall(1)
	if one < 1e9 || one > 1e12 {
		t.Errorf("single-thread wall measured %.1f GB/s, which is not plausible", one/1e9)
	}
	four := MemWall(4)
	if four < one {
		t.Errorf("4 threads (%.1f GB/s) measured slower than 1 (%.1f GB/s)", four/1e9, one/1e9)
	}
	t.Logf("memory wall: 1 thread %.1f GB/s, 4 threads %.1f GB/s", one/1e9, four/1e9)
}

// TestReport: a throughput line must carry the wall it is a
// fraction of, so a reader can check it. A bare "85 tok/s" cannot be argued with.
func TestReport(t *testing.T) {
	s := Report("decode", 520_950_640, 25*time.Millisecond, 6)
	for _, want := range []string{"GB/s", "% of the", "wall"} {
		if !strings.Contains(s, want) {
			t.Errorf("Report() = %q, missing %q", s, want)
		}
	}
	t.Log(s)
}

func TestGuard(t *testing.T) {
	// Only asserts that it runs and explains itself; on a busy or hot box a
	// refusal is the correct answer and must not fail the suite.
	if err := Guard(false); err != nil {
		t.Logf("guard refused (correctly, if the box is busy): %v", err)
	} else {
		t.Log("guard: box is idle and cool enough to measure")
	}
}

func TestStable(t *testing.T) {
	if (Result{Median: 1.0, IQR: 0.05}).Stable() != true {
		t.Error("5% spread should be stable")
	}
	if (Result{Median: 1.0, IQR: 0.2}).Stable() != false {
		t.Error("20% spread should be unstable")
	}
}
