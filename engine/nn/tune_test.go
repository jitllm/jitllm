//go:build amd64 && linux

package nn

import (
	"testing"
	"time"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// The tuner decides which generated kernel every matvec reaches for, so its
// state machine is driven here with synthetic timings rather than measured
// ones. No throughput is asserted -- perf floors live in bench/. What
// is asserted is that the climb stops where the numbers say it should.
func TestPackTunerClimb(t *testing.T) {
	for _, tc := range []struct {
		name string
		// cost returns the token time for a width; the climb should find the
		// smallest width that is not beaten by more than the 2% margin.
		cost func(w int) time.Duration
		want int
	}{
		{"wider always worse", func(w int) time.Duration {
			return time.Duration(w) * time.Millisecond
		}, 2},
		{"peak at 4", func(w int) time.Duration {
			d := map[int]int{2: 100, 3: 90, 4: 80, 5: 95, 6: 99, 7: 99, 8: 99}[w]
			return time.Duration(d) * time.Millisecond
		}, 4},
		{"monotone better to the cap", func(w int) time.Duration {
			return time.Duration(100-8*w) * time.Millisecond
		}, cpu.Pack8},
		{"inside the margin counts as no win", func(w int) time.Duration {
			// pack3 is 1% faster: real, but under tuneMargin, so the safe
			// direction (fewer read streams) wins.
			d := map[int]int{2: 100, 3: 99}[w]
			return time.Duration(d) * time.Millisecond
		}, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// No ResetForTest needed: the pin and the mode are fields on the
			// JIT, so an earlier NewJIT cannot reach this one.
			f := &JIT{shaped: map[shapeKey]*cpu.Code{}}
			tn := newPackTuner(2, 0, TuneAuto, true)
			f.tuner = tn
			if !tn.on {
				t.Fatal("tuner should start armed at width 2")
			}
			for i := 0; i < 500 && tn.on; i++ {
				tn.begin()
				tn.t0 = time.Now().Add(-tc.cost(tn.cur))
				tn.end(f)
			}
			if tn.on {
				t.Fatal("tuner never settled")
			}
			if tn.best != tc.want {
				t.Errorf("settled on pack%d, want pack%d", tn.best, tc.want)
			}
			if tn.cur != tn.best {
				t.Errorf("current width %d does not match settled %d", tn.cur, tn.best)
			}
		})
	}
}

// A box too noisy to separate the candidates must fall back to the table rather
// than cache whichever arm got lucky.
func TestPackTunerRejectsNoise(t *testing.T) {
	f := &JIT{shaped: map[shapeKey]*cpu.Code{}}
	tn := newPackTuner(2, 0, TuneAuto, true)
	f.tuner = tn
	spread := []int{10, 15, 20, 25, 30} // IQR/median = 0.5, far past the gate
	for i := 0; i < 500 && tn.on; i++ {
		tn.begin()
		tn.t0 = time.Now().Add(-time.Duration(spread[i%len(spread)]) * time.Millisecond)
		tn.end(f)
	}
	if tn.on {
		t.Fatal("tuner never settled")
	}
	if tn.best != 2 {
		t.Errorf("noise moved the width to pack%d; it must stay at the table's 2", tn.best)
	}
	if tn.key != "" {
		t.Error("a rejected measurement must not be cached")
	}
}

func TestMedianIQRGate(t *testing.T) {
	tight := []float64{1.00, 1.01, 1.02, 1.03, 1.04}
	if _, ok := medianIQRf(tight); !ok {
		t.Error("tight series rejected")
	}
	loose := []float64{1.0, 1.5, 2.0, 2.5, 3.0}
	if _, ok := medianIQRf(loose); ok {
		t.Error("dispersed series accepted")
	}
}
