package oracle

import (
	"math"
	"math/rand"
	"runtime"
	"testing"
)

// fastExp must match math.Exp closely over the range softmax actually uses:
// arguments are (score - max), so at most zero, and anything below about -40
// contributes nothing to the sum regardless.
func TestFastExpAccuracy(t *testing.T) {
	worst, at := 0.0, 0.0
	for _, x := range []float64{0, -1e-12, -0.5, -1, -2, -5, -10, -20, -40, -80, -200, -700, -800} {
		got, want := fastExp(x), math.Exp(x)
		if want == 0 {
			if got != 0 {
				t.Errorf("fastExp(%g) = %g, want 0", x, got)
			}
			continue
		}
		if rel := math.Abs(got-want) / want; rel > worst {
			worst, at = rel, x
		}
	}
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 200000; i++ {
		x := -rng.Float64() * 60
		got, want := fastExp(x), math.Exp(x)
		if rel := math.Abs(got-want) / want; rel > worst {
			worst, at = rel, x
		}
	}
	t.Logf("worst relative error %.3e at x=%g", worst, at)
	// Bound from the degree-7 truncation term, r^8/40320 at |r| <= ln2/2,
	// which is about 5e-9; 2e-8 leaves room for the range reduction.
	if worst > 2e-8 {
		t.Errorf("fastExp worst relative error %.3e at x=%g exceeds 1e-8", worst, at)
	}
}

// Softmax must still be a distribution, and must agree with a math.Exp
// implementation of itself.
func TestSoftmaxMatchesExact(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, n := range []int{1, 2, 17, 821, 4096} {
		x := make([]float64, n)
		ref := make([]float64, n)
		for i := range x {
			v := rng.NormFloat64() * 12
			x[i], ref[i] = v, v
		}
		Softmax(x)

		mx := ref[0]
		for _, v := range ref {
			if v > mx {
				mx = v
			}
		}
		var sum float64
		for i, v := range ref {
			ref[i] = math.Exp(v - mx)
			sum += ref[i]
		}
		var total, worst float64
		for i := range ref {
			ref[i] /= sum
			total += x[i]
			if d := math.Abs(x[i] - ref[i]); d > worst {
				worst = d
			}
		}
		if math.Abs(total-1) > 1e-12 {
			t.Errorf("n=%d: softmax sums to %.15g", n, total)
		}
		// Probabilities, so the deviation is absolute and bounded by the exp
		// error times one. 1e-10 is two orders above what degree 7 gives and
		// three below anything a real bug produces.
		if worst > 1e-10 {
			t.Errorf("n=%d: max absolute deviation from an exact softmax is %.3e", n, worst)
		}
	}
}

func BenchmarkExp(b *testing.B) {
	x := make([]float64, 1024)
	for i := range x {
		x[i] = -float64(i%60) - 0.25
	}
	b.Run("math.Exp", func(b *testing.B) {
		var s float64
		for i := 0; i < b.N; i++ {
			for _, v := range x {
				s += math.Exp(v)
			}
		}
		runtime.KeepAlive(s)
	})
	b.Run("fastExp", func(b *testing.B) {
		var s float64
		for i := 0; i < b.N; i++ {
			for _, v := range x {
				s += fastExp(v)
			}
		}
		runtime.KeepAlive(s)
	})
}

// TestSoftmaxOnlineAgrees keeps the benchmark's B arm honest: a wrong answer
// would make the measurement that rejected it meaningless.
func TestSoftmaxOnlineAgrees(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for _, n := range []int{1, 2, 17, 128, 821} {
		for _, shape := range []string{"unsorted", "ascending", "descending", "equal"} {
			a := make([]float64, n)
			for i := range a {
				switch shape {
				case "unsorted":
					a[i] = rng.NormFloat64() * 4
				case "ascending":
					a[i] = float64(i) * 8 / float64(n)
				case "descending":
					a[i] = -float64(i) * 8 / float64(n)
				case "equal":
					a[i] = 1.25
				}
			}
			b := append([]float64(nil), a...)
			Softmax(a)
			SoftmaxOnline(b)
			// Relative, at 1e-8: the online form compounds fastExp's error
			// through its rescales, so the two do not agree to the last bit.
			for i := range a {
				if d := math.Abs(a[i] - b[i]); d > 1e-8*math.Abs(a[i])+1e-300 {
					t.Fatalf("n=%d %s: element %d differs by %.3e relative (%v vs %v)",
						n, shape, i, d/math.Abs(a[i]), a[i], b[i])
				}
			}
		}
	}
}
