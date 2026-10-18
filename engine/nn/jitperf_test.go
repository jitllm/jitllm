//go:build amd64 && linux

package nn

import (
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/format/quant"
)

// TestMatVecScaling isolates the JIT tier from the model, so a scaling number
// can be attributed to the pool, the kernel, or the shapes the model issues.
//
// Logging only: no perf floors in tests.
func TestMatVecScaling(t *testing.T) {
	requireRowMajorHost(t, quant.Q4_0)
	if testing.Short() {
		t.Skip("allocates 128 MB")
	}
	const k = 2048
	const rowBytes = k / 32 * 18
	const rows = 128 << 20 / rowBytes // ~128 MB, far past L3

	rng := rand.New(rand.NewSource(3))
	w := make([]byte, rows*rowBytes)
	for i := range w {
		w[i] = byte(rng.Intn(256))
	}
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	out := make([]float32, rows)

	for _, n := range []int{1, 2, 4, 6} {
		f := NewJIT(k, rows, []quant.Type{quant.Q4_0}, WithSched(sched.WithCores(n)))
		if f == nil {
			t.Skip("no fast tier")
		}
		best := time.Hour
		for i := 0; i < 4; i++ {
			start := time.Now()
			if !f.MatVec(out, quant.Q4_0, w, x, rows, k) {
				t.Fatal("MatVec declined")
			}
			if d := time.Since(start); d < best {
				best = d
			}
		}
		f.Close()
		gbs := float64(rows) * rowBytes / best.Seconds() / 1e9
		t.Logf("%d core(s): %6.2f GB/s  (%v for %d rows)", n, gbs, best.Round(time.Millisecond), rows)
	}
}

// TestMatVecShapeSensitivity: the model issues matvecs from 256 rows (gemma's
// GQA k/v projections) to 256128 (its tied lm_head).
//
// Caveat: this repeats the same matvec, so any weight set under the L3 is
// measured hot in cache, not from DRAM as a real decode reads it. Compare
// shapes against each other, never against the memory wall.
func TestMatVecShapeSensitivity(t *testing.T) {
	if testing.Short() {
		t.Skip("allocates")
	}
	rng := rand.New(rand.NewSource(5))
	os.Unsetenv("JITLLM_CORES")

	// gemma's actual per-layer mix, plus two controls.
	for _, sh := range []struct {
		name    string
		rows, k int
	}{
		{"gemma q/o", 2048, 2048},
		{"gemma k/v (GQA)", 256, 2048},
		{"gemma gate/up", 16384, 2048},
		{"gemma ffn_down", 2048, 16384},
		{"gemma lm_head", 256128, 2048},
		{"control large", 131072, 2048},
	} {
		rows, kk := sh.rows, sh.k
		rowBytes := kk / 32 * 18
		x := make([]float32, kk)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		w := make([]byte, rows*rowBytes)
		for i := range w {
			w[i] = byte(rng.Intn(251))
		}
		out := make([]float32, rows)
		f := NewJIT(kk, rows, []quant.Type{quant.Q4_0})
		if f == nil {
			t.Skip("no fast tier")
		}
		best := time.Hour
		reps := max(1, 1<<23/(rows*rowBytes/1024+1))
		for i := 0; i < 4; i++ {
			start := time.Now()
			for r := 0; r < reps; r++ {
				f.MatVec(out, quant.Q4_0, w, x, rows, kk)
			}
			if d := time.Since(start) / time.Duration(reps); d < best {
				best = d
			}
		}
		f.Close()
		gbs := float64(rows) * float64(rowBytes) / best.Seconds() / 1e9
		t.Logf("%-16s %6d x %5d (%6.1f MB): %6.2f GB/s  %v/call",
			sh.name, rows, kk, float64(rows*rowBytes)/1e6, gbs, best.Round(time.Microsecond))
	}
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var b [8]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	return string(b[i:])
}
