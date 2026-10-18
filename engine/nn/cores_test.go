//go:build amd64 && linux

package nn

import (
	"math/rand"
	"os"
	"testing"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/dev/bench"
)

// TestABCoreSets compares pool layouts paired and interleaved: measured in
// sequence the thermal drift swamps the result, so both pools are built up
// front and their calls alternated within one process.
func TestABCoreSets(t *testing.T) {
	requireRowMajorHost(t, quant.Q4_0)
	if testing.Short() {
		t.Skip("allocates 256 MB")
	}
	const k = 2048
	rowBytes := k / 32 * 18
	rows := 256 << 20 / rowBytes

	rng := rand.New(rand.NewSource(23))
	w := make([]byte, rows*rowBytes)
	for i := range w {
		w[i] = byte(rng.Intn(256))
	}
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	out := make([]float32, rows)

	mk := func(set string) *JIT {
		return NewJIT(k, rows, []quant.Type{quant.Q4_0}, WithSched(sched.WithCoreSet(set)))
	}
	base := mk("p")
	if base == nil {
		t.Skip("no fast tier")
	}
	defer base.Close()

	bytes := uint64(rows) * uint64(rowBytes)
	// The result is checked, because a declined kernel times nothing and
	// reads as an impossible bandwidth.
	run := func(f *JIT) func(int) {
		return func(n int) {
			for i := 0; i < n; i++ {
				if !f.MatVec(out, quant.Q4_0, w, x, rows, k) {
					panic("MatVec declined: this arm timed an empty loop")
				}
			}
		}
	}
	for _, set := range []string{"pe", "smt", "all"} {
		other := mk(set)
		if other == nil {
			continue
		}
		r := bench.AB(
			bench.Case{Name: "p(" + itoa(base.Workers()) + ")", BytesPerIter: bytes, Threads: base.Workers(), Fn: run(base)},
			bench.Case{Name: set + "(" + itoa(other.Workers()) + ")", BytesPerIter: bytes, Threads: other.Workers(), Fn: run(other)},
			20, 1)
		t.Logf("%-4s %v", set, r)
		other.Close()
	}
	os.Unsetenv("JITLLM_CORESET")
}
