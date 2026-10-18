//go:build jitllmbench && amd64 && linux

// This file is a measurement instrument, not a gate, so it is opt-in by build
// tag rather than by a t.Skip that would put a green line in every run:
//
//	go test -tags jitllmbench -run <Name> ./<pkg>
//
// Any JITLLM_* variable the test reads still selects its parameters.

package nn

import (
	"math/rand"
	"os"
	"testing"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/dev/bench"
)

// TestABInterleave settles whether 4-row interleaving is a win on a
// DRAM-resident working set, paired and interleaved.
func TestABInterleave(t *testing.T) {
	requireRowMajorHost(t, quant.Q4_0)
	if testing.Short() {
		t.Skip("allocates 256 MB")
	}
	if err := bench.Guard(false); err != nil {
		t.Skipf("box is not quiet enough to measure: %v", err)
	}
	const k = 2048
	const rowBytes = k / 32 * 18
	const rows = 256 << 20 / rowBytes // 256 MB: cannot sit in a 24 MB L3

	rng := rand.New(rand.NewSource(11))
	w := make([]byte, rows*rowBytes)
	for i := range w {
		w[i] = byte(rng.Intn(256))
	}
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	out := make([]float32, rows)
	os.Unsetenv("JITLLM_CORES")

	f := NewJIT(k, rows, []quant.Type{quant.Q4_0})
	if f == nil {
		t.Skip("no fast tier")
	}
	defer f.Close()
	saved := f.code4[quant.Q4_0]

	one := bench.Case{Name: "1-row", BytesPerIter: uint64(rows) * rowBytes, Threads: f.Workers(),
		Fn: func(n int) {
			f.code4[quant.Q4_0] = nil // force the single-row path
			for i := 0; i < n; i++ {
				f.MatVec(out, quant.Q4_0, w, x, rows, k)
			}
		}}
	four := bench.Case{Name: "4-row", BytesPerIter: uint64(rows) * rowBytes, Threads: f.Workers(),
		Fn: func(n int) {
			f.code4[quant.Q4_0] = saved
			for i := 0; i < n; i++ {
				f.MatVec(out, quant.Q4_0, w, x, rows, k)
			}
		}}

	r := bench.AB(one, four, 20, 1)
	t.Logf("%d cores, %d MB DRAM-resident: %v", f.Workers(), rows*rowBytes>>20, r)
	if !r.Stable() {
		t.Logf("spread too wide to conclude anything")
	}
	f.code4[quant.Q4_0] = saved
}
