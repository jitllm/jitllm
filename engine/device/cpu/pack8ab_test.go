//go:build jitllmbench && amd64 && linux

// This file is a measurement instrument, not a gate, so it is opt-in by build
// tag rather than run by every go test (AGENTS.md RULE 4): bench.AB's physics
// guard panics when a loaded box reads a low memory wall, which took the
// package's correctness gates down with it.
//
//	go test -tags jitllmbench -run <Name> ./jit/cpu

package cpu

import (
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/dev/bench"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
)

// TestABPack8 measures interleave width on a DRAM-resident set, paired.
func TestABPack8(t *testing.T) {
	if testing.Short() {
		t.Skip("allocates 128 MB")
	}
	if ggufDeclined(t, quant.Q8_0) {
		return
	}
	const k = 2048
	nb := k / 32
	rowBytes := RowBytes(quant.Q8_0, k)
	rows := 128 << 20 / rowBytes

	rng := rand.New(rand.NewSource(17))
	w := make([]byte, rows*rowBytes)
	for i := range w {
		w[i] = byte(rng.Intn(256))
	}
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	q := make([]int8, k)
	pairs := make([]float32, 2*nb)
	oracle.QuantizeQ8(q, pairs, x, BiasC(quant.Q8_0))
	konst := KernelConst(quant.Q8_0)
	out := make([]float32, rows)

	mk := func(width int) (*Code, int) {
		var b []byte
		var err error
		if width == 1 {
			b, err = Emit(Spec{W: quant.Q8_0, Rows: 1, Accs: 1, Cols: 1})
		} else {
			b, err = EmitInterleaved(quant.Q8_0, width, nb)
		}
		if err != nil {
			t.Fatal(err)
		}
		c := mustMap(t, b)
		return c, len(b)
	}
	one, n1 := mk(1)
	defer one.Close()
	eight, n8 := mk(8)
	defer eight.Close()
	t.Logf("1-row %d bytes, 8-row %d bytes", n1, n8)

	chunk := RowsPerCall(nb) / 8 * 8
	run := func(c *Code, width int) func(int) {
		return func(iters int) {
			for i := 0; i < iters; i++ {
				for r := 0; r < rows; r += chunk {
					n := min(chunk, rows-r)
					n = n / width * width
					if n == 0 {
						continue
					}
					args := Args{Out: &out[r], W: &w[r*rowBytes], A: &q[0], AScale: &pairs[0],
						Rows: int64(n / width), K: int64(nb), RowStr: int64(rowBytes), Scr: &konst[0]}
					c.Call(&args)
				}
			}
		}
	}
	bytes := uint64(rows) * uint64(rowBytes)
	r := bench.AB(
		bench.Case{Name: "1-row", BytesPerIter: bytes, Threads: 1, Fn: run(one, 1)},
		bench.Case{Name: "8-row (pack8)", BytesPerIter: bytes, Threads: 1, Fn: run(eight, 8)},
		20, 1)
	t.Logf("1 core, %d MB DRAM-resident: %v", bytes>>20, r)
}
