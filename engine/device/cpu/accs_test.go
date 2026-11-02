//go:build jitllmbench && amd64 && linux

// This file is a measurement instrument, not a gate, so it is opt-in by build
// tag rather than run by every go test (AGENTS.md RULE 4): bench.AB's physics
// guard panics when a loaded box reads a low memory wall, which took the
// package's correctness gates down with it.
//
//	go test -tags jitllmbench -run <Name> ./jit/cpu

package cpu

import (
	"github.com/jitllm/jitllm/internal/oracle"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/dev/bench"
)

// TestABKQuantAccs measures whether the k-quant kernel is latency-bound.
//
// Each of the eight groups in a super-block issued two VFMADD231PS into one
// register, so a 144-byte Q4_K super-block carried 16 serially dependent FMAs.
// At 4-cycle FMA latency that is 64 cycles of dependency against a static issue
// floor around 40-45, which predicts a large win from splitting the chain and
// nothing at all if the loop is actually issue- or memory-bound.
//
// One core on purpose: this is the diagnostic, not the shipping number. At the
// shipping thread count the kernel may be DRAM-bound and a one-core win may not
// convert, so the end-to-end number is the one that decides.
func TestABKQuantAccs(t *testing.T) {
	if testing.Short() {
		t.Skip("allocates 128 MB")
	}
	for _, wt := range []quant.Type{quant.Q4_K, quant.Q5_K} {
		t.Run(wt.String(), func(t *testing.T) {
			requireRowMajorExec(t, wt)
			abAccs(t, wt)
		})
	}
}

func abAccs(t *testing.T, wt quant.Type) {
	const k = 2048
	nb := BlocksPerRow(wt, k) // super-blocks of 256
	rowBytes := RowBytes(wt, k)
	rows := 128 << 20 / rowBytes

	rng := rand.New(rand.NewSource(23))
	w := make([]byte, rows*rowBytes)
	for i := range w {
		w[i] = byte(rng.Intn(256))
	}
	// Sane fp16 d and dmin in every super-block. Random bits there seed NaN and
	// denormal scales, and a denormal multiplicand is a timing artefact with
	// nothing to do with the dependency chain under test.
	for off := 0; off+4 <= len(w); off += int(wt.BlockBytes()) {
		w[off], w[off+1] = 0x00, 0x34 // 0.25
		w[off+2], w[off+3] = 0x00, 0x30
	}

	x := make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	q := make([]int8, k)
	pairs := make([]float32, 2*(k/32))
	if err := oracle.QuantizeQ8(q, pairs, x, BiasC(wt)); err != nil {
		t.Fatal(err)
	}
	half := make([]float32, k/16)
	oracle.QuantizeHalfSums(half, q, BiasC(wt))
	konst := KernelConst(wt)
	out := make([]float32, rows)

	mk := func(accs int8) *Code {
		b, err := Emit(Spec{W: wt, Rows: 1, Accs: accs, Cols: 1})
		if err != nil {
			t.Fatal(err)
		}
		c := mustMap(t, b)
		return c
	}
	chunk := RowsPerCall(nb)
	run := func(c *Code) func(int) {
		return func(iters int) {
			for i := 0; i < iters; i++ {
				for r := 0; r < rows; r += chunk {
					n := min(chunk, rows-r)
					args := Args{Out: &out[r], W: &w[r*rowBytes], A: &q[0], AScale: &pairs[0],
						Rows: int64(n), K: int64(nb), RowStr: int64(rowBytes),
						Scr: &konst[0], AHalf: &half[0]}
					c.Call(&args)
				}
			}
		}
	}

	one := mk(1)
	defer one.Close()
	bytes := uint64(rows) * uint64(rowBytes)
	for _, accs := range []int8{2, 4} {
		c := mk(accs)
		r := bench.AB(
			bench.Case{Name: "accs1", BytesPerIter: bytes, Threads: 1, Fn: run(one)},
			bench.Case{Name: "accs" + string(rune('0'+accs)), BytesPerIter: bytes, Threads: 1, Fn: run(c)},
			21, 1)
		t.Logf("%s %s", wt, r)
		c.Close()
	}
}
