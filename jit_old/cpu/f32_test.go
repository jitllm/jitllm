package cpu_test

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestF32MatVecMatchesReference holds the generated F32 kernel to the same bar
// as every other kernel in jit_old/cpu: NMSE against a float64 evaluation of the
// same arithmetic. The other kernel tests name their formats explicitly, so
// F32 needs its own.
func TestF32MatVecMatchesReference(t *testing.T) {
	for _, s := range []struct{ rows, k int }{
		{1, 32}, {128, 2048}, // the 30B's router
		{4, 1024},  // the 4x0.6B's
		{8, 4096},  // Mixtral's
		{37, 512},  // rows not a multiple of anything
		{2048, 64}, // many rows, short k
	} {
		// The host tier's row-major emitter, not Emit: Emit is the x86 AVX2
		// emitter and exists on every arch, so calling it would generate AVX2 on
		// an arm64 or SSE-only host. Native().RowMajor is EmitNative on the primary
		// tier (the bytes this gate always tested) and the SSE float matvec, on
		// the same Args, on an SSE host.
		code, err := cpu.Native().RowMajor(cpu.Spec{W: quant.F32, Rows: 1, Accs: 1, Cols: 1})
		if err != nil {
			t.Fatalf("Emit: %v", err)
		}
		c, err := cpu.Map(code)
		if err != nil {
			t.Fatalf("Map: %v", err)
		}
		defer c.Close()

		rng := rand.New(rand.NewSource(int64(s.rows*7919 + s.k)))
		w := make([]byte, s.rows*s.k*4)
		fw := unsafe.Slice((*float32)(unsafe.Pointer(&w[0])), s.rows*s.k)
		for i := range fw {
			// Heavy-tailed: uniform alone hides cancellation.
			fw[i] = float32(rng.NormFloat64() / (0.2 + math.Abs(rng.NormFloat64())))
		}
		x := make([]float32, s.k)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		out := make([]float32, s.rows)

		args := cpu.Args{
			Out: &out[0], W: &w[0],
			A:      (*int8)(unsafe.Pointer(&x[0])),
			Rows:   int64(s.rows),
			K:      int64(s.k), // BlocksPerRow, as every other kernel takes it
			RowStr: int64(s.k * 4),
		}
		c.Call(&args)

		var sse, sy2 float64
		for r := 0; r < s.rows; r++ {
			var want float64
			for i := 0; i < s.k; i++ {
				want += float64(fw[r*s.k+i]) * float64(x[i])
			}
			d := float64(out[r]) - want
			sse += d * d
			sy2 += want * want
		}
		nmse := 0.0
		if sy2 > 0 {
			nmse = sse / sy2
		}
		// 1e-10, not the 1e-14 an f64 kernel would reach: the accumulator is
		// float32 by design, so ~1e-13 is the floor, and 1e-10 leaves room
		// for cancellation without letting a real error through.
		if nmse > 1e-10 || math.IsNaN(nmse) {
			t.Errorf("F32 rows=%d k=%d: NMSE %.3e exceeds 1e-10", s.rows, s.k, nmse)
		}
	}
}
