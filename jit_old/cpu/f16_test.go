package cpu_test

import (
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestF16MatVecMatchesReference gates the F16 matvec against float64.
//
// Mixtral's `ffn_gate_inp` is F16 where other mixtures' routers are F32; the
// shapes are the real ones (8x4096 is Mixtral's router).
func TestF16MatVecMatchesReference(t *testing.T) {
	for _, s := range []struct{ rows, k int }{
		{8, 4096}, // Mixtral's router
		{1, 32},   // the narrowest legal shape
		{128, 2048},
		{37, 512},  // rows not a multiple of anything
		{2048, 64}, // many rows, short k
	} {
		// The host tier's row-major emitter, not Emit: Emit is the x86 AVX2
		// emitter and exists on every arch, so calling it would generate AVX2 on
		// an arm64 or SSE-only host. Native().RowMajor is EmitNative on the primary
		// tier (the bytes this gate always tested) and the SSE float matvec, on
		// the same Args, on an SSE host.
		code, err := cpu.Native().RowMajor(cpu.Spec{W: quant.F16, Rows: 1, Accs: 1, Cols: 1})
		if err != nil {
			t.Fatalf("Emit: %v", err)
		}
		c, err := cpu.Map(code)
		if err != nil {
			t.Fatalf("Map: %v", err)
		}

		rng := rand.New(rand.NewSource(int64(s.rows*7919 + s.k)))
		w := make([]byte, s.rows*s.k*2)
		hw := unsafe.Slice((*uint16)(unsafe.Pointer(&w[0])), s.rows*s.k)
		want := make([]float64, s.rows*s.k) // the f16 values EXACTLY, as f64
		for i := range hw {
			// Random f16 bit patterns, not f32 rounded down: the reference is
			// exactly what the kernel reads, and subnormals and the whole
			// exponent range are swept. Exponent 0x1F (NaN, Inf) is excluded.
			b := uint16(rng.Intn(1 << 16))
			if (b>>10)&0x1F == 0x1F {
				b &^= 1 << 14
			}
			hw[i] = b
			want[i] = quant.DecodeHalf(b)
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
			RowStr: int64(s.k * 2),
		}
		c.Call(&args)
		c.Close()

		var sse, sy2 float64
		for r := 0; r < s.rows; r++ {
			var ref float64
			for i := 0; i < s.k; i++ {
				ref += want[r*s.k+i] * float64(x[i])
			}
			d := float64(out[r]) - ref
			sse, sy2 = sse+d*d, sy2+ref*ref
		}
		if nmse := sse / sy2; nmse > 1e-12 {
			t.Errorf("%dx%d: NMSE %.3e against the float64 reference", s.rows, s.k, nmse)
		}
	}
}
