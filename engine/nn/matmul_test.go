//go:build amd64 || arm64

package nn

import (
	"github.com/jitllm/jitllm/jit_old/cpu"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
)

// TestMatMulMatchesMatVec holds the batched prefill path to the decode path it
// replaces: for every token, MatMul must produce what MatVec produces.
//
// That is the right oracle rather than a fresh float64 reference. The two paths
// use different kernels, different activation layouts and different bias
// corrections -- MatVec reads the {scale, negSum} pairs while the GEMM uses a
// separate 128*sum(q) array -- so agreement between them exercises the whole
// contract. They quantize identically, so the only permitted difference is
// float32 summation order.
func TestMatMulMatchesMatVec(t *testing.T) {
	ran := 0
	for _, wt := range quant.PackedTypes {
		// Formats with no row-major kernel (Q5_0) get no subtest rather than a
		// skip; the count below stops the filter from emptying the sweep.
		if !cpu.Supported(wt) {
			continue
		}
		ran++
		t.Run(wt.String(), func(t *testing.T) {
			requireRowMajorHost(t, wt)
			matmulVsMatvec(t, wt)
		})
	}
	if ran == 0 {
		t.Fatal("no format has a row-major kernel; this gate proved nothing")
	}
}

func matmulVsMatvec(t *testing.T, wt quant.Type) {
	// One tile, so a failure names one kernel.
	f := NewJIT(16384, 20000, []quant.Type{wt}, WithTune(TuneOff))
	if f == nil {
		t.Skip("no generated tier")
	}
	defer f.Close()
	shapes := []struct{ rows, k, ntok int }{
		{64, 64, 5}, {256, 256, 8}, {2048, 2048, 5}, {2048, 2048, 24},
		{257, 2048, 33}, {2048, 5632, 7}, {64, 2048, 48},
		// gemma's actual shapes, including a 16384-row FFN and a k=16384
		// down-projection.
		{2048, 2048, 5}, {256, 2048, 5}, {16384, 2048, 5}, {2048, 16384, 5},
	}
	if wt.BlockElems() == 256 {
		// k must tile the 256-element super-block. tinyllama's shapes.
		shapes = []struct{ rows, k, ntok int }{
			{64, 256, 5}, {256, 512, 8}, {2048, 2048, 5}, {2048, 2048, 24},
			{257, 2048, 33}, {5632, 2048, 7}, {2048, 5632, 5}, {256, 2048, 48},
		}
	}
	for _, shape := range shapes {
		rng := rand.New(rand.NewSource(int64(shape.rows*31 + shape.k + shape.ntok)))
		bb := int(wt.BlockBytes())
		w := make([]byte, shape.rows*(shape.k/32)*bb)
		for i := range w {
			w[i] = byte(rng.Intn(256))
		}
		for off := 0; off+2 <= len(w); off += bb {
			w[off], w[off+1] = 0x00, 0x34
		}
		x := make([]float32, shape.ntok*shape.k)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		batched := make([]float32, shape.ntok*shape.rows)
		if !f.MatMul(batched, wt, w, x, shape.rows, shape.k, shape.ntok) {
			t.Fatalf("rows=%d k=%d ntok=%d: MatMul declined", shape.rows, shape.k, shape.ntok)
		}
		one := make([]float32, shape.rows)
		var sse, sy2 float64
		for n := 0; n < shape.ntok; n++ {
			f.NewInput()
			if !f.MatVec(one, wt, w, x[n*shape.k:(n+1)*shape.k], shape.rows, shape.k) {
				t.Fatal("MatVec declined")
			}
			for r := 0; r < shape.rows; r++ {
				d := float64(batched[n*shape.rows+r] - one[r])
				sse += d * d
				sy2 += float64(one[r]) * float64(one[r])
			}
		}
		nmse := 0.0
		if sy2 > 0 {
			nmse = sse / sy2
		}
		t.Logf("  rows=%-6d k=%-6d ntok=%-3d NMSE %.3e", shape.rows, shape.k, shape.ntok, nmse)
		if nmse > 1e-10 || math.IsNaN(nmse) {
			t.Errorf("rows=%d k=%d ntok=%d: NMSE %.3e between MatMul and MatVec",
				shape.rows, shape.k, shape.ntok, nmse)
		}
	}
}

// TestMatMulDeclinesPaddedBatch is the violation this policy is worth nothing
// without. The narrowest GEMM tile is eight tokens and a shorter batch is
// zero-padded to fill it, so at four real tokens the kernel does twice the work
// for the same answer and loses to the per-token loop. Declining is a supported
// answer and the caller loops MatVec.
func TestMatMulDeclinesPaddedBatch(t *testing.T) {
	requireRowMajorHost(t, quant.Q4_0)
	f := NewJIT(4096, 4096, []quant.Type{quant.Q8_0}, WithTune(TuneOff))
	if f == nil {
		t.Skip("no generated tier")
	}
	defer f.Close()
	const rows, k = 256, 256
	w := make([]byte, rows*(k/32)*int(quant.Q8_0.BlockBytes()))
	for _, ntok := range []int{2, 3, 4} {
		x := make([]float32, ntok*k)
		out := make([]float32, ntok*rows)
		if f.MatMul(out, quant.Q8_0, w, x, rows, k, ntok) {
			t.Errorf("ntok=%d: MatMul took a batch that is more than half padding", ntok)
		}
	}
	// And it must still take the width the tile actually fits.
	x := make([]float32, 8*k)
	out := make([]float32, 8*rows)
	if !f.MatMul(out, quant.Q8_0, w, x, rows, k, 8) {
		t.Error("ntok=8 fills an eight-wide tile exactly and should be taken")
	}
}
