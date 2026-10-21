//go:build amd64 || arm64

package cpu_test

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestFloatMatVecEveryK runs the row-major F32, F16 and BF16 kernels at k that
// exercise each of their three loops alone and together -- single elements
// only (1, 3, 7), quads or octets only (8, 16), groups only (32, 64), and all
// three (13, 45, 100, 2047) -- against a float64 sum over the weights exactly
// as stored. The output carries a guard, and rows past Rows must not move.
// Small k (a tiny HF model's k=8 or 16) must be served too.
func TestFloatMatVecEveryK(t *testing.T) {
	for _, typ := range []quant.Type{quant.F32, quant.F16, quant.BF16} {
		// The host tier's kernel: EmitNative on the primary tier, the SSE float
		// matvec (same Args) on an SSE host -- see TestF32MatVecMatchesReference.
		code, err := cpu.Native().RowMajor(cpu.Spec{W: typ, Rows: 1, Accs: 1, Cols: 1})
		if err != nil {
			t.Fatalf("%s: %v", typ, err)
		}
		c, err := cpu.Map(code)
		if err != nil {
			t.Fatal(err)
		}
		es := int(typ.BlockBytes())
		for _, k := range []int{1, 3, 7, 8, 16, 32, 64, 13, 45, 100, 2047} {
			const rows = 5
			rng := rand.New(rand.NewSource(int64(k)))
			w := make([]byte, rows*k*es)
			wv := make([]float64, rows*k) // the weight as the kernel must read it
			for i := range wv {
				v := float32(rng.NormFloat64())
				switch typ {
				case quant.F32:
					*(*float32)(unsafe.Pointer(&w[4*i])) = v
					wv[i] = float64(v)
				case quant.F16:
					h := quant.EncodeHalf(v)
					*(*uint16)(unsafe.Pointer(&w[2*i])) = h
					wv[i] = quant.DecodeHalf(h)
				default:
					b := uint16(math.Float32bits(v) >> 16)
					*(*uint16)(unsafe.Pointer(&w[2*i])) = b
					wv[i] = float64(math.Float32frombits(uint32(b) << 16))
				}
			}
			x := make([]float32, k)
			for i := range x {
				x[i] = float32(rng.NormFloat64())
			}
			out := make([]float32, rows+4)
			for i := range out {
				out[i] = -7
			}
			c.Call(&cpu.Args{Out: &out[0], W: &w[0], A: (*int8)(unsafe.Pointer(&x[0])),
				Rows: rows, K: int64(k), RowStr: int64(k * es)})
			for r := 0; r < rows; r++ {
				var want float64
				for i := 0; i < k; i++ {
					want += wv[r*k+i] * float64(x[i])
				}
				if d := math.Abs(float64(out[r]) - want); d > 1e-4*(1+math.Abs(want))*math.Sqrt(float64(k)) {
					t.Fatalf("%s k=%d row %d: %v, want %v", typ, k, r, out[r], want)
				}
			}
			for i := rows; i < len(out); i++ {
				if out[i] != -7 {
					t.Fatalf("%s k=%d: wrote row %d past Rows", typ, k, i)
				}
			}
		}
		c.Close()
	}
}
