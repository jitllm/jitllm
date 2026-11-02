//go:build amd64 || arm64

package nn

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
)

// TestMatMulFloatMatchesMatVec: the float GEMM over a batch, tiled by
// cpu.MaxFloatTokens with a ragged last tile and split over the pool by rows,
// against the float matvec per token, with a guard after the output; and the
// GEMM was what ran (its call counter).
func TestMatMulFloatMatchesMatVec(t *testing.T) {
	for _, typ := range []quant.Type{quant.F32, quant.F16, quant.BF16} {
		const nrows, k = 37, 100
		j := NewJIT(k, nrows, []quant.Type{typ, quant.F32})
		if j == nil {
			t.Fatal("no code generator")
		}
		rng := rand.New(rand.NewSource(3))
		es := int(typ.BlockBytes())
		w := make([]byte, nrows*k*es)
		for i := 0; i < nrows*k; i++ {
			v := float32(rng.NormFloat64())
			switch typ {
			case quant.F32:
				*(*float32)(unsafe.Pointer(&w[4*i])) = v
			case quant.F16:
				*(*uint16)(unsafe.Pointer(&w[2*i])) = quant.EncodeHalf(v)
			default:
				*(*uint16)(unsafe.Pointer(&w[2*i])) = uint16(math.Float32bits(v) >> 16)
			}
		}
		for _, ntok := range []int{2, 3, 7, 8, 9, 15, 16, 17, 20} {
			x := make([]float32, ntok*k)
			for i := range x {
				x[i] = float32(rng.NormFloat64())
			}
			got := make([]float32, ntok*nrows+5)
			for i := range got {
				got[i] = -7
			}
			calls := j.FloatMatMulCalls()
			if !j.MatMulFloat(got, typ, w, x, nrows, k, ntok) {
				t.Fatalf("%s ntok=%d: declined", typ, ntok)
			}
			if j.FloatMatMulCalls() != calls+1 {
				t.Fatalf("%s ntok=%d: the GEMM's counter did not move", typ, ntok)
			}
			want := make([]float32, nrows)
			for tok := 0; tok < ntok; tok++ {
				if !j.MatVec(want, typ, w, x[tok*k:(tok+1)*k], nrows, k) {
					t.Fatal("no matvec")
				}
				for r := range want {
					if d := math.Abs(float64(got[tok*nrows+r] - want[r])); d > 1e-4*(1+math.Abs(float64(want[r]))) {
						t.Fatalf("%s ntok=%d token %d row %d: %v, the matvec %v", typ, ntok, tok, r, got[tok*nrows+r], want[r])
					}
				}
			}
			for i := ntok * nrows; i < len(got); i++ {
				if got[i] != -7 {
					t.Fatalf("%s ntok=%d: wrote past the output at %d", typ, ntok, i)
				}
			}
		}
		j.Close()
	}
}
