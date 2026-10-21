//go:build amd64 || arm64

package nn

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// TestRaggedProductsMatchThePaddedRows holds a packed product over a ragged
// activation -- kx floats under a weight packed at the next whole outer
// block, the quantizer taking the tail (ragged.go) -- bit for bit to the
// same product over the rows padded with zeros to the packed width, for
// every packed type, one row (MatVecPacked) and several (MatMulPacked), at
// widths either side of a block. The violation fills the staged tail with
// NaN, which must reach the products.
func TestRaggedProductsMatchThePaddedRows(t *testing.T) { raggedProducts(t) }

func raggedProducts(t *testing.T) {
	ran := 0
	for _, gt := range quant.PackedTypes {
		for _, kx := range []int{7, 31, 33, 65, 143, 144, 145, 200, 257} {
			k := raggedK(gt, kx)
			if k == kx {
				continue
			}
			const nrows, ntok = 64, 3
			rng := rand.New(rand.NewSource(int64(kx)))
			q, _ := kernels.QuantOf(gt)
			src := make([]byte, uint64(nrows*k)/gt.BlockElems()*gt.BlockBytes())
			for i := range src {
				src[i] = byte(rng.Intn(256))
			}
			if !quant.PlantScales(gt, src, 0) {
				t.Fatalf("%s: no scale planter", gt)
			}
			qs, d, sc, err := kernels.PackWeights(q, src, nrows, k)
			if err != nil {
				t.Fatal(err)
			}
			pk := &Packed{QS: u32b(qs), D: u32b(d), SC: u32b(sc)}
			x := make([]float32, ntok*kx)
			for i := range x {
				x[i] = float32(rng.NormFloat64())
			}
			padded := make([]float32, ntok*k)
			for r := 0; r < ntok; r++ {
				copy(padded[r*k:], x[r*kx:(r+1)*kx])
			}
			j := NewJIT(k, nrows, []quant.Type{gt}, WithGEMMTokens(-1), WithTune(TuneOff), WithGEMMExact(true))
			if j == nil {
				t.Fatal("no JIT")
			}
			one := func(x []float32, k int) []float32 {
				j.NewInput()
				out := make([]float32, nrows)
				if !j.MatVecPacked(out, gt, pk, x, nrows, k) {
					t.Fatalf("%s k=%d: MatVecPacked declined", gt, k)
				}
				return out
			}
			many := func(x []float32, k int) []float32 {
				j.NewInput()
				out := make([]float32, ntok*nrows)
				if !j.MatMulPacked(out, gt, pk, x, nrows, k, ntok) {
					t.Fatalf("%s k=%d: MatMulPacked declined", gt, k)
				}
				return out
			}
			before := RaggedRows.Load()
			same := func(what string, got, want []float32) {
				for i := range want {
					if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						t.Fatalf("%s %s kx=%d (packed %d): output %d is %g, the padded rows' %g",
							gt, what, kx, k, i, got[i], want[i])
					}
				}
			}
			same("one row", one(x[:kx], kx), one(padded[:k], k))
			same("rows", many(x, kx), many(padded, k))
			if RaggedRows.Load() == before {
				t.Fatalf("%s kx=%d: no row's tail was staged, so the ragged path did not run", gt, kx)
			}
			j.ragViolation = true
			bad := many(x, kx)
			j.ragViolation = false
			nan := 0
			for _, v := range bad {
				if math.IsNaN(float64(v)) {
					nan++
				}
			}
			if nan == 0 {
				t.Errorf("%s kx=%d: a NaN tail reached no product: the gate does not see the tail", gt, kx)
			}
			j.Close()
			ran++
		}
	}
	if ran == 0 {
		t.Fatal("no ragged shape ran")
	}
}
