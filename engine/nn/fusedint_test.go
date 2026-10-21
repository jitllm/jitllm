//go:build amd64 || arm64

package nn

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestFusedIntFoldMatchesTheGEMMExactly holds the decode matvec's integer
// super-block fold (cpu.EmitA64PackedMatVecFusedWin) to the prefill GEMM's
// integer form bit for bit, token by token. The two are written to be the same
// arithmetic -- int32 sums of sc*dot, the same f32 bias chain, the same two
// final FMLAs -- and that is what lets a prefilled cache and a decoded one
// agree exactly on arm64.
//
// It also checks the fold was selected: the float form (GEMMExact) must differ
// from it somewhere, or the equality would prove nothing about the fold.
func TestFusedIntFoldMatchesTheGEMMExactly(t *testing.T) {
	compared := 0
	for _, gt := range []quant.Type{quant.Q4_K, quant.Q5_K, quant.Q3_K, quant.Q6_K} {
		if _, err := cpu.Native().PackedFusedWin(gt, 256); err != nil {
			t.Logf("%s: no integer-fold fused kernel on this host (%v)", gt, err)
			continue
		}
		for _, sh := range []struct{ nrows, k, ntok int }{{768, 768, 5}, {512, 2048, 3}, {48, 256, 7}} {
			nrows, k, ntok := sh.nrows, sh.k, sh.ntok
			pk := packOne(t, gt, nrows, k)
			rng := rand.New(rand.NewSource(int64(nrows + 3*k + int(gt))))
			x := make([]float32, ntok*k)
			for i := range x {
				x[i] = float32(rng.NormFloat64())
			}
			decode := func(opts ...Option) []float32 {
				j := NewJIT(k, nrows, []quant.Type{gt}, append(opts, WithTune(TuneOff), WithMatVecPick(2))...)
				if j == nil {
					t.Fatal("no JIT")
				}
				defer j.Close()
				if j.actWindow != 256 {
					t.Fatalf("%s: activation window %d, want 256", gt, j.actWindow)
				}
				out := make([]float32, ntok*nrows)
				for i := 0; i < ntok; i++ {
					j.NewInput()
					if !j.MatVecPacked(out[i*nrows:(i+1)*nrows], gt, pk, x[i*k:(i+1)*k], nrows, k) {
						t.Fatalf("%s: the matvec declined", gt)
					}
				}
				return out
			}
			got, float := decode(), decode(WithGEMMExact(true))

			j := NewJIT(k, nrows, []quant.Type{gt}, WithTune(TuneOff))
			want := make([]float32, ntok*nrows)
			if !j.MatMulPacked(want, gt, pk, x, nrows, k, ntok) || j.PackedGEMMCalls() != 1 {
				j.Close()
				t.Fatalf("%s %dx%d: the weight-stationary GEMM did not run", gt, nrows, k)
			}
			j.Close()

			same, finite := true, 0
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(float[i]) {
					same = false
				}
				if !math.IsNaN(float64(want[i])) && !math.IsInf(float64(want[i]), 0) {
					finite++
				}
			}
			if finite == 0 {
				t.Fatalf("%s: every GEMM value is non-finite -- a degenerate oracle", gt)
			}
			if same {
				t.Fatalf("%s %dx%d: the integer-fold matvec equals the float one bit for bit -- "+
					"the fold was not selected", gt, nrows, k)
			}
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("%s %dx%d: token %d row %d: decode %v, GEMM %v", gt, nrows, k,
						i/nrows, i%nrows, got[i], want[i])
				}
			}
			compared++
		}
	}
	if compared == 0 {
		t.Skip("no integer-fold fused kernel on this host")
	}
	t.Logf("%d (format, shape) pairs: decode == prefill GEMM bit for bit", compared)
}
