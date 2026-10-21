package nn

import (
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// TestMatVecPackedTakesALongerActivation hands MatVecPacked an activation
// longer than k -- what a hybrid's attention output projection passes, a
// scratch at the model width for a k of NHead*HeadDim -- and requires the
// kernel to run on its first k and agree with the exact-length call.
func TestMatVecPackedTakesALongerActivation(t *testing.T) {
	const rows, k = 32, 64
	q, _ := kernels.QuantOf(quant.Q4_0)
	src := make([]byte, rows*k/32*18)
	rng := rand.New(rand.NewSource(3))
	rng.Read(src)
	quant.PlantScales(quant.Q4_0, src, 0)
	qs, d, sc, err := kernels.PackWeights(q, src, rows, k)
	if err != nil {
		t.Fatal(err)
	}
	p := &Packed{QS: u32b(qs), D: u32b(d), SC: u32b(sc)}
	x := make([]float32, 4*k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	j := NewJIT(4*k, rows, []quant.Type{quant.Q4_0})
	defer j.Close()
	long, exact := make([]float32, rows), make([]float32, rows)
	if !j.MatVecPacked(long, quant.Q4_0, p, x, rows, k) {
		t.Fatal("declined an activation longer than k")
	}
	j.NewInput()
	if !j.MatVecPacked(exact, quant.Q4_0, p, x[:k], rows, k) {
		t.Fatal("declined the exact-length activation")
	}
	for r := range long {
		if long[r] != exact[r] {
			t.Fatalf("row %d: %v with the long activation, %v with x[:k]", r, long[r], exact[r])
		}
	}
}
