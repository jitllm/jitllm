package oracle_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/internal/oracle"
)

// TestMatMulIsMatVecPerRow: each activation row of MatMul is MatVec of that
// row, bit for bit, on every packed format, at row counts that are not a
// power of two; and a row swapped in x moves only that row's outputs.
func TestMatMulIsMatVecPerRow(t *testing.T) {
	const nrows, rows = 7, 5
	for _, f := range packedRefFormats() {
		k := int(f.g.BlockElems()) * 2
		w := make([]byte, uint64(nrows*k)/f.g.BlockElems()*f.g.BlockBytes())
		rng := rand.New(rand.NewSource(int64(len(f.name))))
		rng.Read(w)
		plant(t, f.g, w, 3)
		x := make([]float32, rows*k)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		got := make([]float32, rows*nrows)
		if err := oracle.MatMul(got, f.g, w, x, rows, nrows, k); err != nil {
			t.Fatalf("%s: %v", f.name, err)
		}
		for i := range rows {
			want := make([]float32, nrows)
			if err := oracle.MatVec(want, f.g, w, x[i*k:(i+1)*k], nrows, k); err != nil {
				t.Fatalf("%s: %v", f.name, err)
			}
			for r, v := range want {
				if math.Float32bits(got[i*nrows+r]) != math.Float32bits(v) || math.IsNaN(float64(v)) {
					t.Fatalf("%s: row %d output %d: MatMul %v, MatVec %v", f.name, i, r, got[i*nrows+r], v)
				}
			}
		}
		copy(x[2*k:3*k], x[4*k:5*k])
		moved := make([]float32, rows*nrows)
		if err := oracle.MatMul(moved, f.g, w, x, rows, nrows, k); err != nil {
			t.Fatal(err)
		}
		for i := range rows {
			for r := range nrows {
				same := moved[i*nrows+r] == got[i*nrows+r]
				if want := i != 2 || got[2*nrows+r] == got[4*nrows+r]; same != want {
					t.Fatalf("%s: row %d output %d changed %v after row 2 took row 4's activations", f.name, i, r, !same)
				}
				if i == 2 && moved[2*nrows+r] != got[4*nrows+r] {
					t.Fatalf("%s: row 2 output %d is %v, row 4's %v", f.name, r, moved[2*nrows+r], got[4*nrows+r])
				}
			}
		}
	}
}
