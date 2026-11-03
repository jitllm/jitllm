package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestRaggedTowerProductsMatchTheOracle holds the products of a matrix whose
// k is not a whole number of blocks (the openjev test model's tower, 144
// wide, packed at 160) to a float64 dot product over its 144 columns: one row
// through mv and three through mm, each row's tail quantized as zeros (nn's
// ragged quantizer, whose violation nn.TestRaggedProductsMatchThePaddedRows
// holds). The packed columns past 144 must be zero.
func TestRaggedTowerProductsMatchTheOracle(t *testing.T) {
	text, proj := testmodels.Path("openjev/tinyopenjev-for-testing-Q8_0.gguf"), testmodels.Path("openjev/mmproj-tinyopenjev-for-testing-Q8_0.gguf")
	if !vlmAvailable(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (RULE 11)", text)
	}
	m, err := Open(jlmOfPair(t, text, proj), noTune())
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	li := int(m.container.H.NBlocks)
	if err := m.pageIn(li); err != nil {
		t.Fatal(err)
	}
	l := &m.layers[li]
	for _, w := range []tensor{l.wq, l.wk, l.wv, l.wo, l.up} {
		if w.k != 144 || w.kp != 160 {
			t.Fatalf("%s is read at k=%d packed at %d, want 144 at 160: this gate proves nothing", w.name(), w.k, w.kp)
		}
	}
	w := l.wq
	ref := make([]float32, w.rows*w.k)
	row := make([]float32, w.kp)
	for r := 0; r < w.rows; r++ {
		nn.RowPacked32JIT(row, w.typ, w.packed, r, w.rows, w.kp)
		for c := w.k; c < w.kp; c++ {
			if row[c] != 0 {
				t.Fatalf("row %d column %d of the padding is %g, not zero", r, c, row[c])
			}
		}
		copy(ref[r*w.k:], row[:w.k])
	}
	const n = 3
	x := make([]float32, n*w.k)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*0.37)) + 0.5
	}
	want := make([]float64, n*w.rows)
	scale := 0.0
	for tk := 0; tk < n; tk++ {
		for r := 0; r < w.rows; r++ {
			d := 0.0
			for c := 0; c < w.k; c++ {
				d += float64(ref[r*w.k+c]) * float64(x[tk*w.k+c])
			}
			want[tk*w.rows+r] = d
			scale = max(scale, math.Abs(d))
		}
	}
	s := m.NewInstance(64)
	defer s.Close()
	worst := func(got []float32) float64 {
		e := 0.0
		for i, g := range got {
			e = max(e, math.Abs(float64(g)-want[i]))
		}
		return e / scale
	}
	// Q8_0 activations: a few parts in a thousand of the largest product.
	const bound = 0.01
	one := make([]float32, w.rows)
	if err := s.mv(one, w, x[:w.k]); err != nil {
		t.Fatal(err)
	}
	all := make([]float32, n*w.rows)
	if err := s.mm(all, w, x, n); err != nil {
		t.Fatal(err)
	}
	e1, en := worst(one), worst(all)
	t.Logf("relative to the largest product: mv %.5f, mm %.5f", e1, en)
	if !(e1 <= bound && en <= bound) { // a NaN fails too
		t.Errorf("the padded products are %.5f and %.5f from the oracle, past %g", e1, en, bound)
	}
}
