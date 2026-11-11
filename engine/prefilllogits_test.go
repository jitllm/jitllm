package engine

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/model"
)

// TestPrefillLogitsMatchesForward: PrefillLogits' row i is the logits plain decode
// has after token i -- a prefill of the first token and Forward over the
// rest -- within the prompt forms' band of the decode forms, every row.
func TestPrefillLogitsMatchesForward(t *testing.T) {
	if _, err := os.Stat(models()["stories15M"]); err != nil {
		t.Fatalf("MODEL MISSING: %v", err)
	}
	m, err := model.Open(jlmOf(t, models()["stories15M"]))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := []int32{1, 450, 3014, 471, 263, 931, 29892, 727, 471, 263}
	nv, nd := m.Cfg.NVocab, m.Cfg.NEmbd
	a := testInstance(t, m, 64)
	defer a.Close()
	rows, hid := make([]float32, len(ids)*nv), make([]float32, len(ids)*nd)
	if err := a.PrefillLogits(ids, rows, hid); err != nil {
		t.Fatal(err)
	}
	b := testInstance(t, m, 64)
	defer b.Close()
	lg, err := b.Prefill(ids[:1])
	if err != nil {
		t.Fatal(err)
	}
	for i := range ids {
		if i > 0 {
			if lg, err = b.Forward(ids[i]); err != nil {
				t.Fatal(err)
			}
		}
		if e, worst := nmseOf(rows[i*nv:(i+1)*nv], lg); e > 1e-3 || e != e {
			t.Fatalf("row %d: NMSE %.3g (worst %.3g) against plain decode", i, e, worst)
		}
	}
}
