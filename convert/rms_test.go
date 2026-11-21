package convert

import (
	"math"
	"testing"
)

// TestRMSNormalisesEachRow holds the rms primitive to each row over its root
// mean square at an f32 epsilon, accumulated in float64 and rounded once, and
// refuses a packed tensor and an epsilon the file does not state.
func TestRMSNormalisesEachRow(t *testing.T) {
	env := mjEnv(map[string]any{"eps": 1e-6})
	vals := []float32{1, 2, 3, -4, 0.5, 2}
	v, err := mjRun(t, env, `[{"rms": "eps"}]`, mjF32([]uint64{2, 3}, vals...))
	if err != nil {
		t.Fatal(err)
	}
	got := mjFloats(v)
	for r := 0; r < 2; r++ {
		row := vals[r*3 : r*3+3]
		var ss float64
		for _, x := range row {
			ss += float64(x) * float64(x)
		}
		k := 1 / math.Sqrt(ss/3+float64(float32(1e-6)))
		for j, x := range row {
			if want := float32(float64(x) * k); got[r*3+j] != want {
				t.Fatalf("row %d element %d is %v, want %v", r, j, got[r*3+j], want)
			}
		}
	}
	q, err := mjRun(t, env, `[{"type": "q8"}]`, mjF32([]uint64{1, 32}, make([]float32, 32)...))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := mjRun(t, env, `[{"rms": "eps"}]`, q); err == nil {
		t.Error("the rms of a packed tensor was taken")
	}
	if _, err := mjRun(t, env, `[{"rms": "missing"}]`, mjF32([]uint64{3}, vals[:3]...)); err == nil {
		t.Error("an rms at an epsilon the file does not state was taken")
	}
}
