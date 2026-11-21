package convert

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestLinearPadAndWeightedRMS holds the linear, pad and weighted rms
// primitives to the reference arithmetic, and a float64 value to a type step
// before it is stored.
func TestLinearPadAndWeightedRMS(t *testing.T) {
	env := mjEnv(map[string]any{"eps": 1e-6})
	x := []float32{1, -2, 3, 0.5, 4, -1}
	w := []float32{0.5, 1, -1, 2, 0, 3}
	v, err := mjRun(t, env, `[{"linear": true}]`, mjF32([]uint64{2, 3}, x...), mjF32([]uint64{6}, w...))
	if err != nil {
		t.Fatal(err)
	}
	got := mjFloats(v)
	if len(v.shape) != 2 || v.shape[0] != 2 || v.shape[1] != 2 {
		t.Fatalf("a [2][3] by [2][3] linear is %v", v.shape)
	}
	for i := range 2 {
		for r := range 2 {
			var acc float64
			for j := range 3 {
				acc += float64(w[r*3+j]) * float64(x[i*3+j])
			}
			if want := float32(acc); got[i*2+r] != want {
				t.Fatalf("row %d output %d is %v, want %v", i, r, got[i*2+r], want)
			}
		}
	}
	nw := []float32{2, -1, 0.5}
	v, err = mjRun(t, env, `[{"rms": "eps"}]`, mjF32([]uint64{6}, x...), mjF32([]uint64{3}, nw...))
	if err != nil {
		t.Fatal(err)
	}
	got = mjFloats(v)
	for r := range 2 {
		var ss float64
		for _, y := range x[r*3 : r*3+3] {
			ss += float64(y) * float64(y)
		}
		k := 1 / math.Sqrt(ss/3+float64(float32(1e-6)))
		for j := range 3 {
			if want := float32(float64(x[r*3+j]) * (k * float64(nw[j]))); got[r*3+j] != want {
				t.Fatalf("weighted row %d element %d is %v, want %v", r, j, got[r*3+j], want)
			}
		}
	}
	v, err = mjRun(t, env, `[{"pad": [1, 4]}]`, mjF32([]uint64{2, 3}, x...))
	if err != nil {
		t.Fatal(err)
	}
	if got, want := mjFloats(v), []float32{1, -2, 3, 0, 0.5, 4, -1, 0}; len(v.shape) != 2 || v.shape[1] != 4 || !slices.Equal(got, want) {
		t.Fatalf("[2][3] padded to 4 is %v %v, want %v", v.shape, got, want)
	}
	for _, bad := range []struct {
		what, pipe string
		in         []mjVal
	}{
		{"a linear of one tensor", `[{"linear": true}]`, []mjVal{mjF32([]uint64{2, 3}, x...)}},
		{"a linear by a weight its k does not divide", `[{"linear": true}]`, []mjVal{mjF32([]uint64{2, 3}, x...), mjF32([]uint64{4}, w[:4]...)}},
		{"a weighted rms whose rows the weight does not divide", `[{"rms": "eps"}]`, []mjVal{mjF32([]uint64{6}, x...), mjF32([]uint64{4}, w[:4]...)}},
		{"a pad to a multiple of 0", `[{"pad": [1, 0]}]`, []mjVal{mjF32([]uint64{2, 3}, x...)}},
		{"a pad of an axis past the shape", `[{"pad": [2, 4]}]`, []mjVal{mjF32([]uint64{2, 3}, x...)}},
	} {
		if _, err := mjRun(t, env, bad.pipe, bad.in...); err == nil {
			t.Errorf("%s ran", bad.what)
		}
	}
	wide := mjVal{shape: []uint64{2}, typ: mjF64, data: mjStore([]float64{1.0 / 3, 2}, mjF64)}
	v, err = mjRun(t, env, `[{"map": "x * 3"}]`, wide)
	if err != nil {
		t.Fatal(err)
	}
	if v.typ != mjF64 || mjFloats64(v)[0] != 1.0/3*3 {
		t.Fatalf("a float64 map is %v %v", v.typ, mjFloats64(v))
	}
	tensor := jlm.Tensor{Name: "t"}
	if err := mjSetShape(&tensor, v); err == nil {
		t.Error("a float64 value was stored")
	}
	if v, err = mjRun(t, env, `[{"type": "f32"}]`, v); err != nil || v.typ != jlm.TypeF32 || mjFloats(v)[0] != float32(1.0/3*3) {
		t.Fatalf("a float64 value typed f32 is %v %v (%v)", v.typ, mjFloats(v), err)
	}
}
