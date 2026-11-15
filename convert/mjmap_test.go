package convert

import (
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestModelJSONMapOps holds every operation a map composes to its arithmetic:
// each function on one tensor, subtraction and division over two, a third
// tensor, a config value with its default; and refuses a ?? over a tensor's
// element (it reads the config) and a tensor not in hand.
func TestModelJSONMapOps(t *testing.T) {
	val := func(f ...float32) mjVal {
		return mjVal{shape: []uint64{uint64(len(f))}, typ: jlm.TypeF32, data: f32AsBytes(f)}
	}
	x, y, z := val(1, 4, 9, 0.25), val(2, 2, 3, 0.5), val(10, 20, 30, 40)
	env := exprEnv(func(p string) any {
		if p == "eps" {
			return 0.5
		}
		return nil
	})
	each := func(f func(i int) float64) []float64 {
		out := make([]float64, 4)
		for i := range out {
			out[i] = f(i)
		}
		return out
	}
	xs, ys, zs := mjFloats(x), mjFloats(y), mjFloats(z)
	for _, r := range []struct {
		src  string
		vs   []mjVal
		want []float64
	}{
		{"sqrt(x)", []mjVal{x}, []float64{1, 2, 3, 0.5}},
		{"log(x)", []mjVal{x}, each(func(i int) float64 { return math.Log(float64(xs[i])) })},
		{"tanh(x)", []mjVal{x}, each(func(i int) float64 { return math.Tanh(float64(xs[i])) })},
		{"softplus(x)", []mjVal{x}, each(func(i int) float64 { return math.Log1p(math.Exp(float64(xs[i]))) })},
		{"exp(-x)", []mjVal{x}, each(func(i int) float64 { return math.Exp(-float64(xs[i])) })},
		{"x - y", []mjVal{x, y}, each(func(i int) float64 { return float64(xs[i]) - float64(ys[i]) })},
		{"x / y", []mjVal{x, y}, each(func(i int) float64 { return float64(xs[i]) / float64(ys[i]) })},
		{"x * y + z", []mjVal{x, y, z}, each(func(i int) float64 { return float64(xs[i])*float64(ys[i]) + float64(zs[i]) })},
		{"x + (eps ?? 1)", []mjVal{x}, each(func(i int) float64 { return float64(xs[i]) + 0.5 })},
		{"x + (missing ?? 1)", []mjVal{x}, each(func(i int) float64 { return float64(xs[i]) + 1 })},
	} {
		out, err := mjMap(r.src, env, r.vs)
		if err != nil {
			t.Errorf("%s: %v", r.src, err)
			continue
		}
		got := mjFloats(out)
		for i := range r.want {
			if got[i] != float32(r.want[i]) {
				t.Errorf("%s: element %d is %v, want %v", r.src, i, got[i], float32(r.want[i]))
				break
			}
		}
	}
	for _, r := range []struct {
		src, err string
		vs       []mjVal
	}{
		{"x ?? 1", "reads a tensor", []mjVal{x}},
		{"x + z", "names tensor 3 of 2", []mjVal{x, y}},
	} {
		if _, err := mjMap(r.src, env, r.vs); err == nil || !strings.Contains(err.Error(), r.err) {
			t.Errorf("%s: %v, want an error saying %q", r.src, err, r.err)
		}
	}
}
