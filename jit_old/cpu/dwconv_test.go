//go:build amd64 || arm64

package cpu_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// dwShapes are the depthwise rows the gates sweep: every filter MobileNet-V5
// runs (3 and 5, at strides 1 and 2), a 1x1, ragged channel counts on both
// hosts' vector widths, and an output row whose window reaches the padded
// row's last sample.
func dwShapes() []cpu.DWShape {
	var out []cpu.DWShape
	for _, k := range []int{1, 3, 5} {
		for _, st := range []int{1, 2} {
			for _, ch := range []int{1, 3, 4, 5, 7, 8, 9, 13, 16, 17, 33, 64} {
				for _, wout := range []int{1, 4, 7} {
					out = append(out, cpu.DWShape{K: k, Stride: st, WP: (wout-1)*st + k, Chans: ch, WOut: wout})
				}
			}
		}
	}
	return out
}

// dwRef is the depthwise row written out plainly, in float64.
func dwRef(s cpu.DWShape, in, w []float32) []float64 {
	out := make([]float64, s.WOut*s.Chans)
	for x := 0; x < s.WOut; x++ {
		for c := 0; c < s.Chans; c++ {
			var acc float64
			for ky := 0; ky < s.K; ky++ {
				for kx := 0; kx < s.K; kx++ {
					acc += float64(in[(ky*s.WP+x*s.Stride+kx)*s.Chans+c]) * float64(w[(ky*s.K+kx)*s.Chans+c])
				}
			}
			out[x*s.Chans+c] = acc
		}
	}
	return out
}

// dwInputs fills a padded window of K rows and a filter.
func dwInputs(s cpu.DWShape, seed int64) (in, w []float32) {
	rnd := rand.New(rand.NewSource(seed))
	in = make([]float32, s.K*s.WP*s.Chans)
	w = make([]float32, s.K*s.K*s.Chans)
	for i := range in {
		in[i] = float32(rnd.NormFloat64())
	}
	for i := range w {
		w[i] = float32(rnd.NormFloat64() * 0.3)
	}
	return in, w
}

// TestEmitDWConvMatchesReference gates the host tier's depthwise row at every
// shape dwShapes names, with a guard after the output: a tail that writes a
// whole vector, a stride that walks the wrong window, or a filter read at the
// wrong tap each fail it. Then it runs a filter whose taps are transposed
// (kx, ky), which the gate must see.
func TestEmitDWConvMatchesReference(t *testing.T) {
	const guard = 8
	for i, s := range dwShapes() {
		code := onHost(t)(hostTable().DWConv(s))
		in, w := dwInputs(s, int64(i))
		want := dwRef(s, in, w)
		out := make([]float32, s.WOut*s.Chans+guard)
		for j := range out {
			out[j] = -1234.5
		}
		code.Call(&cpu.Args{Out: &out[0], AScale: &w[0], Q32: &in[0]})
		code.Close()
		for j, v := range want {
			if d := math.Abs(float64(out[j]) - v); d > 1e-4*(1+math.Abs(v)) {
				t.Fatalf("%+v: out[%d] = %v, want %v", s, j, out[j], v)
			}
		}
		for j := s.WOut * s.Chans; j < len(out); j++ {
			if out[j] != -1234.5 {
				t.Fatalf("%+v: wrote past the row at %d", s, j)
			}
		}
	}
	// The violation: the taps read transposed.
	s := cpu.DWShape{K: 3, Stride: 2, WP: 9, Chans: 13, WOut: 4}
	in, w := dwInputs(s, 99)
	wt := make([]float32, len(w))
	for ky := 0; ky < 3; ky++ {
		for kx := 0; kx < 3; kx++ {
			copy(wt[(kx*3+ky)*s.Chans:(kx*3+ky+1)*s.Chans], w[(ky*3+kx)*s.Chans:(ky*3+kx+1)*s.Chans])
		}
	}
	code := onHost(t)(hostTable().DWConv(s))
	defer code.Close()
	out := make([]float32, s.WOut*s.Chans)
	code.Call(&cpu.Args{Out: &out[0], AScale: &wt[0], Q32: &in[0]})
	want := dwRef(s, in, w)
	worst := 0.0
	for j, v := range want {
		worst = math.Max(worst, math.Abs(float64(out[j])-v))
	}
	if worst < 1e-2 {
		t.Fatalf("a transposed filter reads within %.2e: the gate cannot see the tap order", worst)
	}
}

// TestEmitDWConvRefusesAnOverreachingRow is the shape check: a row whose last
// window reads past the padded row is refused, not emitted.
func TestEmitDWConvRefusesAnOverreachingRow(t *testing.T) {
	if _, err := cpu.EmitDWConv(cpu.DWShape{K: 3, Stride: 2, WP: 8, Chans: 4, WOut: 4}); err == nil {
		t.Fatal("a 3-wide filter at stride 2 over 4 outputs needs 9 samples, and 8 was accepted")
	}
}
