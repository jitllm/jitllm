package backend_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestQ4EKernelsMatchTheLoops runs Qwen3.8-Flash-Next's device kernels
// (kernels/q4e.go) against the loops engine/model/q4e.go and ngram.go
// transcribe, at a width that is not a multiple of a workgroup and over
// several rows, on every backend present.
func TestQ4EKernelsMatchTheLoops(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const d, h, rows = 40, 4, 3
	const eps = 1e-5
	val := func(n int, s float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(math.Sin(s*float64(i+1)) * (1 + float64(i%5)))
		}
		return v
	}
	sig := func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }
	x := val(h*rows*d, 0.37) // the streams, stream-major
	w := val(h*d, 0.11)      // a grouped norm
	gate := val(rows*h*d, 0.23)
	y := val(rows*d, 0.53)
	inj := val(rows*h, 0.71)
	// stream k of row r, stream-major; row-major (r*h+k)*d.
	sm := func(k, r int) int { return (k*rows + r) * d }
	rm := func(r, k int) int { return (r*h + k) * d }
	normed := func(src []float32, at int, wk int) []float64 {
		var ss float64
		for i := 0; i < d; i++ {
			ss += float64(src[at+i]) * float64(src[at+i])
		}
		sc := 1 / math.Sqrt(ss/d+eps)
		out := make([]float64, d)
		for i := range out {
			out[i] = float64(src[at+i]) * sc * float64(w[wk*d+i])
		}
		return out
	}
	wantXN := make([]float64, rows*h*d)
	for r := 0; r < rows; r++ {
		for k := 0; k < h; k++ {
			copy(wantXN[rm(r, k):], normed(x, sm(k, r), k))
		}
	}
	xnF := make([]float32, len(wantXN))
	for i, v := range wantXN {
		xnF[i] = float32(v)
	}
	wantMix := make([]float64, rows*d)
	for r := 0; r < rows; r++ {
		for i := 0; i < d; i++ {
			var s float64
			for k := 0; k < h; k++ {
				s += sig(float64(gate[rm(r, k)+i])) * float64(xnF[rm(r, k)+i])
			}
			wantMix[r*d+i] = s / h
		}
	}
	wantW := make([]float64, rows*h)
	wF := make([]float32, rows*h)
	for i := range wantW {
		wantW[i] = 2 * sig(float64(inj[i])/h)
		wF[i] = float32(wantW[i])
	}
	wantPlace := make([]float64, h*rows*d)
	for k := 0; k < h; k++ {
		for r := 0; r < rows; r++ {
			for i := 0; i < d; i++ {
				wantPlace[sm(k, r)+i] = float64(x[sm(k, r)+i]) + float64(wF[r*h+k])*float64(y[r*d+i])
			}
		}
	}
	type job struct {
		name  string
		k     func() (*ir.Kernel, error)
		grid  int
		width int
		in    [][]float32
		want  []float64
		tol   float64
	}
	jobs := []job{
		{"group norm, stream-major", func() (*ir.Kernel, error) { return mainkernels.Q4EGroupNorm(d, h, rows, eps, true) },
			rows * h, mainkernels.RMSNormGroup, [][]float32{x, w}, wantXN, 1e-5},
		{"gate mix", func() (*ir.Kernel, error) { return mainkernels.Q4EGateMix(d, h, rows, false) },
			(rows*d + 127) / 128, 128, [][]float32{xnF, gate}, wantMix, 1e-5},
		{"inject", func() (*ir.Kernel, error) { return mainkernels.Q4EInject(rows*h, h, mainkernels.Q4EPlaceGated) },
			1, 128, [][]float32{inj}, wantW, 1e-6},
		{"place", func() (*ir.Kernel, error) { return mainkernels.Q4EPlace(d, h, rows) },
			(h*rows*d + 127) / 128, 128, [][]float32{x, y, wF}, wantPlace, 1e-5},
	}
	for _, dv := range devs {
		defer dv.Close()
		t.Run(dv.API(), func(t *testing.T) {
			g := newGPU(t, dv)
			defer g.free()
			for _, j := range jobs {
				kk, err := j.k()
				if err != nil {
					t.Fatal(err)
				}
				kern, err := dv.Compile(kk)
				if err != nil {
					t.Fatalf("%s: compile: %v", j.name, err)
				}
				var bufs []backend.Buf
				for _, in := range j.in {
					bufs = append(bufs, g.up(f32bytes(in)))
				}
				out := g.up(make([]byte, len(j.want)*4))
				bufs = append(bufs, out)
				if err := kern.Launch(j.grid, j.width, bufs...); err != nil {
					t.Fatal(err)
				}
				raw := make([]byte, len(j.want)*4)
				if err := out.Read(raw); err != nil {
					t.Fatal(err)
				}
				for i, wv := range j.want {
					got := float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:])))
					if !(math.Abs(got-wv) <= j.tol*(1+math.Abs(wv))) {
						t.Fatalf("%s: element %d is %g, want %g", j.name, i, got, wv)
					}
				}
				kern.Close()
			}
		})
	}
}
