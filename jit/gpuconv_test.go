package jit_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testlock"
	"github.com/jitllm/jitllm/jit"
)

// TestComposedGPUGridConv is TestComposedCPUGridConv on every device
// backend: the composer's depthwise convolution over a picture's grid
// (graph.OpConv with GridW) at ragged grids, channels and 3x3 and 5x5 taps,
// a grid of one row and one column among them, held to the oracle with the
// words past the result untouched, and the filter read transposed seen.
func TestComposedGPUGridConv(t *testing.T) {
	testlock.GPU(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate proved nothing")
	}
	const guard = 64
	cases := [][5]int{{1, 1, 1, 3, 1}, {1, 7, 5, 3, 1}, {7, 1, 9, 3, 1}, {4, 6, 3, 3, 1}, {5, 5, 17, 5, 1}, {9, 7, 33, 3, 1},
		{8, 8, 64, 5, 1}, {3, 3, 1, 5, 1}, {16, 16, 128, 3, 1},
		{8, 8, 64, 3, 2}, {7, 9, 5, 5, 2}, {5, 5, 17, 3, 2}, {16, 16, 128, 5, 2}, {1, 7, 5, 3, 2}, {7, 1, 9, 3, 2}, {2, 2, 3, 3, 2}}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			r := jit.Default()
			dev := gpuDevice(d)
			rng := rand.New(rand.NewSource(37))
			right := 0
			for _, cs := range cases {
				h, w, c, k, s := cs[0], cs[1], cs[2], cs[3], cs[4]
				name := fmt.Sprintf("%dx%d/%d channels/%dx%d/stride %d", h, w, c, k, k, s)
				out := graph.Result{Name: "x.conv", Width: c}
				if s > 1 {
					out.Rows = ((h + s - 1) / s) * ((w + s - 1) / s)
				}
				b := graph.New()
				x0 := b.In("x", c)
				b.Out(b.Emit(graph.OpConv, []graph.ValueID{x0}, []graph.Result{out},
					[]graph.Binding{{Role: jlm.RoleVMNDwMid, K: c, Rows: k * k}}, graph.Attrs{Taps: k, GridW: w, Stride: s}, nil, nil)[0])
				_, codes, err := r.Form(jit.Request{Graph: b.G,
					Formats: map[jlm.Role]jlm.Type{jlm.RoleVMNDwMid: jlm.TypeF32},
					Step:    jit.Step{Rows: h * w, Page: 64, Keys: 1}, Device: dev})
				if err != nil {
					t.Errorf("%s: %v", name, err)
					continue
				}
				op, err := compileLone(d, codes)
				if err != nil {
					t.Errorf("%s: %v", name, err)
					continue
				}
				x := make([]float32, h*w*c)
				for i := range x {
					x[i] = float32(rng.NormFloat64())
				}
				filt := make([]float32, k*k*c)
				for i := range filt {
					filt[i] = float32(rng.NormFloat64())
				}
				want := make([]float32, ((h+s-1)/s)*((w+s-1)/s)*c)
				oracle.DWConv32(want, x, h, w, c, k, s, filt)
				run := func(filt []float32) ([]float32, error) {
					res, tail, err := op.run(x, len(want), guard, map[jlm.Role][]float32{jlm.RoleVMNDwMid: filt})
					if err != nil {
						return nil, err
					}
					for i, v := range tail {
						if v != guardWord {
							return nil, fmt.Errorf("the guard word %d past the result was written", i)
						}
					}
					return res, nil
				}
				got, err := run(filt)
				if err != nil {
					t.Errorf("%s: %v", name, err)
					op.close()
					continue
				}
				if e := gpuNMSE(got, want); math.IsNaN(e) || !(e < 1e-10) {
					t.Errorf("%s: NMSE %.3e against the oracle", name, e)
					op.close()
					continue
				}
				if k > 1 && h*w >= 4 {
					tr := make([]float32, len(filt))
					for ky := range k {
						for kx := range k {
							copy(tr[(kx*k+ky)*c:(kx*k+ky+1)*c], filt[(ky*k+kx)*c:(ky*k+kx+1)*c])
						}
					}
					if bad, err := run(tr); err != nil {
						t.Errorf("%s: the violation did not run: %v", name, err)
					} else if e := gpuNMSE(bad, want); !(e > 1e-6) {
						t.Errorf("%s: the filter transposed reads %.3e: the gate cannot see it", name, e)
					}
				}
				op.close()
				right++
			}
			if right != len(cases) {
				t.Fatalf("%d of %d grid convolutions held to the oracle", right, len(cases))
			}
		})
	}
}
