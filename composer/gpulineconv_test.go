package composer_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPULineConv is TestComposedCPULineConv on every device
// backend: the composer's depthwise convolution over a line of rows
// (graph.OpConv with no GridW) held to the oracle with the words past the
// result untouched, and the taps read reversed seen.
func TestComposedGPULineConv(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate proved nothing")
	}
	const guard = 64
	cases := [][4]int{{1, 1, 3, 1}, {2, 3, 5, 1}, {7, 5, 3, 1}, {9, 33, 5, 1}, {16, 64, 3, 1}, {31, 17, 7, 1},
		{8, 17, 3, 2}, {7, 9, 5, 2}, {2, 3, 3, 2}, {33, 64, 5, 2}}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			reg := composer.Default()
			dev := gpuDevice(d)
			rng := rand.New(rand.NewSource(41))
			right := 0
			for _, cs := range cases {
				r, c, k, s := cs[0], cs[1], cs[2], cs[3]
				name := fmt.Sprintf("%d rows/%d channels/%d taps/stride %d", r, c, k, s)
				out := graph.Result{Name: "x.conv", Width: c}
				if s > 1 {
					out.Rows = (r + s - 1) / s
				}
				b := graph.New()
				x0 := b.In("x", c)
				b.Out(b.Emit(graph.OpConv, []graph.ValueID{x0}, []graph.Result{out},
					[]graph.Binding{{Role: jlm.RoleVMNDwMid, K: c, Rows: k}}, graph.Attrs{Taps: k, Stride: s}, nil, nil)[0])
				_, codes, err := reg.Form(composer.Request{Graph: b.G,
					Formats: map[jlm.Role]jlm.Type{jlm.RoleVMNDwMid: jlm.TypeF32},
					Step:    composer.Step{Rows: r, Page: 64, Keys: 1}, Device: dev})
				if err != nil {
					t.Errorf("%s: %v", name, err)
					continue
				}
				op, err := compileLone(d, codes)
				if err != nil {
					t.Errorf("%s: %v", name, err)
					continue
				}
				x := make([]float32, r*c)
				for i := range x {
					x[i] = float32(rng.NormFloat64())
				}
				filt := make([]float32, k*c)
				for i := range filt {
					filt[i] = float32(rng.NormFloat64())
				}
				want := make([]float32, ((r+s-1)/s)*c)
				oracle.DWConvLine32(want, x, r, c, k, s, filt)
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
				if r > 1 {
					rev := make([]float32, len(filt))
					for i := range k {
						copy(rev[(k-1-i)*c:(k-i)*c], filt[i*c:(i+1)*c])
					}
					if bad, err := run(rev); err != nil {
						t.Errorf("%s: the violation did not run: %v", name, err)
					} else if e := gpuNMSE(bad, want); !(e > 1e-6) {
						t.Errorf("%s: the taps reversed read %.3e: the gate cannot see it", name, e)
					}
				}
				op.close()
				right++
			}
			if right != len(cases) {
				t.Fatalf("%d of %d line convolutions held to the oracle", right, len(cases))
			}
		})
	}
}
