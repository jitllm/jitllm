package composer_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUUpsample is TestComposedCPUUpsample on every device
// backend, at ragged grids, channels and factors: every output row its
// nearest input row's, exactly, the words past the result untouched, and
// the grid read transposed seen.
func TestComposedGPUUpsample(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate proved nothing")
	}
	const guard = 64
	cases := [][4]int{{3, 4, 5, 2}, {1, 1, 1, 2}, {1, 7, 3, 3}, {6, 2, 33, 2}, {24, 24, 40, 2}}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			r := composer.Default()
			dev := gpuDevice(d)
			seen := false
			for _, cs := range cases {
				h, w, c, s := cs[0], cs[1], cs[2], cs[3]
				name := fmt.Sprintf("%dx%d/%d channels/by %d", h, w, c, s)
				b := graph.New()
				x0 := b.In("x", c)
				b.Out(b.Emit(graph.OpUpsample, []graph.ValueID{x0}, []graph.Result{{Name: "up", Width: c, Rows: h * w * s * s}}, nil,
					graph.Attrs{Span: s, GridW: w}, nil, nil)[0])
				_, codes, err := r.Form(composer.Request{Graph: b.G, Step: composer.Step{Rows: h * w, Page: 64, Keys: 1}, Device: dev})
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
					x[i] = float32(math.Sin(0.37*float64(i) + 0.1))
				}
				want := func(transposed bool) []float32 {
					out := make([]float32, 0, h*w*s*s*c)
					for y := range h * s {
						for col := range w * s {
							p := (y/s)*w + col/s
							if transposed {
								p = (col/s)%h*w + (y/s)%w
							}
							out = append(out, x[p*c:(p+1)*c]...)
						}
					}
					return out
				}
				res, tail, err := op.run(x, h*w*s*s*c, guard, nil)
				op.close()
				if err != nil {
					t.Errorf("%s: %v", name, err)
					continue
				}
				for i, v := range tail {
					if v != guardWord {
						t.Errorf("%s: the guard word %d past the result was written", name, i)
						break
					}
				}
				off := func(w []float32) int {
					n := 0
					for i := range w {
						if res[i] != w[i] {
							n++
						}
					}
					return n
				}
				if n := off(want(false)); n != 0 {
					t.Errorf("%s: %d of %d values are not their nearest input row's", name, n, len(res))
					continue
				}
				if off(want(true)) > 0 {
					seen = true
				}
				t.Logf("%s: every row its nearest input's", name)
			}
			if !seen {
				t.Fatal("VIOLATION: the transposed grid reads the same on every case")
			}
		})
	}
}
