package composer

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUGridConvPadded holds graph.OpConv with PyTorch's padding
// (Offset) to its definition on the CPU: output (i, j) of channel c the
// taps over input (s*i-p+ky, s*j-p+kx), zero off the grid, in float64. Even
// sides at stride two are where it parts from TensorFlow's SAME padding (the
// default, which pads after), so a convolution that ignored Offset fails.
func TestComposedCPUGridConvPadded(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	dev := cpuDevice(hostArm())
	for _, cs := range [][5]int{{4, 6, 3, 3, 2}, {8, 8, 16, 3, 2}, {5, 7, 4, 3, 2}, {6, 4, 8, 3, 1}} {
		h, w, c, k, s := cs[0], cs[1], cs[2], cs[3], cs[4]
		p := k / 2
		t.Run(fmt.Sprintf("%dx%d/%d/%dx%d/stride %d", h, w, c, k, k, s), func(t *testing.T) {
			ho, wo := (h+s-1)/s, (w+s-1)/s
			b := graph.New()
			x := b.In("x", c)
			out := graph.Result{Name: "x.conv", Width: c}
			if s > 1 {
				out.Rows = ho * wo
			}
			b.Out(b.Emit(graph.OpConv, []graph.ValueID{x}, []graph.Result{out},
				[]graph.Binding{{Role: jlm.RoleVMNDwMid, K: c, Rows: k * k}}, graph.Attrs{Taps: k, GridW: w, Stride: s, Offset: p}, nil, nil)[0])
			_, codes, err := Default().Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleVMNDwMid: jlm.TypeF32},
				Step: Step{Rows: h * w, Page: 64, Keys: 1, ActWindow: 32}, Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			in := make([]float32, h*w*c)
			for i := range in {
				in[i] = float32(rng.NormFloat64())
			}
			filt := make([]float32, k*k*c)
			for i := range filt {
				filt[i] = float32(rng.NormFloat64())
			}
			got, err := runCPUVectors(hostArm(), codes, in, ho*wo*c, [3][]uint32{f32Words(filt)}, nil)
			if err != nil {
				t.Fatal(err)
			}
			worst := 0.0
			for i := range ho {
				for j := range wo {
					for ch := range c {
						want := 0.0
						for ky := range k {
							for kx := range k {
								y, xx := s*i-p+ky, s*j-p+kx
								if y < 0 || y >= h || xx < 0 || xx >= w {
									continue
								}
								want += float64(filt[(ky*k+kx)*c+ch]) * float64(in[(y*w+xx)*c+ch])
							}
						}
						d := math.Abs(float64(got[(i*wo+j)*c+ch]) - want)
						if math.IsNaN(d) || d > 1e-5*max(1, math.Abs(want)) {
							t.Fatalf("(%d, %d) channel %d: %g, want %g", i, j, ch, got[(i*wo+j)*c+ch], want)
						}
						worst = max(worst, d)
					}
				}
			}
			t.Logf("worst %.2e", worst)
		})
	}
}
