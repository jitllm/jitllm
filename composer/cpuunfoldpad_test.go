package composer

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUUnfoldPadded holds graph.OpUnfold with PyTorch's padding
// (Offset) to its definition on the CPU: output row (i, j)'s element
// (ky*k+kx)*c+ch is channel ch of input (s*i-p+ky, s*j-p+kx), zero off the
// grid and past the taps. Even sides at stride two are where it parts from
// TensorFlow's SAME padding, so an unfold that ignored Offset fails.
func TestComposedCPUUnfoldPadded(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	dev := cpuDevice(hostArm())
	for _, cs := range [][5]int{{6, 8, 1, 3, 2}, {4, 6, 3, 3, 2}, {5, 7, 2, 3, 2}, {6, 4, 2, 3, 1}} {
		h, w, c, k, s := cs[0], cs[1], cs[2], cs[3], cs[4]
		p := k / 2
		t.Run(fmt.Sprintf("%dx%d/%d/%dx%d/stride %d", h, w, c, k, k, s), func(t *testing.T) {
			ho, wo := (h+s-1)/s, (w+s-1)/s
			kw := k*k*c + 2
			b := graph.New()
			x := b.In("x", c)
			out := graph.Result{Name: "x.unfold", Width: kw}
			if s > 1 {
				out.Rows = ho * wo
			}
			b.Out(b.Emit(graph.OpUnfold, []graph.ValueID{x}, []graph.Result{out}, nil,
				graph.Attrs{Taps: k, GridW: w, Stride: s, Offset: p}, nil, nil)[0])
			_, codes, err := Default().Form(Request{Graph: b.G, Step: Step{Rows: h * w, Page: 64, Keys: 1, ActWindow: 32}, Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			in := make([]float32, h*w*c)
			for i := range in {
				in[i] = float32(rng.NormFloat64())
			}
			got, err := runCPUBound(hostArm(), codes, in, ho*wo*kw, [3][]uint32{}, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			for i := range ho {
				for j := range wo {
					for e := range kw {
						want := float32(0)
						if e < k*k*c {
							tap, ch := e/c, e%c
							y, xx := s*i-p+tap/k, s*j-p+tap%k
							if y >= 0 && y < h && xx >= 0 && xx < w {
								want = in[(y*w+xx)*c+ch]
							}
						}
						if g := got[(i*wo+j)*kw+e]; g != want {
							t.Fatalf("row (%d, %d) element %d: %g, want %g", i, j, e, g, want)
						}
					}
				}
			}
		})
	}
}
