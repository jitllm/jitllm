package composer

import (
	"testing"

	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUClipMask holds graph.OpClipMask to its definition on the
// CPU: a clip of 205 samples every 10 is 20 valid frames, 10, 5 and 3 after
// one, two and three halvings; a grid 4 wide of 12 grid rows keeps the
// rows of the grid rows below that length and zeroes the rest, at two
// channels and at 64 (a vector form). A mask that did not halve is the
// violation.
func TestComposedCPUClipMask(t *testing.T) {
	const n, hop, gw, gr = 205, 10, 4, 12
	req := make([]uint32, RequestPicW+1)
	req[RequestPicW] = n
	for _, w := range []int{2, 64} {
		x := make([]float32, gw*gr*w)
		for i := range x {
			x[i] = float32(i) + 1
		}
		clipMaskCase(t, x, w, req, n, hop, gw, gr)
	}
}

// clipMaskCase is TestComposedCPUClipMask at rows of w.
func clipMaskCase(t *testing.T, x []float32, w int, req []uint32, n, hop, gw, gr int) {
	t.Helper()
	for span, keep := range []int{20, 10, 5, 3} {
		b := graph.NewBlock(-1)
		in := b.In("x", w)
		b.Out(b.Emit(graph.OpClipMask, []graph.ValueID{in}, []graph.Result{{Name: "masked", Width: w}}, nil,
			graph.Attrs{Stride: hop, Span: span, GridW: gw}, nil, nil)[0])
		got, err := runCPUBound(hostArm(), slotForm(t, b.G, gw*gr), x, len(x), [3][]uint32{}, nil, func(o Operand) []uint32 {
			if o.Name == "request" {
				return req
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		for r := range gw * gr {
			for e := range w {
				want := x[r*w+e]
				if r/gw >= keep {
					want = 0
				}
				if g := got[r*w+e]; g != want {
					t.Fatalf("%d wide, %d halvings: row %d element %d: %g, want %g (%d grid rows kept)", w, span, r, e, g, want, keep)
				}
			}
		}
	}
}
