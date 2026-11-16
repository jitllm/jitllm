package composer

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/convert/gather"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// statedGather is a stated gather's attributes as the engine lowers them.
func statedGather(op jlm.TowerOp) graph.Attrs { return graph.StatedGather(op.Expr, op.Span) }

// TestComposedCPUClipMask holds gather.ClipMask to its definition on the
// CPU: a clip of 205 samples every 10 is 20 valid frames, 10, 5 and 3 after
// one, two and three halvings; a grid 4 wide of 12 grid rows keeps the
// rows of the grid rows below that length and zeroes the rest, at two
// channels and at 64 (a vector form). A mask stated one halving short, at
// twice the stride or on a grid one row narrower is the violation it must
// see wherever it keeps other rows; a stride of 0, a negative halving count
// or a negative width is refused.
func TestComposedCPUClipMask(t *testing.T) {
	const n, hop, gw, gr = 205, 10, 4, 12
	req := make([]uint32, RequestPicW+1)
	req[RequestPicW] = n
	for _, w := range []int{2, 64} {
		x := make([]float32, gw*gr*w)
		for i := range x {
			x[i] = float32(i) + 1
		}
		for span, keep := range []int{20, 10, 5, 3} {
			if bad := clipMaskCase(t, x, w, req, gw, gr, keep, gw, hop, span); bad != "" {
				t.Fatal(bad)
			}
			wrong := [][3]int{{gw, 2 * hop, span}}
			if span > 0 {
				wrong = append(wrong, [3]int{gw, hop, span - 1})
			}
			if keep < gr {
				wrong = append(wrong, [3]int{gw - 1, hop, span})
			}
			for _, v := range wrong {
				if clipMaskCase(t, x, w, req, gw, gr, keep, v[0], v[1], v[2]) == "" {
					t.Fatalf("%d wide, %d halvings: a mask stated as %v kept the rows the stated one does", w, span, v)
				}
			}
		}
	}
	for _, v := range [][3]int{{gw, 0, 1}, {gw, hop, -1}, {-1, hop, 1}} {
		if _, err := gather.ClipMask(v[0], v[1], v[2]); err == nil {
			t.Errorf("a clip mask stated as %v is not refused", v)
		}
	}
}

// clipMaskCase runs gather.ClipMask(mw, hop, span) over x, rows of w, and
// says how it differs from keeping the rows of the first keep grid rows of a
// grid gw wide ("" when it does not).
func clipMaskCase(t *testing.T, x []float32, w int, req []uint32, gw, gr, keep, mw, hop, span int) string {
	t.Helper()
	m, err := gather.ClipMask(mw, hop, span)
	if err != nil {
		t.Fatal(err)
	}
	b := graph.NewBlock(-1)
	in := b.In("x", w)
	b.Out(b.Emit(graph.OpGatherRows, []graph.ValueID{in}, []graph.Result{{Name: "masked", Width: w}}, nil,
		statedGather(m), nil, nil)[0])
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
				return fmt.Sprintf("%d wide, %d halvings: row %d element %d: %g, want %g (%d grid rows kept)", w, span, r, e, g, want, keep)
			}
		}
	}
	return ""
}
