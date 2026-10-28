package jit

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestComposedCPURopeShift forms OpRope turned at the row's history
// position less Shift (graph.Attrs.Shift), its angles from the rotary's
// planes, on the CPU: the last NRot of each head turned in neighbouring
// pairs (DeepSeek V4's entry, turned at its window's first position), and
// back where Inverse, held to the definition with the elements before the
// turned ones the input's bit for bit. The row's own position's angles must
// part from it.
func TestComposedCPURopeShift(t *testing.T) {
	const heads, hd, nrot, pos, shift = 2, 32, 16, 11, 3
	for _, inv := range []bool{false, true} {
		w := heads * hd
		b := graph.New()
		x := b.In("x", w)
		y := b.Emit(graph.OpRope, []graph.ValueID{x}, []graph.Result{{Name: "y", Width: w}}, nil,
			graph.Attrs{HeadDim: hd, NRot: nrot, Base: 10000, Layout: graph.RopeNorm, Offset: hd - nrot, Shift: shift, Inverse: inv}, nil, nil)[0]
		b.Out(y)
		_, codes, err := Default().Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{},
			Step: Step{Rows: 1, Page: 64, Keys: 64, Seqs: 1}, Device: cpuDevice()})
		if err != nil {
			t.Fatal(err)
		}
		rng := rand.New(rand.NewSource(13))
		in := make([]float32, w)
		for i := range in {
			in[i] = float32(rng.NormFloat64())
		}
		out := make([]float32, w)
		for i := range out {
			out[i] = float32(math.NaN())
		}
		planes := nn.Rope{NRot: nrot, Base: 10000}.TabPlanes(nrot / 2)
		bufs := map[string][]uint32{"x": f32Words(append([]float32(nil), in...)), b.G.Values[y].Name: f32Words(out),
			"rope.planes": f32Words(planes), "rows": {0, 0, pos + 1, pos}, "rows.n": {1}}
		if err := runCPUCalls(codes, b.G, bufs); err != nil {
			t.Fatal(err)
		}
		for _, bd := range codes[len(codes)-1].Binds {
			switch o := bd.Operand; {
			case !bd.Writes:
			case o.Kind == OperandScratch:
				copy(out, wordsF32(bufs["own:"+o.Name]))
			case o.Kind == OperandValue && o.Value != y:
				copy(out, wordsF32(bufs[b.G.Values[o.Value].Name]))
			}
		}
		turn := func(at int) []float64 {
			want := make([]float64, w)
			for i := range in {
				want[i] = float64(in[i])
			}
			for h := range heads {
				base := h*hd + hd - nrot
				for p := range nrot / 2 {
					a := float64(at) * math.Pow(10000, -2*float64(p)/float64(nrot))
					c, s := math.Cos(a), math.Sin(a)
					if inv {
						s = -s
					}
					x0, x1 := float64(in[base+2*p]), float64(in[base+2*p+1])
					want[base+2*p], want[base+2*p+1] = x0*c-x1*s, x0*s+x1*c
				}
			}
			return want
		}
		worst := func(want []float64) float64 {
			m := 0.0
			for i := range want {
				m = math.Max(m, math.Abs(float64(out[i])-want[i]))
				if math.IsNaN(float64(out[i])) {
					return math.Inf(1)
				}
			}
			return m
		}
		e := worst(turn(pos - shift))
		t.Logf("inverse %v: worst element off by %.3e", inv, e)
		if e > 1e-5 {
			t.Fatalf("inverse %v: the turn at %d is off by %.3e", inv, pos-shift, e)
		}
		for h := range heads {
			for j := range hd - nrot {
				if i := h*hd + j; math.Float32bits(out[i]) != math.Float32bits(in[i]) {
					t.Fatalf("head %d element %d is %g, the input's %g", h, j, out[i], in[i])
				}
			}
		}
		if e := worst(turn(pos)); !(e > 1e-3) {
			t.Errorf("against the turn at the row's own position the worst element is off by %.3e: the gate cannot see the shift", e)
		}
	}
}
