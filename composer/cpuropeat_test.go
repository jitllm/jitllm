package composer

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// TestComposedCPURopeAtWindow executes a rotary turning each row at its
// window's first position on the CPU: OpRopeTable's value at each row's
// position floored to a multiple of Span, from the rotary's planes, and
// OpRope by that value, the head's tail turned (Offset) and its head kept.
// Held to the definition row by row; the same rows turned at their own
// positions must fail it.
func TestComposedCPURopeAtWindow(t *testing.T) {
	for _, sh := range []struct{ heads, hd, nrot, span int }{{2, 16, 8, 4}, {1, 12, 4, 128}, {3, 8, 8, 3}, {2, 8, 8, 4}} {
		t.Run(fmt.Sprintf("%dx%d/rot%d/every%d", sh.heads, sh.hd, sh.nrot, sh.span), func(t *testing.T) {
			composedCPURopeAtWindow(t, sh.heads, sh.hd, sh.nrot, sh.span)
		})
	}
}

func composedCPURopeAtWindow(t *testing.T, H, hd, nrot, span int) {
	const R, capRows, base = 7, 8, 10000.0
	run := func(span int) ([]float32, []float64) {
		b := graph.New()
		x := b.In("x", H*hd)
		turn := graph.Attrs{HeadDim: hd, NRot: nrot, Base: base, Layout: graph.RopeNorm, Offset: hd - nrot}
		tab := b.RopeTableAt(turn, span)
		b.Out(b.Emit(graph.OpRope, []graph.ValueID{x, tab}, []graph.Result{{Name: "y", Width: H * hd}}, nil, turn, nil, nil)[0])
		g := b.G
		_, codes, err := Default().Form(Request{Graph: g, Rotaries: []graph.Rotary{{NRot: nrot, Base: base}}, Apart: true,
			Step:   Step{Rows: capRows, Page: 8, Keys: 32, ActWindow: 32, Seqs: 1},
			Device: Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(hostArm().tg.Features().ISA()), ExactDiv: true}})
		if err != nil {
			t.Fatal(err)
		}
		rng := rand.New(rand.NewSource(17))
		in := make([]float32, capRows*H*hd)
		for i := range in {
			in[i] = float32(rng.NormFloat64())
		}
		pos := []uint32{R}
		for r := range R {
			pos = append(pos, uint32(37+5*r))
		}
		planes := nn.Rope{NRot: nrot, Base: base}.TabPlanes(nrot / 2)
		out := make([]float32, capRows*H*hd)
		bufs := map[string][]uint32{"x": f32Words(in), "rows.n": {R}, "q.off": make([]uint32, capRows*4)}
		for _, c := range codes {
			for _, bd := range c.Binds {
				switch o := bd.Operand; {
				case o.Kind == OperandTable && o.Table == TablePos:
					bufs[o.Name] = pos
				case o.Kind == OperandTable && o.Table == TablePlanes:
					bufs[o.Name] = f32Words(planes)
				}
			}
		}
		bufs[g.Values[g.Out[0]].Name] = f32Words(out)
		if err := runCPURows(hostArm(), codes, g, bufs, R); err != nil {
			t.Fatal(err)
		}
		for _, bd := range codes[len(codes)-1].Binds {
			if o := bd.Operand; bd.Writes && o.Kind == OperandScratch {
				copy(out, wordsF32(bufs["own:"+o.Name]))
			}
		}
		want := make([]float64, R*H*hd)
		for r := range R {
			at := float64(int(pos[1+r]) / span * span)
			for h := range H {
				row := in[(r*H+h)*hd:]
				for i := range hd {
					want[(r*H+h)*hd+i] = float64(row[i])
				}
				for p := range nrot / 2 {
					th := at * math.Pow(base, -2*float64(p)/float64(nrot))
					a, b := hd-nrot+2*p, hd-nrot+2*p+1
					x0, x1 := float64(row[a]), float64(row[b])
					want[(r*H+h)*hd+a] = x0*math.Cos(th) - x1*math.Sin(th)
					want[(r*H+h)*hd+b] = x0*math.Sin(th) + x1*math.Cos(th)
				}
			}
		}
		return out, want
	}
	nmse := func(got []float32, want []float64) float64 {
		var num, den float64
		for i := range want {
			d := float64(got[i]) - want[i]
			num += d * d
			den += want[i] * want[i]
		}
		return num / den
	}
	got, want := run(span)
	e := nmse(got, want)
	t.Logf("NMSE %.3e against the definition", e)
	if math.IsNaN(e) || e > 1e-10 {
		t.Fatalf("the rotary at each row's window start: NMSE %.3e", e)
	}
	if own, _ := run(1); !(nmse(own, want) > 1e-6) {
		t.Errorf("turned at their own positions the rows read the same: the gate cannot see the window")
	}
}
