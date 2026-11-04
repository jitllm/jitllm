package composer

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPURegroup gathers each S x S block of a grid's rows into one
// row on the CPU (graph.OpRegroup), the output stating its own rows, and
// holds every value to the gather's definition. The guard past the output
// stays as it was, and a grid stated a patch wider is the violation it must
// see. A row drop over an input of rows of its own (graph.OpDropRows) is held
// to its rows after the dropped ones, the guard untouched.
func TestComposedCPURegroup(t *testing.T) {
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	for _, c := range []struct{ gw, gh, S, E int }{{8, 8, 2, 5}, {12, 6, 3, 4}, {4, 4, 4, 8}} {
		n := c.gw * c.gh
		out, W := n/(c.S*c.S), c.S*c.S*c.E
		b := graph.NewBlock(-1)
		x := b.In("rows", c.E)
		b.Out(b.Emit(graph.OpRegroup, []graph.ValueID{x}, []graph.Result{{Name: "grouped", Width: W, Rows: out}},
			nil, graph.Attrs{Span: c.S}, nil, nil)[0])
		if err := graph.Verify(b.G, graph.Facts{}); err != nil {
			t.Fatal(err)
		}
		_, codes, err := r.Form(Request{Graph: b.G, Step: Step{Rows: n, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
		if err != nil {
			t.Fatalf("%+v: %v", c, err)
		}
		in := make([]float32, n*c.E)
		for i := range in {
			in[i] = float32(i%97) + float32(i/97)*0.25
		}
		run := func(gw int) []float32 {
			req := make([]uint32, RequestRows+n)
			req[RequestGrid] = uint32(gw)
			got, err := runCPUBound(hostArm(), codes, in, out*W, [3][]uint32{}, nil, func(o Operand) []uint32 {
				if o.Name == "request" {
					return req
				}
				return nil
			})
			if err != nil {
				t.Fatalf("%+v: %v", c, err)
			}
			return got
		}
		got := run(c.gw)
		gsw := c.gw / c.S
		for row := range out {
			for j := range W {
				slot, e := j/c.E, j%c.E
				src := ((row/gsw)*c.S+slot/c.S)*c.gw + (row%gsw)*c.S + slot%c.S
				if want := in[src*c.E+e]; got[row*W+j] != want {
					t.Fatalf("%+v: row %d element %d is %v, want input row %d's %v", c, row, j, got[row*W+j], src, want)
				}
			}
		}
		for i, v := range got[len(got):cap(got)] {
			if !math.IsNaN(float64(v)) {
				t.Fatalf("%+v: the guard's element %d past the %d rows was written (%v)", c, i, out, v)
			}
		}
		bad, same := run(c.gw+c.S), true
		for i := range got {
			same = same && bad[i] == got[i]
		}
		if same {
			t.Fatalf("%+v: a grid stated a patch wider gathers the same rows", c)
		}
		t.Logf("%dx%d by %d, %d of width %d: the gather's rows, the guard untouched", c.gw, c.gh, c.S, out, W)
	}
	for _, c := range []struct{ rows, E, drop int }{{16, 20, 1}, {9, 3, 4}} {
		out := c.rows - c.drop
		b := graph.NewBlock(-1)
		x := b.In("rows", c.E)
		b.G.Values[x].Rows = c.rows
		b.Out(b.Emit(graph.OpDropRows, []graph.ValueID{x}, []graph.Result{{Name: "kept", Width: c.E, Rows: out}}, nil,
			graph.Attrs{Offset: c.drop}, nil, nil)[0])
		if err := graph.Verify(b.G, graph.Facts{}); err != nil {
			t.Fatal(err)
		}
		_, codes, err := r.Form(Request{Graph: b.G, Step: Step{Rows: c.rows, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
		if err != nil {
			t.Fatalf("a drop of %d from %d: %v", c.drop, c.rows, err)
		}
		in := make([]float32, c.rows*c.E)
		for i := range in {
			in[i] = float32(i) + 0.5
		}
		got, err := runCPUBound(hostArm(), codes, in, out*c.E, [3][]uint32{}, nil, nil)
		if err != nil {
			t.Fatalf("a drop of %d from %d: %v", c.drop, c.rows, err)
		}
		for i := range got {
			if want := in[c.drop*c.E+i]; got[i] != want {
				t.Fatalf("a drop of %d from %d: element %d is %v, want the input's %v", c.drop, c.rows, i, got[i], want)
			}
		}
		for i, v := range got[len(got):cap(got)] {
			if !math.IsNaN(float64(v)) {
				t.Fatalf("a drop of %d from %d: the guard's element %d was written (%v)", c.drop, c.rows, i, v)
			}
		}
		t.Logf("a drop of %d from %d rows of %d: the rows after them, the guard untouched", c.drop, c.rows, c.E)
	}
}
