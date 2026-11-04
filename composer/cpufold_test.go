package composer

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUFoldFollowsTheCall folds each k consecutive rows of a step's
// into one (graph.OpFold, the output stating its own rows) and calls the form
// for fewer rows than it was formed for, as a picture of its own size runs
// its class's form: the live rows' share of the output is each its k input
// rows, and no output row past it is written. A code not held to the call
// writes the rest: the same form run over every call, told the step's every
// row, is the violation the check must see.
func TestComposedCPUFoldFollowsTheCall(t *testing.T) {
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	const R, E, k = 64, 6, 4
	b := graph.NewBlock(-1)
	x := b.In("rows", E)
	y := b.Emit(graph.OpFold, []graph.ValueID{x}, []graph.Result{{Name: "folded", Width: k * E, Rows: R / k}},
		nil, graph.Attrs{Span: k}, nil, nil)[0]
	b.Out(y)
	if err := graph.Verify(b.G, graph.Facts{}); err != nil {
		t.Fatal(err)
	}
	_, codes, err := r.Form(Request{Graph: b.G, Step: Step{Rows: R, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
	if err != nil {
		t.Fatal(err)
	}
	held := 0
	for _, c := range codes {
		if c.Live == R {
			held++
		}
	}
	if held == 0 {
		t.Fatalf("none of the fold's %d codes is held to the call's rows", len(codes))
	}
	in := make([]uint32, R*E)
	for i := range in {
		in[i] = math.Float32bits(float32(i) + 0.5)
	}
	nan := math.Float32bits(float32(math.NaN()))
	run := func(live int, every bool) (wrong, past int) {
		out := make([]uint32, R*E)
		for i := range out {
			out[i] = nan
		}
		n := live
		if every {
			n = R
		}
		bufs := map[string][]uint32{b.G.Values[x].Name: in, b.G.Values[y].Name: out, "rows.n": {uint32(n)}}
		if every {
			err = runCPUCalls(hostArm(), codes, b.G, bufs)
		} else {
			err = runCPURows(hostArm(), codes, b.G, bufs, live)
		}
		if err != nil {
			t.Fatal(err)
		}
		for row := range R / k {
			for j := range k * E {
				got := out[row*k*E+j]
				switch {
				case row >= live/k && got != nan:
					past++
				case row < live/k && got != in[(row*k+j/E)*E+j%E]:
					wrong++
				}
			}
		}
		return wrong, past
	}
	for _, live := range []int{R, 40, 8} {
		if wrong, past := run(live, false); wrong > 0 || past > 0 {
			t.Fatalf("a call of %d rows: %d elements of its rows wrong, %d past them written", live, wrong, past)
		}
		t.Logf("a call of %d of %d rows: its %d folded rows, none past them", live, R, live/k)
	}
	if _, past := run(8, true); past == 0 {
		t.Fatal("VIOLATION: every call run for 8 live rows wrote nothing past their 2 folded rows")
	}
}
