package composer

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUAppendRow runs five rows six wide with a table's row after
// them (graph.OpAppendRow) on the CPU: the input's rows as they were, then
// the table's row 1 exactly. Reading the table's row 0 is the violation.
func TestComposedCPUAppendRow(t *testing.T) {
	const R, E, rows, at = 5, 6, 3, 1
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	b := graph.NewBlock(-1)
	x := b.In("rows", E)
	b.Out(b.Emit(graph.OpAppendRow, []graph.ValueID{x}, []graph.Result{{Name: "marked", Width: E, Rows: R + 1}},
		[]graph.Binding{{Role: jlm.RoleVMNHardEmbd, K: E, Rows: rows}}, graph.Attrs{Offset: at}, nil, nil)[0])
	if err := graph.Verify(b.G, graph.Facts{Inventory: map[jlm.Role]graph.Shape{jlm.RoleVMNHardEmbd: {K: E, Rows: rows}}}); err != nil {
		t.Fatal(err)
	}
	_, codes, err := r.Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleVMNHardEmbd: jlm.TypeF32},
		Step: Step{Rows: R, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
	if err != nil {
		t.Fatal(err)
	}
	in := make([]float32, R*E)
	for i := range in {
		in[i] = float32(math.Sin(0.3*float64(i) + 0.2))
	}
	table := make([]float32, rows*E)
	for i := range table {
		table[i] = float32(10 + i)
	}
	got, err := runCPUBound(hostArm(), codes, in, (R+1)*E, [3][]uint32{}, map[jlm.Role][]float32{jlm.RoleVMNHardEmbd: table}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := range R * E {
		if got[i] != in[i] {
			t.Fatalf("element %d of the input's rows is %v, want %v", i, got[i], in[i])
		}
	}
	other := 0
	for j := range E {
		if got[R*E+j] != table[at*E+j] {
			t.Fatalf("the appended row's element %d is %v, want the table's row %d's %v", j, got[R*E+j], at, table[at*E+j])
		}
		if got[R*E+j] != table[j] {
			other++
		}
	}
	t.Logf("%d rows and the table's row %d after them, exactly; %d of its %d elements differ from row 0's", R, at, other, E)
	if other == 0 {
		t.Fatal("VIOLATION: the table's row 0 reads the same")
	}
}

// TestComposedCPURegroupStatedGrid gathers each 2x2 block of an 8x8 grid
// whose width the graph states (graph.Attrs.GridW) while the request names a
// grid half as wide: the gather follows the stated width. The same op
// reading the request's width is the violation.
func TestComposedCPURegroupStatedGrid(t *testing.T) {
	const gw, gh, S, E, wrong = 8, 8, 2, 5, 4
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	n := gw * gh
	out, W := n/(S*S), S*S*E
	in := make([]float32, n*E)
	for i := range in {
		in[i] = float32(i%97) + float32(i/97)*0.25
	}
	run := func(stated int) []float32 {
		b := graph.NewBlock(-1)
		x := b.In("rows", E)
		b.Out(b.Emit(graph.OpRegroup, []graph.ValueID{x}, []graph.Result{{Name: "grouped", Width: W, Rows: out}},
			nil, graph.Attrs{Span: S, GridW: stated}, nil, nil)[0])
		_, codes, err := r.Form(Request{Graph: b.G, Step: Step{Rows: n, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
		if err != nil {
			t.Fatal(err)
		}
		req := make([]uint32, RequestRows+n)
		req[RequestGrid] = wrong
		got, err := runCPUBound(hostArm(), codes, in, out*W, [3][]uint32{}, nil, func(o Operand) []uint32 {
			if o.Name == "request" {
				return req
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		return got
	}
	got := run(gw)
	gsw := gw / S
	for row := range out {
		for j := range W {
			slot, e := j/E, j%E
			src := ((row/gsw)*S+slot/S)*gw + (row%gsw)*S + slot%S
			if want := in[src*E+e]; got[row*W+j] != want {
				t.Fatalf("row %d element %d is %v, want input row %d's %v", row, j, got[row*W+j], src, want)
			}
		}
	}
	bad, differ := run(0), 0
	for i := range got {
		if bad[i] != got[i] {
			differ++
		}
	}
	t.Logf("an %dx%d grid stated %d wide gathers its own blocks though the request says %d; read from the request, %d values differ",
		gw, gh, gw, wrong, differ)
	if differ == 0 {
		t.Fatal("VIOLATION: the request's width gathers the same rows")
	}
}
