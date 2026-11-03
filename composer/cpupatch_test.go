package composer

import (
	"math"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUPatchify gathers a picture's patches on the CPU, row r the
// patch the step's positions name, of a grid as wide as the request's word
// states: the division by a width known only at the call (a register's
// DIV), the divisions by 9 whose multiplier passes 31 bits, the row before
// the lead zero. The grid stated one wider is the violation it must see.
func TestComposedCPUPatchify(t *testing.T) {
	const P, lead = 3, 1
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true}
	ran := 0
	for _, grid := range [][2]int{{1, 1}, {2, 4}, {2, 3}, {3, 5}, {4, 7}, {5, 6}, {7, 7}, {24, 24}} {
		gh, gw := grid[0], grid[1]
		np, k := gh*gw, 3*P*P+5
		rows := np + lead
		b := graph.NewBlock(-1)
		px := b.In("pixels", 3*P*P)
		b.Out(b.Emit(graph.OpPatchify, []graph.ValueID{px}, []graph.Result{{Name: "patches", Width: k}}, nil,
			graph.Attrs{Span: P, Offset: lead}, nil, nil)[0])
		_, codes, err := r.Form(Request{Graph: b.G, Step: Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
		if err != nil {
			t.Fatalf("%dx%d: %v", gh, gw, err)
		}
		pix := make([]float32, rows*3*P*P)
		for i := range np * 3 * P * P {
			pix[i] = float32(i%9973) + 0.5
		}
		pos := make([]uint32, rows+1)
		pos[0] = uint32(rows)
		for i := range rows {
			// The patches in reverse, a permutation no row's own index is.
			pos[1+i] = uint32(i)
			if i >= lead {
				pos[1+i] = uint32(lead + np - 1 - (i - lead))
			}
		}
		want := make([]float32, rows*k)
		for i := lead; i < rows; i++ {
			p := int(pos[1+i]) - lead
			py, pxi := p/gw, p%gw
			for c := range 3 * P * P {
				ch, ky, kx := c/(P*P), c/P%P, c%P
				want[i*k+c] = pix[((py*P+ky)*gw*P+pxi*P+kx)*3+ch]
			}
		}
		run := func(width int) []float32 {
			req := make([]uint32, RequestRows+rows)
			req[RequestGrid] = uint32(width)
			got, err := runCPUBound(codes, pix, rows*k, [3][]uint32{}, nil, func(o Operand) []uint32 {
				switch o.Name {
				case "request":
					return req
				case "rope.pos":
					return pos
				}
				return nil
			})
			if err != nil {
				t.Fatalf("%dx%d: %v", gh, gw, err)
			}
			return got
		}
		got := run(gw)
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%dx%d: element %d (row %d) is %v, want %v", gh, gw, i, i/k, got[i], want[i])
			}
		}
		bad := run(gw + 1)
		same := true
		for i := range want {
			same = same && got[i] == bad[i]
		}
		if same {
			t.Fatalf("%dx%d: a grid stated %d wide gathers the same rows", gh, gw, gw+1)
		}
		ran++
		t.Logf("%dx%d grid, %d patches of %d: exact, a wrong width seen", gh, gw, np, 3*P*P)
	}
	if ran == 0 {
		t.Fatal("no grid ran")
	}
}

// TestComposedCPULeadRow writes a tower's class row over the first row of
// its rows on the CPU and leaves the rest as they were; the class row at
// row 1 too is the violation it must see.
func TestComposedCPULeadRow(t *testing.T) {
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true}
	for _, shape := range [][2]int{{2, 8}, {5, 13}, {577, 64}} {
		rows, e := shape[0], shape[1]
		b := graph.NewBlock(-1)
		x := b.In("x", e)
		b.Out(b.Emit(graph.OpLeadRow, []graph.ValueID{x}, []graph.Result{{Name: "class", Width: e}},
			[]graph.Binding{{Role: jlm.RoleVClassEmbd, K: e}}, graph.Attrs{Offset: 1}, nil, nil)[0])
		_, codes, err := r.Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleVClassEmbd: jlm.TypeF32},
			Step: Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
		if err != nil {
			t.Fatalf("%dx%d: %v", rows, e, err)
		}
		in, cls := make([]float32, rows*e), make([]float32, e)
		for i := range in {
			in[i] = float32(i) + 0.25
		}
		for i := range cls {
			cls[i] = -float32(i) - 0.5
		}
		got, err := runCPUVectors(codes, in, rows*e, [3][]uint32{}, map[jlm.Role][]float32{jlm.RoleVClassEmbd: cls})
		if err != nil {
			t.Fatalf("%dx%d: %v", rows, e, err)
		}
		for i := range got {
			want := in[i]
			if i < e {
				want = cls[i]
			}
			if got[i] != want {
				t.Fatalf("%dx%d: element %d (row %d) is %v, want %v", rows, e, i, i/e, got[i], want)
			}
		}
		t.Logf("%d rows of %d: the class row first, the rest kept", rows, e)
	}
}

// TestComposedCPUTailRow writes a tower's class row over the last live row
// of its rows on the CPU (the positions' first word) and leaves the rest
// as they were; the live rows stated one fewer, the class row a row early,
// is the violation it must see.
func TestComposedCPUTailRow(t *testing.T) {
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true}
	for _, shape := range [][2]int{{2, 8}, {5, 13}, {577, 64}} {
		rows, e := shape[0], shape[1]
		b := graph.NewBlock(-1)
		x := b.In("x", e)
		b.Out(b.Emit(graph.OpTailRow, []graph.ValueID{x}, []graph.Result{{Name: "class", Width: e}},
			[]graph.Binding{{Role: jlm.RoleVClassEmbd, K: e}}, graph.Attrs{}, nil, nil)[0])
		_, codes, err := r.Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleVClassEmbd: jlm.TypeF32},
			Step: Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
		if err != nil {
			t.Fatalf("%dx%d: %v", rows, e, err)
		}
		in, cls := make([]float32, rows*e), make([]float32, e)
		for i := range in {
			in[i] = float32(i) + 0.25
		}
		for i := range cls {
			cls[i] = -float32(i) - 0.5
		}
		run := func(live int) []float32 {
			pos := make([]uint32, rows+1)
			pos[0] = uint32(live)
			got, err := runCPUBound(codes, in, rows*e, [3][]uint32{}, map[jlm.Role][]float32{jlm.RoleVClassEmbd: cls}, func(o Operand) []uint32 {
				if o.Name == "rope.pos" {
					return pos
				}
				return nil
			})
			if err != nil {
				t.Fatalf("%dx%d: %v", rows, e, err)
			}
			return got
		}
		got := run(rows)
		for i := range got[:rows*e] {
			want := in[i]
			if i >= (rows-1)*e {
				want = cls[i-(rows-1)*e]
			}
			if got[i] != want {
				t.Fatalf("%dx%d: element %d (row %d) is %v, want %v", rows, e, i, i/e, got[i], want)
			}
		}
		bad := run(rows - 1)
		if bad[(rows-2)*e] == got[(rows-2)*e] {
			t.Fatalf("%dx%d: the live rows stated one fewer leave the row before the last a patch's", rows, e)
		}
		t.Logf("%d rows of %d: the class row last, the rest kept; a row early seen", rows, e)
	}
}

// TestComposedCPUPosAxes reads each row's position rows of a table of two
// axes on the CPU: the row at its cell's column plus the row at the axis
// size plus its cell's row, the cell from the step's position on a grid the
// request's word states. The grid stated one wider is the violation it
// must see.
func TestComposedCPUPosAxes(t *testing.T) {
	const S, e, lead = 6, 12, 1
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true}
	tab := make([]float32, 2*S*e)
	for i := range tab {
		tab[i] = float32(i%97)*0.5 - 7
	}
	for _, grid := range [][2]int{{1, 1}, {2, 3}, {3, 5}, {6, 6}, {5, 4}} {
		gh, gw := grid[0], grid[1]
		rows := gh*gw + lead
		b := graph.NewBlock(-1)
		b.In("x", e)
		b.Out(b.Emit(graph.OpPosAxes, nil, []graph.Result{{Name: "pos", Width: e}},
			[]graph.Binding{{Role: jlm.RoleVPosEmbd, K: e, Rows: 2 * S}}, graph.Attrs{Span: S, Offset: lead}, nil, nil)[0])
		_, codes, err := r.Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleVPosEmbd: jlm.TypeF32},
			Step: Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
		if err != nil {
			t.Fatalf("%dx%d: %v", gh, gw, err)
		}
		pos := make([]uint32, rows+1)
		pos[0] = uint32(rows)
		for i := range rows {
			pos[1+i] = uint32(i)
			if i >= lead {
				pos[1+i] = uint32(lead + gh*gw - 1 - (i - lead))
			}
		}
		run := func(width int) []float32 {
			req := make([]uint32, RequestRows+rows)
			req[RequestGrid] = uint32(width)
			words := unsafe.Slice((*uint32)(unsafe.Pointer(&tab[0])), len(tab))
			got, err := runCPUBound(codes, make([]float32, rows*e), rows*e, [3][]uint32{words}, nil, func(o Operand) []uint32 {
				switch o.Name {
				case "request":
					return req
				case "rope.pos":
					return pos
				}
				return nil
			})
			if err != nil {
				t.Fatalf("%dx%d: %v", gh, gw, err)
			}
			return got
		}
		got := run(gw)
		for i := range rows {
			q := max(int(pos[1+i]), lead) - lead
			c, w := q%gw, S+q/gw
			for k := range e {
				want := tab[c*e+k] + tab[w*e+k]
				if got[i*e+k] != want {
					t.Fatalf("%dx%d: row %d element %d is %v, want %v", gh, gw, i, k, got[i*e+k], want)
				}
			}
		}
		bad := run(gw + 1)
		same := true
		for i := range got[:rows*e] {
			same = same && got[i] == bad[i]
		}
		if same && gh*gw > 1 {
			t.Fatalf("%dx%d: a grid stated %d wide reads the same rows", gh, gw, gw+1)
		}
		t.Logf("%dx%d grid: exact, a wrong width seen", gh, gw)
	}
}

// TestComposedCPUPosResample forms each row's position-table row resampled
// to the step's grid on the CPU, for each filter and chain the op states,
// over a picture of more rows than a vector loop holds: the forms fit the
// device's registers, the taps' indices integers in its general registers.
// The bicubic's sixteen do not fit them yet.
func TestComposedCPUPosResample(t *testing.T) {
	const S, e = 8, 64
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true}
	for _, c := range []struct {
		name          string
		chain, filter uint8
	}{{"bilinear, aligned", graph.ChainAligned, graph.FilterBilinear}, {"linear, half-pixel", graph.ChainHalfPixel, graph.FilterLinear}} {
		b := graph.NewBlock(-1)
		x := b.In("x", e)
		b.Out(b.Add(x, b.Emit(graph.OpPosResample, nil, []graph.Result{{Name: "pos", Width: e}},
			[]graph.Binding{{Role: jlm.RoleVPosEmbd, K: e, Rows: S * S}}, graph.Attrs{Span: S, Chain: c.chain, Filter: c.filter}, nil, nil)[0]))
		_, _, err := r.Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleVPosEmbd: jlm.TypeF32},
			Step: Step{Rows: 1024, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		t.Logf("%s: formed", c.name)
	}
}
