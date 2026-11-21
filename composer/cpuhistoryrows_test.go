package composer

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
)

// HistoryPoolGraph is a history pool over generic ops, as a compressor
// states one: a history of rows of rw appended from the graph's inputs k
// and v, then per segment {windows back, value offset, gate offset, bias
// offset} the gate rows (OpHistoryRows) plus a slice of the bias table
// (span rows of the segments' widest bias), the values weighted by their
// softmax per channel over every segment's positions (ColumnSoftmaxSum).
func HistoryPoolGraph(span, width, rw int, segs [][4]uint64, bias jlm.Role) *graph.Graph {
	bw := 0
	for _, s := range segs {
		bw = max(bw, int(s[3])+width)
	}
	b := graph.New()
	h := graph.StateRef{State: graph.StateKV, Version: b.KVAppend(b.In("k", rw), b.In("v", rw), 0)}
	var scores, values []graph.ValueID
	for _, s := range segs {
		g := b.HistoryRows(h, graph.Attrs{Span: span, Width: rw, Offset: int(s[2]), Local: int(s[0]), Const: -math.MaxFloat32}, width)
		t := b.Emit(graph.OpSlice, nil, []graph.Result{{Name: "pool.bias", Width: span * width}}, []graph.Binding{{Role: bias, K: bw, Rows: span}},
			graph.Attrs{HeadDim: bw, Offset: int(s[3]), Width: width}, nil, nil)[0]
		scores = append(scores, b.Emit(graph.OpMap, []graph.ValueID{g, t}, []graph.Result{{Name: "pool.score", Width: span * width}}, nil,
			graph.Attrs{Fn: graph.MapAdd}, nil, nil)[0])
		values = append(values, b.HistoryRows(h, graph.Attrs{Span: span, Width: rw, Offset: int(s[1]), Local: int(s[0])}, width))
	}
	b.Out(b.ColumnSoftmaxSum(scores, values, span, width))
	return b.G
}

// HistoryRowsCases are the shapes the history rows gates sweep: windows
// of span positions, local windows back, width elements of a row of rw
// from offset, const where a position holds none.
func HistoryRowsCases() []HistoryRowsCase {
	return []HistoryRowsCase{
		{Span: 4, Local: 0, RW: 10, Offset: 3, Width: 5, Const: 7.5},
		{Span: 2, Local: 7, RW: 6, Offset: 0, Width: 6, Const: -3},
		{Span: 3, Local: 5, RW: 9, Offset: 4, Width: 5, Const: 0.25},
		{Span: 1, Local: 14, RW: 7, Offset: 6, Width: 1, Const: -1},
		{Span: 5, Local: 3, RW: 12, Offset: 1, Width: 7, Const: 2},
	}
}

// HistoryRowsCase is one of HistoryRowsCases.
type HistoryRowsCase struct {
	Span, Local, RW, Offset, Width int
	Const                          float32
}

// Name names the case.
func (c HistoryRowsCase) Name() string {
	return fmt.Sprintf("span%d-back%d-%dof%dfrom%d", c.Span, c.Local, c.Width, c.RW, c.Offset)
}

// Graph is the case's graph: an append of k and v, then OpHistoryRows.
func (c HistoryRowsCase) Graph() *graph.Graph {
	b := graph.New()
	h := graph.StateRef{State: graph.StateKV, Version: b.KVAppend(b.In("k", c.RW), b.In("v", c.RW), 0)}
	b.Out(b.HistoryRows(h, graph.Attrs{Span: c.Span, Width: c.RW, Offset: c.Offset, Local: c.Local, Const: float64(c.Const)}, c.Width))
	return b.G
}

// Want is the definition for the rows at positions p0.. of history rows:
// per row, window position and element, the element of the position's row,
// or Const where the window starts before the first position or the
// position is past the row's own. It also counts both kinds of element.
func (c HistoryRowsCase) Want(rows [][]float32, p0, n int) (out []float32, held, consts int) {
	out = make([]float32, n*c.Span*c.Width)
	for r := range n {
		t := p0 + r
		ws := t/c.Span*c.Span - c.Local*c.Span
		for j := range c.Span {
			p := ws + j
			for e := range c.Width {
				x := c.Const
				if ws >= 0 && p <= t {
					x, held = rows[p][c.Offset+e], held+1
				} else {
					consts++
				}
				out[(r*c.Span+j)*c.Width+e] = x
			}
		}
	}
	return out, held, consts
}

// TestComposedCPUHistoryRows executes OpHistoryRows alone on the CPU over a
// chunk of rows appended to a paged binary16 history: every element of
// every row's window, Local windows back, held to the definition exactly,
// a Const where the position holds none and the row's elements from Offset
// where it does. A write position moved by one must fail it.
func TestComposedCPUHistoryRows(t *testing.T) {
	for _, c := range HistoryRowsCases() {
		t.Run(c.Name(), func(t *testing.T) { composedCPUHistoryRows(t, c) })
	}
}

func composedCPUHistoryRows(t *testing.T, c HistoryRowsCase) {
	const page, p0, R, capRows, pages = 8, 13, 7, 8, 4
	form := func(g *graph.Graph) ([]Code, error) {
		_, codes, err := Default().Form(Request{Graph: g,
			Step:   Step{Rows: capRows, Page: page, Keys: pages * page, ActWindow: 32, Seqs: 1},
			Device: Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(hostArm().tg.Features().ISA()), ExactDiv: true}})
		return codes, err
	}
	g := c.Graph()
	codes, err := form(g)
	if err != nil {
		t.Fatal(err)
	}
	past := c
	past.Offset = c.RW - c.Width + 1
	if _, err := form(past.Graph()); err == nil {
		t.Errorf("history rows of %d from %d of a row of %d are formed", c.Width, past.Offset, c.RW)
	}
	rng := rand.New(rand.NewSource(3))
	rows := make([][]float32, p0+R)
	for p := range rows {
		rows[p] = make([]float32, c.RW)
		for e := range rows[p] {
			rows[p][e] = float32(quant.DecodeHalf(quant.EncodeHalf(float32(rng.NormFloat64() * 2))))
		}
	}
	k := make([]float32, capRows*c.RW)
	for r := range R {
		copy(k[r*c.RW:], rows[p0+r])
	}
	pool := func() []uint32 {
		out := make([]uint32, pages*page*c.RW/2)
		for p := range p0 {
			for d, x := range rows[p] {
				h, i := uint32(quant.EncodeHalf(x)), p*c.RW+d
				if i%2 == 0 {
					out[i/2] = out[i/2]&0xFFFF0000 | h
				} else {
					out[i/2] = out[i/2]&0xFFFF | h<<16
				}
			}
		}
		return out
	}
	W := c.Span * c.Width
	run := func(shift uint32) []float32 {
		out := make([]float32, capRows*W)
		desc := make([]uint32, capRows*4)
		for r := range R {
			p := uint32(p0+r) + shift
			copy(desc[r*4:], []uint32{0, 0, p + 1, p})
		}
		bufs := map[string][]uint32{"k": f32Words(k), "v": f32Words(k), "rows": desc, "pages": {0, 1, 2, 3}, "rows.n": {R},
			"kv.k": pool(), "kv.v": pool(), "q.off": make([]uint32, capRows*4)}
		bufs[g.Values[g.Out[0]].Name] = f32Words(out)
		if err := runCPURows(hostArm(), codes, g, bufs, R); err != nil {
			t.Fatal(err)
		}
		for _, b := range codes[len(codes)-1].Binds {
			if o := b.Operand; b.Writes && o.Kind == OperandScratch {
				copy(out, wordsF32(bufs["own:"+o.Name]))
			}
		}
		return out[:R*W]
	}
	want, held, consts := c.Want(rows, p0, R)
	if held == 0 || consts == 0 {
		t.Fatalf("the case reads %d held elements and %d consts: it cannot tell the two apart", held, consts)
	}
	got := run(0)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("row %d position %d element %d: %v, want %v (%d held, %d consts)", i/W, i%W/c.Width, i%c.Width, got[i], want[i], held, consts)
		}
	}
	moved, same := run(1), true
	for i := range want {
		same = same && moved[i] == want[i]
	}
	if same {
		t.Errorf("with the write position moved every element agrees: the gate cannot see the window")
	}
}

// TestComposedCPUColumnReduce executes OpReduce with HeadDim on the CPU:
// a row of blocks of Span rows of HeadDim, each column of each block
// combined by add, max and min, held to the definition at every block,
// ragged widths among them. An output not one element per column of a
// block is refused.
func TestComposedCPUColumnReduce(t *testing.T) {
	for _, fn := range []graph.MapFn{graph.MapAdd, graph.MapMax, graph.MapMin} {
		for _, sh := range ColumnReduceShapes() {
			for _, rows := range []int{1, 3} {
				t.Run(fmt.Sprintf("%v/%dx%dx%d/%d", fn, sh[0], sh[1], sh[2], rows), func(t *testing.T) {
					composedCPUColumnReduce(t, fn, sh[0], sh[1], sh[2], rows)
				})
			}
		}
	}
}

// ColumnReduceShapes are the column reduce gates' {blocks, Span, HeadDim}.
func ColumnReduceShapes() [][3]int {
	return [][3]int{{1, 1, 1}, {3, 4, 5}, {2, 7, 3}, {4, 1, 9}, {1, 6, 8}, {5, 3, 17}}
}

// ColumnReduceGraph is OpReduce by fn over a row of g blocks of s rows of
// d, out wide.
func ColumnReduceGraph(fn graph.MapFn, g, s, d, out int) (*graph.Graph, graph.ValueID, graph.ValueID) {
	b := graph.New()
	in := b.In("x", g*s*d)
	y := b.Emit(graph.OpReduce, []graph.ValueID{in}, []graph.Result{{Name: "y", Width: out}}, nil, graph.Attrs{Fn: fn, Span: s, HeadDim: d}, nil, nil)[0]
	b.Out(y)
	return b.G, in, y
}

// ColumnReduceWant is the definition: Out[b*d+j] combines In[(b*s+i)*d+j].
func ColumnReduceWant(fn graph.MapFn, x []float32, rows, g, s, d int) []float64 {
	out := make([]float64, rows*g*d)
	for r := range rows {
		for b := range g {
			for j := range d {
				acc := float64(x[r*g*s*d+b*s*d+j])
				for i := 1; i < s; i++ {
					v := float64(x[r*g*s*d+(b*s+i)*d+j])
					switch fn {
					case graph.MapAdd:
						acc += v
					case graph.MapMax:
						acc = math.Max(acc, v)
					default:
						acc = math.Min(acc, v)
					}
				}
				out[(r*g+b)*d+j] = acc
			}
		}
	}
	return out
}

func composedCPUColumnReduce(t *testing.T, fn graph.MapFn, G, S, D, rows int) {
	g, in, y := ColumnReduceGraph(fn, G, S, D, G*D)
	_, codes, err := Default().Form(Request{Graph: g, Formats: map[jlm.Role]jlm.Type{}, Step: Step{Rows: rows, Page: 64, Keys: 1},
		Device: cpuDevice(hostArm()), Apart: true})
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(int64(G*100 + S*10 + D)))
	x := make([]float32, rows*G*S*D)
	for i := range x {
		x[i] = float32(rng.NormFloat64() * 3)
	}
	vals := map[graph.ValueID][]uint32{in: append([]uint32(nil), f32Words(x)...)}
	if err := runCPUGridValues(hostArm(), codes, g, vals, [3][]uint32{}, nil, func(graph.ValueID) int { return rows }, nil); err != nil {
		t.Fatal(err)
	}
	want := ColumnReduceWant(fn, x, rows, G, S, D)
	for i, w := range want {
		got := float64(math.Float32frombits(vals[y][i]))
		if math.IsNaN(got) || math.Abs(got-w) > 1e-5*(1+math.Abs(w)) {
			t.Fatalf("row %d block %d column %d: %g, want %g", i/(G*D), i%(G*D)/D, i%D, got, w)
		}
	}
	for _, out := range []int{G*D + 1, D} {
		if out == G*D {
			continue
		}
		bad, _, _ := ColumnReduceGraph(fn, G, S, D, out)
		if _, _, err := Default().Form(Request{Graph: bad, Formats: map[jlm.Role]jlm.Type{}, Step: Step{Rows: rows, Page: 64, Keys: 1},
			Device: cpuDevice(hostArm()), Apart: true}); err == nil {
			t.Errorf("a column reduce of %d blocks of %d columns into %d is formed", G, D, out)
		}
	}
}
