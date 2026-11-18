package composer

import (
	"fmt"
	"math"
	"testing"

	"github.com/jitllm/jitllm/graph"
)

// IterateCase is an iteration over a row of N elements, carried Trips
// trips: Words its shift-and-xor chain of three values over words, Take a
// walk of a table (each trip the word the one before named), else its
// counter and the multiply-add that reads it, over floats.
type IterateCase struct {
	Words, Take bool
	N, Trips    int
}

// iterateTable is the walked table's words: one cycle through them all.
const iterateTable = 1024

func (c IterateCase) String() string {
	kind := "floats"
	switch {
	case c.Take:
		kind = "take"
	case c.Words:
		kind = "words"
	}
	return fmt.Sprintf("%s_%dx%d", kind, c.N, c.Trips)
}

// IterateCases are the gate's iterations: ragged rows, one trip, a few and
// a thousand.
func IterateCases() []IterateCase {
	var cs []IterateCase
	for _, kind := range []IterateCase{{Words: true}, {}, {Take: true}} {
		for _, n := range []int{1, 7, 8, 33, 256} {
			for _, trips := range []int{1, 3, 64, 1000} {
				c := kind
				c.N, c.Trips = n, trips
				cs = append(cs, c)
			}
		}
	}
	return cs
}

// Taps is how many values c carries.
func (c IterateCase) Taps() int {
	switch {
	case c.Take:
		return 1
	case c.Words:
		return 3
	}
	return 2
}

// IterateGraph is c's graph. Over words, value 0 steps by a shift-and-xor
// (<<13, >>17, <<5), value 1 adds value 0 after the trip, value 2 takes the
// exclusive or of value 1 before it shifted right by 3: no trip commutes
// with the next. Over floats, value 0 counts the trips and value 1 is
// 0.75 of itself plus value 0 before the trip.
func IterateGraph(c IterateCase) *graph.Graph {
	in := graph.Arg{Kind: graph.ArgIn}
	k := func(v float64) graph.Arg { return graph.Arg{Kind: graph.ArgConst, Const: v} }
	st := func(j int) graph.Arg { return graph.Arg{Kind: graph.ArgState, Index: j} }
	tm := func(j int) graph.Arg { return graph.Arg{Kind: graph.ArgTerm, Index: j} }
	var terms []graph.Term
	elem := graph.ElemF32
	if c.Take {
		b := graph.New()
		x := b.In("x", c.N)
		b.G.Values[x].Elem = graph.ElemU32
		tab := b.In("table", iterateTable)
		b.G.Values[tab].Elem = graph.ElemU32
		terms = []graph.Term{
			{Fn: graph.MapAdd, Args: [3]graph.Arg{in, k(0)}},
			{Fn: graph.MapTake, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 1}, st(0)}},
		}
		y := b.Emit(graph.OpIterate, []graph.ValueID{x, tab}, []graph.Result{{Name: "y", Width: c.N}}, nil,
			graph.Attrs{Taps: 1, Span: c.Trips, Expr: terms}, nil, nil)[0]
		b.G.Values[y].Elem = graph.ElemU32
		b.Out(y)
		return b.G
	}
	if c.Words {
		elem = graph.ElemU32
		terms = []graph.Term{
			{Fn: graph.MapAdd, Args: [3]graph.Arg{in, k(1)}},
			{Fn: graph.MapXor, Args: [3]graph.Arg{in, k(0x9e3779b9)}},
			{Fn: graph.MapMul, Args: [3]graph.Arg{in, k(7)}},
			{Fn: graph.MapShl, Args: [3]graph.Arg{st(0), k(13)}},
			{Fn: graph.MapXor, Args: [3]graph.Arg{st(0), tm(3)}},
			{Fn: graph.MapShr, Args: [3]graph.Arg{tm(4), k(17)}},
			{Fn: graph.MapXor, Args: [3]graph.Arg{tm(4), tm(5)}},
			{Fn: graph.MapShl, Args: [3]graph.Arg{tm(6), k(5)}},
			{Fn: graph.MapShr, Args: [3]graph.Arg{st(1), k(3)}},
			{Fn: graph.MapXor, Args: [3]graph.Arg{tm(6), tm(7)}},
			{Fn: graph.MapAdd, Args: [3]graph.Arg{st(1), tm(9)}},
			{Fn: graph.MapXor, Args: [3]graph.Arg{st(2), tm(8)}},
		}
	} else {
		terms = []graph.Term{
			{Fn: graph.MapAdd, Args: [3]graph.Arg{in, k(0)}},
			{Fn: graph.MapMul, Args: [3]graph.Arg{in, k(0.5)}},
			{Fn: graph.MapAdd, Args: [3]graph.Arg{st(0), k(1)}},
			{Fn: graph.MapFma, Args: [3]graph.Arg{st(1), k(0.75), st(0)}},
		}
	}
	b := graph.New()
	x := b.In("x", c.N)
	b.G.Values[x].Elem = elem
	y := b.Emit(graph.OpIterate, []graph.ValueID{x}, []graph.Result{{Name: "y", Width: c.N * c.Taps()}}, nil,
		graph.Attrs{Taps: c.Taps(), Span: c.Trips, Expr: terms}, nil, nil)[0]
	b.G.Values[y].Elem = elem
	b.Out(y)
	return b.G
}

// IterateInputs are c's inputs by name: its row, words or floats, each
// element its own start, and for a walk the table.
func IterateInputs(c IterateCase) map[string][]uint32 {
	x := make([]uint32, c.N)
	for i := range x {
		switch {
		case c.Take:
			x[i] = uint32(i*37) % iterateTable
		case c.Words:
			x[i] = uint32(i)*2654435761 + 12345
		default:
			x[i] = math.Float32bits(float32(i%11)*0.25 - 1)
		}
	}
	in := map[string][]uint32{"x": x}
	if c.Take {
		// One cycle: the order of a multiplier coprime to the table's size.
		tab := make([]uint32, iterateTable)
		order := make([]uint32, iterateTable)
		for i := range order {
			order[i] = uint32(i*389+7) % iterateTable
		}
		for i, at := range order {
			tab[at] = order[(i+1)%iterateTable]
		}
		in["table"] = tab
	}
	return in
}

// IterateWant is c's output, its trips taken in order.
func IterateWant(c IterateCase, in map[string][]uint32) []uint32 {
	n := c.Taps()
	want := make([]uint32, c.N*n)
	for i, w := range in["x"] {
		if c.Take {
			for range c.Trips {
				w = in["table"][w]
			}
			want[i] = w
			continue
		}
		if c.Words {
			s0, s1, s2 := w+1, w^0x9e3779b9, w*7
			for range c.Trips {
				d := s0 ^ s0<<13
				d ^= d >> 17
				n0 := d ^ d<<5
				s0, s1, s2 = n0, s1+n0, s2^s1>>3
			}
			want[i*n], want[i*n+1], want[i*n+2] = s0, s1, s2
			continue
		}
		f := math.Float32frombits(w)
		s0, s1 := f, f*0.5
		for range c.Trips {
			s0, s1 = s0+1, float32(math.FMA(float64(s1), 0.75, float64(s0)))
		}
		want[i*n], want[i*n+1] = math.Float32bits(s0), math.Float32bits(s1)
	}
	return want
}

// IterateMismatch is the first element of got that is not want's, words
// exact and floats within 1e-6 of their size, "" for none.
func IterateMismatch(c IterateCase, got, want []uint32) string {
	for i := range want {
		if c.Words || c.Take {
			if got[i] != want[i] {
				return fmt.Sprintf("element %d (value %d of %d) is %#x, want %#x", i/c.Taps(), i%c.Taps(), i/c.Taps(), got[i], want[i])
			}
			continue
		}
		g, w := float64(math.Float32frombits(got[i])), float64(math.Float32frombits(want[i]))
		if math.IsNaN(g) || math.Abs(g-w) > 1e-6*max(1, math.Abs(w)) {
			return fmt.Sprintf("element %d (value %d) is %v, want %v", i/c.Taps(), i%c.Taps(), g, w)
		}
	}
	return ""
}

// TestComposedCPUIterate holds an iteration on the CPU device to its trips
// taken in order: a shift-and-xor chain over words, no two trips of which
// commute, matches the definition exactly at every row width and trip
// count, and a counter with the multiply-add reading it matches within
// 1e-6. A loop whose trips were taken apart, regrouped or cut short
// answers otherwise.
func TestComposedCPUIterate(t *testing.T) { gateComposedCPUIterate(t, hostArm()) }

func gateComposedCPUIterate(t *testing.T, cpuOn cpuArm) {
	dev := Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(cpuOn.tg.Features().ISA()), ExactDiv: true}
	for _, c := range IterateCases() {
		t.Run(c.String(), func(t *testing.T) {
			g := IterateGraph(c)
			_, codes, err := Default().Form(Request{Graph: g, Step: Step{Rows: 1, Page: 64, Keys: 1}, Device: dev, Apart: true})
			if err != nil {
				t.Fatal(err)
			}
			in := IterateInputs(c)
			got := make([]uint32, c.N*c.Taps())
			bufs := map[string][]uint32{g.Values[g.Out[0]].Name: got}
			for name, w := range in {
				bufs[name] = w
			}
			if err := runCPUCalls(cpuOn, codes, g, bufs); err != nil {
				t.Fatal(err)
			}
			if bad := IterateMismatch(c, got, IterateWant(c, in)); bad != "" {
				t.Fatalf("%d code(s): %s", len(codes), bad)
			}
		})
	}
}
