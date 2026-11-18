package composer

import (
	"fmt"
	"math"
	"testing"

	"github.com/jitllm/jitllm/graph"
)

// IterateCase is an iteration over a row of N elements, carried Trips
// trips: Words its shift-and-xor chain of three values over words, else
// its counter and the multiply-add that reads it, over floats.
type IterateCase struct {
	Words    bool
	N, Trips int
}

func (c IterateCase) String() string {
	kind := "floats"
	if c.Words {
		kind = "words"
	}
	return fmt.Sprintf("%s_%dx%d", kind, c.N, c.Trips)
}

// IterateCases are the gate's iterations: ragged rows, one trip, a few and
// a thousand.
func IterateCases() []IterateCase {
	var cs []IterateCase
	for _, words := range []bool{true, false} {
		for _, n := range []int{1, 7, 8, 33, 256} {
			for _, trips := range []int{1, 3, 64, 1000} {
				cs = append(cs, IterateCase{Words: words, N: n, Trips: trips})
			}
		}
	}
	return cs
}

// Taps is how many values c carries.
func (c IterateCase) Taps() int {
	if c.Words {
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

// IterateInput is c's row: words or floats, each element its own start.
func IterateInput(c IterateCase) []uint32 {
	x := make([]uint32, c.N)
	for i := range x {
		if c.Words {
			x[i] = uint32(i)*2654435761 + 12345
		} else {
			x[i] = math.Float32bits(float32(i%11)*0.25 - 1)
		}
	}
	return x
}

// IterateWant is c's output, its trips taken in order.
func IterateWant(c IterateCase, x []uint32) []uint32 {
	n := c.Taps()
	want := make([]uint32, c.N*n)
	for i, w := range x {
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
		if c.Words {
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
			x := IterateInput(c)
			got := make([]uint32, c.N*c.Taps())
			if err := runCPUCalls(cpuOn, codes, g, map[string][]uint32{"x": x, g.Values[g.Out[0]].Name: got}); err != nil {
				t.Fatal(err)
			}
			if bad := IterateMismatch(c, got, IterateWant(c, x)); bad != "" {
				t.Fatalf("%d code(s): %s", len(codes), bad)
			}
		})
	}
}
