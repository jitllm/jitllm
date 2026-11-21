package composer

import (
	"fmt"
	"math"
	"testing"

	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
)

// SpanCase is an iteration over a row of N elements carried Trips trips,
// two of its terms each summed over runs of Span elements of the row: two
// floats in one output, value j of element i at i*2+j, where Apart each in
// an output of its own, where Mean each value's step reading the other's
// and its own run's mean, and where Take a word index walking a float table
// beside a float summing the run's takes, each in an output of its own.
type SpanCase struct {
	Apart, Mean, Take bool
	Span, N, Trips    int
}

func (c SpanCase) String() string {
	kind := "interleaved"
	switch {
	case c.Take:
		kind = "take"
	case c.Mean:
		kind = "mean"
	case c.Apart:
		kind = "apart"
	}
	return fmt.Sprintf("%s_%dof%dx%d", kind, c.Span, c.N, c.Trips)
}

// SpanCases are the gate's iterations for runs up to most elements, a walk
// of a table's for runs up to takes and the mean's up to means: rows of one
// run, of a few and ragged against every vector, a row that does not fill
// its last vector, and one of many vectors; one trip, a few and many.
func SpanCases(most, takes, means int) []SpanCase {
	var cs []SpanCase
	for span := 1; span <= most; span *= 2 {
		for _, runs := range []int{1, 2, 3, 5, 12} {
			for _, trips := range []int{1, 3, 200} {
				for _, kind := range []SpanCase{{}, {Apart: true}, {Mean: true}, {Take: true}} {
					if kind.Take && span > takes || kind.Mean && span > means {
						continue
					}
					c := kind
					c.Span, c.N, c.Trips = span, runs*span, trips
					cs = append(cs, c)
				}
			}
		}
	}
	return cs
}

// SpanGraph is c's graph. Over floats, the two values turn through a
// fixed angle a trip, value 1 first reflected about its run's mean (its
// negation summed over the run, times minus two over the run's length, less
// itself): no value grows or fades, and every trip mixes the run. With
// Mean, value 0 is three quarters of itself averaged over its run plus value
// 1, and value 1 half of itself plus one, less a quarter of that averaged
// over its run: one run's sum feeds every element's step, which a form
// holding a run's elements as copies shares among them. With Take,
// value 0 is a word index stepped by 389 and wrapped at the table, and value
// 1 half of itself plus the table's element at the index summed over the
// run, over twice the run's length.
func SpanGraph(c SpanCase) *graph.Graph {
	in := graph.Arg{Kind: graph.ArgIn}
	k := func(v float64) graph.Arg { return graph.Arg{Kind: graph.ArgConst, Const: v} }
	st := func(j int) graph.Arg { return graph.Arg{Kind: graph.ArgState, Index: j} }
	tm := func(j int) graph.Arg { return graph.Arg{Kind: graph.ArgTerm, Index: j} }
	sum := func(j int) graph.Arg { return graph.Arg{Kind: graph.ArgSum, Index: j} }
	inv := 1 / float64(c.Span)
	b := graph.New()
	x := b.In("x", c.N)
	if c.Take {
		b.G.Values[x].Elem = graph.ElemU32
		tab := b.In("ftable", iterateTable)
		terms := []graph.Term{
			{Fn: graph.MapAdd, Args: [3]graph.Arg{in, k(0)}},
			{Fn: graph.MapMul, Args: [3]graph.Arg{in, k(0.25)}},
			{Fn: graph.MapTake, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 1}, st(0)}},
			{Fn: graph.MapMul, Args: [3]graph.Arg{st(1), k(0.5)}},
			{Fn: graph.MapFma, Args: [3]graph.Arg{sum(2), k(0.5 * inv), tm(3)}},
			{Fn: graph.MapAdd, Args: [3]graph.Arg{st(0), k(389)}},
			{Fn: graph.MapAnd, Args: [3]graph.Arg{tm(5), k(iterateTable - 1)}},
		}
		ys := b.Emit(graph.OpIterate, []graph.ValueID{x, tab}, []graph.Result{{Name: "y0", Width: c.N}, {Name: "y1", Width: c.N}}, nil,
			graph.Attrs{Taps: 2, Span: c.Trips, Width: c.Span, Words: []uint64{6, 4}, Expr: terms}, nil, nil)
		b.G.Values[ys[0]].Elem = graph.ElemU32
		b.Out(ys[0])
		b.Out(ys[1])
		return b.G
	}
	if c.Mean {
		terms := []graph.Term{
			{Fn: graph.MapAdd, Args: [3]graph.Arg{in, k(0)}},
			{Fn: graph.MapMul, Args: [3]graph.Arg{in, k(0.5)}},
			{Fn: graph.MapMul, Args: [3]graph.Arg{st(0), k(0.75)}},
			{Fn: graph.MapFma, Args: [3]graph.Arg{sum(2), k(inv), st(1)}},
			{Fn: graph.MapFma, Args: [3]graph.Arg{st(1), k(0.5), k(1)}},
			{Fn: graph.MapFma, Args: [3]graph.Arg{sum(4), k(-0.25 * inv), tm(4)}},
		}
		b.Out(b.Emit(graph.OpIterate, []graph.ValueID{x}, []graph.Result{{Name: "y", Width: 2 * c.N}}, nil,
			graph.Attrs{Taps: 2, Span: c.Trips, Width: c.Span, Words: []uint64{3, 5}, Expr: terms}, nil, nil)[0])
		return b.G
	}
	terms := []graph.Term{
		{Fn: graph.MapAdd, Args: [3]graph.Arg{in, k(0)}},
		{Fn: graph.MapMul, Args: [3]graph.Arg{in, k(0.5)}},
		{Fn: graph.MapMul, Args: [3]graph.Arg{st(1), k(-1)}},
		{Fn: graph.MapFma, Args: [3]graph.Arg{sum(2), k(-2 * inv), tm(2)}},
		{Fn: graph.MapMul, Args: [3]graph.Arg{st(0), k(0.6)}},
		{Fn: graph.MapFma, Args: [3]graph.Arg{tm(3), k(-0.8), tm(4)}},
		{Fn: graph.MapMul, Args: [3]graph.Arg{st(0), k(-1)}},
		{Fn: graph.MapMul, Args: [3]graph.Arg{tm(3), k(0.6)}},
		{Fn: graph.MapFma, Args: [3]graph.Arg{tm(6), k(-0.8), tm(7)}},
	}
	a := graph.Attrs{Taps: 2, Span: c.Trips, Width: c.Span, Words: []uint64{5, 8}, Expr: terms}
	if c.Apart {
		ys := b.Emit(graph.OpIterate, []graph.ValueID{x}, []graph.Result{{Name: "y0", Width: c.N}, {Name: "y1", Width: c.N}}, nil, a, nil, nil)
		b.Out(ys[0])
		b.Out(ys[1])
		return b.G
	}
	b.Out(b.Emit(graph.OpIterate, []graph.ValueID{x}, []graph.Result{{Name: "y", Width: 2 * c.N}}, nil, a, nil, nil)[0])
	return b.G
}

// SpanInputs are c's inputs by name: its row, each element its own start,
// and for a walk the float table.
func SpanInputs(c SpanCase) map[string][]uint32 {
	x := make([]uint32, c.N)
	for i := range x {
		if c.Take {
			x[i] = uint32(i*37) % iterateTable
			continue
		}
		x[i] = math.Float32bits(float32(math.Sin(float64(i)*0.37)) * 1.5)
	}
	in := map[string][]uint32{"x": x}
	if c.Take {
		tab := make([]uint32, iterateTable)
		for i := range tab {
			tab[i] = math.Float32bits(float32(i%17)*0.125 - 1)
		}
		in["ftable"] = tab
	}
	return in
}

// SpanGot is c's values as SpanWant lays them out, value j of element i at
// i*2+j, from its outputs in order.
func SpanGot(c SpanCase, outs [][]uint32) []uint32 {
	if len(outs) == 1 {
		return outs[0][:2*c.N]
	}
	got := make([]uint32, 2*c.N)
	for j, o := range outs {
		for i := range c.N {
			got[i*2+j] = o[i]
		}
	}
	return got
}

// SpanWant is c's output, its trips taken in order, each run's sums the
// oracle's.
func SpanWant(c SpanCase, in map[string][]uint32) []uint32 {
	n := c.N
	inv := float32(1 / float64(c.Span))
	fma := func(a, b, x float32) float32 { return float32(math.FMA(float64(a), float64(b), float64(x))) }
	sums := func(v []float32) []float32 {
		out := make([]float32, len(v))
		if err := oracle.SpanSums(out, v, c.Span); err != nil {
			panic(err)
		}
		return out
	}
	want := make([]uint32, 2*n)
	if c.Take {
		w := append([]uint32(nil), in["x"]...)
		f := make([]float32, n)
		for i := range f {
			f[i] = float32(int32(w[i])) * 0.25
		}
		tk := make([]float32, n)
		for range c.Trips {
			for i := range tk {
				tk[i] = math.Float32frombits(in["ftable"][w[i]])
			}
			s := sums(tk)
			for i := range f {
				f[i] = fma(s[i], float32(0.5*float64(inv)), f[i]*0.5)
				w[i] = (w[i] + 389) & (iterateTable - 1)
			}
		}
		for i := range n {
			want[2*i], want[2*i+1] = w[i], math.Float32bits(f[i])
		}
		return want
	}
	if c.Mean {
		s0, s1 := make([]float32, n), make([]float32, n)
		for i, b := range in["x"] {
			x := math.Float32frombits(b)
			s0[i], s1[i] = x, x*0.5
		}
		a, d := make([]float32, n), make([]float32, n)
		for range c.Trips {
			for i := range n {
				a[i], d[i] = s0[i]*0.75, fma(s1[i], 0.5, 1)
			}
			sa, sd := sums(a), sums(d)
			for i := range n {
				s0[i], s1[i] = fma(sa[i], inv, s1[i]), fma(sd[i], float32(-0.25*float64(inv)), d[i])
			}
		}
		for i := range n {
			want[2*i], want[2*i+1] = math.Float32bits(s0[i]), math.Float32bits(s1[i])
		}
		return want
	}
	s0, s1 := make([]float32, n), make([]float32, n)
	for i, b := range in["x"] {
		x := math.Float32frombits(b)
		s0[i], s1[i] = x, x*0.5
	}
	h := make([]float32, n)
	for range c.Trips {
		for i := range n {
			h[i] = -s1[i]
		}
		sh := sums(h)
		for i := range n {
			r := fma(sh[i], -2*inv, h[i])
			s0[i], s1[i] = fma(r, -0.8, s0[i]*0.6), fma(-s0[i], -0.8, r*0.6)
		}
	}
	for i := range n {
		want[2*i], want[2*i+1] = math.Float32bits(s0[i]), math.Float32bits(s1[i])
	}
	return want
}

// SpanMismatch is the first value of got that is not want's, a word exact
// and a float within 1e-5 of its size, "" for none.
func SpanMismatch(c SpanCase, got, want []uint32) string {
	for i := range want {
		if c.Take && i%2 == 0 {
			if got[i] != want[i] {
				return fmt.Sprintf("element %d's index is %d, want %d", i/2, got[i], want[i])
			}
			continue
		}
		g, w := float64(math.Float32frombits(got[i])), float64(math.Float32frombits(want[i]))
		if math.IsNaN(g) || math.IsNaN(w) || math.Abs(g-w) > 1e-5*max(1, math.Abs(w)) {
			return fmt.Sprintf("element %d's value %d is %v, want %v", i/2, i%2, g, w)
		}
	}
	return ""
}

// spanGuard is the words past each output a gate poisons and holds.
const spanGuard = 16

// SpanForm is what c's final kernels hold: loops whose values are vectors
// of lanes, exchanges across lanes, and sums over a workgroup.
type SpanForm struct{ Wide, Shuffles, GroupSums int }

// SpanFormOf counts ks's wide loops, exchanges and group sums.
func SpanFormOf(ks []*Kernel) SpanForm {
	var f SpanForm
	for _, k := range ks {
		for _, l := range k.Loops {
			if l.Wide > 0 {
				f.Wide++
			}
		}
		for _, nd := range k.Nodes {
			switch nd.Op {
			case OpShuffle:
				f.Shuffles++
			case OpGroupSum:
				f.GroupSums++
			}
		}
	}
	return f
}

// TestComposedCPUIterateSpans holds an iteration summing terms over runs
// of its row on the CPU device to its trips taken in order, every run's
// sums the oracle's: floats within 1e-5, the walked index exact, at runs
// of one element to two vectors, rows of one run to many and ragged against
// the vector, each output poisoned and guarded past its end. A row that
// fills a run and a vector, read and stored by its elements, runs as
// vectors summing each run across lanes, with a vector again at the end of
// a row that does not fill its last; any other runs a trip a run, its
// elements copies. Each case's form is asserted, so a case that ran the
// other form does not pass for this one. A form holds every copy's values in
// registers and the binding spills none: the walk runs to runs of four
// elements, its copies scalar, and the mean to runs of two vectors.
func TestComposedCPUIterateSpans(t *testing.T) { gateComposedCPUIterateSpans(t, hostArm()) }

func gateComposedCPUIterateSpans(t *testing.T, cpuOn cpuArm) {
	dev := Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(cpuOn.tg.Features().ISA()), ExactDiv: true}
	L := 8
	if NarrowsCPU(dev) {
		L = 4
	}
	for _, c := range SpanCases(16, 4, 2*L) {
		t.Run(c.String(), func(t *testing.T) {
			g := SpanGraph(c)
			req := Request{Graph: g, Step: Step{Rows: 1, Page: 64, Keys: 1}, Device: dev, Apart: true}
			ks, _, err := Default().FormKernels(req)
			if err != nil {
				t.Fatal(err)
			}
			f := SpanFormOf(ks)
			vector := !c.Take && c.N >= max(c.Span, L)
			switch {
			case vector && f.Wide == 0:
				t.Fatalf("a row of %d in runs of %d formed no vectors: %+v", c.N, c.Span, f)
			case vector && c.Span > 1 && f.Shuffles == 0:
				t.Fatalf("runs of %d in vectors of %d formed no exchange across lanes: %+v", c.Span, L, f)
			case vector && c.N%L != 0 && c.Span < L && f.Wide != 4:
				t.Fatalf("a row of %d formed %d wide loops, not a vector again at its end: %+v", c.N, f.Wide, f)
			case !vector && (f.Wide != 0 || f.Shuffles != 0):
				t.Fatalf("a row of %d in runs of %d formed vectors: %+v", c.N, c.Span, f)
			}
			_, codes, err := Default().Form(req)
			if err != nil {
				t.Fatal(err)
			}
			in := SpanInputs(c)
			bufs := map[string][]uint32{}
			var outs [][]uint32
			for _, v := range g.Out {
				o := make([]uint32, g.Values[v].Width+spanGuard)
				for i := range o {
					o[i] = 0x7fc0dead
				}
				outs = append(outs, o)
				bufs[g.Values[v].Name] = o
			}
			for name, w := range in {
				bufs[name] = w
			}
			if err := runCPUCalls(cpuOn, codes, g, bufs); err != nil {
				t.Fatal(err)
			}
			for j, o := range outs {
				for _, w := range o[g.Values[g.Out[j]].Width:] {
					if w != 0x7fc0dead {
						t.Fatalf("output %d was written past its end: %#x", j, w)
					}
				}
			}
			if bad := SpanMismatch(c, SpanGot(c, outs), SpanWant(c, in)); bad != "" {
				t.Fatalf("%d code(s), %+v: %s", len(codes), f, bad)
			}
		})
	}
}
