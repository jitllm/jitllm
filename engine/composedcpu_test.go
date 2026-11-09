package engine

import (
	"encoding/binary"
	"fmt"
	"maps"
	"math"
	"math/rand"
	"runtime"
	"slices"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
)

// The statements the front end makes of the attention output gate, of an
// activation from a block's stored numbers and of a norm inside a sublayer,
// executed as the composer forms them for the host's CPU and held to the
// oracle. Each bound is the engine's own for the function, or a multiple of
// one, and each gate runs its violations through the same comparison and must
// see them. Where main has a kernel for the function it runs on the same
// inputs under the same bound.

// expBound is the relative error of the exp the composer builds on. It is a
// degree-5 Taylor sum on the reduced argument |r| <= ln2/2 (composer/pass_exp.go),
// whose remainder is 3.24e-6 of exp(r) at r = -ln2/2, and a few units in the
// last place for the rest. Main's gate for its own exp (engine/device/cpu/exp_test.go)
// holds 3e-6 on the points it samples, which do not include that edge.
const expBound = 3.5e-6

// ulp is one unit in the last place of a float32 in [1, 2).
const ulp = 1.0 / (1 << 23)

// guardBits is the word kept after every output: a kernel that writes past the
// width it was given changes it.
const guardBits = 0x7FC0DEAD

// cpuForm forms g for the host's CPU. Its kernels write an output over an
// input nothing reads after, which a block's residual relies on; here an output
// is a buffer of the gate's own, so that pass is off.
func cpuForm(t *testing.T, g *graph.Graph, roles []jlm.Role) []composer.Code {
	t.Helper()
	formats := map[jlm.Role]jlm.Type{}
	for _, r := range roles {
		formats[r] = jlm.TypeF32
	}
	reg := composer.Default().WithoutInt8(composer.DeviceCPU)
	p := reg.Profiles[composer.DeviceCPU]
	p.Off = map[string]bool{"inplace": true}
	reg.Profiles[composer.DeviceCPU] = p
	_, codes, err := reg.Form(composer.Request{Graph: g, Formats: formats,
		Step:   composer.Step{Rows: 1, Page: 64, Keys: 1, ActWindow: 32},
		Device: composer.Device{Kind: composer.DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}})
	if err != nil {
		t.Fatalf("forming for the CPU: %v", err)
	}
	return codes
}

// cpuExec runs codes in order. A value in bufs is the caller's buffer, a
// vector in vecs and a weight in planes the container's under its role, and
// every other operand a block of NaN the form must write before it reads.
func cpuExec(t *testing.T, codes []composer.Code, vecs map[jlm.Role][]float32, planes map[jlm.Role][3][]uint32, bufs map[graph.ValueID][]float32) {
	t.Helper()
	words := func(f []float32) []uint32 { return unsafe.Slice((*uint32)(unsafe.Pointer(&f[0])), len(f)) }
	reach := map[composer.Operand]int{}
	for _, c := range codes {
		for _, b := range c.Binds {
			reach[b.Operand] = max(reach[b.Operand], int(b.Bytes)/4+16, 1<<10)
		}
	}
	own := map[composer.Operand][]uint32{}
	var keep [][]uint32
	for _, c := range codes {
		if c.CPU == nil {
			t.Fatalf("%s has no CPU code", c.Name)
		}
		code, err := cpu.New().Map(c.CPU)
		if err != nil {
			t.Fatalf("mapping %s: %v", c.Name, err)
		}
		// Each of the code's calls, as one worker runs them (cpuLaunch): a
		// shared kernel's rows in one call, any other at each coordinate of
		// its calls' space, the last axis fastest.
		n, rows := 1, 0
		for _, k := range c.Calls {
			n *= k
		}
		if c.Share > 0 {
			rows, n = c.ShareRows(1)
		}
		coord := func(at, ax int) int {
			if rows > 0 {
				return at * rows / max(c.ShareUnit, 1)
			}
			for a := len(c.Calls) - 1; a > ax; a-- {
				at /= c.Calls[a]
			}
			return at % c.Calls[ax]
		}
		for at := range max(n, 1) {
			var args cpu.Args
			base := unsafe.Pointer(&args)
			for _, a := range c.Args {
				if a.Field < 0 || uintptr(a.Field)+8 > unsafe.Sizeof(args) {
					t.Fatalf("%s: a call word at %d past the argument block", c.Name, a.Field)
				}
				slot := (*uint64)(unsafe.Add(base, a.Field))
				switch {
				case a.Share:
					*slot = uint64(min(rows, c.Share-at*rows))
					continue
				case a.Coord > 0:
					*slot = uint64(coord(at, a.Coord-1))
					continue
				case a.Length || a.Value:
					*slot = uint64(a.Extent)
					continue
				}
				var w []uint32
				o := a.Operand
				switch {
				case a.Words != nil:
					w = append([]uint32(nil), a.Words...)
				case o.Kind == composer.OperandValue && bufs[o.Value] != nil:
					w = words(bufs[o.Value])
				case o.Kind == composer.OperandVector && vecs[o.Role] != nil:
					w = words(vecs[o.Role])
				case o.Kind == composer.OperandWeight && o.Region >= 0 && o.Region < 3 && planes[o.Role][o.Region] != nil:
					w = planes[o.Role][o.Region]
				case o.Kind == composer.OperandVector || o.Kind == composer.OperandWeight:
					t.Fatalf("%s binds %+v, which the gate does not hold", c.Name, o)
				default:
					if own[o] == nil {
						own[o] = make([]uint32, reach[o])
						for i := range own[o] {
							own[o][i] = 0x7fc00000
						}
					}
					w = own[o]
				}
				keep = append(keep, w)
				if a.Word {
					*slot = uint64(w[a.Offset])
					continue
				}
				off, elem := a.Offset, 4
				if a.Elem > 0 {
					elem = a.Elem
				}
				// An address moves a unit of each axis it has a step along
				// (cpuWords).
				for ax, per := range a.Per {
					off += coord(at, ax) * per
				}
				*slot = uint64(uintptr(unsafe.Pointer(&w[0]))) + uint64(off*elem)
			}
			code.Call(&args)
		}
		code.Close()
	}
	runtime.KeepAlive(keep)
}

// cpuRun forms g and runs it once.
func cpuRun(t *testing.T, g *graph.Graph, vecs map[jlm.Role][]float32, bufs map[graph.ValueID][]float32) {
	t.Helper()
	var roles []jlm.Role
	for r := range vecs {
		roles = append(roles, r)
	}
	cpuExec(t, cpuForm(t, g, roles), vecs, nil, bufs)
}

// guarded is an output row of n elements, NaN so that one the form leaves
// unwritten shows, with guard words after it in the same array.
func guarded(n int) (row, back []float32) {
	back = make([]float32, n+8)
	for i := range back {
		if i < n {
			back[i] = float32(math.NaN())
		} else {
			back[i] = math.Float32frombits(guardBits)
		}
	}
	return back[:n], back
}

// checkGuard fails t if what changed the guard words after its n elements.
func checkGuard(t *testing.T, what string, back []float32, n int) {
	t.Helper()
	for _, v := range back[n:] {
		if math.Float32bits(v) != guardBits {
			t.Fatalf("%s wrote past its %d elements", what, n)
		}
	}
}

// relErr is got's distance from want over want's size, 0 where both are the
// same infinity, and infinite where either is not a number.
func relErr(got, want float32) float64 {
	g, w := float64(got), float64(want)
	switch {
	case math.IsNaN(g) || math.IsNaN(w):
		return math.Inf(1)
	case math.IsInf(g, 0) || math.IsInf(w, 0):
		if g == w {
			return 0
		}
		return math.Inf(1)
	case w == 0:
		return math.Abs(g)
	}
	return math.Abs(g-w) / math.Abs(w)
}

// worstRel is the largest relErr of got against want, and where.
func worstRel(got, want []float32) (worst float64, at int) {
	for i := range want {
		if e := relErr(got[i], want[i]); e > worst || math.IsNaN(e) {
			worst, at = e, i
		}
	}
	return worst, at
}

// sweep is x from the places an activation's branches meet and the places
// they cancel -- zero, the Taylor window's edge at one half, a decade of every
// size below it -- and random sizes between, ragged against the lane width.
func sweep(seed int64, limit float64) []float32 {
	xs := []float32{0}
	for _, e := range []float64{1e-30, 1e-12, 1e-8, 1e-6, 1e-4, 1e-3, 0.01, 0.1, 0.25, 0.4999, 0.5, 0.5001, 0.75, 1, 2, 5, 10, 20, 40, 80} {
		if e <= limit {
			xs = append(xs, float32(e), float32(-e))
		}
	}
	rng := rand.New(rand.NewSource(seed))
	for len(xs) < 137 {
		xs = append(xs, float32(rng.NormFloat64()*math.Exp2(rng.Float64()*23-20)))
	}
	return xs
}

// mapOver is one OpMap of x by a, with the bound vector w where it is set.
func mapOver(b *graph.Builder, name string, a graph.Attrs, w []graph.Binding, x ...graph.ValueID) graph.ValueID {
	return b.Emit(graph.OpMap, x, []graph.Result{{Name: name, Width: b.G.Values[x[0]].Width}}, w, a, nil, nil)[0]
}

// unary runs the map fn over xs as the composer forms it.
func unary(t *testing.T, fn graph.MapFn, xs []float32) []float32 {
	t.Helper()
	b := graph.New()
	x := b.In("x", len(xs))
	y := mapOver(b, "y", graph.Attrs{Fn: fn}, nil, x)
	b.Out(y)
	row, back := guarded(len(xs))
	cpuRun(t, b.G, nil, map[graph.ValueID][]float32{x: xs, y: row})
	checkGuard(t, fn.String(), back, len(xs))
	return row
}

// naiveExpm1 is g with each expm1 stated as exp(x) - 1: the form that cancels
// every digit of exp(x) against the one near zero.
func naiveExpm1(g *graph.Graph) *graph.Graph {
	out := *g
	out.Values = slices.Clone(g.Values)
	out.Ops = nil
	for _, o := range g.Ops {
		if o.Kind == graph.OpExpr {
			// One map a term, as an expression whose one map the CPU's
			// registers refuse is formed (composer.ErrRegisterPressure).
			o.Attr.Expr = naiveExpm1Terms(o.Attr.Expr)
			out.Ops = append(out.Ops, exprAsMaps(o, &out.Values)...)
			continue
		}
		if o.Kind != graph.OpMap || o.Attr.Fn != graph.MapExpm1 {
			out.Ops = append(out.Ops, o)
			continue
		}
		out.Values = append(out.Values, graph.Value{Name: "naive.exp", Width: g.Values[o.Out[0]].Width})
		e := graph.ValueID(len(out.Values) - 1)
		out.Ops = append(out.Ops,
			graph.Op{Kind: graph.OpMap, In: o.In, Out: []graph.ValueID{e}, Attr: graph.Attrs{Fn: graph.MapExp}},
			graph.Op{Kind: graph.OpMap, In: []graph.ValueID{e}, Out: o.Out, Attr: graph.Attrs{Fn: graph.MapSub, Const: 1}})
	}
	return &out
}

// TestComposedCPUTanhErf holds the composer's tanh and erf, the maps a stated
// GELU reads, to math.Tanh and math.Erf over the sweep, at absolute bounds
// derived from the composed exp's (expBound, relative): tanh's 1 - 2/(e + 1)
// moves by at most half of e's relative error, erf's 1 - p(t) t e^(-z^2) by
// all of it, its product being at most one, beside erf's rational
// approximation's 1.5e-7 (Abramowitz and Stegun 7.1.26), and each a few
// units of 1 in its subtractions. Each held to the other's definition is
// wrong by its whole value.
func TestComposedCPUTanhErf(t *testing.T) {
	const units = 4 * 0x1p-23
	xs := sweep(13, 80)
	ref := map[graph.MapFn]func(float64) float64{graph.MapTanh: math.Tanh, graph.MapErf: math.Erf}
	bounds := map[graph.MapFn]float64{graph.MapTanh: expBound/2 + units, graph.MapErf: expBound + 1.5e-7 + units}
	for _, fn := range []graph.MapFn{graph.MapTanh, graph.MapErf} {
		bound := bounds[fn]
		got := unary(t, fn, xs)
		other := graph.MapErf
		if fn == graph.MapErf {
			other = graph.MapTanh
		}
		worst, wrong := 0.0, 0.0
		for i, x := range xs {
			e := math.Abs(float64(got[i]) - ref[fn](float64(x)))
			if math.IsNaN(e) {
				e = math.Inf(1)
			}
			worst = max(worst, e)
			wrong = max(wrong, math.Abs(float64(got[i])-ref[other](float64(x))))
		}
		t.Logf("%v: worst absolute error %.3g over %d inputs (bound %g); %.3g from %v", fn, worst, len(xs), bound, wrong, other)
		if !(worst <= bound) {
			t.Errorf("%v is %.3g from its definition, bound %g", fn, worst, bound)
		}
		if !(wrong > 1e3*bound) {
			t.Errorf("%v is only %.3g from %v: the comparison cannot tell them apart", fn, wrong, other)
		}
	}
}

// exprAsMaps is expression o as one OpMap a term, in order: each term's
// operands its inputs, then a bound scalar, then a constant, as OpMap reads
// them, each term but the last a value of its own.
func exprAsMaps(o graph.Op, values *[]graph.Value) []graph.Op {
	width := (*values)[o.Out[0]].Width
	ids := make([]graph.ValueID, len(o.Attr.Expr))
	var ops []graph.Op
	for k, t := range o.Attr.Expr {
		m := graph.Op{Kind: graph.OpMap, Attr: graph.Attrs{Fn: t.Fn}}
		for _, a := range t.Args[:t.Fn.Arity()] {
			switch a.Kind {
			case graph.ArgIn:
				m.In = append(m.In, o.In[a.Index])
			case graph.ArgTerm:
				m.In = append(m.In, ids[a.Index])
			case graph.ArgWeight:
				m.Weight, m.Attr.Offset, m.Attr.Width = []graph.Binding{o.Weight[a.Index]}, a.Elem, 1
			case graph.ArgConst:
				m.Attr.Const = a.Const
			}
		}
		if k == len(o.Attr.Expr)-1 {
			m.Out = o.Out
		} else {
			*values = append(*values, graph.Value{Name: fmt.Sprintf("naive.t%d", k), Width: width})
			ids[k] = graph.ValueID(len(*values) - 1)
			m.Out = []graph.ValueID{ids[k]}
		}
		ops = append(ops, m)
	}
	return ops
}

// naiveExpm1Terms is terms with each expm1 term stated as exp(x) - 1, the
// terms after it renumbered.
func naiveExpm1Terms(terms []graph.Term) []graph.Term {
	var out []graph.Term
	at := make([]int, len(terms))
	for k, t := range terms {
		for j := range t.Args {
			if t.Args[j].Kind == graph.ArgTerm {
				t.Args[j].Index = at[t.Args[j].Index]
			}
		}
		if t.Fn == graph.MapExpm1 {
			t.Fn = graph.MapExp
			out = append(out, t)
			t = term(graph.MapSub, argTerm(len(out)-1), argK(1))
		}
		at[k] = len(out)
		out = append(out, t)
	}
	return out
}

// rotateWindows is g with each map's window of the bound vector k wide, and
// each expression's element of it, moved one element along: the numbers read
// at the wrong places.
func rotateWindows(g *graph.Graph, k int) *graph.Graph {
	out := *g
	out.Ops = slices.Clone(g.Ops)
	for i := range out.Ops {
		o := &out.Ops[i]
		if o.Kind == graph.OpMap && o.Attr.Width > 0 {
			o.Attr.Offset = (o.Attr.Offset + 1) % k
		}
		if o.Kind == graph.OpExpr {
			o.Attr.Expr = slices.Clone(o.Attr.Expr)
			for t := range o.Attr.Expr {
				for j := range o.Attr.Expr[t].Args {
					if a := &o.Attr.Expr[t].Args[j]; a.Kind == graph.ArgWeight {
						a.Elem = (a.Elem + 1) % k
					}
				}
			}
		}
	}
	return &out
}

// TestComposedCPUExpm1 holds the composer's expm1 to math.Expm1 in float64
// over the sweep. The composed exp is held to its derived bound first, and the
// bounds past the Taylor window are multiples of it: exp(x) - 1 amplifies
// exp's relative error by at most e^x/(e^x - 1) = 2.54 at the window's edge.
// Within the window the form is a Taylor sum of a few units in the last place,
// so its bound is the same 4e-7 the first version of this gate held every
// input to. The naive exp(x) - 1 over the same sweep is wrong by its whole
// value near zero, so the comparison sees a form that lost the small branch.
func TestComposedCPUExpm1(t *testing.T) {
	const window = 4e-7
	past := 2.6*expBound + ulp
	xs := sweep(11, 80)
	n := len(xs)
	exp := unary(t, graph.MapExp, xs)
	expWant := make([]float32, n)
	want := make([]float32, n)
	for i, x := range xs {
		expWant[i] = float32(math.Exp(float64(x)))
		want[i] = float32(math.Expm1(float64(x)))
	}
	if e, at := worstRel(exp, expWant); !(e <= expBound) {
		t.Fatalf("the composed exp is %.3g from math.Exp at %g, past the bound %g the bounds below stand on", e, xs[at], expBound)
	} else {
		t.Logf("exp: worst relative error %.3g (at %g), bound %g", e, xs[at], expBound)
	}
	// worst is the largest relative error within the Taylor window and past it.
	worst := func(got []float32) (in, out float64) {
		for i, x := range xs {
			e := relErr(got[i], want[i])
			if math.Abs(float64(x)) < 0.5 {
				in = max(in, e)
			} else {
				out = max(out, e)
			}
		}
		return in, out
	}
	in, out := worst(unary(t, graph.MapExpm1, xs))
	t.Logf("expm1: worst relative error %.3g within the window (bound %.3g) and %.3g past it (bound %.3g), over %d inputs", in, window, out, past, n)
	if !(in <= window) || !(out <= past) {
		t.Errorf("expm1 is %.3g within the window and %.3g past it from math.Expm1, bounds %.3g and %.3g", in, out, window, past)
	}

	b := graph.New()
	x := b.In("x", n)
	y := mapOver(b, "y", graph.Attrs{Fn: graph.MapExpm1}, nil, x)
	b.Out(y)
	row, back := guarded(n)
	cpuRun(t, naiveExpm1(b.G), nil, map[graph.ValueID][]float32{x: xs, y: row})
	checkGuard(t, "the naive expm1", back, n)
	bad, _ := worst(row)
	t.Logf("exp(x) - 1: worst relative error %.3g within the window", bad)
	if !(bad > 1e3*window) {
		t.Errorf("exp(x) - 1 is only %.3g from math.Expm1 within the window: the comparison cannot see a lost small branch", bad)
	}
}

// TestComposedCPUMapWindow holds a map's bound vector window to the vector's
// elements: a window of one is the scalar at its offset, a window of two is
// two groups of the row. A window one element off is a different product.
func TestComposedCPUMapWindow(t *testing.T) {
	const n = 8
	vec := []float32{2, 3, 5, 7}
	xs := make([]float32, n)
	for i := range xs {
		xs[i] = float32(i + 1)
	}
	run := func(off, width int) []float32 {
		b := graph.New()
		x := b.In("x", n)
		y := mapOver(b, "y", graph.Attrs{Fn: graph.MapMul, Offset: off, Width: width},
			[]graph.Binding{{Role: jlm.RoleXIELU, K: len(vec)}}, x)
		b.Out(y)
		row, back := guarded(n)
		cpuRun(t, b.G, map[jlm.Role][]float32{jlm.RoleXIELU: vec}, map[graph.ValueID][]float32{x: xs, y: row})
		checkGuard(t, "the windowed map", back, n)
		return row
	}
	for _, c := range []struct{ off, width int }{{0, 1}, {1, 1}, {2, 1}, {3, 1}, {0, 2}, {2, 2}, {0, 4}, {0, 0}} {
		want := make([]float32, n)
		for i, x := range xs {
			if c.width == 0 {
				want[i] = x * vec[i/(n/len(vec))]
			} else {
				want[i] = x * vec[c.off+i/(n/c.width)]
			}
		}
		got := run(c.off, c.width)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("window %d from %d: element %d is %g, want %g", c.width, c.off, i, got[i], want[i])
			}
		}
		if c.width == 1 {
			// The violation: one element along reads another scalar.
			if bad := run((c.off+1)%len(vec), 1); bad[0] == want[0] {
				t.Errorf("window 1 from %d and from %d read the same product: the comparison cannot see a wrong window", c.off, (c.off+1)%len(vec))
			}
		}
	}
}

// TestComposedCPUHeadSlices holds the split of a double-width projection to
// the per-head deinterleave -- head h's query is [h*2d, h*2d+d), its gate the
// d after -- and the gated output to sigmoid(gate)*attention. The gated
// output's bound is exp's, since 1/(1 + e^-g) carries exp's relative error at
// most, and a unit in the last place for each of the quotient and the product.
// Main's kernel for the gate runs on the same rows under the same bound. A
// query read from the gate's half, or a gate from the query's, is another
// output.
func TestComposedCPUHeadSlices(t *testing.T) {
	bound := expBound + ulp
	for _, c := range []struct{ heads, d int }{{4, 8}, {3, 7}, {2, 64}} {
		qdim := c.heads * c.d
		rng := rand.New(rand.NewSource(int64(c.heads*100 + c.d)))
		qg := make([]float32, 2*qdim)
		att := make([]float32, qdim)
		for i := range qg {
			qg[i] = float32(rng.NormFloat64() * 2)
		}
		for i := range att {
			att[i] = float32(rng.NormFloat64())
		}
		wantQ, wantOut, gate := make([]float32, qdim), make([]float32, qdim), make([]float32, qdim)
		for h := range c.heads {
			for j := range c.d {
				wantQ[h*c.d+j] = qg[h*2*c.d+j]
				gate[h*c.d+j] = qg[h*2*c.d+c.d+j]
				wantOut[h*c.d+j] = float32(float64(att[h*c.d+j]) / (1 + math.Exp(-float64(gate[h*c.d+j]))))
			}
		}
		run := func(qOff, gateOff int) (q, out, qBack, outBack []float32) {
			b := graph.New()
			in := b.In("qg", 2*qdim)
			a := b.In("att", qdim)
			slice := func(name string, off int) graph.ValueID {
				return b.Emit(graph.OpSlice, []graph.ValueID{in}, []graph.Result{{Name: name, Width: qdim}}, nil,
					graph.Attrs{HeadDim: 2 * c.d, Offset: off, Width: c.d}, nil, nil)[0]
			}
			qv, gv := slice("q", qOff), slice("gate", gateOff)
			ov := sigmoidGate(b, gv, a, qdim)
			b.Out(qv, ov)
			q, qBack = guarded(qdim)
			out, outBack = guarded(qdim)
			cpuRun(t, b.G, nil, map[graph.ValueID][]float32{in: qg, a: att, qv: q, ov: out})
			return q, out, qBack, outBack
		}
		q, out, qBack, outBack := run(0, c.d)
		checkGuard(t, "the query split", qBack, qdim)
		checkGuard(t, "the gated output", outBack, qdim)
		if e, at := worstRel(q, wantQ); e != 0 {
			t.Fatalf("%d heads of %d: the query split is %.3g from the deinterleave, at %d", c.heads, c.d, e, at)
		}
		e, at := worstRel(out, wantOut)
		if !(e <= bound) {
			t.Fatalf("%d heads of %d: the gated output is %.3g from sigmoid(gate)*attention, at %d, bound %.3g", c.heads, c.d, e, at, bound)
		}
		t.Logf("%d heads of %d: the split is exact, the gated output %.3g from the definition, bound %.3g", c.heads, c.d, e, bound)
		badQ, _, _, _ := run(c.d, c.d)
		if e, _ := worstRel(badQ, wantQ); !(e > 1e-2) {
			t.Errorf("%d heads of %d: a query read from the gate's half is only %.3g from the query", c.heads, c.d, e)
		}
		_, badOut, _, _ := run(0, 0)
		if e, _ := worstRel(badOut, wantOut); !(e > 1e-2) {
			t.Errorf("%d heads of %d: a gate read from the query's half moves the output by only %.3g", c.heads, c.d, e)
		}
	}
}

// xieluEngineTol is the error the engine's gate allows the CPU xIELU kernel at
// x, a few units in the last place of each term the result is a sum of: the
// function engine/device/cpu's TestXIELUMatchesTheOracle holds main's kernel to, copied
// from it. The device kernel's gate (backend's TestXIELUMatchesTheOracle) takes
// 2e-6 of the same sum, for the backend's own exp.
func xieluEngineTol(x float64, p [4]float32) float64 {
	ap, an, b, e := float64(p[0]), float64(p[1]), float64(p[2]), float64(p[3])
	m := math.Min(x, e)
	return 4e-7*(math.Abs(ap*x*x)+math.Abs(b*x)+math.Abs(an)*(math.Abs(math.Expm1(m)-m)+math.Abs(m-x)+math.Abs(m))) + 1e-30
}

// xieluTol is that bound with exp's error added where the form takes exp. Below
// -1/2 the activation carries exp(m), and exp is a degree-5 Taylor sum good to
// expBound, not to a few units in the last place. The engine's gate passes
// main's kernel on the inputs it samples and does not sample the arguments
// where exp's remainder peaks, one of them -1.038: on this gate's inputs
// main's own kernel reads up to 1.6 times the engine's bound (the log has it
// for each set of numbers, beside the composed form's, which reads the same).
// Elsewhere the engine's bound stands.
func xieluTol(x float64, p [4]float32) float64 {
	t := xieluEngineTol(x, p)
	if m := math.Min(x, float64(p[3])); m <= -0.5 {
		t += expBound * math.Abs(float64(p[1])) * math.Exp(m)
	}
	return t
}

// xieluNumbers are engine/device/cpu's three sets of the block's effective numbers
// (alpha_p, alpha_n, beta, eps) and three of like size, an eps wide enough that
// min(x, eps) clamps a visible band of inputs and alphas far apart so a swap is
// loud.
func xieluNumbers() [][4]float32 {
	return [][4]float32{
		{0.8, 0.95, 0.5, -1e-6}, {1.31, 0.42, 0.37, -0.25}, {0.05, 2.7, 0.9, -0.0625},
		{0.9, 0.6, 0.5, -1e-6}, {1.3, 0.8, 0.4, -0.02}, {0.2, 1.1, 0.7, -0.5},
	}
}

// xieluInputs mixes the regions the kernel distinguishes, as engine/device/cpu's gate
// does: the polynomial's [-1/2, 0], exp below it, values near zero where
// expm1 - x cancels, the eps band, deep negatives where exp underflows, and
// positives.
func xieluInputs(n int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	x := make([]float32, n)
	for i := range x {
		switch i % 6 {
		case 0:
			x[i] = float32(rng.NormFloat64() * 3)
		case 1:
			x[i] = -float32(rng.Float64()) * 0.5
		case 2:
			x[i] = float32(rng.NormFloat64() * 1e-4)
		case 3:
			x[i] = -0.5 - float32(rng.Float64())*4
		case 4:
			x[i] = -20 - float32(rng.Float64())*100
		default:
			x[i] = float32(rng.Float64()) * 5
		}
	}
	return x
}

// expEdges are the arguments, for eight periods of exp's range reduction
// below zero, where the reduced argument sits just inside either edge of
// [-ln2/2, ln2/2] and exp's Taylor remainder peaks.
func expEdges() []float32 {
	var xs []float32
	for n := -1; n >= -8; n-- {
		for _, r := range []float64{-math.Ln2 / 2 * 0.9999, math.Ln2 / 2 * 0.9999} {
			xs = append(xs, float32(float64(n)*math.Ln2+r))
		}
	}
	return xs
}

// xieluGraph is the activation the converter states for Apertus's ungated
// FFN, as the front end copies it, over a row of n.
func xieluGraph(t *testing.T, n int) (g *graph.Graph, x, y graph.ValueID) {
	t.Helper()
	e, err := convert.Activation(&jlm.Config{NLayer: 1}, 0, false, true)
	if err != nil {
		t.Fatal(err)
	}
	b := graph.New()
	x = b.In("x", n)
	if y, err = b.Stated("act", &e, []graph.ValueID{x}, n, graph.Facts{Inventory: map[jlm.Role]graph.Shape{jlm.RoleXIELU: {K: 4}}}); err != nil {
		t.Fatal(err)
	}
	b.Out(y)
	return b.G, x, y
}

// TestComposedCPUXIELU holds the activation the front end states from a
// block's four numbers to the oracle's xIELU (a float64 expm1, so another
// method from the form's Taylor window), at ragged widths with a guard after
// the output, under the bound the engine holds main's CPU kernel to with exp's
// error added where the form takes exp (xieluTol). Main's kernel runs on the
// same inputs under the same bound, and the log says how far each stands from
// the engine's own bound and from the result. Each way of reading the numbers
// wrongly -- the alphas exchanged, beta dropped, eps ignored, the windows of
// the stored vector one element off, expm1 stated as exp - 1 -- leaves the
// bound by a wide margin.
func TestComposedCPUXIELU(t *testing.T) {
	type family struct {
		name string
		xs   []float32
	}
	var fams []family
	for _, n := range []int{1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100} {
		fams = append(fams, family{fmt.Sprintf("width %d", n), xieluInputs(n, int64(n*31))})
	}
	fams = append(fams, family{"the sweep", sweep(23, 80)}, family{"exp's edges", expEdges()})
	kinds := []string{"ragged widths", "the sweep", "exp's edges"}

	// dist is the largest distance of got from the oracle as a fraction of the
	// bound, of the engine's own bound, and of the result: the last is the
	// metric this gate was first written with, and no float32 evaluation of
	// the terms meets a bound of it where they cancel.
	type dist struct{ bound, engine, result float64 }
	distOf := func(xs, got []float32, p [4]float32) (d dist) {
		for i, v := range xs {
			x := float64(v)
			want := oracle.XIELU(x, float64(p[0]), float64(p[1]), float64(p[2]), float64(p[3]))
			e := math.Abs(float64(got[i]) - want)
			if math.IsNaN(e) {
				e = math.Inf(1)
			}
			d.bound = max(d.bound, e/xieluTol(x, p))
			d.engine = max(d.engine, e/xieluEngineTol(x, p))
			if want != 0 {
				d.result = max(d.result, e/math.Abs(want))
			}
		}
		return d
	}
	worse := func(a, b dist) dist {
		return dist{max(a.bound, b.bound), max(a.engine, b.engine), max(a.result, b.result)}
	}
	// figs is the worst of each family kind for each set of numbers.
	type fig struct{ comp dist }
	figs := make([][]fig, len(kinds))
	for k := range figs {
		figs[k] = make([]fig, len(xieluNumbers()))
	}
	// A violation is asked to be seen where it moves the result. An eps one
	// part in a million under zero clamps nothing, so ignoring it changes
	// nothing (the engine's own exception); and exp(x) - 1 for expm1 is wrong
	// only near zero, which min(x, eps) reaches only where eps does.
	seenWhere := func(name string, p [4]float32) bool {
		switch name {
		case "eps ignored":
			return p[3] <= -1e-3
		case "expm1 stated as exp(x) - 1":
			return p[3] > -1e-3
		}
		return true
	}
	seen := map[string][]float64{}
	for _, fam := range fams {
		n, kind := len(fam.xs), 0
		if k := slices.Index(kinds, fam.name); k >= 0 {
			kind = k
		}
		g, x, y := xieluGraph(t, n)
		codes := cpuForm(t, g, []jlm.Role{jlm.RoleXIELU})
		t.Logf("%s: %d kernels", fam.name, len(codes))
		run := func(codes []composer.Code, p [4]float32) []float32 {
			row, back := guarded(n)
			cpuExec(t, codes, map[jlm.Role][]float32{jlm.RoleXIELU: p[:]}, nil, map[graph.ValueID][]float32{x: fam.xs, y: row})
			checkGuard(t, fam.name, back, n)
			return row
		}
		naive := cpuForm(t, naiveExpm1(g), []jlm.Role{jlm.RoleXIELU})
		rotated := cpuForm(t, rotateWindows(g, 4), []jlm.Role{jlm.RoleXIELU})
		for pi, p := range xieluNumbers() {
			got := run(codes, p)
			dc := distOf(fam.xs, got, p)
			f := &figs[kind][pi]
			f.comp = worse(f.comp, dc)
			if !(dc.bound <= 1) {
				t.Errorf("%s, numbers %v: the composed form is %.3g of the bound from the oracle", fam.name, p, dc.bound)
			}
			for name, v := range map[string]struct {
				codes []composer.Code
				p     [4]float32
			}{
				"the alphas exchanged":       {codes, [4]float32{p[1], p[0], p[2], p[3]}},
				"beta dropped":               {codes, [4]float32{p[0], p[1], 0, p[3]}},
				"eps ignored":                {codes, [4]float32{p[0], p[1], p[2], float32(math.Inf(1))}},
				"eps moved by a quarter":     {codes, [4]float32{p[0], p[1], p[2], p[3] - 0.25}},
				"the windows one along":      {rotated, p},
				"expm1 stated as exp(x) - 1": {naive, p},
			} {
				if !seenWhere(name, p) {
					continue
				}
				if seen[name] == nil {
					seen[name] = make([]float64, len(xieluNumbers()))
				}
				seen[name][pi] = max(seen[name][pi], distOf(fam.xs, run(v.codes, v.p), p).bound)
			}
		}
	}
	for k, kind := range kinds {
		for pi, p := range xieluNumbers() {
			f := figs[k][pi]
			t.Logf("%s, numbers %v: composed %.3f of the bound, %.3f of the engine's, %.3g of the result",
				kind, p, f.comp.bound, f.comp.engine, f.comp.result)
		}
	}
	for _, name := range slices.Sorted(maps.Keys(seen)) {
		weakest, strongest := math.Inf(1), 0.0
		for pi, p := range xieluNumbers() {
			if seenWhere(name, p) {
				weakest, strongest = min(weakest, seen[name][pi]), max(strongest, seen[name][pi])
			}
		}
		t.Logf("violation %q: %.3g of the bound at the least, %.3g at the most, over the sets of numbers", name, weakest, strongest)
		if weakest <= 100 {
			t.Errorf("%s: only %.3g of the bound from the oracle for some set of numbers: the comparison cannot see it", name, weakest)
		}
	}
}

// rowsNMSE is the squared distance of got from want over want's squared size.
func rowsNMSE(got, want []float32) float64 {
	var num, den float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		num, den = num+d*d, den+float64(want[i])*float64(want[i])
	}
	return num / den
}

// TestComposedCPUSubNorm holds a norm inside a sublayer -- the attention's
// output before the output projection, the gated product before down -- to the
// oracle: an RMSNorm with the block's own scale, then the next projection's
// matrix product. The bound is the NMSE TestComposedCPUMatVec (jit) holds the
// composer's projection to. The scale all ones, or the norm left out, leaves
// it by orders of magnitude. It is the only execution of these norms: no
// container reaches them (TestFrontStatesTheNormsInsideASublayer).
func TestComposedCPUSubNorm(t *testing.T) {
	const eps, bound = 1e-5, 1e-9
	for _, c := range []struct {
		name       string
		norm, proj jlm.Role
		k, rows    int
	}{
		{"the attention's output before the output projection", jlm.RoleAttnSubNorm, jlm.RoleAttnOut, 96, 40},
		{"the gated product before down", jlm.RoleFFNSubNorm, jlm.RoleFFNDown, 160, 24},
	} {
		t.Run(c.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(c.k)))
			x, scale, ones := make([]float32, c.k), make([]float32, c.k), make([]float32, c.k)
			for i := range x {
				x[i], scale[i], ones[i] = float32(rng.NormFloat64()*3), float32(0.5+rng.Float64()), 1
			}
			wb := make([]byte, 4*c.rows*c.k)
			for i := 0; i < c.rows*c.k; i++ {
				binary.LittleEndian.PutUint32(wb[4*i:], math.Float32bits(float32(rng.NormFloat64())))
			}
			qs, d, sc, err := layout.PackWeights(layout.Float32, wb, c.rows, c.k)
			if err != nil {
				t.Fatal(err)
			}
			normed := make([]float32, c.k)
			oracle.RMSNorm32(normed, x, scale, eps)
			want := make([]float32, c.rows)
			if err := oracle.MatVec(want, quant.F32, wb, normed, c.rows, c.k); err != nil {
				t.Fatal(err)
			}
			run := func(withNorm bool, vec []float32) []float32 {
				b := graph.New()
				in := b.In("x", c.k)
				row := in
				if withNorm {
					row = b.Norm(in, c.norm, eps)
				}
				y := b.MatVec(row, c.proj, c.rows)
				b.Out(y)
				roles := []jlm.Role{c.proj}
				if withNorm {
					roles = append(roles, c.norm)
				}
				out, back := guarded(c.rows)
				cpuExec(t, cpuForm(t, b.G, roles), map[jlm.Role][]float32{c.norm: vec},
					map[jlm.Role][3][]uint32{c.proj: {qs, d, sc}}, map[graph.ValueID][]float32{in: x, y: out})
				checkGuard(t, c.name, back, c.rows)
				return out
			}
			e := rowsNMSE(run(true, scale), want)
			t.Logf("NMSE %.3g against the oracle, bound %g", e, float64(bound))
			if math.IsNaN(e) || !(e < bound) {
				t.Fatalf("NMSE %.3g against the oracle, bound %g", e, float64(bound))
			}
			if bad := rowsNMSE(run(true, ones), want); !(bad > 1e3*bound) {
				t.Errorf("a scale of ones is only %.3g from the oracle: the comparison cannot see the norm's own weight", bad)
			}
			if bad := rowsNMSE(run(false, nil), want); !(bad > 1e3*bound) {
				t.Errorf("the norm left out is only %.3g from the oracle: the comparison cannot see it", bad)
			}
		})
	}
}

// TestComposedCPUHeadSheets runs the two statements a latent attention block
// adds, formed for the host's CPU: a matvec of each head of a row through its
// own sheet of a bank (an expert matvec with no selection), held to the sum
// at f64 within the f32 sum's worst rounding, and two rows joined head by
// head, held exact. The bank's sheets in another order, and the rows joined
// end to end, must each be seen.
func TestComposedCPUHeadSheets(t *testing.T) {
	for _, c := range []struct{ heads, k, rows, tail int }{{4, 16, 16, 8}, {3, 7, 5, 2}, {2, 64, 128, 64}} {
		rng := rand.New(rand.NewSource(int64(c.heads*1000 + c.k*10 + c.rows)))
		x := make([]float32, c.heads*c.k)
		bank := make([]float32, c.heads*c.rows*c.k)
		tail := make([]float32, c.heads*c.tail)
		for _, s := range [][]float32{x, bank, tail} {
			for i := range s {
				s[i] = float32(rng.NormFloat64())
			}
		}
		want := make([]float32, c.heads*c.rows)
		bound := make([]float64, c.heads*c.rows)
		for h := range c.heads {
			for r := range c.rows {
				sum, mag := 0.0, 0.0
				for j := range c.k {
					p := float64(bank[(h*c.rows+r)*c.k+j]) * float64(x[h*c.k+j])
					sum, mag = sum+p, mag+math.Abs(p)
				}
				want[h*c.rows+r], bound[h*c.rows+r] = float32(sum), float64(c.k+1)*ulp/2*mag+math.Abs(sum)*ulp
			}
		}
		joined := make([]float32, c.heads*(c.rows+c.tail))
		for h := range c.heads {
			copy(joined[h*(c.rows+c.tail):], want[h*c.rows:(h+1)*c.rows])
			copy(joined[h*(c.rows+c.tail)+c.rows:], tail[h*c.tail:(h+1)*c.tail])
		}
		run := func(sheets []float32, heads int) (y, out []float32) {
			b := graph.New()
			in := b.In("x", c.heads*c.k)
			tv := b.In("tail", c.heads*c.tail)
			yv := b.Emit(graph.OpExpertMatVec, []graph.ValueID{in}, []graph.Result{{Name: "y", Width: c.heads * c.rows}},
				[]graph.Binding{{Role: jlm.RoleAttnKB, K: c.k, Rows: c.rows, Sheets: c.heads}}, graph.Attrs{}, nil, nil)[0]
			ov := b.Emit(graph.OpConcat, []graph.ValueID{yv, tv}, []graph.Result{{Name: "joined", Width: c.heads * (c.rows + c.tail)}}, nil,
				graph.Attrs{NHead: heads}, nil, nil)[0]
			b.Out(ov)
			y, yBack := guarded(c.heads * c.rows)
			out, oBack := guarded(c.heads * (c.rows + c.tail))
			w := unsafe.Slice((*uint32)(unsafe.Pointer(&sheets[0])), len(sheets))
			cpuExec(t, cpuForm(t, b.G, []jlm.Role{jlm.RoleAttnKB}), nil, map[jlm.Role][3][]uint32{jlm.RoleAttnKB: {w}},
				map[graph.ValueID][]float32{in: x, tv: tail, yv: y, ov: out})
			checkGuard(t, "the head matvec", yBack, len(y))
			checkGuard(t, "the head-wise join", oBack, len(out))
			return y, out
		}
		y, out := run(bank, c.heads)
		worst := 0.0
		for i := range want {
			e := math.Abs(float64(y[i]) - float64(want[i]))
			if !(e <= bound[i]) {
				t.Fatalf("%d heads of %d into %d: element %d is %.3g from the sum, bound %.3g", c.heads, c.k, c.rows, i, e, bound[i])
			}
			worst = max(worst, e/max(bound[i], 1e-30))
		}
		for h := range c.heads {
			copy(joined[h*(c.rows+c.tail):], y[h*c.rows:(h+1)*c.rows])
		}
		if i := slices.IndexFunc(joined, func(v float32) bool { return false }); !slices.Equal(out, joined) {
			t.Fatalf("%d heads: the join is not each head's rows then its tail (%d)", c.heads, i)
		}
		t.Logf("%d heads of %d into %d: the head matvec at %.3g of its bound, the join exact", c.heads, c.k, c.rows, worst)
		rev := make([]float32, len(bank))
		for h := range c.heads {
			copy(rev[h*c.rows*c.k:], bank[(c.heads-1-h)*c.rows*c.k:(c.heads-h)*c.rows*c.k])
		}
		if bad, _ := run(rev, c.heads); slices.Equal(bad, y) {
			t.Errorf("%d heads: the bank's sheets reversed leave the head matvec as it was", c.heads)
		}
		if _, bad := run(bank, 0); slices.Equal(bad, out) {
			t.Errorf("%d heads: the rows joined end to end leave the head-wise join as it was", c.heads)
		}
	}
}

// TestComposedCPUReduceTree holds a pairwise reduce (graph.OrderTree, a
// stated activation's mean) to its definition: each level combines element
// i with element i+w/2, from the group's width rounded up to a power of two,
// an element whose pair is past the group passed on alone; to the bit, at
// ragged widths and several groups to a row, nothing written past the
// output. The violations: values whose sequential sum differs from the
// pairwise one, and negative zeros, which a pad of +0 would turn positive.
func TestComposedCPUReduceTree(t *testing.T) {
	tree := func(xs []float32, fn func(a, b float32) float32) float32 {
		v := slices.Clone(xs)
		w := 1
		for w < len(v) {
			w <<= 1
		}
		for n := len(v); w > 1; w /= 2 {
			for i := range w / 2 {
				if i+w/2 < n {
					v[i] = fn(v[i], v[i+w/2])
				}
			}
			n = w / 2
		}
		return v[0]
	}
	run := func(xs []float32, fn graph.MapFn, span int) []float32 {
		b := graph.New()
		x := b.In("x", len(xs))
		y := b.Emit(graph.OpReduce, []graph.ValueID{x}, []graph.Result{{Name: "y", Width: len(xs) / span}}, nil,
			graph.Attrs{Fn: fn, Order: graph.OrderTree, Span: span}, nil, nil)[0]
		b.Out(y)
		row, back := guarded(len(xs) / span)
		cpuRun(t, b.G, nil, map[graph.ValueID][]float32{x: xs, y: row})
		checkGuard(t, "the pairwise reduce", back, len(xs)/span)
		return row
	}
	sum := func(a, b float32) float32 { return a + b }
	maxOf := func(a, b float32) float32 { return max(a, b) }
	rng := rand.New(rand.NewSource(7))
	for _, n := range []int{1, 2, 3, 4, 5, 7, 8, 16, 33, 60, 64, 100, 256} {
		for _, groups := range []int{1, 3} {
			xs := make([]float32, n*groups)
			for i := range xs {
				xs[i] = float32(rng.NormFloat64()) * float32(math.Pow(10, float64(rng.Intn(6))))
			}
			for _, c := range []struct {
				name string
				fn   graph.MapFn
				ref  func(a, b float32) float32
			}{{"sum", graph.MapAdd, sum}, {"max", graph.MapMax, maxOf}} {
				got := run(xs, c.fn, n)
				for g := range groups {
					if want := tree(xs[g*n:(g+1)*n], c.ref); got[g] != want {
						t.Fatalf("a pairwise %s of %d, group %d of %d, is %g, want %g", c.name, n, g, groups, got[g], want)
					}
				}
			}
		}
	}
	for _, xs := range [][]float32{{1e8, 1, -1e8, 1}, {1e8, 1, -1e8}} {
		seq := float32(0)
		for _, v := range xs {
			seq += v
		}
		got := run(xs, graph.MapAdd, len(xs))[0]
		if want := tree(xs, sum); got != want || got == seq {
			t.Fatalf("the pairwise sum of %v is %g: pairwise %g, in order %g", xs, got, want, seq)
		}
	}
	negz := float32(math.Copysign(0, -1))
	if got := run([]float32{negz, negz, negz}, graph.MapAdd, 3)[0]; got != 0 || !math.Signbit(float64(got)) {
		t.Fatalf("the pairwise sum of three negative zeros is %g, sign bit %v: an element without a pair was combined", got, math.Signbit(float64(got)))
	}
}
