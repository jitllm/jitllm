package composer

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
)

// GridRowsCase is a grid, its channels, a window's side and stride, and the
// expansion width the row-wise ops widen the grid's rows to.
type GridRowsCase struct{ H, W, C, K, S, E int }

func (g GridRowsCase) String() string {
	return fmt.Sprintf("%dx%d/%d channels/%dx%d/stride %d/%d wide", g.H, g.W, g.C, g.K, g.K, g.S, g.E)
}

// Out is the case's output grid's rows.
func (g GridRowsCase) Out() int { return ((g.H + g.S - 1) / g.S) * ((g.W + g.S - 1) / g.S) }

// GridRowsCases is ragged grids at MobileNet-V5's strides: odd sides, one
// row and one column, channels off any vector.
func GridRowsCases() []GridRowsCase {
	return []GridRowsCase{{8, 8, 16, 3, 2, 32}, {7, 9, 5, 3, 2, 12}, {5, 5, 17, 5, 2, 9}, {9, 7, 8, 3, 1, 24},
		{1, 7, 5, 3, 2, 7}, {7, 1, 9, 3, 2, 4}, {2, 2, 3, 3, 2, 5}, {16, 16, 32, 3, 2, 64}}
}

// GridRowsGraph is the row-wise ops over a depthwise convolution's grid:
// the grid (OpConv at the case's stride), widened (OpMatVec), normed
// (OpNorm), through a stated GELU (OpExpr), narrowed back (OpMatVec) and
// added to the grid (OpAdd) -- an inverted residual's ops over a grid of
// rows of its own.
func GridRowsGraph(g GridRowsCase) (*graph.Graph, error) {
	b := graph.New()
	x := b.In("x", g.C)
	res := graph.Result{Name: "dw", Width: g.C}
	if g.S > 1 {
		res.Rows = g.Out()
	}
	dw := b.Emit(graph.OpConv, []graph.ValueID{x}, []graph.Result{res},
		[]graph.Binding{{Role: jlm.RoleVMNDwMid, K: g.C, Rows: g.K * g.K}}, graph.Attrs{Taps: g.K, GridW: g.W, Stride: g.S}, nil, nil)[0]
	z := b.MatVec(dw, jlm.RoleVMNPwExp, g.E)
	n := b.Norm(z, jlm.RoleVMNPwExpNorm, 1e-6)
	var e jlm.ExprBuilder
	e.GELUTanh(jlm.Operand(0))
	act := jlm.Expr{Terms: e.T}
	a, err := b.Stated("act", &act, []graph.ValueID{n}, g.E, graph.Facts{})
	if err != nil {
		return nil, err
	}
	o := b.MatVec(a, jlm.RoleVMNPwProj, g.C)
	b.Out(b.Add(o, dw))
	return b.G, graph.Verify(b.G, graph.Facts{Inventory: map[jlm.Role]graph.Shape{
		jlm.RoleVMNDwMid: {K: g.C, Rows: g.K * g.K}, jlm.RoleVMNPwExp: {K: g.C, Rows: g.E},
		jlm.RoleVMNPwExpNorm: {K: g.E}, jlm.RoleVMNPwProj: {K: g.E, Rows: g.C}}})
}

// GridRowsWant is GridRowsGraph's result by the oracle.
func GridRowsWant(g GridRowsCase, x, filt, w1, n1, w2 []float32) []float32 {
	m := g.Out()
	dw := make([]float32, m*g.C)
	oracle.DWConv32(dw, x, g.H, g.W, g.C, g.K, g.S, filt)
	out := make([]float32, m*g.C)
	z, a := make([]float32, g.E), make([]float32, g.C)
	for r := range m {
		row := dw[r*g.C : (r+1)*g.C]
		oracle.MatVec(z, quant.F32, F32Bytes(w1), row, g.E, g.C)
		oracle.RMSNorm32(z, z, n1, 1e-6)
		for i := range z {
			z[i] = float32(oracle.GELUTanh(float64(z[i])))
		}
		oracle.MatVec(a, quant.F32, F32Bytes(w2), z, g.C, g.E)
		for i := range a {
			out[r*g.C+i] = a[i] + row[i]
		}
	}
	return out
}

func F32Bytes(x []float32) []byte { return unsafe.Slice((*byte)(unsafe.Pointer(&x[0])), 4*len(x)) }

func RandRow(rng *rand.Rand, n int, scale float64) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(rng.NormFloat64() * scale)
	}
	return x
}

// gridRowsRun runs codes formed from g over x, the graph's values each at
// its own rows, the weights by role; nil and the error where a guard word
// past any value was written.
func gridRowsRun(cpuOn cpuArm, codes []Code, g *graph.Graph, stepRows int, x []float32, roles map[jlm.Role][]float32, planes [3][]uint32) ([]float32, error) {
	vals := map[graph.ValueID][]uint32{g.In[0]: f32Words(x)}
	byRole := map[jlm.Role][]uint32{}
	for r, v := range roles {
		byRole[r] = f32Words(v)
	}
	rows := func(v graph.ValueID) int {
		if r := g.Values[v].Rows; r > 0 {
			return r
		}
		return stepRows
	}
	if err := runCPUGridValues(cpuOn, codes, g, vals, planes, nil, rows, byRole); err != nil {
		return nil, err
	}
	return wordsF32(vals[g.Out[0]]), nil
}

// Misrowed is r with kind's decomposition run at rows rather than the
// grid's own, its values' rows unstated: a stride-2 grid read at the
// input's row count, the violation TestComposedCPUGridRows must see.
func Misrowed(r *Registry, kind graph.OpKind, rows int) *Registry {
	v := *r
	v.cache = newFormCache()
	v.Decompositions = map[graph.OpKind]Decomposition{}
	for k, d := range r.Decompositions {
		v.Decompositions[k] = d
	}
	d := r.Decompositions[kind]
	v.Decompositions[kind] = func(o graph.Op, c *Context) ([]Prim, error) {
		cc := *c
		cc.Step.Rows = rows
		g := *c.Graph
		g.Values = slices.Clone(g.Values)
		for _, id := range append(slices.Clone(o.In), o.Out...) {
			g.Values[id].Rows = 0
		}
		cc.Graph = &g
		return d(o, &cc)
	}
	return &v
}

// TestComposedCPUGridRows executes the row-wise ops over a grid of rows of
// its own on the CPU device -- a matrix product, an RMSNorm, a stated
// activation and an add over a stride-2 depthwise convolution's rows
// (GridRowsGraph), and a packed matrix product over one -- held to the
// oracle, in the float and the int8 forms, with a guard of poisoned words
// past every value. With each op's decomposition run at the input's rows in
// place of the grid's, the gate must fail: a guard written, or the result
// parted from the oracle.
func TestComposedCPUGridRows(t *testing.T) {
	rng := rand.New(rand.NewSource(41))
	dev := cpuDevice(hostArm())
	right := 0
	kinds := []graph.OpKind{graph.OpMatVec, graph.OpNorm, graph.OpLayerNorm, graph.OpExpr, graph.OpAdd}
	seen := map[graph.OpKind]int{}
	for _, g := range GridRowsCases() {
		gr, err := GridRowsGraph(g)
		if err != nil {
			t.Fatalf("%v: %v", g, err)
		}
		x := RandRow(rng, g.H*g.W*g.C, 1)
		filt := RandRow(rng, g.K*g.K*g.C, 0.5)
		w1, n1 := RandRow(rng, g.E*g.C, 1/math.Sqrt(float64(g.C))), RandRow(rng, g.E, 1)
		w2 := RandRow(rng, g.C*g.E, 1/math.Sqrt(float64(g.E)))
		want := GridRowsWant(g, x, filt, w1, n1, w2)
		roles := map[jlm.Role][]float32{jlm.RoleVMNDwMid: filt, jlm.RoleVMNPwExp: w1, jlm.RoleVMNPwExpNorm: n1, jlm.RoleVMNPwProj: w2}
		formats := map[jlm.Role]jlm.Type{jlm.RoleVMNDwMid: jlm.TypeF32, jlm.RoleVMNPwExp: jlm.TypeF32,
			jlm.RoleVMNPwExpNorm: jlm.TypeF32, jlm.RoleVMNPwProj: jlm.TypeF32}
		step := Step{Rows: g.H * g.W, Page: 64, Keys: 1, ActWindow: 32}
		form := func(r *Registry) ([]Code, error) {
			_, codes, err := r.Form(Request{Graph: gr, Formats: formats, Step: step, Device: dev})
			return codes, err
		}
		codes, err := form(Default().WithoutInt8(DeviceCPU))
		if err != nil {
			t.Errorf("%v: %v", g, err)
			continue
		}
		got, err := gridRowsRun(hostArm(), codes, gr, step.Rows, x, roles, [3][]uint32{})
		if err != nil {
			t.Errorf("%v: %v", g, err)
			continue
		}
		if e := rowsNMSE(got, want); math.IsNaN(e) || !(e < 1e-10) {
			t.Errorf("%v: NMSE %.3e against the oracle", g, e)
			continue
		}
		right++
		if g.S == 1 {
			continue
		}
		for _, k := range kinds {
			codes, err := form(Misrowed(Default().WithoutInt8(DeviceCPU), k, step.Rows))
			if err != nil {
				t.Logf("%v: %s at the input's rows refused to form: %v", g, k, err)
				seen[k]++
				continue
			}
			bad, err := gridRowsRun(hostArm(), codes, gr, step.Rows, x, roles, [3][]uint32{})
			if err != nil {
				t.Logf("%v: %s at the input's rows: %v", g, k, err)
				seen[k]++
				continue
			}
			e := rowsNMSE(bad, want)
			t.Logf("%v: %s at the input's %d rows in place of the grid's %d reads NMSE %.3e", g, k, step.Rows, g.Out(), e)
			if e > 1e-6 || math.IsNaN(e) {
				seen[k]++
			}
			// The op alone over the grid, its output the graph's.
			lone, lwant := GridRowsLone(g, k, x, filt, n1)
			_, lcodes, err := Default().WithoutInt8(DeviceCPU).Form(Request{Graph: lone, Formats: formats, Step: step, Device: dev})
			if err != nil {
				t.Errorf("%v: %s alone: %v", g, k, err)
				continue
			}
			lgot, err := gridRowsRun(hostArm(), lcodes, lone, step.Rows, x, roles, [3][]uint32{})
			if err != nil {
				t.Errorf("%v: %s alone: %v", g, k, err)
				continue
			}
			if e := rowsNMSE(lgot, lwant); math.IsNaN(e) || !(e < 1e-10) {
				t.Errorf("%v: %s alone over the grid reads NMSE %.3e", g, k, e)
			}
			_, lcodes, err = Misrowed(Default().WithoutInt8(DeviceCPU), k, step.Rows).Form(Request{Graph: lone, Formats: formats, Step: step, Device: dev})
			if err != nil {
				seen[k]++
				continue
			}
			lbad, err := gridRowsRun(hostArm(), lcodes, lone, step.Rows, x, roles, [3][]uint32{})
			if err != nil {
				t.Logf("%v: %s alone at the input's rows: %v", g, k, err)
				seen[k]++
				continue
			}
			le := rowsNMSE(lbad, lwant)
			t.Logf("%v: %s alone at the input's rows reads NMSE %.3e", g, k, le)
			if le > 1e-6 || math.IsNaN(le) {
				seen[k]++
			}
		}
	}
	if right != len(GridRowsCases()) {
		t.Fatalf("%d of %d grids' row-wise ops held to the oracle", right, len(GridRowsCases()))
	}
	t.Logf("%d grids' row-wise ops held to the oracle; the violations seen by op %v", right, seen)
	for _, k := range kinds {
		if seen[k] == 0 {
			t.Errorf("%s read at the input's rows over a stride-2 grid is seen at no grid: the gate cannot see it", k)
		}
	}

	// A packed matrix product over a stride-2 grid's rows, float and int8.
	const c, e = 256, 96
	for _, typ := range []quant.Type{quant.Q8_0, quant.Q4_K} {
		q, _ := layout.QuantOf(typ)
		jt, _ := jlm.TypeOf(typ)
		wb := cpuTestWeights(typ, q, e, c, rng)
		qs, d, sc, err := layout.PackWeights(q, wb, e, c)
		if err != nil {
			t.Fatalf("%v: %v", typ, err)
		}
		planes := [3][]uint32{qs, d, sc}
		for _, arm := range []struct {
			name string
			r    *Registry
			int8 bool
		}{{"float", Default().WithoutInt8(DeviceCPU), false}, {"int8", withCPUInt8(), true}} {
			h, w, k := 9, 7, 3
			m := ((h + 1) / 2) * ((w + 1) / 2)
			b := graph.New()
			xv := b.In("x", c)
			dw := b.Emit(graph.OpConv, []graph.ValueID{xv}, []graph.Result{{Name: "dw", Width: c, Rows: m}},
				[]graph.Binding{{Role: jlm.RoleVMNDwMid, K: c, Rows: k * k}}, graph.Attrs{Taps: k, GridW: w, Stride: 2}, nil, nil)[0]
			b.Out(b.MatVec(dw, jlm.RoleVMNPwExp, e))
			x := RandRow(rng, h*w*c, 1)
			filt := RandRow(rng, k*k*c, 0.5)
			step := Step{Rows: h * w, Page: 64, Keys: 1, ActWindow: 32}
			formats := map[jlm.Role]jlm.Type{jlm.RoleVMNDwMid: jlm.TypeF32, jlm.RoleVMNPwExp: jt}
			_, codes, err := arm.r.Form(Request{Graph: b.G, Formats: formats, Step: step, Device: dev})
			if err != nil {
				t.Fatalf("%v/%s: %v", typ, arm.name, err)
			}
			got, err := gridRowsRun(hostArm(), codes, b.G, step.Rows, x, map[jlm.Role][]float32{jlm.RoleVMNDwMid: filt}, planes)
			if err != nil {
				t.Fatalf("%v/%s: %v", typ, arm.name, err)
			}
			dwv := make([]float32, m*c)
			oracle.DWConv32(dwv, x, h, w, c, k, 2, filt)
			want := make([]float32, m*e)
			for r := range m {
				row := dwv[r*c : (r+1)*c]
				if arm.int8 {
					row = int8Row(row, 32)
				}
				if err := oracle.MatVec(want[r*e:(r+1)*e], typ, wb, row, e, c); err != nil {
					t.Fatal(err)
				}
			}
			if nm := rowsNMSE(got, want); math.IsNaN(nm) || !(nm < 1e-9) {
				t.Fatalf("%v/%s: a packed product over a stride-2 grid's %d rows reads NMSE %.3e", typ, arm.name, m, nm)
			}
			_, codes, err = Misrowed(arm.r, graph.OpMatVec, step.Rows).Form(Request{Graph: b.G, Formats: formats, Step: step, Device: dev})
			if err == nil {
				var bad []float32
				if bad, err = gridRowsRun(hostArm(), codes, b.G, step.Rows, x, map[jlm.Role][]float32{jlm.RoleVMNDwMid: filt}, planes); err == nil {
					if nm := rowsNMSE(bad, want); !(nm > 1e-6) {
						t.Fatalf("%v/%s: the product at the input's %d rows reads NMSE %.3e: the gate cannot see it", typ, arm.name, step.Rows, nm)
					}
				}
			}
			t.Logf("%v/%s: a packed product over a stride-2 grid's %d rows held to the oracle; at the input's %d rows: %v", typ, arm.name, m, step.Rows, err)
		}
	}
}

// TestComposedCPUUnfold executes OpUnfold and a matrix product over it on
// the CPU device -- a full convolution, MobileNet-V5's edge residual's --
// at ragged grids, channels, 3x3 and 5x5 windows and strides one and two,
// the unfolded row padded to a whole quantization block, held to the
// oracle's convolution with every value's guard words poisoned. The filter
// transposed and the grid read a column narrower must fail it.
func TestComposedCPUUnfold(t *testing.T) {
	rng := rand.New(rand.NewSource(43))
	dev := cpuDevice(hostArm())
	right := 0
	cases := GridRowsCases()
	for _, g := range cases {
		kp := (g.K*g.K*g.C + 31) / 32 * 32
		build := func(gw int) *graph.Graph {
			b := graph.New()
			x := b.In("x", g.C)
			res := graph.Result{Name: "unfold", Width: kp}
			if g.S > 1 {
				res.Rows = g.Out()
			}
			u := b.Emit(graph.OpUnfold, []graph.ValueID{x}, []graph.Result{res}, nil, graph.Attrs{Taps: g.K, GridW: gw, Stride: g.S}, nil, nil)[0]
			b.Out(b.MatVec(u, jlm.RoleVMNConvExp, g.E))
			return b.G
		}
		gr := build(g.W)
		if err := graph.Verify(gr, graph.Facts{Inventory: map[jlm.Role]graph.Shape{jlm.RoleVMNConvExp: {K: kp, Rows: g.E}}}); err != nil {
			t.Fatalf("%v: %v", g, err)
		}
		step := Step{Rows: g.H * g.W, Page: 64, Keys: 1, ActWindow: 32}
		formats := map[jlm.Role]jlm.Type{jlm.RoleVMNConvExp: jlm.TypeF32}
		x := RandRow(rng, g.H*g.W*g.C, 1)
		wt := RandRow(rng, g.E*kp, 1/math.Sqrt(float64(g.K*g.K*g.C)))
		want := make([]float32, g.Out()*g.E)
		oracle.Conv32(want, x, g.H, g.W, g.C, g.K, g.S, wt, g.E, kp)
		run := func(gr *graph.Graph, wt []float32) ([]float32, error) {
			_, codes, err := Default().WithoutInt8(DeviceCPU).Form(Request{Graph: gr, Formats: formats, Step: step, Device: dev})
			if err != nil {
				return nil, err
			}
			return gridRowsRun(hostArm(), codes, gr, step.Rows, x, map[jlm.Role][]float32{jlm.RoleVMNConvExp: wt}, [3][]uint32{})
		}
		clean := make([]float32, len(wt))
		copy(clean, wt)
		for o := range g.E {
			for j := g.K * g.K * g.C; j < kp; j++ {
				clean[o*kp+j] = 0
			}
		}
		got, err := run(gr, clean)
		if err != nil {
			t.Errorf("%v: %v", g, err)
			continue
		}
		if e := rowsNMSE(got, want); math.IsNaN(e) || !(e < 1e-10) {
			t.Errorf("%v: NMSE %.3e against the oracle's convolution", g, e)
			continue
		}
		right++
		if g.H*g.W < 4 {
			continue
		}
		tr := make([]float32, len(clean))
		for o := range g.E {
			for ky := range g.K {
				for kx := range g.K {
					copy(tr[o*kp+(kx*g.K+ky)*g.C:o*kp+(kx*g.K+ky+1)*g.C], clean[o*kp+(ky*g.K+kx)*g.C:o*kp+(ky*g.K+kx+1)*g.C])
				}
			}
		}
		if bad, err := run(gr, tr); err != nil {
			t.Errorf("%v: the violation did not run: %v", g, err)
		} else if e := rowsNMSE(bad, want); !(e > 1e-6) {
			t.Errorf("%v: the filter transposed reads %.3e: the gate cannot see it", g, e)
		}
		if g.S == 1 && g.W > 1 && (g.H*g.W)%(g.W-1) == 0 {
			if bad, err := run(build(g.W-1), clean); err != nil {
				t.Errorf("%v: the violation did not run: %v", g, err)
			} else if e := rowsNMSE(bad, want); !(e > 1e-6) {
				t.Errorf("%v: the grid read a column narrower reads %.3e: the gate cannot see it", g, e)
			}
		}
	}
	if right != len(cases) {
		t.Fatalf("%d of %d full convolutions held to the oracle", right, len(cases))
	}
	t.Logf("%d full convolutions held to the oracle", right)
}

// GridRowsLone is op kind alone over a stride-2 depthwise convolution's
// grid, its output the graph's, and that output by the oracle: a matrix
// product of the grid by the norm's vector as a one-row matrix, the RMS
// norm, the LayerNorm, GELU, and the grid added to itself.
func GridRowsLone(g GridRowsCase, kind graph.OpKind, x, filt, n1 []float32) (*graph.Graph, []float32) {
	m := g.Out()
	b := graph.New()
	xv := b.In("x", g.C)
	dw := b.Emit(graph.OpConv, []graph.ValueID{xv}, []graph.Result{{Name: "dw", Width: g.C, Rows: m}},
		[]graph.Binding{{Role: jlm.RoleVMNDwMid, K: g.C, Rows: g.K * g.K}}, graph.Attrs{Taps: g.K, GridW: g.W, Stride: g.S}, nil, nil)[0]
	ref := make([]float32, m*g.C)
	oracle.DWConv32(ref, x, g.H, g.W, g.C, g.K, g.S, filt)
	var want []float32
	switch kind {
	case graph.OpMatVec:
		b.Out(b.MatVec(dw, jlm.RoleVMNDwMid, 1))
		want = make([]float32, m)
		for r := range m {
			oracle.MatVec(want[r:r+1], quant.F32, F32Bytes(filt[:g.C]), ref[r*g.C:(r+1)*g.C], 1, g.C)
		}
		b.G.Ops[1].Weight[0].Rows = 1
	case graph.OpNorm:
		b.Out(b.Emit(graph.OpNorm, []graph.ValueID{dw}, []graph.Result{{Name: "n", Width: g.C}},
			[]graph.Binding{{Role: jlm.RoleVMNDwMid, K: g.C}}, graph.Attrs{Eps: 1e-6}, nil, nil)[0])
		want = make([]float32, m*g.C)
		for r := range m {
			oracle.RMSNorm32(want[r*g.C:(r+1)*g.C], ref[r*g.C:(r+1)*g.C], filt[:g.C], 1e-6)
		}
	case graph.OpLayerNorm:
		b.Out(b.LayerNorm(dw, jlm.RoleVMNDwMid, 1e-6))
		want = make([]float32, m*g.C)
		for r := range m {
			oracle.LayerNorm32(want[r*g.C:(r+1)*g.C], ref[r*g.C:(r+1)*g.C], filt[:g.C], nil, 1e-6)
		}
	case graph.OpExpr:
		var e jlm.ExprBuilder
		e.GELUTanh(jlm.Operand(0))
		act := jlm.Expr{Terms: e.T}
		a, _ := b.Stated("act", &act, []graph.ValueID{dw}, g.C, graph.Facts{})
		b.Out(a)
		want = make([]float32, m*g.C)
		for i, v := range ref {
			want[i] = float32(oracle.GELUTanh(float64(v)))
		}
	case graph.OpAdd:
		b.Out(b.Add(dw, dw))
		want = make([]float32, m*g.C)
		for i, v := range ref {
			want[i] = v + v
		}
	}
	return b.G, want
}
