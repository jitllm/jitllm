package composer_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUGateFromAReduction forms the n-gram block's gate on every
// GPU with every factor of it computed in the graph: a head's dot of two
// rows reduced over its D elements, the gate a function of the dot, g_k =
// gate_k * v, their concatenation normed a head at a time, and out_k = x_k +
// g_k + norm[k*D:(k+1)*D]. The concatenation is read in a later kernel than
// the one reducing the dots, and each g_k where its output is, so a chain
// computing the gate where it is read must give both the dots it stores.
func TestComposedGPUGateFromAReduction(t *testing.T) {
	testlock.GPU(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			dev := gpuDevice(d)
			rng := rand.New(rand.NewSource(47))
			ran := 0
			for _, s := range []struct{ h, d int }{{2, 8}, {4, 64}, {4, 96}} {
				for _, R := range []int{1, 8, 64, 512} {
					worst, _, err := gateFromAReduction(composer.Default(), d, dev, s.h, s.d, R, rng)
					if err != nil {
						t.Fatalf("%+v at %d rows: %v", s, R, err)
					}
					if !(worst < 1e-4) {
						t.Fatalf("%+v at %d rows: %g from the oracle", s, R, worst)
					}
					ran++
				}
			}
			t.Logf("%d forms held to the oracle", ran)
		})
	}
}

// gateFromAReduction forms TestComposedGPUGateFromAReduction's graph of H
// heads of D through reg at a step of R rows, runs it on d and returns its
// worst distance from the oracle and the codes it ran.
func gateFromAReduction(reg *composer.Registry, d backend.Device, dev composer.Device, H, D, R int, rng *rand.Rand) (float64, []composer.Code, error) {
	C := H * D
	b := graph.New()
	xs := make([]graph.ValueID, H)
	for k := range xs {
		xs[k] = b.In("x", D)
	}
	key, query, v := b.In("key", C), b.In("query", C), b.In("v", D)
	kq := b.Emit(graph.OpMap, []graph.ValueID{key, query}, []graph.Result{{Name: "kq", Width: C}}, nil, graph.Attrs{Fn: graph.MapMul}, nil, nil)[0]
	dot := b.Emit(graph.OpReduce, []graph.ValueID{kq}, []graph.Result{{Name: "dot", Width: H}}, nil, graph.Attrs{Fn: graph.MapAdd, Span: D}, nil, nil)[0]
	in0 := graph.Arg{Kind: graph.ArgIn, Index: 0}
	term := func(i int) graph.Arg { return graph.Arg{Kind: graph.ArgTerm, Index: i} }
	konst := func(x float64) graph.Arg { return graph.Arg{Kind: graph.ArgConst, Const: x} }
	gate := b.Expr("gate", H, []graph.Term{
		{Fn: graph.MapMul, Args: [3]graph.Arg{in0, konst(1 / math.Sqrt(float64(D)))}},
		{Fn: graph.MapAbs, Args: [3]graph.Arg{term(0)}},
		{Fn: graph.MapMax, Args: [3]graph.Arg{term(1), konst(float64(graph.SqrtSigmoidFloor))}},
		{Fn: graph.MapSqrt, Args: [3]graph.Arg{term(2)}},
		{Fn: graph.MapLess, Args: [3]graph.Arg{konst(0), term(0)}},
		{Fn: graph.MapLess, Args: [3]graph.Arg{term(0), konst(0)}},
		{Fn: graph.MapSub, Args: [3]graph.Arg{term(4), term(5)}},
		{Fn: graph.MapMul, Args: [3]graph.Arg{term(6), term(3)}},
		{Fn: graph.MapSigmoid, Args: [3]graph.Arg{term(7)}},
	}, nil, dot)
	gv := make([]graph.ValueID, H)
	for k := range gv {
		gk := b.Emit(graph.OpSlice, []graph.ValueID{gate}, []graph.Result{{Name: "gk", Width: 1}}, nil, graph.Attrs{Offset: k, Width: 1}, nil, nil)[0]
		gv[k] = b.Emit(graph.OpMap, []graph.ValueID{gk, v}, []graph.Result{{Name: "gv", Width: D}}, nil, graph.Attrs{Fn: graph.MapMul}, nil, nil)[0]
	}
	cat := b.Emit(graph.OpConcat, gv, []graph.Result{{Name: "gvs", Width: C}}, nil, graph.Attrs{}, nil, nil)[0]
	const eps = 1e-6
	hn := b.Emit(graph.OpHeadNorm, []graph.ValueID{cat}, []graph.Result{{Name: "normed", Width: C}}, nil,
		graph.Attrs{Eps: eps, HeadDim: D}, nil, nil)[0]
	outs := make([]graph.ValueID, H)
	for k := range outs {
		nk := b.Emit(graph.OpSlice, []graph.ValueID{hn}, []graph.Result{{Name: "nk", Width: D}}, nil, graph.Attrs{Offset: k * D, Width: D}, nil, nil)[0]
		outs[k] = b.Add(xs[k], b.Add(gv[k], nk))
	}
	b.Out(outs...)
	fr, err := formRecWith(reg, d, dev, b.G, R, nil)
	if err != nil {
		return 0, nil, err
	}
	defer fr.close()
	rnd := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		return x
	}
	fr.vals = map[graph.ValueID][]uint32{}
	xv := make([][]float32, H)
	for k := range H {
		xv[k] = rnd(R * D)
		fr.vals[xs[k]] = f32w(xv[k])
	}
	kv, qv, vv := rnd(R*C), rnd(R*C), rnd(R*D)
	fr.vals[key], fr.vals[query], fr.vals[v] = f32w(kv), f32w(qv), f32w(vv)
	for id, val := range b.G.Values {
		if _, ok := fr.vals[graph.ValueID(id)]; !ok {
			fr.vals[graph.ValueID(id)] = poisoned(R * val.Width)
		}
	}
	fr.tables = map[string][]uint32{"rows.n": {uint32(R)}}
	if err := fr.launch(); err != nil {
		return 0, fr.codes, err
	}
	worst := 0.0
	for r := range R {
		g := make([]float64, C)
		for k := range H {
			var dt float64
			for e := range D {
				dt += float64(kv[r*C+k*D+e]) * float64(qv[r*C+k*D+e])
			}
			t0 := dt / math.Sqrt(float64(D))
			sg := math.Sqrt(max(math.Abs(t0), float64(graph.SqrtSigmoidFloor)))
			switch {
			case t0 < 0:
				sg = -sg
			case t0 == 0:
				sg = 0
			}
			gk := 1 / (1 + math.Exp(-sg))
			for i := range D {
				g[k*D+i] = gk * float64(vv[r*D+i])
			}
		}
		for k := range H {
			var ss float64
			for i := range D {
				ss += g[k*D+i] * g[k*D+i]
			}
			rs := 1 / math.Sqrt(ss/float64(D)+eps)
			got := wf32(fr.vals[outs[k]])
			for i := range D {
				want := float64(xv[k][r*D+i]) + g[k*D+i] + g[k*D+i]*rs
				worst = max(worst, math.Abs(float64(got[r*D+i])-want))
			}
		}
	}
	return worst, fr.codes, nil
}
