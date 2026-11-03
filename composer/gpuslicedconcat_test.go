package composer_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUSlicedConcat forms, on every GPU at a step of one row and
// of 8, values of a row that are read both whole and through a function of
// their concatenation sliced back apart: g_k = gate_k * v for each of h
// groups, and out_k = x_k + g_k + silu(concat(g))[k*d:(k+1)*d] -- the n-gram
// block's shape -- and holds every output to the oracle.
func TestComposedGPUSlicedConcat(t *testing.T) {
	testlock.GPU(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			dev := gpuDevice(d)
			rng := rand.New(rand.NewSource(31))
			for _, s := range []struct{ h, d int }{{2, 8}, {4, 64}, {4, 96}, {3, 7}} {
				for _, R := range []int{1, 8} {
					H, D := s.h, s.d
					b := graph.New()
					xs := make([]graph.ValueID, H)
					for k := range xs {
						xs[k] = b.In("x", D)
					}
					v, gate := b.In("v", D), b.In("gate", H)
					gv := make([]graph.ValueID, H)
					for k := range gv {
						gk := b.Emit(graph.OpSlice, []graph.ValueID{gate}, []graph.Result{{Name: "gk", Width: 1}}, nil, graph.Attrs{Offset: k, Width: 1}, nil, nil)[0]
						gv[k] = b.Emit(graph.OpMap, []graph.ValueID{gk, v}, []graph.Result{{Name: "gv", Width: D}}, nil, graph.Attrs{Fn: graph.MapMul}, nil, nil)[0]
					}
					cat := b.Emit(graph.OpConcat, gv, []graph.Result{{Name: "gvs", Width: H * D}}, nil, graph.Attrs{}, nil, nil)[0]
					act := b.Expr("act", H*D, []graph.Term{
						{Fn: graph.MapSigmoid, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0}}},
						{Fn: graph.MapMul, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0}, {Kind: graph.ArgTerm, Index: 0}}},
					}, nil, cat)
					outs := make([]graph.ValueID, H)
					for k := range outs {
						ck := b.Emit(graph.OpSlice, []graph.ValueID{act}, []graph.Result{{Name: "ck", Width: D}}, nil, graph.Attrs{Offset: k * D, Width: D}, nil, nil)[0]
						outs[k] = b.Add(xs[k], b.Add(gv[k], ck))
					}
					b.Out(outs...)
					fr, err := formRec(d, dev, b.G, R, map[jlm.Role]jlm.Type{})
					if err != nil {
						t.Fatalf("%+v at %d rows: %v", s, R, err)
					}
					rnd := func(n int) []float32 {
						x := make([]float32, n)
						for i := range x {
							x[i] = float32(rng.NormFloat64())
						}
						return x
					}
					xv := make([][]float32, H)
					fr.vals = map[graph.ValueID][]uint32{}
					for k := range H {
						xv[k] = rnd(R * D)
						fr.vals[xs[k]] = f32w(xv[k])
						fr.vals[outs[k]] = poisoned(R * D)
					}
					vv, gg := rnd(R*D), rnd(R*H)
					fr.vals[v], fr.vals[gate] = f32w(vv), f32w(gg)
					for id, val := range b.G.Values {
						if _, ok := fr.vals[graph.ValueID(id)]; !ok {
							fr.vals[graph.ValueID(id)] = poisoned(R * val.Width)
						}
					}
					clear(fr.bufs)
					if err := fr.launch(); err != nil {
						fr.close()
						t.Fatalf("%+v at %d rows: %v", s, R, err)
					}
					worst := 0.0
					for k := range H {
						got := wf32(fr.vals[outs[k]])
						for r := range R {
							for i := range D {
								g := float64(gg[r*H+k]) * float64(vv[r*D+i])
								want := float64(xv[k][r*D+i]) + g + g/(1+math.Exp(-g))
								worst = max(worst, math.Abs(float64(got[r*D+i])-want))
							}
						}
					}
					fr.close()
					if !(worst < 1e-4) {
						t.Fatalf("%+v at %d rows: %g from the oracle", s, R, worst)
					}
				}
			}
		})
	}
}

// TestComposedGPUSlicedConcatTwoReaders is TestComposedGPUSlicedConcat with
// the concatenation read twice, as the n-gram block reads it: by a per-group
// RMS norm, whose sum enters the sliced function at a weight of zero, and by
// the function the slices take. Read by the norm too, the concatenation must
// still hold its own values when the slices read it.
func TestComposedGPUSlicedConcatTwoReaders(t *testing.T) {
	testlock.GPU(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			dev := gpuDevice(d)
			rng := rand.New(rand.NewSource(37))
			for _, s := range []struct{ h, d int }{{2, 8}, {4, 64}, {4, 96}, {3, 7}} {
				for _, R := range []int{1, 8} {
					H, D := s.h, s.d
					b := graph.New()
					xs := make([]graph.ValueID, H)
					for k := range xs {
						xs[k] = b.In("x", D)
					}
					v, gate := b.In("v", D), b.In("gate", H)
					gv := make([]graph.ValueID, H)
					for k := range gv {
						gk := b.Emit(graph.OpSlice, []graph.ValueID{gate}, []graph.Result{{Name: "gk", Width: 1}}, nil, graph.Attrs{Offset: k, Width: 1}, nil, nil)[0]
						gv[k] = b.Emit(graph.OpMap, []graph.ValueID{gk, v}, []graph.Result{{Name: "gv", Width: D}}, nil, graph.Attrs{Fn: graph.MapMul}, nil, nil)[0]
					}
					cat := b.Emit(graph.OpConcat, gv, []graph.Result{{Name: "gvs", Width: H * D}}, nil, graph.Attrs{}, nil, nil)[0]
					hn := b.Emit(graph.OpHeadNorm, []graph.ValueID{cat}, []graph.Result{{Name: "normed", Width: H * D}}, nil,
						graph.Attrs{Eps: 1e-6, HeadDim: D}, nil, nil)[0]
					z := b.Emit(graph.OpReduce, []graph.ValueID{hn}, []graph.Result{{Name: "kept", Width: 1}}, nil, graph.Attrs{Fn: graph.MapAdd}, nil, nil)[0]
					z = b.Emit(graph.OpMap, []graph.ValueID{z}, []graph.Result{{Name: "kept.zero", Width: 1}}, nil, graph.Attrs{Fn: graph.MapMul, Const: 0}, nil, nil)[0]
					kept := b.Emit(graph.OpMap, []graph.ValueID{cat, z}, []graph.Result{{Name: "kept.in", Width: H * D}}, nil, graph.Attrs{Fn: graph.MapAdd}, nil, nil)[0]
					act := b.Expr("act", H*D, []graph.Term{
						{Fn: graph.MapSigmoid, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0}}},
						{Fn: graph.MapMul, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0}, {Kind: graph.ArgTerm, Index: 0}}},
					}, nil, kept)
					outs := make([]graph.ValueID, H)
					for k := range outs {
						ck := b.Emit(graph.OpSlice, []graph.ValueID{act}, []graph.Result{{Name: "ck", Width: D}}, nil, graph.Attrs{Offset: k * D, Width: D}, nil, nil)[0]
						outs[k] = b.Add(xs[k], b.Add(gv[k], ck))
					}
					b.Out(outs...)
					fr, err := formRec(d, dev, b.G, R, map[jlm.Role]jlm.Type{})
					if err != nil {
						t.Fatalf("%+v at %d rows: %v", s, R, err)
					}
					rnd := func(n int) []float32 {
						x := make([]float32, n)
						for i := range x {
							x[i] = float32(rng.NormFloat64())
						}
						return x
					}
					xv := make([][]float32, H)
					fr.vals = map[graph.ValueID][]uint32{}
					for k := range H {
						xv[k] = rnd(R * D)
						fr.vals[xs[k]] = f32w(xv[k])
						fr.vals[outs[k]] = poisoned(R * D)
					}
					vv, gg := rnd(R*D), rnd(R*H)
					fr.vals[v], fr.vals[gate] = f32w(vv), f32w(gg)
					for id, val := range b.G.Values {
						if _, ok := fr.vals[graph.ValueID(id)]; !ok {
							fr.vals[graph.ValueID(id)] = poisoned(R * val.Width)
						}
					}
					clear(fr.bufs)
					if err := fr.launch(); err != nil {
						fr.close()
						t.Fatalf("%+v at %d rows: %v", s, R, err)
					}
					worst := 0.0
					for k := range H {
						got := wf32(fr.vals[outs[k]])
						for r := range R {
							for i := range D {
								g := float64(gg[r*H+k]) * float64(vv[r*D+i])
								want := float64(xv[k][r*D+i]) + g + g/(1+math.Exp(-g))
								worst = max(worst, math.Abs(float64(got[r*D+i])-want))
							}
						}
					}
					fr.close()
					if !(worst < 1e-4) {
						t.Fatalf("%+v at %d rows: %g from the oracle", s, R, worst)
					}
				}
			}
		})
	}
}

// TestComposedGPUSlicedConcatThroughAStateConv is the n-gram block's own
// chain at one row: the concatenation normed per group into a dilated state
// convolution over a window at an offset of the recurrent state's slot,
// whose sum enters the sliced function at a weight of zero, and the
// concatenation itself sliced back apart. The slices must read the
// concatenation's values, whatever the convolution's window and state.
func TestComposedGPUSlicedConcatThroughAStateConv(t *testing.T) {
	testlock.GPU(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			dev := gpuDevice(d)
			rng := rand.New(rand.NewSource(41))
			for _, s := range []struct{ h, d, T, D, off int }{{2, 8, 4, 3, 5}, {4, 64, 4, 3, 2 + 640}, {4, 96, 4, 3, 7}} {
				for _, R := range []int{1, 8, 64} {
					H, D := s.h, s.d
					C := H * D
					W := (s.T - 1) * s.D * C
					span := s.off + W + 3
					b := graph.New()
					xs := make([]graph.ValueID, H)
					for k := range xs {
						xs[k] = b.In("x", D)
					}
					v, gate := b.In("v", D), b.In("gate", H)
					gv := make([]graph.ValueID, H)
					for k := range gv {
						gk := b.Emit(graph.OpSlice, []graph.ValueID{gate}, []graph.Result{{Name: "gk", Width: 1}}, nil, graph.Attrs{Offset: k, Width: 1}, nil, nil)[0]
						gv[k] = b.Emit(graph.OpMap, []graph.ValueID{gk, v}, []graph.Result{{Name: "gv", Width: D}}, nil, graph.Attrs{Fn: graph.MapMul}, nil, nil)[0]
					}
					cat := b.Emit(graph.OpConcat, gv, []graph.Result{{Name: "gvs", Width: C}}, nil, graph.Attrs{}, nil, nil)[0]
					hn := b.Emit(graph.OpHeadNorm, []graph.ValueID{cat}, []graph.Result{{Name: "normed", Width: C}}, nil,
						graph.Attrs{Eps: 1e-6, HeadDim: D}, nil, nil)[0]
					conv := b.Emit(graph.OpStateConv, []graph.ValueID{hn}, []graph.Result{{Name: "conv", Width: C}},
						[]graph.Binding{{Role: jlm.RoleSSMConv1d, K: s.T, Rows: C}}, graph.Attrs{Taps: s.T, Dil: s.D, Offset: s.off, Span: span},
						[]graph.StateRef{{State: graph.StateRecState}}, []graph.StateRef{{State: graph.StateRecState, Version: 1}})[0]
					z := b.Emit(graph.OpReduce, []graph.ValueID{conv}, []graph.Result{{Name: "kept", Width: 1}}, nil, graph.Attrs{Fn: graph.MapAdd}, nil, nil)[0]
					z = b.Emit(graph.OpMap, []graph.ValueID{z}, []graph.Result{{Name: "kept.zero", Width: 1}}, nil, graph.Attrs{Fn: graph.MapMul, Const: 0}, nil, nil)[0]
					kept := b.Emit(graph.OpMap, []graph.ValueID{cat, z}, []graph.Result{{Name: "kept.in", Width: C}}, nil, graph.Attrs{Fn: graph.MapAdd}, nil, nil)[0]
					outs := make([]graph.ValueID, H)
					for k := range outs {
						ck := b.Emit(graph.OpSlice, []graph.ValueID{kept}, []graph.Result{{Name: "ck", Width: D}}, nil, graph.Attrs{Offset: k * D, Width: D}, nil, nil)[0]
						outs[k] = b.Add(xs[k], b.Add(gv[k], ck))
					}
					b.Out(outs...)
					fr, err := formRec(d, dev, b.G, R, map[jlm.Role]jlm.Type{jlm.RoleSSMConv1d: jlm.TypeF32})
					if err != nil {
						t.Fatalf("%+v: %v", s, err)
					}
					rnd := func(n int) []float32 {
						x := make([]float32, n)
						for i := range x {
							x[i] = float32(rng.NormFloat64())
						}
						return x
					}
					xv := make([][]float32, H)
					fr.vals = map[graph.ValueID][]uint32{}
					for k := range H {
						xv[k] = rnd(R * D)
						fr.vals[xs[k]] = f32w(xv[k])
						fr.vals[outs[k]] = poisoned(R * D)
					}
					vv, gg := rnd(R*D), rnd(R*H)
					fr.vals[v], fr.vals[gate] = f32w(vv), f32w(gg)
					for id, val := range b.G.Values {
						if _, ok := fr.vals[graph.ValueID(id)]; !ok {
							fr.vals[graph.ValueID(id)] = poisoned(R * val.Width)
						}
					}
					runs := recRuns{{0}}
					slots := len(runs) + 2
					in, out := make([]uint32, slots*span), make([]uint32, slots*span)
					for i := range in {
						in[i] = math.Float32bits(float32(rng.NormFloat64()))
						out[i] = math.Float32bits(float32(rng.NormFloat64()))
					}
					fr.tables = map[string][]uint32{"rec.state": in, "rec.state.out": out, "rec.rows": runs.desc(1, 1, 1)}
					fr.wts = map[jlm.Role][]uint32{jlm.RoleSSMConv1d: f32w(rnd(C * s.T))}
					clear(fr.bufs)
					if err := fr.launch(); err != nil {
						fr.close()
						t.Fatalf("%+v at %d rows: %v", s, R, err)
					}
					worst := 0.0
					for k := range H {
						got := wf32(fr.vals[outs[k]])
						for i := range D {
							g := float64(gg[k]) * float64(vv[i])
							want := float64(xv[k][i]) + 2*g
							worst = max(worst, math.Abs(float64(got[i])-want))
						}
					}
					fr.close()
					if !(worst < 1e-4) {
						t.Fatalf("%+v at %d rows (one live): %g from the oracle", s, R, worst)
					}
				}
			}
		})
	}
}
