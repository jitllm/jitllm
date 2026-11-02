package jit_test

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
