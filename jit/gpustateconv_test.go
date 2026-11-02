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

// TestComposedGPUStateWindowConv forms OpStateConv whose window lives in the
// recurrent state's slot, dilated -- a window of (Taps-1)*Dil rows at Offset
// into slots of Span floats, a delta rule's tail -- at a step of one row and
// of 8, on every GPU, and holds every row's output and the window each run's
// slot written holds to the oracle.
func TestComposedGPUStateWindowConv(t *testing.T) {
	testlock.GPU(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			dev := gpuDevice(d)
			rng := rand.New(rand.NewSource(29))
			ran := 0
			for _, s := range []struct{ T, D, C, off, span int }{
				{4, 3, 8, 5, 8 + 5 + 3*3*8}, {4, 3, 96, 2 + 640, 640 + 2 + 3*3*96 + 7}, {3, 1, 5, 0, 10},
			} {
				for _, R := range []int{1, 8} {
					b := graph.New()
					x := b.In("x", s.C)
					b.Out(b.Emit(graph.OpStateConv, []graph.ValueID{x}, []graph.Result{{Name: "conv", Width: s.C}},
						[]graph.Binding{{Role: jlm.RoleSSMConv1d, K: s.T, Rows: s.C}}, graph.Attrs{Taps: s.T, Dil: s.D, Offset: s.off, Span: s.span},
						[]graph.StateRef{{State: graph.StateRecState}}, []graph.StateRef{{State: graph.StateRecState, Version: 1}})[0])
					fr, err := formRec(d, dev, b.G, R, map[jlm.Role]jlm.Type{jlm.RoleSSMConv1d: jlm.TypeF32})
					if err != nil {
						t.Fatalf("%+v at %d rows: %v", s, R, err)
					}
					steps := recSteps
					if R == 1 {
						steps = steps[:0:0]
						steps = append(steps, recSteps[0])
						steps[0].n, steps[0].runs = 1, recRuns{{0}}
					}
					for _, st := range steps {
						if st.n > R {
							continue
						}
						worst, err := runStateConv(fr, b.G, s.T, s.D, s.C, s.off, s.span, R, st.n, st.runs, rng)
						if err != nil {
							fr.close()
							t.Fatalf("%+v at %d rows %s: %v", s, R, st.name, err)
						}
						if worst > 1e-5 {
							fr.close()
							t.Fatalf("%+v at %d rows %s: the composed convolution is %g from the oracle", s, R, st.name, worst)
						}
						ran++
					}
					fr.close()
				}
			}
			t.Logf("%d steps held to the oracle", ran)
		})
	}
}

// runStateConv runs one step of fr over the runs, its window at off of
// slots of span floats, and returns the worst distance from the oracle at
// the outputs and at the windows the slots written hold.
func runStateConv(fr *recRun, g *graph.Graph, T, D, C, off, span, R, n int, runs recRuns, rng *rand.Rand) (float64, error) {
	H := (T - 1) * D
	taps := make([]float32, C*T)
	for i := range taps {
		taps[i] = float32(rng.NormFloat64())
	}
	x := make([]float32, R*C)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
		if i >= n*C {
			x[i] = float32(math.NaN())
		}
	}
	slots := len(runs) + 2
	in, out := make([]uint32, slots*span), make([]uint32, slots*span)
	for i := range in {
		in[i] = math.Float32bits(float32(rng.NormFloat64()))
		out[i] = math.Float32bits(float32(math.NaN()))
	}
	fr.vals = map[graph.ValueID][]uint32{g.In[0]: f32w(x), g.Out[0]: poisoned(R * C)}
	fr.tables = map[string][]uint32{"rec.state": in, "rec.state.out": out, "rec.rows": runs.desc(n, 1, 1)}
	fr.wts = map[jlm.Role][]uint32{jlm.RoleSSMConv1d: f32w(taps)}
	clear(fr.bufs)
	if err := fr.launch(); err != nil {
		return 0, err
	}
	got := wf32(fr.vals[g.Out[0]])
	worst := 0.0
	for j, run := range runs {
		base := (1 + j) * span
		seq := make([]float64, 0, (H+len(run))*C)
		for i := range H * C {
			seq = append(seq, float64(math.Float32frombits(in[base+off+i])))
		}
		for _, r := range run {
			for c := range C {
				seq = append(seq, float64(x[r*C+c]))
			}
		}
		for e, r := range run {
			p := H + e
			for c := range C {
				var want float64
				for k := range T {
					want += float64(taps[c*T+k]) * seq[(p-(T-1-k)*D)*C+c]
				}
				worst = max(worst, math.Abs(float64(got[r*C+c])-want))
			}
		}
		last := len(seq)/C - H
		for i := range H * C {
			worst = max(worst, math.Abs(float64(math.Float32frombits(out[base+off+i]))-seq[last*C+i]))
		}
	}
	return worst, nil
}
