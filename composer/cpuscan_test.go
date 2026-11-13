package composer

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
)

// TestComposedCPUDeltaRuleKeepsRows forms the delta rule keeping its rows
// (graph.Attrs.Keeps) at a step of three runs over fewer live rows than the
// form holds, in the scan's loops form: every entry's slot of the keep table
// holds the oracle's state after that entry's row, the entries past the live
// rows untouched, and every row's output and each run's state after are the
// oracle's as in the plain form.
func TestComposedCPUDeltaRuleKeepsRows(t *testing.T) {
	rng := rand.New(rand.NewSource(12))
	const R, live = 8, 6
	runs := [][]int{{1, 3, 0, 4}, {2}, {5}}
	in, out := []int{1, 3, 5}, []int{2, 4, 6}
	kept := 0
	for _, s := range []deltaShape{
		{h: 4, kh: 2, dk: 7, dv: 5},
		{h: 8, kh: 2, dk: 9, dv: 8, chans: true},
		{h: 4, kh: 2, dk: 7, dv: 5, additive: true},
		{h: 2, kh: 1, dk: 128, dv: 4},
	} {
		g := deltaGraph(s)
		for i := range g.Ops {
			g.Ops[i].Attr.Keeps = true
		}
		_, codes, err := Default().Form(Request{Graph: g, Formats: map[jlm.Role]jlm.Type{}, Step: Step{Rows: R, Page: 64, Keys: 1}, Device: cpuDevice(hostArm())})
		if err != nil {
			t.Fatalf("%+v: %v", s, err)
		}
		N := s.h * s.dv * s.dk
		nan := math.Float32bits(float32(math.NaN()))
		pool, keep := make([]uint32, 7*N), make([]uint32, R*N)
		for i := range pool {
			pool[i] = nan
		}
		for i := range keep {
			keep[i] = nan
		}
		refs := make([][]float64, len(runs))
		for j := range runs {
			refs[j] = make([]float64, N)
			for i := range N {
				x := float32(rng.NormFloat64() * 0.3)
				pool[in[j]*N+i], refs[j][i] = math.Float32bits(x), float64(x)
			}
		}
		rnd := func(n int, lo, hi float64) []float32 {
			x := make([]float32, n)
			for i := range x {
				x[i] = float32(lo + (hi-lo)*rng.Float64())
			}
			return x
		}
		nd := s.h
		if s.chans {
			nd = s.h * s.dk
		}
		qk, vv := s.kh*s.dk, s.h*s.dv
		q, k, v := rnd(R*qk, -1, 1), rnd(R*qk, -1, 1), rnd(R*vv, -1, 1)
		d, beta := rnd(R*nd, 0.5, 0.99), rnd(R*s.h, 0.1, 0.9)
		words := func(x []float32) []uint32 { return unsafe.Slice((*uint32)(unsafe.Pointer(&x[0])), len(x)) }
		vals := map[graph.ValueID][]uint32{g.In[0]: words(q), g.In[1]: words(k), g.In[2]: words(v), g.In[3]: words(d), g.In[4]: words(beta)}
		tables := map[string][]uint32{"rec.state": pool, "rec.state.out": pool, "rec.rows": scanDesc(runs, live, in, out), "rec.state.keep": keep}
		gr := *g
		gr.Values = slices.Clone(g.Values)
		for i := range gr.Values {
			gr.Values[i].Width *= R
		}
		if err := runCPUTables(hostArm(), codes, &gr, vals, [3][]uint32{}, tables); err != nil {
			t.Fatalf("%+v: %v", s, err)
		}
		got := wordsF32(vals[g.Out[0]])
		worst, e := 0.0, 0
		for j, r := range runs {
			for _, x := range r {
				want := deltaOracle(s, refs[j], q[x*qk:][:qk], k[x*qk:][:qk], v[x*vv:][:vv], d[x*nd:][:nd], beta[x*s.h:][:s.h])
				for i, w := range want {
					worst = max(worst, math.Abs(float64(got[x*vv+i])-w))
				}
				for i := range N {
					worst = max(worst, math.Abs(float64(math.Float32frombits(keep[e*N+i]))-refs[j][i]))
				}
				e++
			}
			for i := range N {
				worst = max(worst, math.Abs(float64(math.Float32frombits(pool[out[j]*N+i]))-refs[j][i]))
			}
		}
		for i := e * N; i < len(keep); i++ {
			if w := keep[i]; w != nan {
				t.Fatalf("%+v: the form kept %v at word %d, past the step's %d entries", s, math.Float32frombits(w), i, e)
			}
		}
		if !(worst <= 2e-5) {
			t.Fatalf("%+v: the kept scan is %g from the oracle", s, worst)
		}
		kept += e
	}
	t.Logf("%d entries' states kept and held to the oracle", kept)
}

// scanDesc is the run descriptor (kernels.RunDescWords' layout) of runs,
// each a sequence's rows of the step in order, over n live rows: run j
// reads slot in[j] and writes slot out[j], the entries past the runs
// repeating the last.
func scanDesc(runs [][]int, n int, in, out []int) []uint32 {
	w := make([]uint32, 1+7*n)
	w[0] = uint32(n)
	at := 0
	for j := range n {
		s, start := min(j, len(runs)-1), at
		if j >= len(runs) {
			start = at - len(runs[s])
		}
		w[1+2*j], w[2+2*j] = uint32(in[s]), uint32(out[s])
		w[1+2*n+2*j], w[2+2*n+2*j] = uint32(start), uint32(len(runs[s]))
		if j < len(runs) {
			for k, r := range runs[j] {
				w[1+4*n+at+k] = uint32(r)
				w[1+5*n+r] = uint32(at + k)
				w[1+6*n+r] = uint32(j)
			}
			at += len(runs[j])
		}
	}
	return w
}

// TestComposedCPUStateConvRows forms OpStateConv on the CPU device for a
// step of several rows over ragged channel counts and window lengths, and
// runs the step of TestComposedCPUDeltaRuleRows's three sequences, each
// run's window read from its slot of one pool and written to its slot of
// another, as the CPU device binds them (the entries past the runs write a
// run's window again, so the window written is not the window read).
// Every row's output and every run's window after are held to the oracle
// stepping the run's rows in order; the slots no run names stay untouched.
// With each row its own run reading its sequence's slot, it must part.
func TestComposedCPUStateConvRows(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	const R, live = 8, 6
	runs := [][]int{{1, 3, 0, 4}, {2}, {5}}
	parted, shapes := 0, 0
	for _, sh := range [][2]int{{2, 3}, {3, 8}, {4, 7}, {4, 33}, {3, 64}, {5, 9}} {
		shapes++
		T, C := sh[0], sh[1]
		b := graph.New()
		x := b.In("x", C)
		b.Out(b.Emit(graph.OpStateConv, []graph.ValueID{x}, []graph.Result{{Name: "conv", Width: C}},
			[]graph.Binding{{Role: jlm.RoleSSMConv1d, K: T, Rows: C}}, graph.Attrs{Taps: T},
			[]graph.StateRef{{State: graph.StateRecConv}}, []graph.StateRef{{State: graph.StateRecConv, Version: 1}})[0])
		_, codes, err := Default().Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleSSMConv1d: jlm.TypeF32},
			Step: Step{Rows: R, Page: 64, Keys: 1}, Device: cpuDevice(hostArm())})
		if err != nil {
			t.Fatalf("%d taps over %d: %v", T, C, err)
		}
		W := (T - 1) * C
		taps, planes := make([]float32, C*T), make([]float64, C*T)
		for c := range C {
			for k := range T {
				v := float32(rng.NormFloat64())
				taps[c*T+k], planes[k*C+c] = v, float64(v)
			}
		}
		wt := unsafe.Slice((*uint32)(unsafe.Pointer(&taps[0])), len(taps))
		for _, swap := range []bool{false, true} {
			for _, unchained := range []bool{false, true} {
				name := fmt.Sprintf("%d taps over %d slots swapped %v unchained %v", T, C, swap, unchained)
				nan := math.Float32bits(float32(math.NaN()))
				pool, outPool := make([]uint32, 7*W), make([]uint32, 7*W)
				for i := range pool {
					pool[i], outPool[i] = nan, nan
				}
				in, out := []int{1, 3, 5}, []int{1, 3, 5}
				if swap {
					out = []int{2, 4, 6}
				}
				refs := make([][]float64, len(runs))
				for j := range runs {
					refs[j] = make([]float64, W)
					for i := range W {
						v := float32(rng.NormFloat64())
						pool[in[j]*W+i], refs[j][i] = math.Float32bits(v), float64(v)
					}
				}
				run, rin, rout := runs, in, out
				if unchained {
					run, rin, rout = nil, nil, nil
					for j, r := range runs {
						for _, x := range r {
							run, rin, rout = append(run, []int{x}), append(rin, in[j]), append(rout, out[j])
						}
					}
				}
				xs := make([]float32, R*C)
				for i := range xs {
					xs[i] = float32(rng.NormFloat64())
				}
				vals := map[graph.ValueID][]uint32{b.G.In[0]: unsafe.Slice((*uint32)(unsafe.Pointer(&xs[0])), len(xs))}
				tables := map[string][]uint32{"rec.conv": pool, "rec.conv.out": outPool, "rec.rows": scanDesc(run, live, rin, rout)}
				gr := *b.G
				gr.Values = slices.Clone(b.G.Values)
				for i := range gr.Values {
					gr.Values[i].Width *= R
				}
				if err := runCPUTables(hostArm(), codes, &gr, vals, [3][]uint32{wt}, tables); err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				got := wordsF32(vals[b.G.Out[0]])
				worst := 0.0
				for j, r := range runs {
					for _, row := range r {
						x64 := make([]float64, C)
						for i := range C {
							x64[i] = float64(xs[row*C+i])
						}
						want := make([]float64, C)
						oracle.Conv1d(want, refs[j], planes, x64, T, C)
						for i, w := range want {
							worst = max(worst, math.Abs(float64(got[row*C+i])-w))
						}
					}
					if !unchained {
						for i := range W {
							worst = max(worst, math.Abs(float64(math.Float32frombits(outPool[out[j]*W+i]))-refs[j][i]))
						}
					}
				}
				for i := range W {
					if w := outPool[i]; !math.IsNaN(float64(math.Float32frombits(w))) {
						t.Fatalf("%s: the form wrote %v in slot 0, which no run names", name, math.Float32frombits(w))
					}
				}
				if unchained {
					if worst > 1e-3 {
						parted++
					}
					continue
				}
				if !(worst <= 1e-5) {
					t.Fatalf("%s: the composed convolution is %g from the oracle", name, worst)
				}
			}
		}
	}
	if parted != 2*shapes {
		t.Fatalf("the unchained window parted from the oracle %d of %d times: the gate cannot see it", parted, 2*shapes)
	}
	t.Logf("%d shapes held to the oracle over a step of %d live rows in %d runs", shapes, live, len(runs))
}

// TestComposedCPUDeltaRuleRows forms OpDeltaRule on the CPU device for a
// step of several rows -- the scan, its state row in memory and its
// components loops -- over the geometries TestComposedCPUDeltaRule runs and
// a key wider than four vectors, and runs a step of three sequences (a
// prompt's run of four rows out of order in the step, and two of one row),
// the live rows fewer than the form's, each run's state read from its slot
// and written to another or over itself. Every row's output and every
// run's state after are held to the oracle stepping the run's rows in
// order; the slots no run names stay untouched. With each row its own run
// reading its sequence's slot (the state not carried along the prompt), it
// must part.
func TestComposedCPUDeltaRuleRows(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	shapes := []deltaShape{
		{h: 2, kh: 1, dk: 4, dv: 4},
		{h: 4, kh: 2, dk: 7, dv: 5},
		{h: 4, kh: 2, dk: 7, dv: 5, tiled: true},
		{h: 8, kh: 2, dk: 9, dv: 8, chans: true},
		{h: 4, kh: 2, dk: 16, dv: 16, tiled: true, chans: true},
		{h: 2, kh: 2, dk: 37, dv: 3},
		{h: 4, kh: 2, dk: 7, dv: 5, additive: true},
		{h: 8, kh: 1, dk: 16, dv: 1, chans: true, additive: true},
		{h: 5, kh: 1, dk: 3, dv: 1, chans: true, additive: true},
		{h: 2, kh: 1, dk: 128, dv: 4},
		{h: 4, kh: 4, dk: 64, dv: 8, chans: true},
	}
	const R, live = 8, 6
	runs := [][]int{{1, 3, 0, 4}, {2}, {5}}
	parted := 0
	for _, s := range shapes {
		g := deltaGraph(s)
		_, codes, err := Default().Form(Request{Graph: g, Formats: map[jlm.Role]jlm.Type{}, Step: Step{Rows: R, Page: 64, Keys: 1}, Device: cpuDevice(hostArm())})
		if err != nil {
			t.Fatalf("%+v: %v", s, err)
		}
		for _, inPlace := range []bool{false, true} {
			for _, unchained := range []bool{false, true} {
				if unchained && inPlace {
					// Written over its slot read, a row's own run reads the
					// state the row before wrote: the state carried.
					continue
				}
				name := fmt.Sprintf("%+v in place %v unchained %v", s, inPlace, unchained)
				N := s.h * s.dv * s.dk
				nan := math.Float32bits(float32(math.NaN()))
				pool := make([]uint32, 7*N)
				for i := range pool {
					pool[i] = nan
				}
				in, out := []int{1, 3, 5}, []int{2, 4, 6}
				if inPlace {
					out = in
				}
				refs := make([][]float64, len(runs))
				for j := range runs {
					refs[j] = make([]float64, N)
					for i := range N {
						x := float32(rng.NormFloat64() * 0.3)
						pool[in[j]*N+i], refs[j][i] = math.Float32bits(x), float64(x)
					}
				}
				run, rin, rout := runs, in, out
				if unchained {
					run, rin, rout = nil, nil, nil
					for j, r := range runs {
						for _, x := range r {
							run, rin, rout = append(run, []int{x}), append(rin, in[j]), append(rout, out[j])
						}
					}
				}
				rnd := func(n int, lo, hi float64) []float32 {
					x := make([]float32, n)
					for i := range x {
						x[i] = float32(lo + (hi-lo)*rng.Float64())
					}
					return x
				}
				nd := s.h
				if s.chans {
					nd = s.h * s.dk
				}
				qk, vv := s.kh*s.dk, s.h*s.dv
				q, k, v := rnd(R*qk, -1, 1), rnd(R*qk, -1, 1), rnd(R*vv, -1, 1)
				d, beta := rnd(R*nd, 0.5, 0.99), rnd(R*s.h, 0.1, 0.9)
				words := func(x []float32) []uint32 { return unsafe.Slice((*uint32)(unsafe.Pointer(&x[0])), len(x)) }
				vals := map[graph.ValueID][]uint32{g.In[0]: words(q), g.In[1]: words(k), g.In[2]: words(v), g.In[3]: words(d), g.In[4]: words(beta)}
				tables := map[string][]uint32{"rec.state": pool, "rec.state.out": pool, "rec.rows": scanDesc(run, live, rin, rout)}
				// The step's values are R rows each.
				gr := *g
				gr.Values = slices.Clone(g.Values)
				for i := range gr.Values {
					gr.Values[i].Width *= R
				}
				if err := runCPUTables(hostArm(), codes, &gr, vals, [3][]uint32{}, tables); err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				got := wordsF32(vals[g.Out[0]])
				worst := 0.0
				for j, r := range runs {
					for _, x := range r {
						want := deltaOracle(s, refs[j], q[x*qk:][:qk], k[x*qk:][:qk], v[x*vv:][:vv], d[x*nd:][:nd], beta[x*s.h:][:s.h])
						for i, w := range want {
							worst = max(worst, math.Abs(float64(got[x*vv+i])-w))
						}
					}
					if !unchained {
						for i := range N {
							worst = max(worst, math.Abs(float64(math.Float32frombits(pool[out[j]*N+i]))-refs[j][i]))
						}
					}
				}
				for sl := range 7 {
					if sl == 0 {
						for i := range N {
							if w := pool[i]; !math.IsNaN(float64(math.Float32frombits(w))) {
								t.Fatalf("%s: the form wrote %v in slot 0, which no run names", name, math.Float32frombits(w))
							}
						}
					}
				}
				if unchained {
					if worst > 1e-3 {
						parted++
					}
					continue
				}
				if !(worst <= 2e-5) {
					t.Fatalf("%s: the composed scan is %g from the oracle", name, worst)
				}
			}
		}
	}
	if parted != len(shapes) {
		t.Fatalf("the unchained state parted from the oracle %d of %d times: the gate cannot see it", parted, len(shapes))
	}
	t.Logf("%d shapes held to the oracle over a step of %d live rows in %d runs", len(shapes), live, len(runs))
}
