package composer_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUStateConvRowTaps forms OpStateConv with per-row taps (each
// group of channels' own taps a row, added to the static ones: a dynamic
// short convolution) on every GPU, over ragged channel counts, taps and
// groups, its window in the recurrent convolution's state and at an offset
// into the recurrent state's slots, at a step of one row and of eight over
// the recurrent gates' steps. Every live row's output and the window each
// run's slot written holds meet the oracle under the row's summed taps; with
// the per-row taps dropped, or the groups taken interleaved over the
// channels, the oracle must part.
func TestComposedGPUStateConvRowTaps(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			dev := gpuDevice(d)
			rng := rand.New(rand.NewSource(31))
			ran, dropped, interleaved, interleavable := 0, 0, 0, 0
			for _, s := range []struct{ T, C, G, off, span int }{
				{2, 3, 1, -1, 0}, {3, 8, 2, -1, 0}, {4, 7, 7, -1, 0}, {4, 33, 3, 5, 5 + 3*33 + 2}, {3, 64, 4, -1, 0}, {5, 9, 3, 0, 4 * 9},
			} {
				for _, R := range []int{1, 8} {
					b := graph.New()
					x := b.In("x", s.C)
					dt := b.In("taps", s.G*s.T)
					st := graph.StateRecConv
					if s.off >= 0 {
						st = graph.StateRecState
					}
					b.Out(b.Emit(graph.OpStateConv, []graph.ValueID{x, dt}, []graph.Result{{Name: "conv", Width: s.C}},
						[]graph.Binding{{Role: jlm.RoleSSMConv1d, K: s.T, Rows: s.C}}, graph.Attrs{Taps: s.T, Offset: max(s.off, 0), Span: s.span},
						[]graph.StateRef{{State: st}}, []graph.StateRef{{State: st, Version: 1}})[0])
					fr, err := formRec(d, dev, b.G, R, map[jlm.Role]jlm.Type{jlm.RoleSSMConv1d: jlm.TypeF32})
					if err != nil {
						t.Fatalf("%+v at %d rows: %v", s, R, err)
					}
					steps := recSteps()
					if R == 1 {
						steps = steps[2:3]
					}
					for _, step := range steps {
						if step.n > R {
							continue
						}
						worst, err := runStateConvTaps(fr, b.G, s.T, s.C, s.G, s.off, s.span, R, step.n, step.runs, rng)
						if err != nil {
							fr.close()
							t.Fatalf("%+v at %d rows %s: %v", s, R, step.name, err)
						}
						if !(worst[0] <= 1e-5) {
							fr.close()
							t.Fatalf("%+v at %d rows %s: the composed convolution is %g from the oracle", s, R, step.name, worst[0])
						}
						if worst[1] > 1e-3 {
							dropped++
						}
						if s.G > 1 && s.C/s.G > 1 {
							interleavable++
							if worst[2] > 1e-3 {
								interleaved++
							}
						}
						ran++
					}
					fr.close()
				}
			}
			if dropped != ran || interleaved != interleavable {
				t.Fatalf("of %d steps the dropped taps parted %d, the interleaved groups %d of %d: the gate cannot see them",
					ran, dropped, interleaved, interleavable)
			}
			t.Logf("%d steps held to the oracle; every violation parted", ran)
		})
	}
}

// runStateConvTaps runs one step of fr over the runs, each row's taps of G
// groups beside its row, its window in rec.conv or (off >= 0) at off of the
// recurrent state's slots of span floats, and returns the worst distance at
// the live rows' outputs and the windows written from the oracle, from the
// oracle with the per-row taps dropped, and from the oracle with the groups
// interleaved over the channels.
func runStateConvTaps(fr *recRun, g *graph.Graph, T, C, G, off, span, R, n int, runs recRuns, rng *rand.Rand) ([3]float64, error) {
	H := T - 1
	gw := C / G
	taps, dyn, x := make([]float32, C*T), make([]float32, R*G*T), make([]float32, R*C)
	for i := range taps {
		taps[i] = float32(rng.NormFloat64())
	}
	for i := range dyn {
		dyn[i] = float32(rng.NormFloat64())
	}
	for i := range x {
		x[i] = float32(rng.NormFloat64())
		if i >= n*C {
			x[i] = float32(math.NaN())
		}
	}
	name, base0 := "rec.conv", 0
	if off >= 0 {
		name, base0 = "rec.state", off
	} else {
		span = H * C
	}
	slots := len(runs) + 2
	in, out := make([]uint32, slots*span), make([]uint32, slots*span)
	for i := range in {
		in[i] = math.Float32bits(float32(rng.NormFloat64()))
		out[i] = math.Float32bits(float32(math.NaN()))
	}
	fr.vals = map[graph.ValueID][]uint32{g.In[0]: f32w(x), g.In[1]: f32w(dyn), g.Out[0]: poisoned(R * C)}
	fr.tables = map[string][]uint32{name: in, name + ".out": out, "rec.rows": runs.desc(n, 1, 1)}
	fr.wts = map[jlm.Role][]uint32{jlm.RoleSSMConv1d: f32w(taps)}
	clear(fr.bufs)
	if err := fr.launch(); err != nil {
		return [3]float64{}, err
	}
	got := wf32(fr.vals[g.Out[0]])
	groups := [3]func(c int) int{func(c int) int { return c / gw }, nil, func(c int) int { return c % G }}
	var worst [3]float64
	for j, run := range runs {
		base := (1+j)*span + base0
		seq := make([]float64, 0, (H+len(run))*C)
		for i := range H * C {
			seq = append(seq, float64(math.Float32frombits(in[base+i])))
		}
		for _, r := range run {
			for c := range C {
				seq = append(seq, float64(x[r*C+c]))
			}
		}
		for e, r := range run {
			p := H + e
			for c := range C {
				for a, group := range groups {
					var want float64
					for k := range T {
						tap := float64(taps[c*T+k])
						if group != nil {
							tap += float64(dyn[r*G*T+group(c)*T+k])
						}
						want += tap * seq[(p-(T-1-k))*C+c]
					}
					worst[a] = max(worst[a], math.Abs(float64(got[r*C+c])-want))
				}
			}
		}
		last := len(seq)/C - H
		for i := range H * C {
			worst[0] = max(worst[0], math.Abs(float64(math.Float32frombits(out[base+i]))-seq[last*C+i]))
		}
	}
	return worst, nil
}
