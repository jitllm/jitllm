package engine

import (
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/graph"
)

// negatedSample is sample over the logits negated: the token of the
// smallest, a wrong pick's violation.
func negatedSample(s *graph.Graph) *graph.Graph {
	g := cloneGraph(s)
	neg := graph.ValueID(len(g.Values))
	g.Values = append(g.Values, graph.Value{Name: "negated", Width: g.Values[g.In[0]].Width})
	g.Ops = slices.Insert(g.Ops, 0, graph.Op{Kind: graph.OpMap, In: []graph.ValueID{g.In[0]}, Out: []graph.ValueID{neg},
		Attr: graph.Attrs{Fn: graph.MapMul, Const: -1}})
	g.Ops[1].In = []graph.ValueID{neg}
	return g
}

// TestComposedHeadTokens holds the tokens a device head selects -- the greedy
// one (nn.Head.ArgmaxOnly) and the sampler's (nn.Head.SampleK) -- to the
// logits it returns, for a head with a bias, a softcap and a scale. Where the
// device served the head's graph it has applied them all and selects every
// token itself, a bias and a scale included; where it ran its own head it
// declines the requests that a stage left to the host would make wrong, and
// the host selects from the logits it finished. Either way the tokens are the
// host's own, and the counts say which arm ran: a gate that cannot tell the
// two would certify a device that selected before the bias, or a host that
// never asked. With the token served too (ComposedSample) the composer
// selects every greedy token the device does, and over the logits negated
// (a wrong pick) the tokens part from the host's.
func TestComposedHeadTokens(t *testing.T) {
	const gen = 24
	ran := 0
	for _, fx := range headFixtures() {
		t.Run(fx.name, func(t *testing.T) {
			for _, dev := range stepDevices() {
				for _, parts := range []gpu.Parts{gpu.ComposedHead | gpu.ComposedSample} {
					name := dev + "/own head"
					switch {
					case parts&gpu.ComposedSample != 0:
						name = dev + "/composed head and token"
					case parts != 0:
						name = dev + "/composed head"
					}
					t.Run(name, func(t *testing.T) {
						g, err := openGPU(t, gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff))
						if err != nil || g == nil {
							noDevice(t, dev, err)
						}
						defer g.Close()
						m, openOpts := fx.open(t)
						defer m.Close()
						// A head with a bias takes no softcap: the tier leaves the
						// pair to the host.
						biased := m.OutB != nil
						defer cfgMut(m, func(c *model.Config) {
							if !biased {
								c.FinalSoftcap = 1.5
							}
							c.LogitScale = 8
						})()
						stages := m.HeadStages()
						ids := headTokens(m)
						newStateOn := func(g *gpu.GPU) *Instance {
							st := testInstance(t, m, len(ids)+gen+4, append(append([]Option(nil), openOpts...), WithDeviceSample(DeviceSampleOn))...)
							st.SetDeviceFallback(false)
							if err := st.SetDevice(g); err != nil {
								t.Fatal(err)
							}
							if st.GPULayers() != m.Cfg.NLayer || !st.HeadOnDevice() {
								t.Fatalf("wanted every block and the head on %s, got %d of %d blocks and the head %v: %s", dev, st.GPULayers(), m.Cfg.NLayer, st.HeadOnDevice(), g.Err())
							}
							return st
						}
						newState := func() *Instance { return newStateOn(g) }
						// Greedy: the token the device selects against Greedy over
						// the logits of the same position.
						a, b := newState(), newState()
						defer a.Close()
						defer b.Close()
						served := 0
						for p, id := range ids {
							lg, err := a.Forward(id)
							if err != nil {
								t.Fatal(err)
							}
							want := Greedy(lg)
							b.head.Token = -1
							got, err := b.ForwardGreedy(id)
							if err != nil {
								t.Fatal(err)
							}
							if got != want {
								t.Fatalf("pos %d: ForwardGreedy %d, Greedy over the logits %d", p, got, want)
							}
							if b.head.Token >= 0 {
								served++
							}
						}
						wantServed := len(ids)
						if parts == 0 && stages&nn.HeadBias != 0 {
							wantServed = 0
						}
						if served != wantServed {
							t.Fatalf("the device selected %d of %d greedy tokens, want %d (stages %03b, parts %d)", served, len(ids), wantServed, stages, parts)
						}
						if parts&gpu.ComposedSample != 0 {
							ns, ds := g.Stats().ServedPart(gpu.ComposedSample)
							ns += g.Stats().ReplayedPart(gpu.ComposedSample)
							if ns != served || ds != 0 {
								t.Fatalf("the composer selected %d of the %d greedy tokens the device did and was declined %d (%s)", ns, served, ds, declinesOf(g.Stats()))
							}
							// The violation: the token over the logits negated, on
							// a tier of its own that records nothing, so that no
							// recording of the right token replays.
							set := m.FrontGraphs()
							orig := set.CpuSample
							m.Front.Mu.Lock()
							set.CpuSample = negatedSample(orig)
							m.Front.Mu.Unlock()
							vg, err := openGPU(t, gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff),
								gpu.WithConfig(func(c *gpu.Config) { c.NoGraph = true }))
							if err != nil || vg == nil {
								t.Fatalf("%s did not open for the violation: %v", dev, err)
							}
							defer vg.Close()
							vState := func() *Instance {
								st := testInstance(t, m, len(ids)+gen+4, append(append([]Option(nil), openOpts...), WithDeviceSample(DeviceSampleOn))...)
								st.SetDeviceFallback(false)
								if err := st.SetDevice(vg); err != nil {
									t.Fatal(err)
								}
								return st
							}
							va, vb := vState(), vState()
							parted := 0
							for _, id := range ids {
								lg, err := va.Forward(id)
								if err != nil {
									t.Fatal(err)
								}
								got, err := vb.ForwardGreedy(id)
								if err != nil {
									t.Fatal(err)
								}
								if got != Greedy(lg) {
									parted++
								}
							}
							va.Close()
							vb.Close()
							m.Front.Mu.Lock()
							set.CpuSample = orig
							m.Front.Mu.Unlock()
							if vn, _ := vg.Stats().ServedPart(gpu.ComposedSample); vn != len(ids) {
								t.Fatalf("the violation's tier selected %d of %d tokens: it did not run the negated pick (%s)", vn, len(ids), declinesOf(vg.Stats()))
							}
							if parted == 0 {
								t.Fatalf("with the token taken over the logits negated every greedy token is the host's: the gate cannot see a wrong pick")
							}
							t.Logf("%d greedy tokens selected by the composer; over the logits negated %d of %d part", ns, parted, len(ids))
						}
						// The sampler: the same seeded draws over the device's
						// candidates and over the logits it returned.
						k := min(40, m.Cfg.NVocab/4)
						newSampler := func() *Sampler {
							return &Sampler{Temp: 1.3, TopK: k, TopP: 0.9, MinP: 0.01, RepeatPen: 1.3, RepeatLastN: 32, Seed: 4242}
						}
						// draws is gen seeded draws made two ways, from the device's
						// candidates (ForwardSample) and from the logits it returned
						// (Sample over Forward's): how many parted, how many the
						// device selected, and the second's tokens. hide clears the
						// head's stages from the first, so that nothing tells the
						// device a stage is the host's to apply. With ref the second
						// is ref replayed: a violation that changes every instance's
						// draw is held to the clean draws.
						drawsOn := func(g *gpu.GPU, hide bool, ref []int32) (parted int, onDev int64, refs []int32) {
							c := newStateOn(g)
							defer c.Close()
							var d *Instance
							if ref == nil {
								d = newStateOn(g)
								defer d.Close()
							}
							if hide {
								c.head.Stages = 0
							}
							sc, sd := newSampler(), newSampler()
							next := func(i int, lg []float32) int32 {
								if ref != nil {
									return ref[i]
								}
								nd, err := d.Sample(sd, lg)
								if err != nil {
									t.Fatal(err)
								}
								return nd
							}
							lc, err := c.Prefill(ids)
							if err != nil {
								t.Fatal(err)
							}
							var ld []float32
							if d != nil {
								if ld, err = d.Prefill(ids); err != nil {
									t.Fatal(err)
								}
							}
							nc, err := c.Sample(sc, lc)
							if err != nil {
								t.Fatal(err)
							}
							nd := next(0, ld)
							refs = append(refs, nd)
							for i := range gen {
								if nc != nd {
									parted++
								}
								sc.Observe(nd)
								sd.Observe(nd)
								if nc, err = c.ForwardSample(nd, sc); err != nil {
									t.Fatal(err)
								}
								var lg []float32
								if d != nil {
									if lg, err = d.Forward(nd); err != nil {
										t.Fatal(err)
									}
								}
								nd = next(i+1, lg)
								refs = append(refs, nd)
							}
							return parted, c.SampledOnDevice, refs
						}
						draws := func(hide bool) (int, int64) {
							p, n, _ := drawsOn(g, hide, nil)
							return p, n
						}
						before := composedSamples(g.Stats())
						parted, onDev, clean := drawsOn(g, false, nil)
						if parted != 0 {
							t.Fatalf("%d of %d draws from the device's candidates parted from the draws over its logits", parted, gen)
						}
						// Every draw is the device's: the composer serves the
						// prompt's head too, its bias and its scale with it.
						wantSampled := int64(gen)
						if onDev != wantSampled {
							t.Fatalf("the device selected %d of %d sampled tokens, want %d (stages %03b, parts %d)", onDev, gen, wantSampled, stages, parts)
						}
						t.Logf("%d greedy tokens (%d on the device) and %d sampled tokens (%d on the device), stages %03b", len(ids), served, gen, onDev, stages)
						if parts&gpu.ComposedSample != 0 {
							// The composer selects every candidate set the device
							// does: one form, whatever k and the history.
							if n := composedSamples(g.Stats()) - before; n != int(onDev) {
								t.Fatalf("the composer selected %d of the %d candidate sets the device did (%s)", n, onDev, declinesOf(g.Stats()))
							}
							// The violations, each on a tier of its own that
							// records nothing: the k+1 best kept in place of the k
							// (the best dropped), and the penalty on the
							// neighbouring id. Each must part the draws, every set
							// still the composer's.
							set := m.FrontGraphs()
							origWalk := set.CpuWalk
							for _, v := range []struct {
								name string
								f    func(*graph.Graph) *graph.Graph
							}{{"k+1", sampleShifted}, {"the penalty on the neighbouring id", sampleNeighbour}} {
								m.Front.Mu.Lock()
								set.CpuWalk = v.f(origWalk)
								m.Front.Mu.Unlock()
								vg, err := openGPU(t, gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff),
									gpu.WithConfig(func(c *gpu.Config) { c.NoGraph = true }))
								if err != nil || vg == nil {
									t.Fatalf("%s did not open for the violation: %v", dev, err)
								}
								vp, vd, _ := drawsOn(vg, false, clean)
								vn := composedSamples(vg.Stats())
								vg.Close()
								m.Front.Mu.Lock()
								set.CpuWalk = origWalk
								m.Front.Mu.Unlock()
								t.Logf("violation, %s: %d of %d draws parted, %d sets selected by the composer", v.name, vp, gen, vn)
								if vd == 0 || vn < int(vd) || vp == 0 {
									t.Fatalf("violation, %s: %d of %d draws parted, %d on the device, %d by the composer: the gate cannot see it", v.name, vp, gen, vd, vn)
								}
							}
						}
						// The violation: the tier's own head, which leaves the bias
						// and the scale to the host, asked to select as though it
						// had applied them. It does, before them, and the draws part.
						if parts == 0 && stages&(nn.HeadBias|nn.HeadScale) != 0 {
							parted, onDev := draws(true)
							t.Logf("violation, the stages hidden from the device: %d of %d draws parted, %d selected on the device", parted, gen, onDev)
							if onDev != int64(gen) || parted == 0 {
								t.Fatalf("the gate cannot see a device that selects before the host's stages: %d of %d draws parted, %d selected on the device", parted, gen, onDev)
							}
						}
						ran++
					})
				}
			}
		})
	}
	if ran == 0 {
		t.Fatal("no arm ran")
	}
}

// sampleShifted is a sampler graph (the candidates', or the walk over them)
// keeping the k+1 best and dropping the best: the candidates' ranking one
// place wider, read from its second.
func sampleShifted(s *graph.Graph) *graph.Graph {
	g := cloneGraph(s)
	last := -1
	for i, o := range g.Ops {
		if o.Kind == graph.OpTopK && len(o.Out) == 2 && strings.HasPrefix(g.Values[o.Out[1]].Name, "sample.vals.") {
			last = i
		}
	}
	o := g.Ops[last]
	k := o.Attr.TopK
	wide := []graph.ValueID{graph.ValueID(len(g.Values)), graph.ValueID(len(g.Values) + 1)}
	g.Values = append(g.Values, graph.Value{Name: "shifted.place", Width: k + 1, Elem: graph.ElemU32}, graph.Value{Name: "shifted.vals", Width: k + 1})
	outs := o.Out
	o.Attr.TopK, o.Out = k+1, wide
	g.Ops[last] = o
	g.Ops = slices.Insert(g.Ops, last+1,
		graph.Op{Kind: graph.OpSlice, In: []graph.ValueID{wide[0]}, Out: []graph.ValueID{outs[0]}, Attr: graph.Attrs{Offset: 1, Width: k}},
		graph.Op{Kind: graph.OpSlice, In: []graph.ValueID{wide[1]}, Out: []graph.ValueID{outs[1]}, Attr: graph.Attrs{Offset: 1, Width: k}})
	return g
}

// sampleNeighbour is a sampler graph whose window (its second input) is
// read one id on: the penalty on the token after each one the window holds.
func sampleNeighbour(s *graph.Graph) *graph.Graph {
	g := cloneGraph(s)
	win := g.In[1]
	n := g.Values[win].Width
	last, rest, rot := graph.ValueID(len(g.Values)), graph.ValueID(len(g.Values)+1), graph.ValueID(len(g.Values)+2)
	g.Values = append(g.Values, graph.Value{Name: "win.last", Width: 1}, graph.Value{Name: "win.rest", Width: n - 1},
		graph.Value{Name: "win.rot", Width: n})
	for i := range g.Ops {
		for x, v := range g.Ops[i].In {
			if v == win {
				g.Ops[i].In[x] = rot
			}
		}
	}
	g.Ops = slices.Insert(g.Ops, 0,
		graph.Op{Kind: graph.OpSlice, In: []graph.ValueID{win}, Out: []graph.ValueID{last}, Attr: graph.Attrs{Offset: n - 1, Width: 1}},
		graph.Op{Kind: graph.OpSlice, In: []graph.ValueID{win}, Out: []graph.ValueID{rest}, Attr: graph.Attrs{Offset: 0, Width: n - 1}},
		graph.Op{Kind: graph.OpConcat, In: []graph.ValueID{last, rest}, Out: []graph.ValueID{rot}})
	return g
}

// composedSamples is how many steps the composer selected a token or the
// sampler's candidates in, emitted or replayed.
func composedSamples(s gpu.Stats) int {
	n, _ := s.ServedPart(gpu.ComposedSample)
	return n + s.ReplayedPart(gpu.ComposedSample)
}
