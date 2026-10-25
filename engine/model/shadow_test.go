package model

import (
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// shadowModels are the served checkpoints the shadow gate runs, each a block
// the others do not share: a llama, a qwen3 with a norm over q and k, a gemma
// with a scaled embedding and a GELU gate, an OLMo 2 with its norms after the
// sublayers.
var shadowModels = []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Qwen3-0.6B-Q8_0.gguf", "gemma-2b.gguf",
	"newarch/OLMo-2-0425-1B-Instruct-Q4_K_M.gguf"}

// shadowRun is a teacher-forced decode on a tier whose composer shadows the
// blocks (Config.ComposedShadow).
type shadowRun struct {
	logits [][]float32
	st     tier.Stats
	// worst is every block's worst drift after each decode step.
	worst [][]float32
}

// shadowDecode decodes toks teacher-forced after ids on a fresh tier of dev
// whose composer forms every block (Config.Composed) and runs it beside the
// tier's own kernels (tier.WithComposedShadow), reading every block's worst
// drift after each step. float forms the composer's blocks with float
// activations, the reference the tier's own distance is measured to.
func shadowDecode(t *testing.T, m *Model, dev string, ids, toks []int32, noGraph, float bool, extra ...tier.Option) shadowRun {
	t.Helper()
	opts := append([]tier.Option{tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff), tier.WithComposedShadow(),
		tier.WithConfig(func(c *tier.Config) { c.Composed, c.NoGraph, c.ComposedFloat = tier.ComposedBlock, noGraph, float })},
		testTierOpts(t)...)
	g, err := tier.OpenWith(append(opts, extra...)...)
	if err != nil || g == nil {
		noDevice(t, dev, err)
	}
	defer g.Close()
	s := m.NewState(len(ids) + len(toks) + 4)
	defer s.Close()
	s.SetDeviceFallback(false)
	if err := s.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	if s.GPULayers() != m.Cfg.NLayer {
		t.Fatalf("%s took %d of %d blocks: %s", dev, s.GPULayers(), m.Cfg.NLayer, g.Err())
	}
	l, err := s.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	var run shadowRun
	for _, tok := range toks {
		run.logits = append(run.logits, append([]float32(nil), l...))
		if l, err = s.Forward(tok); err != nil {
			t.Fatal(err)
		}
		run.worst = append(run.worst, g.Stats().ShadowWorst)
	}
	run.logits = append(run.logits, append([]float32(nil), l...))
	if s.DeviceDemotions() != 0 {
		t.Fatalf("the device failed %d time(s) and its blocks moved home", s.DeviceDemotions())
	}
	run.st = g.Stats()
	return run
}

// shadowCompared holds a shadow run to having compared every block at every
// step: the device counted one comparison a block and step (a replayed
// recording's included), each drift is a finite number, no part formed at a
// step, and the composer was asked for every block and declined none. Issued
// one launch at a time, the host emitted a form and a comparison for every
// block of every step.
func shadowCompared(t *testing.T, run shadowRun, nb, steps int, noGraph bool) {
	t.Helper()
	st := run.st
	if cmp, _, _ := st.ServedPart(tier.PartShadow); noGraph && cmp != steps*nb {
		t.Fatalf("the host emitted %d comparisons where %d steps of %d blocks are %d", cmp, steps, nb, steps*nb)
	}
	if len(st.ShadowWorst) != nb || len(st.ShadowSteps) != nb {
		t.Fatalf("the stats hold %d drifts and %d step counts for %d blocks", len(st.ShadowWorst), len(st.ShadowSteps), nb)
	}
	for li := range nb {
		w := float64(st.ShadowWorst[li])
		if st.ShadowSteps[li] != steps || math.IsNaN(w) || math.IsInf(w, 0) || w < 0 || w >= graph.DriftCap {
			t.Fatalf("block %d was compared in %d of %d steps, its worst drift %g: the shadow arm did not hold", li, st.ShadowSteps[li], steps, w)
		}
	}
	if st.FormedAtStep != 0 {
		t.Fatalf("%d composed part(s) formed at a step, not at the placement", st.FormedAtStep)
	}
	served, declined, _ := st.ServedPart(tier.ComposedBlock)
	if declined != 0 || noGraph && served != steps*nb {
		t.Fatalf("the composer served %d block steps and was declined %d, where %d steps of %d blocks are %d (%s)",
			served, declined, steps, nb, steps*nb, st.LastErr)
	}
}

// cloneGraph is a copy of g no rewrite of it reaches the original through.
func cloneGraph(g *graph.Graph) *graph.Graph {
	c := *g
	c.Values, c.In, c.Out = slices.Clone(g.Values), slices.Clone(g.In), slices.Clone(g.Out)
	c.Ops = make([]graph.Op, len(g.Ops))
	for i, o := range g.Ops {
		o.In, o.Out = slices.Clone(o.In), slices.Clone(o.Out)
		o.Weight, o.Reads, o.Writes = slices.Clone(o.Weight), slices.Clone(o.Reads), slices.Clone(o.Writes)
		c.Ops[i] = o
	}
	return &c
}

// rewriteBlock puts f's rewrite of a copy of block li's graph in the front
// end's statement of the model, where a tier placing the block takes it, and
// returns what puts the front end's back; nil where f found nothing to
// rewrite. The tier's own kernels do not read the graph, so only the
// composer's form of the block is the rewrite.
func rewriteBlock(t *testing.T, m *Model, li int, f func(*graph.Graph) bool) func() {
	t.Helper()
	set := m.frontGraphs()
	m.front.mu.Lock()
	defer m.front.mu.Unlock()
	old := set.blocks[li]
	if old == nil {
		t.Fatalf("block %d has no graph to rewrite", li)
	}
	g := cloneGraph(old)
	if !f(g) {
		return nil
	}
	set.blocks[li] = g
	return func() {
		m.front.mu.Lock()
		set.blocks[li] = old
		m.front.mu.Unlock()
	}
}

// faultWindow is the window the window fault gives the attention.
const faultWindow = 32

// shadowFault is a rewrite of one block's graph, the composer's form of the
// block alone computing it. first says from which decode step it can differ
// from the block's own graph, given the prompt's length, 0 for any.
type shadowFault struct {
	name  string
	apply func(*graph.Graph) bool
	first func(prompt int) int
}

// shadowRefused is the faults the tier declines by name before the form can
// run them, so the shadow arm never compares the rewritten block: a graph
// window that is not the plan's (the windowed descriptors are the plan's).
var shadowRefused = map[string]string{
	fmt.Sprintf("the attention sees %d positions", faultWindow): "a graph whose window is not the plan's",
}

var shadowFaults = []shadowFault{
	{"the activation is another", func(g *graph.Graph) bool {
		for i := range g.Ops {
			if o := &g.Ops[i]; o.Kind == graph.OpActMul || o.Kind == graph.OpAct {
				if o.Attr.Act == graph.ActReLU2 {
					o.Attr.Act = graph.ActSiLU
				} else {
					o.Attr.Act = graph.ActReLU2
				}
				return true
			}
		}
		return false
	}, func(int) int { return 0 }},
	{fmt.Sprintf("the attention sees %d positions", faultWindow), func(g *graph.Graph) bool {
		for i := range g.Ops {
			if o := &g.Ops[i]; o.Kind == graph.OpAttend && o.Attr.Window == 0 {
				o.Attr.Window = faultWindow
				return true
			}
		}
		return false
	}, func(prompt int) int { return faultWindow - prompt }},
	{"the attention's scale is doubled", func(g *graph.Graph) bool {
		for i := range g.Ops {
			if o := &g.Ops[i]; o.Kind == graph.OpAttend {
				o.Attr.Scale *= 2
				return true
			}
		}
		return false
	}, func(int) int { return 0 }},
	{"the rotary pairs the other way", func(g *graph.Graph) bool {
		for i := range g.Ops {
			if o := &g.Ops[i]; o.Kind == graph.OpRope {
				if o.Attr.Layout == graph.RopeNeox {
					o.Attr.Layout = graph.RopeNorm
				} else {
					o.Attr.Layout = graph.RopeNeox
				}
				return true
			}
		}
		return false
	}, func(int) int { return 0 }},
	{"the key projection reads the value's weights", func(g *graph.Graph) bool {
		var k, v *graph.Op
		for i := range g.Ops {
			switch o := &g.Ops[i]; {
			case o.Kind != graph.OpMatVec:
			case o.Weight[0].Role == jlm.RoleAttnK:
				k = o
			case o.Weight[0].Role == jlm.RoleAttnV:
				v = o
			}
		}
		if k == nil || v == nil || k.Weight[0].Rows != v.Weight[0].Rows {
			return false
		}
		k.Weight[0].Role, v.Weight[0].Role = jlm.RoleAttnV, jlm.RoleAttnK
		return true
	}, func(int) int { return 0 }},
}

// TestComposedShadowHoldsEveryBlock runs the composer's form of every block
// beside the tier's own kernels (tier.WithComposedShadow) on served
// checkpoints and every backend the box has, and holds the pair to three
// things.
//
// The shadow arm observes. The logits it leaves are the tier's own, bit for
// bit at every step, on a tier issuing its launches one at a time and on one
// replaying a recording. That is the history gate: the form appends its key
// and value to the page the step writes before the tier's kernels write the
// same row, so the row that stays is the tier's, and a form that put a second
// row anywhere else would change the logits. Served alone, with nothing
// beside it, the same blocks move them, so the equality can see a difference.
//
// Every block stays inside the spread of the tier's own backends, at every
// depth. The band is measured as TestComposedBlocksServe measures it: the
// tier's distance from the composer's float reference on each backend, here a
// block at a time on the input the tier gave it, the widest of the backends
// at its worst. A block's worst drift from the tier's, with the int8 form,
// may be no more than bandWorst times that, and the median block's no more
// than bandRatio times.
//
// A rewrite of one block's graph, which the composer's form alone computes,
// is flagged at that block and at no other: the stream is the tier's, so the
// blocks after it keep the input they would have had. A rewrite that cannot
// differ before some step is not flagged before it.
func TestComposedShadowHoldsEveryBlock(t *testing.T) {
	ran := 0
	for _, file := range shadowModels {
		t.Run(file, func(t *testing.T) {
			path := testmodels.Path(file)
			if _, err := os.Stat(path); err != nil {
				t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS)", path)
			}
			m, err := Open(jlmOf(t, path), noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			ids := m.Vocab.Encode("The capital of France is", true)
			n := 140
			if c := int(m.Cfg.NCtx); c > 0 {
				n = min(n, c-len(ids)-4)
			}
			toks := make([]int32, n)
			for i := range toks {
				toks[i] = int32((int(ids[i%len(ids)])*7 + i) % m.Vocab.Size())
			}
			nb := m.Cfg.NLayer
			// The tier without the composer, with and without the captured
			// graph, and its distance from the float reference on every
			// backend first: the widest of them is the band.
			plain := map[string]*[2][][]float32{}
			own := map[string][]float32{}
			band := make([]float32, nb)
			var devs []string
			for _, dev := range stepDevices() {
				t.Run(dev+"/reference", func(t *testing.T) {
					p := &[2][][]float32{}
					for g, noGraph := range []bool{false, true} {
						p[g], _, _, _ = composedDecode(t, m, dev, ids, toks, 0, noGraph, false)
					}
					ref := shadowDecode(t, m, dev, ids, toks, true, true)
					shadowCompared(t, ref, nb, len(toks), true)
					for li, w := range ref.st.ShadowWorst {
						if !(w > 0) {
							t.Fatalf("block %d: the tier is %g from the float reference: a degenerate reference proves nothing", li, w)
						}
						band[li] = max(band[li], w)
					}
					t.Logf("the tier's own drift from the float reference, a block's worst over %d steps: %s", len(toks), driftSpread(ref.st.ShadowWorst))
					// Served alone, the same blocks must move the logits, or
					// their equality below could not tell a form that wrote
					// the history from one that did not.
					served, _, _, _ := composedDecode(t, m, dev, ids, toks, tier.ComposedBlock, true, false)
					differ := 0
					for i := range served {
						if !slices.Equal(served[i], p[1][i]) {
							differ++
						}
					}
					if differ == 0 {
						t.Fatalf("served alone the composed blocks leave the tier's logits as they were at every step: the equality below cannot see a form that writes the history")
					}
					t.Logf("served alone the composed blocks change the logits at %d of %d steps", differ, len(served))
					plain[dev], own[dev] = p, ref.st.ShadowWorst
					devs = append(devs, dev)
				})
			}
			bound := make([]float32, nb)
			for li := range bound {
				bound[li] = bandWorst * band[li]
			}
			for _, dev := range devs {
				t.Run(dev, func(t *testing.T) {
					for g, noGraph := range []bool{false, true} {
						name := "graph"
						if noGraph {
							name = "no graph"
						}
						t.Run(name, func(t *testing.T) {
							run := shadowDecode(t, m, dev, ids, toks, noGraph, false)
							shadowCompared(t, run, nb, len(toks), noGraph)
							// A card that records its launches replays them for
							// most of the decode, and the device's count of the
							// comparisons includes every replay: the recorded
							// step carries the arm.
							if !noGraph && strings.HasPrefix(dev, "cuda") {
								if run.st.Captures == 0 {
									t.Fatalf("the launch sequence was never recorded: this arm replayed nothing")
								}
								t.Logf("the launch sequence was recorded %d times; the device counted %d comparisons a block over %d steps",
									run.st.Captures, run.st.ShadowSteps[0], len(toks))
							}
							for i := range run.logits {
								if !slices.Equal(run.logits[i], plain[dev][g][i]) {
									t.Fatalf("step %d: with the shadow arm the logits are not the tier's: it wrote what the tier reads", i)
								}
							}
							ratios := make([]float64, nb)
							for li, w := range run.st.ShadowWorst {
								ratios[li] = float64(w) / float64(own[dev][li])
								if !(w <= bound[li]) {
									t.Errorf("block %d drifts %.3g from the tier's, past %.3g: %.1f times the widest of the tier's backends, %.3g from the float reference (bound %.1f)",
										li, w, bound[li], float64(w)/float64(band[li]), band[li], bandWorst)
								}
							}
							slices.Sort(ratios)
							if med := ratios[nb/2]; !(med <= bandRatio) {
								t.Errorf("at the median block the composed form drifts %.3g times the tier's own distance from the float reference here (bound %.1f)", med, bandRatio)
							}
							t.Logf("the composed blocks' worst drift from the tier's, a block's worst over %d steps: %s", len(toks), driftSpread(run.st.ShadowWorst))
							t.Logf("in this backend's tier's own distance from the float reference: min %.3g, median %.3g, max %.3g over %d blocks (bounds %.1f at the median, %.1f against the widest backend)",
								ratios[0], ratios[nb/2], ratios[nb-1], nb, bandRatio, bandWorst)
							// Every buffer filled with NaN at allocation: a copy
							// or a comparison reading what nothing wrote shows as
							// a changed drift, or a changed logit.
							if noGraph {
								poisoned := shadowDecode(t, m, dev, ids, toks, true, false, tier.WithPoisonScratch(true), tier.WithPoisonKV(true))
								shadowCompared(t, poisoned, nb, len(toks), true)
								for i := range run.logits {
									if !slices.Equal(poisoned.logits[i], run.logits[i]) {
										t.Fatalf("step %d: with every buffer poisoned the logits change: the shadow arm reads what nothing wrote", i)
									}
								}
								if !slices.Equal(poisoned.st.ShadowWorst, run.st.ShadowWorst) {
									t.Fatalf("with every buffer poisoned the drifts change: %v against %v", poisoned.st.ShadowWorst, run.st.ShadowWorst)
								}
							}
							ran++
						})
					}
				})
			}
			k := nb / 2
			for _, f := range shadowFaults {
				for _, dev := range devs {
					t.Run(fmt.Sprintf("%s/block %d: %s", dev, k, f.name), func(t *testing.T) {
						restore := rewriteBlock(t, m, k, f.apply)
						if restore == nil {
							t.Fatalf("block %d has nothing for this rewrite to change", k)
						}
						defer restore()
						run := shadowDecode(t, m, dev, ids, toks, true, false)
						if why := shadowRefused[f.name]; why != "" {
							for i := range run.logits {
								if !slices.Equal(run.logits[i], plain[dev][1][i]) {
									t.Fatalf("step %d: the rewrite reached the tier's logits: the stream is not the tier's", i)
								}
							}
							named := 0
							for i, n := range run.st.BlockDeclines {
								if n == 0 {
									continue
								}
								if tier.DeclineName(i) != why {
									t.Fatalf("%d block step(s) declined because %s, where the rewrite is refused as %q", n, tier.DeclineName(i), why)
								}
								named += n
							}
							if named != len(toks) || run.st.ShadowSteps[k] != 0 {
								t.Fatalf("%d block steps declined as %q and block %d compared in %d, where its %d steps are all refused",
									named, why, k, run.st.ShadowSteps[k], len(toks))
							}
							t.Logf("block %d refused as %q at all %d steps, never compared", k, why, len(toks))
							ran++
							return
						}
						shadowCompared(t, run, nb, len(toks), true)
						for i := range run.logits {
							if !slices.Equal(run.logits[i], plain[dev][1][i]) {
								t.Fatalf("step %d: the rewrite reached the tier's logits: the stream is not the tier's", i)
							}
						}
						first := make([]int, nb)
						for j := range first {
							first[j] = -1
							for i, w := range run.worst {
								if w[j] > bound[j] {
									first[j] = i
									break
								}
							}
						}
						if first[k] < 0 {
							t.Fatalf("block %d drifts %.3g at most from the tier's, inside the bound %.3g: the shadow arm cannot see the rewrite",
								k, run.st.ShadowWorst[k], bound[k])
						}
						if from := f.first(len(ids)); first[k] < from {
							t.Errorf("block %d flagged at step %d, before step %d, where the rewrite can first differ", k, first[k], from)
						}
						for j := range nb {
							if j != k && first[j] >= 0 {
								t.Errorf("block %d flagged at step %d, though only block %d was rewritten", j, first[j], k)
							}
						}
						t.Logf("block %d flagged at step %d, first possible at %d: a drift of %.3g where the bound is %.3g (%.0fx), the widest of the other blocks %.3g",
							k, first[k], max(0, f.first(len(ids))), run.st.ShadowWorst[k], bound[k], run.st.ShadowWorst[k]/bound[k], widestBut(run.st.ShadowWorst, k))
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

// widestBut is the largest of w but its element skip.
func widestBut(w []float32, skip int) float32 {
	var m float32
	for i, v := range w {
		if i != skip {
			m = max(m, v)
		}
	}
	return m
}

// driftSpread is the smallest, median and largest of w, and where the largest is.
func driftSpread(w []float32) string {
	s := slices.Clone(w)
	slices.Sort(s)
	return fmt.Sprintf("min %.3g, median %.3g, max %.3g (block %d) over %d blocks", s[0], s[len(s)/2], s[len(s)-1], slices.Index(w, s[len(s)-1]), len(s))
}
