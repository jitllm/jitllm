package model

import (
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestKimiK3ServesComposed holds the composer to each Kimi-K3 fixture's
// blocks on every device present: the four residual streams (the running
// residual and three checkpoints) bound as views of the step's residual,
// each sublayer's softmax mix of the bank, the checkpoint banked and the
// residual restarted, KDA, MLA and the latent mixture, every block step
// served, within 1e-5 of the host's logits. Each piece taken out of the
// statement (k3ComposedViolations) must move them past 100 times the clean
// worst and past 1e-4. The streams' collapse before the head is the
// host's, and the input rows and a head over every stream are refused by
// name (UncoveredStreams).
func TestKimiK3ServesComposed(t *testing.T) {
	const bound, promptLen = 1e-5, 3
	ran := 0
	for _, fx := range k3Fixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openK3(t, fx.name)
			defer m.Close()
			if !fx.sel(m) {
				t.Fatal("the container does not carry the fixture's graph")
			}
			if c := coveredBlocks(m); c != m.Cfg.NLayer {
				t.Fatalf("the front end states %d of %d blocks: %s (%v)", c, m.Cfg.NLayer, m.Uncovered(0), m.UncoveredErr(0))
			}
			if s := m.frontGraphs(); s.head != nil || s.headWhy != UncoveredStreams || s.embed != nil || s.embedWhy != UncoveredStreams {
				t.Fatalf("the head and the input rows over several streams are %s and %s, want both refused by name", s.headWhy, s.embedWhy)
			}
			ids := g.IDs
			host := m.NewState(len(ids) + 1)
			var want [][]float32
			for _, id := range ids {
				lg, err := host.Forward(id)
				if err != nil {
					t.Fatal(err)
				}
				want = append(want, append([]float32(nil), lg...))
			}
			host.Close()
			prompt, toks := ids[:promptLen], ids[promptLen:]
			run := func(t *testing.T, dev string) float64 {
				t.Helper()
				got, st, _, _ := composedDecode(t, m, dev, prompt, toks, tier.ComposedBlock, true, false)
				served, declined, _ := st.ServedPart(tier.ComposedBlock)
				if n := len(toks) * m.Cfg.NLayer; served != n || declined != 0 {
					t.Fatalf("%d block steps served and %d declined, want %d served (%s)", served, declined, n, declinesOf(st))
				}
				if st.FormedAtStep != 0 {
					t.Fatalf("%d composed part(s) formed at a step, not at the placement", st.FormedAtStep)
				}
				worst := 0.0
				for i := range got {
					e := logitNMSE(got[i], want[promptLen-1+i])
					if math.IsNaN(e) {
						t.Fatalf("step %d: the logits are not finite", i)
					}
					worst = max(worst, e)
				}
				t.Logf("%s: %d block steps served, worst NMSE %.3g", dev, served, worst)
				return worst
			}
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					ran++
					worst := run(t, dev)
					if !(worst < bound) {
						t.Fatalf("worst logit NMSE %.3g against the host, bound %g", worst, bound)
					}
					for _, v := range k3ComposedViolations(t) {
						undo := v.mut(m)
						bad := run(t, dev)
						undo()
						t.Logf("violation %q: worst NMSE %.3g", v.name, bad)
						if !(bad > 100*worst) || !(bad > 1e-4) {
							t.Errorf("%s: the composed arm cannot see it -- worst NMSE %.3g against a clean %.3g", v.name, bad, worst)
						}
					}
				})
			}
		})
	}
	if ran == 0 {
		t.Fatal("no device on this host")
	}
}

// TestKimiK3ServesComposedOnCPU runs each Kimi-K3 fixture on the CPU
// device with the head's mix over the bank composed, teacher-forced on the
// fixture's ids against main's kernels within 1e-5: every head step and
// row of the mix served, every block step declined by name (the CPU's forms
// hold no latent attention's history; the cards serve the blocks,
// TestKimiK3ServesComposed). The input rows need no expansion: a block banks
// each checkpoint before any block reads it. A checkpoint read twice where
// the mix reads the bank (a softmax mix is the same over any order of its
// rows, so a swap is not a violation it can see), and the mix's score vector
// dropped, move the logits past 100 times the clean worst and past 1e-4.
func TestKimiK3ServesComposedOnCPU(t *testing.T) {
	const bound = 1e-5
	for _, fx := range k3Fixtures {
		t.Run(fx.name, func(t *testing.T) {
			gm, g := openK3(t, fx.name)
			ok := fx.sel(gm)
			gm.Close()
			if !ok {
				t.Fatal("the container does not carry the fixture's graph")
			}
			path := jlmOf(t, testmodels.Path(fx.name+".gguf"))
			ref, err := Open(path, noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer ref.Close()
			want := partCPULogits(t, ref, g.IDs, cpuFaultNone, false, 0, 0)
			m, err := Open(path, noTune, WithComposedCPU(true))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if s := m.frontGraphs(); s.expand != nil || s.collapse == nil {
				t.Fatal("the bank's input rows are expanded, or the head's mix over it is not stated")
			}
			run := func(f cpuFault) float64 {
				return worstFrom(want, partCPULogits(t, m, g.IDs, f, true, 0, jit.DeclineLatent))
			}
			worst := run(cpuFaultNone)
			t.Logf("the composed CPU head's mix from main: worst NMSE %.3g over %d steps", worst, len(g.IDs))
			if !(worst < bound) {
				t.Fatalf("the composed CPU is %.3g from main, bound %g", worst, bound)
			}
			bad := run(cpuFaultStreams)
			t.Logf("violation \"a checkpoint read twice where the head's mix reads the bank\": worst NMSE %.3g", bad)
			if !(bad > 100*worst) || !(bad > 1e-4) {
				t.Errorf("a checkpoint read twice: the composed CPU cannot see it -- worst NMSE %.3g against a clean %.3g", bad, worst)
			}
			undo := m.rewriteCollapse(t, func(g *graph.Graph) *graph.Graph {
				return rewriteOps(t, g, func(_ int, o *graph.Op) bool {
					if o.Kind != graph.OpMap || len(o.Weight) == 0 || o.Weight[0].Role != jlm.RoleOutputResScore {
						return false
					}
					*o = graph.Op{Kind: graph.OpMap, In: o.In, Out: o.Out, Attr: graph.Attrs{Fn: graph.MapMul, Const: 1}}
					return true
				})
			})
			bad = run(cpuFaultNone)
			undo()
			t.Logf("violation \"the mix scored without its stored vector\": worst NMSE %.3g", bad)
			if !(bad > 100*worst) || !(bad > 1e-4) {
				t.Errorf("the mix unscored: the composed CPU cannot see it -- worst NMSE %.3g against a clean %.3g", bad, worst)
			}
		})
	}
}

// k3ComposedViolations are k3Violations as the composed arm reads them:
// main's fault set beside the statement the front end makes without the
// piece (its stored vector dropped, or the block's graph rewritten). The
// head's mix is the host's, and a checkpoint every three blocks is a bank of
// other streams than the device was placed with, so neither is here. KDA's
// output gate is held by Kimi-Linear's composed gates.
func k3ComposedViolations(t *testing.T) []c6Violation {
	both := func(f k3Fault, undoS func(*Model) func()) func(*Model) func() {
		return func(m *Model) func() {
			undoC := cfgMut(m, func(c *Config) { c.k3Fault = f })
			undoR := undoS(m)
			return func() { undoR(); undoC() }
		}
	}
	return []c6Violation{
		{"no residual attention: every sublayer reads the running residual", both(k3FaultNoBank, func(m *Model) func() {
			return dropStored(m, jlm.RoleAttnResScore, jlm.RoleFFNResScore)
		})},
		{"the bank scored on raw values", both(k3FaultRawScores, func(m *Model) func() {
			return m.rewriteEveryBlock(t, func(g *graph.Graph) *graph.Graph {
				return rewriteOps(t, g, func(_ int, o *graph.Op) bool {
					if o.Kind != graph.OpHeadNorm || !strings.HasPrefix(g.Values[o.Out[0]].Name, "bank.normed") {
						return false
					}
					*o = graph.Op{Kind: graph.OpMap, In: o.In, Out: o.Out, Attr: graph.Attrs{Fn: graph.MapMul, Const: 1}}
					return true
				})
			})
		})},
		{"a checkpoint block adds its attention instead of restarting", both(k3FaultNoRestart, func(m *Model) func() {
			return m.rewriteEveryBlock(t, func(g *graph.Graph) *graph.Graph { return addedNotRestarted(m, g) })
		})},
		{"the latent sum unnormed", both(k3FaultNoLatentNorm, func(m *Model) func() {
			return dropStored(m, jlm.RoleFFNRoutedNorm)
		})},
		{"the MLA output ungated", both(k3FaultNoMLAGate, func(m *Model) func() {
			return m.restateAt(func(li int, inv map[jlm.Role]graph.Shape) {
				if !m.Cfg.LayerKind(li).Recurrent() {
					delete(inv, jlm.RoleAttnGate)
				}
			})
		})},
		{"the KDA decay unbounded (Kimi-Linear's softplus)", func(m *Model) func() {
			return cfgMut(m, func(c *Config) { c.KDALowerBound = 0 })
		}},
		{"the MLA latent norms at rms_norm_eps (llama.cpp's)", func(m *Model) func() {
			return cfgMut(m, func(c *Config) { c.LatentNormEps = c.RMSEps })
		}},
		{"plain SwiGLU", actSwap(nn.ActSiLU)},
	}
}

// restateAt is restate with the block each inventory is of.
func (m *Model) restateAt(f func(li int, inv map[jlm.Role]graph.Shape)) func() {
	fr := &m.front
	fr.mu.Lock()
	was := fr.restate
	fr.restate = func(block int32, inv map[jlm.Role]graph.Shape) {
		if block != jlm.DenseBlock {
			f(int(block), inv)
		}
	}
	fr.restated++
	fr.mu.Unlock()
	return func() {
		fr.mu.Lock()
		fr.restate = was
		fr.restated++
		fr.mu.Unlock()
	}
}

// rewriteEveryBlock is rewriteBlocks over every block the front end states,
// the rewrite left out where it returns the graph it was given.
func (m *Model) rewriteEveryBlock(t *testing.T, rewrite func(*graph.Graph) *graph.Graph) func() {
	t.Helper()
	s := m.frontGraphs()
	set := *s
	set.blocks = slices.Clone(s.blocks)
	rewritten := 0
	for li, g := range set.blocks {
		if g == nil {
			continue
		}
		if r := rewrite(g); r != g {
			f, _ := m.inventory(int32(li))
			if err := graph.Verify(r, f); err != nil {
				t.Fatalf("the rewrite of block %d does not verify: %v", li, err)
			}
			set.blocks[li], rewritten = r, rewritten+1
		}
	}
	if rewritten == 0 {
		t.Fatal("the rewrite changed no block")
	}
	m.front.mu.Lock()
	was := m.front.set
	m.front.set = &set
	m.front.mu.Unlock()
	return func() {
		m.front.mu.Lock()
		m.front.set = was
		m.front.mu.Unlock()
	}
}

// addedNotRestarted is a checkpoint block's graph with its attention's
// output added to the running residual (the graph's first input) where it
// restarts the residual: every read of the output after the projection
// reads their sum. g itself on a block that banks nothing.
func addedNotRestarted(m *Model, g *graph.Graph) *graph.Graph {
	if _, ok := m.Cfg.resCheckpoint(g.Block); !ok {
		return g
	}
	// The projection is the last op of the attention's output role before
	// the FFN's mix.
	at := -1
	for i, o := range g.Ops {
		if (o.Kind == graph.OpMatVec || o.Kind == graph.OpAdd) && len(o.Weight) > 0 &&
			(o.Weight[0].Role == jlm.RoleAttnOut || o.Weight[0].Role == jlm.RoleSSMOut) {
			at = i
		}
	}
	if at < 0 {
		return g
	}
	out := *g
	out.Values = slices.Clone(g.Values)
	proj := g.Ops[at].Out[0]
	sum := graph.ValueID(len(out.Values))
	out.Values = append(out.Values, graph.Value{Name: "restart.added", Width: g.Values[proj].Width})
	ops := make([]graph.Op, 0, len(g.Ops)+1)
	for i, o := range g.Ops {
		if i > at {
			o.In = slices.Clone(o.In)
			for j, v := range o.In {
				if v == proj {
					o.In[j] = sum
				}
			}
		}
		ops = append(ops, o)
		if i == at {
			ops = append(ops, graph.Op{Kind: graph.OpAdd, In: []graph.ValueID{g.In[0], proj}, Out: []graph.ValueID{sum}})
		}
	}
	out.Ops = ops
	out.Out = slices.Clone(g.Out)
	for j, v := range out.Out {
		if v == proj {
			out.Out[j] = sum
		}
	}
	return &out
}
