package engine

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestKimiK3ServesComposed holds the composer to each Kimi-K3 fixture's
// blocks on every device present: the four residual streams (the running
// residual and three checkpoints) bound as views of the step's residual,
// each sublayer's softmax mix of the bank, the checkpoint banked and the
// residual restarted, KDA, MLA and the latent mixture, every block step
// served, within 1e-5 of the host's logits. Each piece taken out of the
// statement (k3ComposedViolations) must move them past 100 times the clean
// worst and past 1e-4. The input rows are the composer's, the running
// residual's alone (a block banks each checkpoint before any block reads
// it), and the embedding doubled there moves the logits as far. The
// streams' collapse before the head is the host's, and a head over every
// stream is refused by name (UncoveredStreams).
func TestKimiK3ServesComposed(t *testing.T) {
	const bound, promptLen = 1e-5, 3
	ran := 0
	for _, fx := range k3Fixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, g, openOpts := openK3(t, fx.name)
			defer m.Close()
			if !fx.sel(m) {
				t.Fatal("the container does not carry the fixture's graph")
			}
			if c := coveredBlocks(m); c != m.Cfg.NLayer {
				t.Fatalf("the front end states %d of %d blocks: %s (%v)", c, m.Cfg.NLayer, m.Uncovered(0), m.UncoveredErr(0))
			}
			if s := m.FrontGraphs(); s.Head != nil || s.HeadWhy != model.UncoveredStreams || s.Embed == nil {
				t.Fatalf("the head over several streams is %s, want refused by name, and the input rows %s, want stated", s.HeadWhy, s.EmbedWhy)
			}
			ids := g.IDs
			host := testInstance(t, m, len(ids)+1, openOpts...)
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
				got, st, _ := composedDecode(t, m, openOpts, dev, prompt, toks, gpu.ComposedBlock|gpu.ComposedEmbed, true, false)
				served, declined := st.ServedPart(gpu.ComposedBlock)
				if n := len(toks) * m.Cfg.NLayer; served != n || declined != 0 {
					t.Fatalf("%d block steps served and %d declined, want %d served (%s)", served, declined, n, declinesOf(st))
				}
				ne, de := st.ServedPart(gpu.ComposedEmbed)
				if ne += st.ReplayedPart(gpu.ComposedEmbed); ne < len(toks) || de != 0 {
					t.Fatalf("the input rows served %d times and declined %d, want every step's (%s)", ne, de, declinesOf(st))
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
					for _, v := range append(k3ComposedViolations(t), embedDoubled(t)) {
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
// fixture's ids against main's kernels within 1e-5: every head step, row of
// the mix and block step served, its latent blocks among them. The input rows
// need no expansion: a block banks
// each checkpoint before any block reads it. A checkpoint read twice where
// the mix reads the bank (a softmax mix is the same over any order of its
// rows, so a swap is not a violation it can see), and the mix's score vector
// dropped, move the logits past 100 times the clean worst and past 1e-4.
func TestKimiK3ServesComposedOnCPU(t *testing.T) {
	const bound = 1e-5
	for _, fx := range k3Fixtures() {
		t.Run(fx.name, func(t *testing.T) {
			gm, g, _ := openK3(t, fx.name)
			ok := fx.sel(gm)
			gm.Close()
			if !ok {
				t.Fatal("the container does not carry the fixture's graph")
			}
			// The reference is the CPU's composed float forms of the model as
			// stated (main's own arm is gone); its answer against Moonshot's is
			// TestKimiK3MatchesReference's. The arm under test runs the same
			// forms, so what parts it from the reference is a violation.
			path := jlmOf(t, testmodels.Path(fx.name+".gguf"))
			instOpts := []Option{withCPUFloat()}
			ref, err := model.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer ref.Close()
			want := partCPULogits(t, ref, g.IDs, cpuFaultNone, true, ref.Cfg.NLayer, composer.DeclineServes, instOpts...)
			instOpts2 := []Option{withCPUFloat()}
			m, err := model.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if s := m.FrontGraphs(); s.Expand != nil || s.Collapse == nil {
				t.Fatal("the bank's input rows are expanded, or the head's mix over it is not stated")
			}
			run := func(f cpuFault) float64 {
				return worstFrom(want, partCPULogits(t, m, g.IDs, f, true, m.Cfg.NLayer, composer.DeclineServes, instOpts2...))
			}
			worst := run(cpuFaultNone)
			t.Logf("the composed CPU head's mix from the reference: worst NMSE %.3g over %d steps", worst, len(g.IDs))
			if !(worst < bound) {
				t.Fatalf("the composed CPU is %.3g from the reference, bound %g", worst, bound)
			}
			bad := run(cpuFaultStreams)
			t.Logf("violation \"a checkpoint read twice where the head's mix reads the bank\": worst NMSE %.3g", bad)
			if !(bad > 100*worst) || !(bad > 1e-4) {
				t.Errorf("a checkpoint read twice: the composed CPU cannot see it -- worst NMSE %.3g against a clean %.3g", bad, worst)
			}
			undo := rewriteCollapse(m, t, func(g *graph.Graph) *graph.Graph {
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
// other streams than the device was placed with, so neither is here.
func k3ComposedViolations(t *testing.T) []c6Violation {
	return []c6Violation{
		{"no residual attention: every sublayer reads the running residual", unbanked},
		{"the bank scored on raw values", rawBankScores},
		{"a checkpoint block adds its attention instead of restarting", unrestarted},
		{"the latent sum unnormed", latentUnnormed},
		{"the MLA output ungated", latentAttnUngated},
		{"the KDA output ungated", recurrentUngated},
		{"the KDA decay unbounded (Kimi-Linear's softplus)", func(m *model.Model) func() {
			return cfgMut(m, func(c *model.Config) { c.KDALowerBound = 0 })
		}},
		{"the MLA latent norms at rms_norm_eps (llama.cpp's)", func(m *model.Model) func() {
			return cfgMut(m, func(c *model.Config) { c.LatentNormEps = c.RMSEps })
		}},
		{"plain SwiGLU", actSwap(graph.ActSiLU)},
	}
}

// rewriteEveryBlock is rewriteBlocks over every block the front end states,
// the rewrite left out where it returns the graph it was given.
func rewriteEveryBlock(m *model.Model, t *testing.T, rewrite func(*graph.Graph) *graph.Graph) func() {
	t.Helper()
	s := m.FrontGraphs()
	set := *s
	set.Blocks = slices.Clone(s.Blocks)
	rewritten := 0
	for li, g := range set.Blocks {
		if g == nil {
			continue
		}
		if r := rewrite(g); r != g {
			f, _ := m.Inventory(int32(li))
			if err := graph.Verify(r, f); err != nil {
				t.Fatalf("the rewrite of block %d does not verify: %v", li, err)
			}
			set.Blocks[li], rewritten = r, rewritten+1
		}
	}
	if rewritten == 0 {
		t.Fatal("the rewrite changed no block")
	}
	m.Front.Mu.Lock()
	was := m.Front.Set
	m.Front.Set = &set
	m.Front.Mu.Unlock()
	return func() {
		m.Front.Mu.Lock()
		m.Front.Set = was
		m.Front.Mu.Unlock()
	}
}

// embedDoubled is the input rows' graph with each row's embedding doubled.
func embedDoubled(t *testing.T) c6Violation {
	return c6Violation{"the input rows doubled", func(m *model.Model) func() {
		s := m.FrontGraphs()
		if s.Embed == nil {
			t.Fatal("the model states no input rows")
		}
		g := *s.Embed
		g.Values = slices.Clone(s.Embed.Values)
		g.Ops = slices.Clone(s.Embed.Ops)
		out := g.Out[0]
		doubled := graph.ValueID(len(g.Values))
		g.Values = append(g.Values, graph.Value{Name: "embd.doubled", Width: g.Values[out].Width})
		g.Ops = append(g.Ops, graph.Op{Kind: graph.OpMap, In: []graph.ValueID{out}, Out: []graph.ValueID{doubled},
			Attr: graph.Attrs{Fn: graph.MapMul, Const: 2}})
		g.Out = append([]graph.ValueID{doubled}, g.Out[1:]...)
		set := *s
		set.Embed = &g
		m.Front.Mu.Lock()
		was := m.Front.Set
		m.Front.Set = &set
		m.Front.Mu.Unlock()
		return func() {
			m.Front.Mu.Lock()
			m.Front.Set = was
			m.Front.Mu.Unlock()
		}
	}}
}
