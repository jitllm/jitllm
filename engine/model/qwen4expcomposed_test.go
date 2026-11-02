package model

import (
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestQwen4ExpServesComposedOnCPU runs each Qwen3.8-Flash-Next fixture on
// the CPU device with every block composed over its four streams -- each
// sublayer's low-rank gated mix and its placement into every stream, the
// n-gram block, QSA's pooled selection -- and the input rows' copies and the
// head's mix composed too, teacher-forced on the fixture's ids against
// transformers' logits within c6NMSE, and a prompt with its decode the same.
// Each piece of the hyper-connection or of QSA's selection taken out of the
// composed statement alone (q4eComposedViolations), the input rows' streams
// past the first left empty, and stream 1 read for stream 2 where the head's
// mix reads them, moves the logits past 100 times the clean worst and past
// 1e-4.
func TestQwen4ExpServesComposedOnCPU(t *testing.T) {
	for _, fx := range q4eFixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openQ4E(t, fx.name)
			defer m.Close()
			if !fx.sel(m) {
				t.Fatal("the container does not carry the fixture's graph")
			}
			toks := g.IDs
			for li := range m.Cfg.NLayer {
				if why := m.Uncovered(li); why != Covered {
					t.Fatalf("block %d is refused by %q", li, why)
				}
				if g := m.Composing().BlockGraph(li); len(g.Out) != m.Cfg.HCMult {
					t.Fatalf("block %d writes %d streams, want %d", li, len(g.Out), m.Cfg.HCMult)
				}
			}
			golden := func(got [][]float32, at int) float64 {
				w := 0.0
				for i, lg := range got {
					w = max(w, llama4Cmp(g.Pos[at+i].Head, lg))
				}
				return w
			}
			run := func(f cpuFault) float64 {
				return golden(partCPULogits(t, m, toks, f, true, m.Cfg.NLayer, jit.DeclineNoGraph), 0)
			}
			worst := run(cpuFaultNone)
			t.Logf("every block composed: worst NMSE %.3g against transformers over %d steps", worst, len(toks))
			if !(worst < c6NMSE) {
				t.Fatalf("the composed CPU is %.3g from transformers, bound %g", worst, c6NMSE)
			}
			// A prompt's rows and its decode.
			n := len(toks) - 4
			if pworst := golden(q4ePrompt(t, m, toks, n, true), n-1); !(pworst < c6NMSE) {
				t.Fatalf("the composed CPU prompt is %.3g from transformers, bound %g", pworst, c6NMSE)
			}
			check := func(name string, bad float64) {
				t.Helper()
				t.Logf("violation %q: worst NMSE %.3g", name, bad)
				if !(bad > 100*worst) || !(bad > 1e-4) {
					t.Errorf("%s: the composed CPU cannot see it -- worst NMSE %.3g against a clean %.3g", name, bad, worst)
				}
			}
			for _, v := range q4eComposedViolations(t) {
				undo := v.mut(m)
				bad := run(cpuFaultNone)
				undo()
				check(v.name, bad)
			}
			undo := m.rewriteExpansion(t, func(g *graph.Graph) *graph.Graph {
				return rewriteOps(t, g, func(_ int, o *graph.Op) bool {
					if o.Kind != graph.OpSlice {
						return false
					}
					*o = graph.Op{Kind: graph.OpMap, In: o.In, Out: o.Out, Attr: graph.Attrs{Fn: graph.MapMul, Const: 0}}
					return true
				})
			})
			bad := run(cpuFaultNone)
			undo()
			check("the streams past the first start empty", bad)
			check("stream 1 read for stream 2 where the head's mix reads them", run(cpuFaultStreams))
		})
	}
}

// q4ePrompt is m's logits prefilling ids[:n] and decoding four more; on the
// composed arm some block's rows are served and every decline is for having
// no graph.
func q4ePrompt(t *testing.T, m *Model, ids []int32, n int, composed bool) [][]float32 {
	t.Helper()
	s := m.NewInstance(n + 5)
	defer s.Close()
	lg, err := s.Prefill(ids[:n])
	if err != nil {
		t.Fatal(err)
	}
	out := [][]float32{append([]float32(nil), lg...)}
	for i := range 4 {
		if lg, err = s.Forward(ids[n+i]); err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	served, declined := s.CPUBlockSteps()
	if !composed {
		return out
	}
	for i, k := range declined {
		if jit.Decline(i) != jit.DeclineNoGraph && k > 0 {
			t.Errorf("%d block steps declined: %s (%v)", k, jit.DeclineName(i), s.CPUFormErr())
		}
	}
	if served == 0 {
		t.Fatal("no block step of the prompt served")
	}
	t.Logf("a prompt of %d: %d block steps served, %d declined for no graph", n, served, declined[jit.DeclineNoGraph])
	return out
}

// partCPULogits is each step's logits decoding toks; on the composed arm
// `served` blocks serve every step, every other block step is declined for
// why alone, and every head step and streams' row is served.
func partCPULogits(t *testing.T, m *Model, toks []int32, f cpuFault, composed bool, served int, why jit.Decline) [][]float32 {
	t.Helper()
	s := m.NewInstance(len(toks) + 1)
	defer s.Close()
	s.cpu.fault = f
	var out [][]float32
	for _, tk := range toks {
		lg, err := s.Forward(tk)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	got, declined := s.CPUBlockSteps()
	if !composed {
		if got != 0 {
			t.Fatalf("main's arm served %d composed block steps", got)
		}
		return out
	}
	for i, n := range declined {
		if jit.Decline(i) != why && n > 0 {
			t.Errorf("%d block steps declined: %s (%v)", n, jit.DeclineName(i), s.CPUFormErr())
		}
	}
	if want := len(toks) * served; got != want {
		t.Fatalf("%d block steps served, want %d", got, want)
	}
	if want := len(toks) * (m.Cfg.NLayer - served); declined[why] != want {
		t.Fatalf("%d block steps declined for %s, want %d", declined[why], jit.DeclineName(int(why)), want)
	}
	heads, hdecl := s.CPUHeadSteps()
	for i, n := range hdecl {
		if n > 0 {
			t.Errorf("%d head steps declined: %s (%v)", n, jit.DeclineName(i), s.CPUFormErr())
		}
	}
	if heads != len(toks) {
		t.Fatalf("%d head steps served, want %d", heads, len(toks))
	}
	fg, graphs := m.frontGraphs(), 0
	for _, g := range []*graph.Graph{fg.expand, fg.collapse} {
		if g != nil {
			graphs++
		}
	}
	rows, sdecl := s.CPUStreamSteps()
	for i, n := range sdecl {
		if n > 0 {
			t.Errorf("%d rows of the streams' expansion or collapse declined: %s (%v)", n, jit.DeclineName(i), s.CPUFormErr())
		}
	}
	if rows != graphs*len(toks) {
		t.Fatalf("the streams' expansion and collapse served %d rows, want %d", rows, graphs*len(toks))
	}
	return out
}

// q4eComposedViolations are the hyper-connection's pieces taken out of the
// front end's statement alone, main's kernels left whole: the gate read as
// its plain streams, the placement at weight one, the placement without
// its factor of two, and the gate's rank unscaled.
func q4eComposedViolations(t *testing.T) []c6Violation {
	at := func(name string, f func(g *graph.Graph, o *graph.Op) bool) c6Violation {
		return c6Violation{name, func(m *Model) func() {
			return m.rewriteEveryBlock(t, func(g *graph.Graph) *graph.Graph {
				return rewriteOps(t, g, func(_ int, o *graph.Op) bool { return f(g, o) })
			})
		}}
	}
	named := func(g *graph.Graph, o *graph.Op, name string) bool {
		return len(o.Out) > 0 && strings.HasPrefix(g.Values[o.Out[0]].Name, name)
	}
	// qsa breaks each QSA block's pooled selection (graph.Attrs.Pooled) by
	// f, the blocks with none left as they are.
	qsa := func(name string, f func(o *graph.Op)) c6Violation {
		return c6Violation{name, func(m *Model) func() {
			return m.rewriteEveryBlock(t, func(g *graph.Graph) *graph.Graph {
				i := slices.IndexFunc(g.Ops, func(o graph.Op) bool { return o.Kind == graph.OpSelect && o.Attr.Pooled })
				if i < 0 {
					return g
				}
				out := *g
				out.Ops = slices.Clone(g.Ops)
				f(&out.Ops[i])
				return &out
			})
		}}
	}
	return []c6Violation{
		at("the streams mixed by their plain mean", func(g *graph.Graph, o *graph.Op) bool {
			if o.Kind != graph.OpExpr || !named(g, o, "hc.gate") {
				return false
			}
			*o = graph.Op{Kind: graph.OpMap, In: o.In[1:2], Out: o.Out, Attr: graph.Attrs{Fn: graph.MapMul, Const: 1}}
			return true
		}),
		at("the sublayer placed into every stream at weight one", func(g *graph.Graph, o *graph.Op) bool {
			if o.Kind != graph.OpMap || !named(g, o, "hc.placed") {
				return false
			}
			*o = graph.Op{Kind: graph.OpAdd, In: o.In[1:3], Out: o.Out}
			return true
		}),
		at("the placement without its factor of two", func(g *graph.Graph, o *graph.Op) bool {
			if o.Kind != graph.OpMap || !named(g, o, "hc.w") || o.Attr.Fn != graph.MapMul || o.Attr.Const != 2 {
				return false
			}
			o.Attr.Const = 1
			return true
		}),
		at("the gate's rank unscaled", func(g *graph.Graph, o *graph.Op) bool {
			if o.Kind != graph.OpMap || !named(g, o, "hc.lo") {
				return false
			}
			o.Attr.Const = 1
			return true
		}),
		qsa("QSA's blocks not turned at their first positions", func(o *graph.Op) { o.Attr.NRot = 0 }),
		qsa("QSA's blocks scored by their best key, not their pooled key", func(o *graph.Op) {
			o.Attr.Pooled, o.Attr.NRot, o.Weight = false, 0, nil
		}),
		qsa("QSA keeping one block fewer", func(o *graph.Op) { o.Attr.TopK-- }),
		qsa("every key read on the QSA blocks", func(o *graph.Op) { o.Attr.TopK = 64 }),
	}
}

// rewriteExpansion is the front end's set with the input rows' streams'
// expansion rewritten.
func (m *Model) rewriteExpansion(t *testing.T, rewrite func(*graph.Graph) *graph.Graph) func() {
	t.Helper()
	s := m.frontGraphs()
	if s.expand == nil {
		t.Fatal("the model has no expansion to rewrite")
	}
	set := *s
	set.expand = rewrite(s.expand)
	if err := graph.Verify(set.expand, graph.Facts{}); err != nil {
		t.Fatalf("the rewritten expansion does not verify: %v", err)
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

// rewriteCollapse is the front end's set with the head's collapse of the
// streams rewritten.
func (m *Model) rewriteCollapse(t *testing.T, rewrite func(*graph.Graph) *graph.Graph) func() {
	t.Helper()
	s := m.frontGraphs()
	if s.collapse == nil {
		t.Fatal("the model has no collapse to rewrite")
	}
	set := *s
	set.collapse = rewrite(s.collapse)
	f, _ := m.inventory(jlm.DenseBlock)
	if err := graph.Verify(set.collapse, f); err != nil {
		t.Fatalf("the rewritten collapse does not verify: %v", err)
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

// TestQwen4ExpServesComposed holds the composer to each Qwen3.8-Flash-Next
// fixture's stated blocks on every device present: the four streams bound
// as views of the step's residual, each sublayer's gated mix and placement,
// the delta rule's state read at slots carrying the n-gram tail, every
// stated block step served and every other declined for having no graph
// (the n-gram block), within 1e-5 of the host's logits. Each piece
// of the hyper-connection taken out of the statement (q4eComposedViolations)
// moves them past 100 times the clean worst and past 1e-4.
func TestQwen4ExpServesComposed(t *testing.T) {
	const bound, promptLen = 1e-5, 3
	ran := 0
	for _, fx := range q4eFixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openQ4E(t, fx.name)
			defer m.Close()
			if !fx.sel(m) {
				t.Fatal("the container does not carry the fixture's graph")
			}
			covered := coveredBlocks(m)
			if covered == 0 {
				t.Fatal("no block is stated: the gate selects nothing")
			}
			ids := g.IDs
			host := m.NewInstance(len(ids) + 1)
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
				served, declined := st.ServedPart(tier.ComposedBlock)
				if n := len(toks) * covered; served != n || declined != len(toks)*(m.Cfg.NLayer-covered) ||
					st.BlockDeclines[jit.DeclineNoGraph] != declined {
					t.Fatalf("%d block steps served and %d declined, want %d served and the rest declined for no graph (%s)",
						served, declined, n, declinesOf(st))
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
				t.Logf("%s: %d block steps served, %d declined (%s), worst NMSE %.3g", dev, served, declined, declinesOf(st), worst)
				return worst
			}
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					ran++
					worst := run(t, dev)
					if !(worst < bound) {
						t.Fatalf("worst logit NMSE %.3g against the host, bound %g", worst, bound)
					}
					for _, v := range q4eComposedViolations(t) {
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
