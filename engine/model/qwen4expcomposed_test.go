package model

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
)

// TestQwen4ExpServesComposedOnCPU runs each Qwen3.8-Flash-Next fixture on
// the CPU device with the composer's blocks over its four streams, each
// sublayer's low-rank gated mix and its placement into every stream stated
// as ops, the input rows' copies and the head's mix composed too,
// teacher-forced on the fixture's ids against main's kernels within 1e-5.
// The blocks the front end does not state are declined by name and counted:
// the attention blocks select their keys by QSA (UncoveredSelect) and the
// n-gram block (UncoveredNGram). Each piece of the hyper-connection taken out
// of the composed statement alone (q4eComposedViolations), the input rows'
// streams past the first left empty, and stream 1 read for stream 2 where
// the head's mix reads them, moves the logits past 100 times the clean worst and
// past 1e-4.
func TestQwen4ExpServesComposedOnCPU(t *testing.T) {
	const bound = 1e-5
	for _, fx := range q4eFixtures {
		t.Run(fx.name, func(t *testing.T) {
			path := jlmOf(t, testmodels.Path(fx.name+".gguf"))
			ref, err := Open(path, noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer ref.Close()
			if !fx.sel(ref) {
				t.Fatal("the container does not carry the fixture's graph")
			}
			gm, g := openQ4E(t, fx.name)
			gm.Close()
			toks := g.IDs
			want := partCPULogits(t, ref, toks, cpuFaultNone, false, 0, 0)
			m, err := Open(path, noTune, WithComposedCPU(true))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			covered := 0
			for li := range m.Cfg.NLayer {
				switch why := m.Uncovered(li); why {
				case Covered:
					covered++
					if g := m.Composing().BlockGraph(li); len(g.Out) != m.Cfg.HCMult {
						t.Fatalf("block %d writes %d streams, want %d", li, len(g.Out), m.Cfg.HCMult)
					}
				case UncoveredSelect, UncoveredNGram:
					t.Logf("block %d declined: %s", li, why)
				default:
					t.Fatalf("block %d is refused by %q", li, why)
				}
			}
			if covered == 0 {
				t.Fatal("no block is stated: the gate selects nothing")
			}
			worst := worstFrom(want, partCPULogits(t, m, toks, cpuFaultNone, true, covered, jit.DeclineNoGraph))
			t.Logf("%d of %d blocks composed, worst NMSE %.3g over %d steps", covered, m.Cfg.NLayer, worst, len(toks))
			if !(worst < bound) {
				t.Fatalf("the composed CPU is %.3g from main, bound %g", worst, bound)
			}
			for _, v := range q4eComposedViolations(t) {
				undo := v.mut(m)
				bad := worstFrom(want, partCPULogits(t, m, toks, cpuFaultNone, true, covered, jit.DeclineNoGraph))
				undo()
				t.Logf("violation %q: worst NMSE %.3g", v.name, bad)
				if !(bad > 100*worst) || !(bad > 1e-4) {
					t.Errorf("%s: the composed CPU cannot see it -- worst NMSE %.3g against a clean %.3g", v.name, bad, worst)
				}
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
			bad := worstFrom(want, partCPULogits(t, m, toks, cpuFaultNone, true, covered, jit.DeclineNoGraph))
			undo()
			t.Logf("violation \"the streams past the first start empty\": worst NMSE %.3g", bad)
			if !(bad > 100*worst) || !(bad > 1e-4) {
				t.Errorf("the streams start empty: the composed CPU cannot see it -- worst NMSE %.3g against a clean %.3g", bad, worst)
			}
			bad = worstFrom(want, partCPULogits(t, m, toks, cpuFaultStreams, true, covered, jit.DeclineNoGraph))
			t.Logf("violation \"stream 1 read for stream 2 where the head's mix reads them\": worst NMSE %.3g", bad)
			if !(bad > 100*worst) || !(bad > 1e-4) {
				t.Errorf("a stream read twice: the composed CPU cannot see it -- worst NMSE %.3g against a clean %.3g", bad, worst)
			}
		})
	}
}

// partCPULogits is each step's logits decoding toks; on the composed arm
// `served` blocks serve every step, every other block step is declined for
// why alone, and every head step and streams' row is served.
func partCPULogits(t *testing.T, m *Model, toks []int32, f cpuFault, composed bool, served int, why jit.Decline) [][]float32 {
	t.Helper()
	s := m.NewState(len(toks) + 1)
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
