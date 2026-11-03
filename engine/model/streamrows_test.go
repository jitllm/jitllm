package model

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/engine/graph"
)

// TestStreamRowsServeOnTheCard holds the composer to the input rows and the
// head of a model of several residual streams on every device present: the
// input rows' graph writes every stream, each a view of the stream-major
// residual on the card, and the per-layer inputs where the model has them
// (nn.Head.Embed, its tensors nn.Head.Streams); a call of the head alone
// hands the card every stream, which the composer's form of the front end's
// collapse reduces to the row the head reads (nn.Head.Collapse). Gemma 3n's
// blocks are served too, DeepSeek V4's are the tier's (each selects its
// keys). Each decode step's rows and head are served, every head call's
// collapse, the prompt's rows declined by name (the host fills a chunk's
// first stream alone), within 1e-5 of the host's logits; each of
// streamRowsViolations moves them past 100 times the clean worst and past
// 1e-4.
func TestStreamRowsServeOnTheCard(t *testing.T) {
	const bound, promptLen = 1e-5, 3
	cases := []struct {
		name   string
		open   func(*testing.T, string) (*Model, *c6Golden)
		blocks bool
	}{
		{"synth-gemma3n", openGemma3n, true},
	}
	ran := 0
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m, g := tc.open(t, tc.name)
			defer m.Close()
			f := m.frontGraphs()
			outs := m.Cfg.Streams()
			if m.Cfg.PLEDim != 0 {
				outs++
			}
			if f.embed == nil || len(f.embed.Out) != outs || f.collapse == nil {
				t.Fatalf("the front end states the input rows over %d streams as %v (%s) and the collapse as %v",
					m.Cfg.Streams(), f.embed != nil, f.embedWhy, f.collapse != nil)
			}
			host := m.NewInstance(len(g.IDs) + 1)
			var want [][]float32
			for _, id := range g.IDs {
				l, err := host.Forward(id)
				if err != nil {
					t.Fatal(err)
				}
				want = append(want, append([]float32(nil), l...))
			}
			host.Close()
			prompt, toks := g.IDs[:promptLen], g.IDs[promptLen:]
			heads := len(toks)
			parts := gpu.ComposedEmbed | gpu.ComposedHead
			if tc.blocks {
				parts |= gpu.ComposedBlock | gpu.ComposedPrompt
				heads++
			}
			decode := func(t *testing.T, dev string, after func(*Instance)) float64 {
				t.Helper()
				var collapses bool
				got, st, _ := composedDecodeWith(t, m, dev, prompt, toks, parts, true, false, func(s *Instance) {
					collapses = s.head != nil && s.head.Collapses
					if after != nil {
						after(s)
					}
				})
				ne, de := st.ServedPart(gpu.ComposedEmbed)
				nc, dc := st.ServedPart(gpu.CollapsedHead)
				nh, dh := st.ServedPart(gpu.ComposedHead)
				nb, db := st.ServedPart(gpu.ComposedBlock)
				t.Logf("%s: %d input rows, %d collapses, %d heads and %d block steps served; declined %d, %d, %d, %d (%s)",
					dev, ne, nc, nh, nb, de, dc, dh, db, declinesOf(st))
				switch {
				case !collapses:
					t.Fatalf("the card does not take the head's collapse")
				case ne != len(toks) || de != 0:
					t.Fatalf("%d decode steps' input rows served and %d declined, want %d served", ne, de, len(toks))
				case nc != len(toks)+1 || dc != 0 || nh != heads || dh != 0:
					t.Fatalf("%d collapses and %d heads served, %d and %d declined, want %d and %d: the prompt's head collapses whether or not the prompt is asked",
						nc, nh, dc, dh, len(toks)+1, heads)
				case tc.blocks && (nb != len(toks)*m.Cfg.NLayer || db != 0):
					t.Fatalf("%d block steps served and %d declined, want %d", nb, db, len(toks)*m.Cfg.NLayer)
				case st.StepDeclines[composer.DeclineEmbedStreams] != 1:
					t.Fatalf("the prompt's rows declined %d times for %q, want once",
						st.StepDeclines[composer.DeclineEmbedStreams], composer.DeclineName(int(composer.DeclineEmbedStreams)))
				case st.FormedAtStep != 0:
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
				t.Logf("%s: worst NMSE %.3g", dev, worst)
				return worst
			}
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					ran++
					worst := decode(t, dev, nil)
					if !(worst < bound) {
						t.Fatalf("worst logit NMSE %.3g against the host, bound %g", worst, bound)
					}
					for _, v := range streamRowsViolations(t, m) {
						var after func(*Instance)
						undo := func() {}
						if v.state != nil {
							after = v.state
						} else {
							undo = v.mut(m)
						}
						bad := decode(t, dev, after)
						undo()
						t.Logf("violation %q: worst NMSE %.3g", v.name, bad)
						if !(bad > 100*worst) || !(bad > 1e-4) {
							t.Errorf("%s: the card cannot see it -- worst NMSE %.3g against a clean %.3g", v.name, bad, worst)
						}
					}
				})
			}
		})
	}
	if ran == 0 {
		t.Fatal("no device ran")
	}
}

// streamViolation is a violation of the input rows or the collapse over
// several streams: the front end's graphs rewritten (mut), or the instance's
// device wrapped (state).
type streamViolation struct {
	name  string
	mut   func(m *Model) func()
	state func(*Instance)
}

// streamRowsViolations are m's input rows' streams past the first written
// empty, its per-layer inputs empty where it has them, a stream read twice
// where the collapse reads them, and the gather a token off.
func streamRowsViolations(t *testing.T, m *Model) []streamViolation {
	s := m.Cfg.Streams()
	zero := func(outs func(g *graph.Graph) []graph.ValueID) func(m *Model) func() {
		return func(m *Model) func() {
			return m.rewriteEmbed(t, func(g *graph.Graph) *graph.Graph {
				at := outs(g)
				return rewriteOps(t, g, func(_ int, o *graph.Op) bool {
					if len(o.Out) != 1 || !slices.Contains(at, o.Out[0]) || len(o.In) == 0 {
						return false
					}
					*o = graph.Op{Kind: graph.OpMap, In: o.In[:1], Out: o.Out, Attr: graph.Attrs{Fn: graph.MapMul, Const: 0}}
					return true
				})
			})
		}
	}
	vs := []streamViolation{
		{name: "the streams past the first start empty", mut: zero(func(g *graph.Graph) []graph.ValueID { return g.Out[1:s] })},
		{name: "a stream read twice where the collapse reads them", mut: func(m *Model) func() {
			return m.rewriteCollapse(t, func(g *graph.Graph) *graph.Graph {
				return rewriteOps(t, g, func(_ int, o *graph.Op) bool {
					hit := false
					o.In = slices.Clone(o.In)
					for i, v := range o.In {
						if v == g.In[2] {
							o.In[i], hit = g.In[1], true
						}
					}
					return hit
				})
			})
		}},
		{name: "the gather a token off", state: shiftTokens(1, m.Cfg.NVocab)},
	}
	if m.Cfg.PLEDim != 0 {
		vs = append(vs, streamViolation{name: "the per-layer inputs empty",
			mut: zero(func(g *graph.Graph) []graph.ValueID { return g.Out[s:] })})
	}
	return vs
}

// rewriteEmbed is the front end's set with the input rows' graph rewritten.
func (m *Model) rewriteEmbed(t *testing.T, rewrite func(*graph.Graph) *graph.Graph) func() {
	t.Helper()
	s := m.frontGraphs()
	if s.embed == nil {
		t.Fatal("the model has no input rows to rewrite")
	}
	set := *s
	set.embed = rewrite(s.embed)
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
