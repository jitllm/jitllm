package model

import (
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// deltaRewrite is a violation of a linear block's statement: a rewrite of
// its graph that drops one feature, false where the graph has none to drop.
type deltaRewrite struct {
	name string
	fn   func(g *graph.Graph) bool
}

// producer is the op of g that writes v, -1 for none.
func producer(g *graph.Graph, v graph.ValueID) int {
	for i, o := range g.Ops {
		if slices.Contains(o.Out, v) {
			return i
		}
	}
	return -1
}

// rewireIn points input j of op i at v.
func rewireIn(g *graph.Graph, i, j int, v graph.ValueID) {
	g.Ops[i].In = slices.Clone(g.Ops[i].In)
	g.Ops[i].In[j] = v
}

// deltaRewrites are the delta rule's features, each dropped: the decay (the
// state not decayed: beta read in its place), the output gate, the key's
// unit norm and the causal convolution.
func deltaRewrites() []deltaRewrite {
	return []deltaRewrite{
		{"decay", func(g *graph.Graph) bool {
			for i, o := range g.Ops {
				if o.Kind == graph.OpDeltaRule && g.Values[o.In[3]].Width == g.Values[o.In[4]].Width {
					rewireIn(g, i, 3, o.In[4])
					return true
				}
			}
			return false
		}},
		{"gate", func(g *graph.Graph) bool {
			for i, o := range g.Ops {
				if o.Kind == graph.OpMatVec && o.Weight[0].Role == jlm.RoleSSMOut {
					if p := producer(g, o.In[0]); p >= 0 && g.Ops[p].Kind == graph.OpExpr {
						rewireIn(g, i, 0, g.Ops[p].In[1])
						return true
					}
				}
			}
			return false
		}},
		{"key norm", func(g *graph.Graph) bool {
			for _, o := range g.Ops {
				if o.Kind != graph.OpDeltaRule {
					continue
				}
				sc := producer(g, o.In[1])
				if sc < 0 {
					return false
				}
				if n := producer(g, g.Ops[sc].In[0]); n >= 0 && g.Ops[n].Kind == graph.OpHeadNorm {
					rewireIn(g, sc, 0, g.Ops[n].In[0])
					return true
				}
			}
			return false
		}},
		{"convolution", func(g *graph.Graph) bool {
			for _, o := range g.Ops {
				if o.Kind != graph.OpStateConv {
					continue
				}
				for i, r := range g.Ops {
					if j := slices.Index(r.In, o.Out[0]); j >= 0 {
						rewireIn(g, i, j, o.In[0])
						return true
					}
				}
			}
			return false
		}},
	}
}

// rewriteMixers states each linear block of m as rw rewrites it until the
// returned undo, and says how many it rewrote.
func rewriteMixers(m *Model, rw deltaRewrite) (int, func()) {
	f := m.frontGraphs()
	old := slices.Clone(f.blocks)
	n := 0
	for li, g := range f.blocks {
		if g == nil || m.Cfg.LayerKind(li) != jlm.LayerLinearAttn {
			continue
		}
		c := *g
		c.Ops = slices.Clone(g.Ops)
		if rw.fn(&c) {
			f.blocks[li] = &c
			n++
		}
	}
	return n, func() { copy(f.blocks, old) }
}

// linearLogits is m's teacher-forced decode of toks with its attention
// blocks stated by no graph, so main's kernels run them and the composed
// CPU its linear blocks alone: every linear block step served, every
// attention block step declined as having no graph, nothing else.
func linearLogits(t *testing.T, m *Model, toks []int32, f cpuFault) [][]float32 {
	t.Helper()
	return linearRun(t, m, nil, toks, f)
}

// linearRun is linearLogits after a prompt: the prompt's last logits, then
// each step's.
func linearRun(t *testing.T, m *Model, prompt, toks []int32, f cpuFault) [][]float32 {
	t.Helper()
	fs := m.frontGraphs()
	old := slices.Clone(fs.blocks)
	defer copy(fs.blocks, old)
	lin := 0
	for li := range fs.blocks {
		if m.Cfg.LayerKind(li) == jlm.LayerLinearAttn {
			lin++
		} else {
			fs.blocks[li] = nil
		}
	}
	s := m.NewInstance(len(prompt) + len(toks) + 1)
	defer s.Close()
	s.cpu.fault = f
	var out [][]float32
	if len(prompt) > 0 {
		lg, err := s.Prefill(prompt)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	for _, tk := range toks {
		lg, err := s.Forward(tk)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	served, declined := s.CPUBlockSteps()
	for i, n := range declined {
		if n > 0 && composer.Decline(i) != composer.DeclineNoGraph {
			t.Fatalf("%d block steps declined: %s (%v)", n, composer.DeclineName(i), s.CPUFormErr())
		}
	}
	// A prompt's chunk is one block step of each block, its rows the
	// linear block's one at a time.
	want := len(toks) * lin
	if len(prompt) > 0 {
		want += lin
	}
	if served < want || lin == 0 {
		t.Fatalf("%d linear block steps served, want at least %d (%d linear blocks)", served, want, lin)
	}
	return out
}

// TestComposedCPUDeltaFeaturesAreLoadBearing holds a hybrid's decode on the
// composed CPU, its linear blocks the composer's delta rule, to the CPU's
// float reference within llama.cpp's spread from it (TestComposedCPUBlocksServe's
// band), and drops each of the delta rule's
// features from the statement in turn -- the decay, the output gate, the
// key's norm, the convolution -- and binds each block to another block's
// state (cpuFaultRec): each must leave the band.
func TestComposedCPUDeltaFeaturesAreLoadBearing(t *testing.T) {
	const steps, page = 40, 32
	path := jlmOf(t, testmodels.Path("qwen35/Qwen3.5-0.8B-Q4_K_M.gguf"))
	open := func(opts ...Option) *Model {
		t.Helper()
		m, err := Open(path, append([]Option{noTune()}, opts...)...)
		if err != nil {
			t.Fatal(err)
		}
		m.setKVPageForTest(page)
		return m
	}
	src := fixtureSource(testmodels.Path("qwen35/Qwen3.5-0.8B-Q4_K_M.gguf"))
	m := open()
	defer m.Close()
	toks := cpuTokens(t, m, steps)
	want, band := cpuReference(t, path, src, page, toks)
	got := linearLogits(t, m, toks, cpuFaultNone)
	med, worst, widest := cpuBandTo(t, want, got, band)
	t.Logf("the composed delta rule: %.3f of the band at the median, worst NMSE %.3g where the band's widest is %.3g", med, worst, widest)
	if !(med <= bandRatio) || !(worst <= bandWorst*widest) {
		t.Fatalf("the composed delta rule leaves the band: median ratio %.3f (bound %.1f), worst %.3g (bound %.1f x %.3g)",
			med, bandRatio, worst, bandWorst, widest)
	}
	check := func(what string, bad [][]float32) {
		bm, bw, _ := cpuBandTo(t, want, bad, band)
		t.Logf("%s: median ratio %.3f, worst %.3g", what, bm, bw)
		if bm <= bandRatio && bw <= bandWorst*widest {
			t.Errorf("%s stays in the band (median ratio %.3f, worst %.3g): the gate cannot see it", what, bm, bw)
		}
	}
	for _, rw := range deltaRewrites() {
		n, undo := rewriteMixers(m, rw)
		if n == 0 {
			undo()
			t.Fatalf("no linear block states a %s to drop: the violation selects nothing", rw.name)
		}
		bad := linearLogits(t, m, toks, cpuFaultNone)
		undo()
		check(fmt.Sprintf("the delta rule without its %s (%d blocks)", rw.name, n), bad)
	}
	check("every block on the first linear block's state", linearLogits(t, m, toks, cpuFaultRec))

	// A prompt of several chunks, the recurrence carried across each
	// boundary, and the decode after it.
	text := strings.Repeat("The river rose every spring until the bridges were islands, "+
		"and the clerk counted barrels of salt on the upper floor. ", 60)
	ids := m.Vocab.Encode(text, true)[:700]
	run := func(m *Model) [][]float32 {
		s := m.NewInstance(len(ids) + len(toks) + 1)
		defer s.Close()
		lg, err := s.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		out := [][]float32{append([]float32(nil), lg...)}
		for _, tk := range toks[:4] {
			if lg, err = s.Forward(tk); err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
		}
		return out
	}
	fm := cpuFloat(t, path, page)
	defer fm.Close()
	pw := run(fm)
	pb := lcppBand(t, src, fm.Cfg.NLayer, pw, lcppSeq{ids, toks[:4], nil})
	pg := linearRun(t, m, ids, toks[:4], cpuFaultNone)
	pm, pwst, pwide := cpuBandTo(t, pw, pg, pb)
	t.Logf("a prompt of %d and its decode: %.3f of the band at the median, worst NMSE %.3g where the band's widest is %.3g", len(ids), pm, pwst, pwide)
	if !(pm <= bandRatio) || !(pwst <= bandWorst*pwide) {
		t.Fatalf("the composed delta rule's prompt leaves the band: median ratio %.3f, worst %.3g (bound %.1f x %.3g)", pm, pwst, bandWorst, pwide)
	}
	bad := linearRun(t, m, ids, toks[:4], cpuFaultChunk)
	bm, bw, _ := cpuBandTo(t, pw, bad, pb)
	t.Logf("the recurrence dropped at each chunk boundary: median ratio %.3f, worst %.3g", bm, bw)
	if bm <= bandRatio && bw <= bandWorst*pwide {
		t.Errorf("the recurrence dropped at each chunk boundary stays in the band (median ratio %.3f, worst %.3g): the gate cannot see it", bm, bw)
	}
}
