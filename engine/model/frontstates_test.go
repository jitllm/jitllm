package model

import (
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// The block features the front end states as ops -- the gate on the
// attention's output, an activation from a block's stored numbers, the norms
// inside a sublayer -- each held to the graph's shape, formed for a CPU and a
// GPU device, and refused by name where the container and the statement
// disagree.

// opsOf is the ops of g of kind k, in order.
func opsOf(g *graph.Graph, k graph.OpKind) []graph.Op {
	var out []graph.Op
	for _, o := range g.Ops {
		if o.Kind == k {
			out = append(out, o)
		}
	}
	return out
}

// producerOf is the op of g that writes v.
func producerOf(t *testing.T, g *graph.Graph, v graph.ValueID) graph.Op {
	t.Helper()
	for _, o := range g.Ops {
		if slices.Contains(o.Out, v) {
			return o
		}
	}
	t.Fatalf("no op writes value %d", v)
	return graph.Op{}
}

// readerOf is the first op of g whose first weight is role's.
func readerOf(t *testing.T, g *graph.Graph, role jlm.Role) graph.Op {
	t.Helper()
	for _, o := range g.Ops {
		if len(o.Weight) > 0 && o.Weight[0].Role == role {
			return o
		}
	}
	t.Fatalf("no op binds %s", role)
	return graph.Op{}
}

// formBlocks has the composer form each of blocks of m the front end states,
// on its own, for a CPU and a GPU device, and lowers the GPU's: a block of a
// model with unstated blocks cannot form as the whole model (composer.Front).
func formBlocks(t *testing.T, m *Model, blocks []int) {
	t.Helper()
	f := m.Composing()
	step := composer.Step{Rows: 1, Page: 64, Keys: 16, ActWindow: 32}
	for _, d := range []composer.Device{{Kind: composer.DeviceGPU, Slots: 80 * 2048, Lanes: 32}, composedCPUDevice(6)} {
		reg := composer.Default()
		for _, li := range blocks {
			run := []composer.PartPlan{{Part: composer.Part{Kind: composer.PartBlock, Block: li, Graph: f.BlockGraph(li)}, Device: d}}
			ks, err := reg.Kernels(f, run, 0, step)
			if err != nil {
				t.Fatalf("%v: block %d: %v", d.Kind, li, err)
			}
			if d.Kind == composer.DeviceGPU {
				for _, k := range ks {
					if _, err := reg.Lowerings[d.Kind].Lower(k, d); err != nil {
						t.Fatalf("lowering %s of block %d: %v", k.Name, li, err)
					}
				}
			}
			t.Logf("%v: block %d formed into %d kernels", d.Kind, li, len(ks))
		}
	}
}

// TestFrontStatesTheAttentionOutputGate holds a hybrid's attention blocks --
// a q projection of twice the width, each head's query then its gate, and the
// attention's output multiplied by the sigmoid of the gate before the output
// projection -- to the ops that say so, and forms them, with its recurrent
// blocks stated as delta rules and formed; a mixture's blocks are stated too.
// A model read as ungated is refused, not given a q of the wrong width.
func TestFrontStatesTheAttentionOutputGate(t *testing.T) {
	for _, c := range []struct {
		name string
		opt  hyOpt
		attn Uncovered
	}{{"dense", hyOpt{}, Covered}, {"tiled", hyOpt{tiled: true}, Covered}, {"mixture", hyOpt{moe: true}, Covered}} {
		t.Run(c.name, func(t *testing.T) {
			m := hybridModelOpt(t, c.opt)
			var stated []int
			attn, mixers := 0, 0
			for li := range m.Cfg.NLayer {
				if m.Cfg.LayerKind(li) != jlm.LayerFullAttn {
					mixers++
					checkDeltaMixer(t, m, li)
					stated = append(stated, li)
					continue
				}
				attn++
				if why := m.Uncovered(li); why != c.attn {
					t.Fatalf("attention block %d is refused by %q (%v), want %q", li, why, m.UncoveredErr(li), c.attn)
				}
				if c.attn == Covered {
					checkOutputGate(t, m, li)
					stated = append(stated, li)
				}
			}
			if attn == 0 || mixers == 0 {
				t.Fatalf("the fixture has %d attention blocks and %d recurrent mixers: it is no hybrid", attn, mixers)
			}
			formBlocks(t, m, stated)
		})
	}
	t.Run("read as ungated", func(t *testing.T) {
		m := hybridModelOpt(t, hyOpt{})
		m.Cfg.AttnOutGate = false
		n := 0
		for li := range m.Cfg.NLayer {
			if m.Cfg.LayerKind(li) != jlm.LayerFullAttn {
				continue
			}
			n++
			why, err := m.Uncovered(li), m.UncoveredErr(li)
			if why != UncoveredUnverified || err == nil || !strings.Contains(err.Error(), jlm.RoleAttnQ.String()) {
				t.Fatalf("block %d with a stored double-width q read as ungated is refused by %q (%v), want the q projection's shape named", li, why, err)
			}
		}
		if n == 0 {
			t.Fatal("the fixture has no attention block")
		}
	})
}

// TestFrontStatesTheDeltaRules holds the linear blocks of a mixture hybrid
// and of Kimi Delta Attention -- a decay per key channel through a
// low-rank pair, a low-rank output gate, a sigmoid gate -- to a delta
// rule's statement, and forms them on a CPU and a GPU device.
func TestFrontStatesTheDeltaRules(t *testing.T) {
	for _, file := range []string{"synth-qwen35moe-hybrid-mtp.gguf", "synth-kimilinear"} {
		t.Run(file, func(t *testing.T) {
			m := frontModel(t, file)
			var stated []int
			for li := range m.Cfg.NLayer {
				if m.Cfg.LayerKind(li) != jlm.LayerLinearAttn {
					continue
				}
				checkDeltaMixer(t, m, li)
				stated = append(stated, li)
			}
			if len(stated) == 0 {
				t.Fatal("the model states no delta rule: the gate selects nothing")
			}
			if m.Cfg.ChanDecay() {
				rule := opsOf(m.Composing().BlockGraph(stated[0]), graph.OpDeltaRule)[0]
				if w := m.Composing().BlockGraph(stated[0]).Values[rule.In[3]].Width; w != rule.Attr.NHead*rule.Attr.HeadDim {
					t.Fatalf("a decay of %d, where a channel's decay is %d", w, rule.Attr.NHead*rule.Attr.HeadDim)
				}
			}
			formBlocks(t, m, stated[:min(2, len(stated))])
		})
	}
}

// checkDeltaMixer holds recurrent block li to a delta rule's statement: one
// state convolution over the mixed projection and one delta rule, each
// stepping its own state, at the container's geometry and pairing, and the
// mixer projected by the stored output projection.
func checkDeltaMixer(t *testing.T, m *Model, li int) {
	t.Helper()
	if why := m.Uncovered(li); why != Covered {
		t.Fatalf("block %d, a delta rule, is refused by %q (%v)", li, why, m.UncoveredErr(li))
	}
	g := m.Composing().BlockGraph(li)
	dg := m.Cfg.delta()
	convs, rules := opsOf(g, graph.OpStateConv), opsOf(g, graph.OpDeltaRule)
	if len(convs) != 1 || len(rules) != 1 {
		t.Fatalf("block %d states %d state convolutions and %d delta rules, want one each", li, len(convs), len(rules))
	}
	if c := convs[0]; c.In[0] != readerOf(t, g, jlm.RoleAttnQKV).Out[0] || c.Attr.Taps != dg.conv ||
		len(c.Writes) != 1 || c.Writes[0].State != graph.StateRecConv {
		t.Fatalf("block %d: the convolution %+v is not the mixed projection's over its own window of %d", li, c, dg.conv)
	}
	a := rules[0].Attr
	if a.NHead != dg.vHeads || a.NKVHead != dg.kHeads || a.HeadDim != dg.kDim || a.Tiled != dg.tiled ||
		len(rules[0].Writes) != 1 || rules[0].Writes[0].State != graph.StateRecState {
		t.Fatalf("block %d: a delta rule of %+v over %+v, want the container's geometry", li, a, dg)
	}
	readerOf(t, g, jlm.RoleSSMOut)
}

// checkOutputGate holds block li's graph to the gate's statement.
func checkOutputGate(t *testing.T, m *Model, li int) {
	t.Helper()
	g := m.Composing().BlockGraph(li)
	hd := m.Cfg.HeadDimAt(li)
	qdim := m.Cfg.NHead * hd
	q := readerOf(t, g, jlm.RoleAttnQ)
	if q.Weight[0].Rows != 2*qdim {
		t.Fatalf("block %d: the q projection is %d rows, want the double width %d", li, q.Weight[0].Rows, 2*qdim)
	}
	var halves []graph.Op
	for _, s := range opsOf(g, graph.OpSlice) {
		if s.In[0] == q.Out[0] {
			halves = append(halves, s)
		}
	}
	if len(halves) != 2 {
		t.Fatalf("block %d: %d slices of the projection, want its query and its gate", li, len(halves))
	}
	var gate graph.Op
	var offsets []int
	for _, s := range halves {
		a := s.Attr
		if s.In[0] != q.Out[0] || a.HeadDim != 2*hd || a.Width != hd {
			t.Fatalf("block %d: a slice of %d from %d in heads of %d is not a half of the q projection's heads", li, a.Width, a.Offset, a.HeadDim)
		}
		offsets = append(offsets, a.Offset)
		if a.Offset == hd {
			gate = s
		}
	}
	slices.Sort(offsets)
	if !slices.Equal(offsets, []int{0, hd}) {
		t.Fatalf("block %d: the halves are taken from %v, want the query at 0 and the gate at %d", li, offsets, hd)
	}
	var gated []graph.Op
	for _, o := range opsOf(g, graph.OpExpr) {
		if isOutGate(o) && slices.ContainsFunc(halves, func(s graph.Op) bool { return s.Out[0] == o.In[0] }) {
			gated = append(gated, o)
		}
	}
	if len(gated) != 1 || gated[0].In[0] != gate.Out[0] {
		t.Fatalf("block %d: %d sigmoid gates, the one wanted on the gate half", li, len(gated))
	}
	if att := producerOf(t, g, gated[0].In[1]); att.Kind != graph.OpAttend {
		t.Fatalf("block %d: the gate multiplies a %s, want the attention's output", li, att.Kind)
	}
	if wo := readerOf(t, g, jlm.RoleAttnOut); wo.In[0] != gated[0].Out[0] {
		t.Fatalf("block %d: the output projection reads %d, want the gated output", li, wo.In[0])
	}
}

// TestFrontStatesTheActivationFromTheBlocksNumbers holds an ungated MLP whose
// activation the container states over four stored numbers a block to one
// expression of the up projection that reads each number as a scalar of the
// stored vector; it forms. A model whose container lacks the numbers is
// refused, naming them, not given an activation of nothing.
func TestFrontStatesTheActivationFromTheBlocksNumbers(t *testing.T) {
	m := frontModel(t, "synth-apertus.gguf")
	for li := range m.Cfg.NLayer {
		if why := m.Uncovered(li); why != Covered {
			t.Fatalf("block %d is refused by %q (%v)", li, why, m.UncoveredErr(li))
		}
		g := m.Composing().BlockGraph(li)
		exprs := opsOf(g, graph.OpExpr)
		if len(exprs) != 1 {
			t.Fatalf("block %d: %d expressions, want the activation's one", li, len(exprs))
		}
		o := exprs[0]
		if len(o.Weight) != 1 || o.Weight[0].Role != jlm.RoleXIELU || o.Weight[0].K != 4 {
			t.Fatalf("block %d: the activation binds %v, want the four numbers of %s", li, o.Weight, jlm.RoleXIELU)
		}
		offsets := map[int]int{}
		expm1 := 0
		for _, tm := range o.Attr.Expr {
			if tm.Fn == graph.MapExpm1 {
				expm1++
			}
			for _, a := range tm.Args {
				if a.Kind == graph.ArgWeight {
					offsets[a.Elem]++
				}
			}
		}
		if len(offsets) != 4 || expm1 != 1 {
			t.Fatalf("block %d: the numbers are read at %v and expm1 is taken %d times, want each of the four numbers and one expm1", li, offsets, expm1)
		}
	}
	formBlocks(t, m, []int{0, m.Cfg.NLayer - 1})
	undo := dropStored(m, jlm.RoleXIELU)
	defer undo()
	why, err := m.Uncovered(0), m.UncoveredErr(0)
	if why != UncoveredAct || err == nil || !strings.Contains(err.Error(), jlm.RoleXIELU.String()) {
		t.Fatalf("a block without its stored numbers is refused by %q (%v), want them named", why, err)
	}
}

// TestFrontStatesTheNormsInsideASublayer holds the two norms a model applies
// inside a sublayer -- the attention's output before the output projection and
// the product of the gate and up projections before down -- to their place in
// the graph, and forms them; a norm of the wrong width is refused.
//
// No container reaches this statement yet. A model stores these norms only
// where it quantizes its projection inputs per token (model.go refuses one
// without the other), and the front end refuses that first, by name
// (UncoveredTokenAct), so the graph is read here with the flag taken from the
// config. Until a later change states the quantization, no block of any model
// is served through these norms, and no device gate runs them.
func TestFrontStatesTheNormsInsideASublayer(t *testing.T) {
	m := frontModel(t, "synth-bitnet-f16.gguf")
	if !m.Cfg.TokenAct {
		t.Fatal("the fixture does not quantize per token: this gate is about a model that does")
	}
	m.Cfg.TokenAct = false
	var stated []int
	for li := range m.Cfg.NLayer {
		if why := m.Uncovered(li); why != Covered {
			t.Fatalf("block %d is refused by %q (%v)", li, why, m.UncoveredErr(li))
		}
		g := m.Composing().BlockGraph(li)
		sub := readerOf(t, g, jlm.RoleAttnSubNorm)
		if p := producerOf(t, g, sub.In[0]); p.Kind != graph.OpAttend {
			t.Fatalf("block %d: the attention's inner norm reads a %s, want the attention's output", li, p.Kind)
		}
		if wo := readerOf(t, g, jlm.RoleAttnOut); wo.In[0] != sub.Out[0] {
			t.Fatalf("block %d: the output projection reads %d, want the inner norm's output", li, wo.In[0])
		}
		sub = readerOf(t, g, jlm.RoleFFNSubNorm)
		if p := producerOf(t, g, sub.In[0]); p.Kind != graph.OpExpr || len(p.In) != 2 {
			t.Fatalf("block %d: the FFN's inner norm reads a %s, want the gated product", li, p.Kind)
		}
		if down := readerOf(t, g, jlm.RoleFFNDown); down.In[0] != sub.Out[0] {
			t.Fatalf("block %d: the down projection reads %d, want the inner norm's output", li, down.In[0])
		}
		stated = append(stated, li)
	}
	formBlocks(t, m, stated[:min(2, len(stated))])
	undo := m.restate(func(inv map[jlm.Role]graph.Shape) {
		inv[jlm.RoleAttnSubNorm] = graph.Shape{K: inv[jlm.RoleAttnSubNorm].K - 1}
	})
	defer undo()
	why, err := m.Uncovered(0), m.UncoveredErr(0)
	if why != UncoveredUnverified || err == nil || !strings.Contains(err.Error(), jlm.RoleAttnSubNorm.String()) {
		t.Fatalf("an inner norm stored at another width is refused by %q (%v), want it named", why, err)
	}
}

// TestFrontStatesTheQwen35FixturesAttention holds a converted hybrid: its
// recurrent mixers are refused by their kind, and its attention blocks -- an
// output gate and a rotary over several position axes, interleaved -- are
// stated, with the output gate's ops and every rotary carrying the sections
// the container states, and form.
func TestFrontStatesTheQwen35FixturesAttention(t *testing.T) {
	m := frontModel(t, "synth-qwen35-hybrid-mtp.gguf")
	if !m.Cfg.AttnOutGate || m.Cfg.RopeSections == [4]int{} {
		t.Fatal("the fixture lost its output gate or its rotary sections: this gate is about both")
	}
	var stated []int
	mixers := 0
	for li := range m.Cfg.NLayer {
		if m.Cfg.LayerKind(li) != jlm.LayerFullAttn {
			checkDeltaMixer(t, m, li)
			stated = append(stated, li)
			mixers++
			continue
		}
		if why := m.Uncovered(li); why != Covered {
			t.Fatalf("attention block %d is refused by %q (%v)", li, why, m.UncoveredErr(li))
		}
		var ropes, gates int
		for _, o := range m.Composing().BlockGraph(li).Ops {
			switch {
			case o.Kind == graph.OpRope:
				ropes++
				if o.Attr.Sections != m.Cfg.RopeSections || o.Attr.Interleaved != m.Cfg.RopeInterleaved {
					t.Fatalf("block %d: a rotary over sections %v (interleaved %v), where the container states %v (%v)",
						li, o.Attr.Sections, o.Attr.Interleaved, m.Cfg.RopeSections, m.Cfg.RopeInterleaved)
				}
			case isOutGate(o):
				gates++
			}
		}
		if ropes != 2 || gates != 1 {
			t.Fatalf("block %d states %d rotaries and %d output gates, want 2 and 1", li, ropes, gates)
		}
		stated = append(stated, li)
	}
	if len(stated) == 0 || mixers == 0 {
		t.Fatalf("the fixture has %d attention blocks and %d recurrent mixers: it is no hybrid", len(stated), mixers)
	}
	formBlocks(t, m, stated)
}
