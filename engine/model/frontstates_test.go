package model

import (
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
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
// model with unstated blocks cannot form as the whole model (jit.Front).
func formBlocks(t *testing.T, m *Model, blocks []int) {
	t.Helper()
	f := m.Composing()
	step := jit.Step{Rows: 1, Page: 64, Keys: 16, ActWindow: 32}
	for _, d := range []jit.Device{{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32}, {Kind: jit.DeviceCPU, Slots: 6, Lanes: 8}} {
		reg := jit.Default()
		for _, li := range blocks {
			run := []jit.PartPlan{{Part: jit.Part{Kind: jit.PartBlock, Block: li, Graph: f.BlockGraph(li)}, Device: d}}
			ks, err := reg.Kernels(f, run, 0, step)
			if err != nil {
				t.Fatalf("%v: block %d: %v", d.Kind, li, err)
			}
			if d.Kind == jit.DeviceGPU {
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
// projection -- to the ops that say so, and forms them, while its recurrent
// blocks stay refused by their kind and a mixture's attention blocks by theirs.
// A model read as ungated is refused, not given a q of the wrong width.
func TestFrontStatesTheAttentionOutputGate(t *testing.T) {
	for _, c := range []struct {
		name string
		opt  hyOpt
		attn Uncovered
	}{{"dense", hyOpt{}, Covered}, {"tiled", hyOpt{tiled: true}, Covered}, {"mixture", hyOpt{moe: true}, UncoveredMixture}} {
		t.Run(c.name, func(t *testing.T) {
			m := hybridModelOpt(t, c.opt)
			var stated []int
			attn, mixers := 0, 0
			for li := range m.Cfg.NLayer {
				if m.Cfg.LayerKind(li) != jlm.LayerFullAttn {
					mixers++
					if why := m.Uncovered(li); why != UncoveredKind {
						t.Fatalf("block %d, a recurrent mixer, is refused by %q, want %q", li, why, UncoveredKind)
					}
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
	halves := opsOf(g, graph.OpSlice)
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
	for _, o := range opsOf(g, graph.OpActMul) {
		if o.Attr.Act == graph.ActSigmoid {
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
// activation is stated by four stored numbers a block to maps of the up
// projection that read each number as a scalar of the stored vector, and no
// activation op; it forms. A model whose container lacks the numbers is
// refused, not given an activation of nothing.
func TestFrontStatesTheActivationFromTheBlocksNumbers(t *testing.T) {
	m := frontModel(t, "synth-apertus.gguf")
	for li := range m.Cfg.NLayer {
		if why := m.Uncovered(li); why != Covered {
			t.Fatalf("block %d is refused by %q (%v)", li, why, m.UncoveredErr(li))
		}
		g := m.Composing().BlockGraph(li)
		if n := len(opsOf(g, graph.OpAct)) + len(opsOf(g, graph.OpActMul)); n != 0 {
			t.Fatalf("block %d: %d activation ops, want the maps of the numbers", li, n)
		}
		offsets := map[int]int{}
		expm1 := 0
		for _, o := range opsOf(g, graph.OpMap) {
			if o.Attr.Fn == graph.MapExpm1 {
				expm1++
			}
			for _, w := range o.Weight {
				if w.Role != jlm.RoleXIELU || w.K != 4 || o.Attr.Width != 1 {
					t.Fatalf("block %d: a map binds %s of %d with a window of %d, want one scalar of the four", li, w.Role, w.K, o.Attr.Width)
				}
				offsets[o.Attr.Offset]++
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
	if why != UncoveredUnverified || err == nil || !strings.Contains(err.Error(), jlm.RoleXIELU.String()) {
		t.Fatalf("a block without its stored numbers is refused by %q (%v), want them named", why, err)
	}
}

// TestFrontStatesTheNormsInsideASublayer holds the two norms a model applies
// inside a sublayer -- the attention's output before the output projection and
// the product of the gate and up projections before down -- to their place in
// the graph, and forms them. The model quantizes its projection inputs per
// token, which the front end refuses by name, so the statement is read with
// that taken from the config; a norm of the wrong width is refused.
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
		if p := producerOf(t, g, sub.In[0]); p.Kind != graph.OpActMul {
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

// TestFrontNamesWhatTheQwen35FixtureStillLacks holds a converted hybrid: its
// recurrent mixers are refused by their kind, and its attention blocks, whose
// output gate the front end states, by the next thing they carry that it does
// not, a rotary over several position axes.
func TestFrontNamesWhatTheQwen35FixtureStillLacks(t *testing.T) {
	m := frontModel(t, "synth-qwen35-hybrid-mtp.gguf")
	if !m.Cfg.AttnOutGate || m.Cfg.RopeSections == [4]int{} {
		t.Fatal("the fixture lost its output gate or its rotary sections: this gate is about both")
	}
	attn, mixers := 0, 0
	for li := range m.Cfg.NLayer {
		want := UncoveredKind
		if m.Cfg.LayerKind(li) == jlm.LayerFullAttn {
			want = UncoveredRopeAxes
			attn++
		} else {
			mixers++
		}
		if why := m.Uncovered(li); why != want {
			t.Fatalf("block %d is refused by %q (%v), want %q", li, why, m.UncoveredErr(li), want)
		}
	}
	if attn == 0 || mixers == 0 {
		t.Fatalf("the fixture has %d attention blocks and %d recurrent mixers: it is no hybrid", attn, mixers)
	}
}
