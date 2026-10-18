package graph

import (
	"math"

	"github.com/jitllm/jitllm/format/jlm"
)

// CrossBlock is a decision head's block (Clef's joint schema head): rows that
// read a memory. A routing block's rows (the options) cross-attend to the
// memory normed by its own LayerNorm; a joint block's (the questions) first
// self-attend, then cross-attend to the memory as it is. Every sublayer is
// pre-norm with a biased LayerNorm and adds into the rows; the MLP is an
// ungated erf GELU, every projection biased.
type CrossBlock struct {
	Index              int // the head block's index
	Width, Heads, NFFN int
	Eps                float64
	Routing            bool
}

// Facts is what the verifier holds the block's graph to: a Dense stating its
// widths, its epsilon, its bidirectional attention and the cross attention.
func (b *CrossBlock) Facts(inv map[jlm.Role]Shape) Facts {
	return Facts{Config: Dense{Block: b.Index, NEmbd: b.Width, NHead: b.Heads, NKVHead: b.Heads, HeadDim: b.Width / b.Heads,
		NFFN: b.NFFN, Eps: b.Eps, Act: ActGELUErf, LayerNorm: true, NonCausal: true, NoRope: true, Cross: true,
		Scale: invSqrt(b.Width / b.Heads)}, Inventory: inv}
}

// Build is the block's graph: In[0] the rows, In[1] the memory.
func (b *CrossBlock) Build() *Graph {
	g := &Graph{Version: Version, Block: b.Index}
	w := b.Width
	x, mem := g.value("rows", w), g.value("memory", w)
	g.In = []ValueID{x, mem}
	vec := func(r jlm.Role) Binding { return Binding{Role: r, K: w} }
	ln := func(name string, in ValueID, wr, br jlm.Role) ValueID {
		out := g.value(name, w)
		g.op(Op{Kind: OpLayerNorm, In: []ValueID{in}, Out: []ValueID{out}, Weight: []Binding{vec(wr), vec(br)}, Attr: Attrs{Eps: b.Eps}})
		return out
	}
	mv := func(name string, in ValueID, wr, br jlm.Role, k, rows int) ValueID {
		out := g.value(name, rows)
		g.op(Op{Kind: OpMatVec, In: []ValueID{in}, Out: []ValueID{out},
			Weight: []Binding{{Role: wr, K: k, Rows: rows}, {Role: br, K: rows}}})
		return out
	}
	add := func(name string, a, y ValueID) ValueID {
		out := g.value(name, w)
		g.op(Op{Kind: OpAdd, In: []ValueID{a, y}, Out: []ValueID{out}})
		return out
	}
	hd := w / b.Heads
	if !b.Routing {
		h := ln("self_in", x, jlm.RoleDecSelfNorm, jlm.RoleDecSelfNormBias)
		q := mv("self_q", h, jlm.RoleDecSelfQ, jlm.RoleDecSelfQBias, w, w)
		k := mv("self_k", h, jlm.RoleDecSelfK, jlm.RoleDecSelfKBias, w, w)
		v := mv("self_v", h, jlm.RoleDecSelfV, jlm.RoleDecSelfVBias, w, w)
		g.op(Op{Kind: OpKVAppend, In: []ValueID{k, v}, Reads: []StateRef{{"kv", 0}}, Writes: []StateRef{{"kv", 1}}})
		a := g.value("self_attn", w)
		g.op(Op{Kind: OpAttend, In: []ValueID{q}, Out: []ValueID{a}, Reads: []StateRef{{"kv", 1}},
			Attr: Attrs{HeadDim: hd, NHead: b.Heads, NKVHead: b.Heads, Scale: invSqrt(hd), Bidir: true}})
		x = add("x_self", x, mv("self_out", a, jlm.RoleDecSelfOut, jlm.RoleDecSelfOutBias, w, w))
	} else {
		mem = ln("memory_normed", mem, jlm.RoleDecCrossKVNorm, jlm.RoleDecCrossKVNormBias)
	}
	h := ln("cross_in", x, jlm.RoleDecCrossNorm, jlm.RoleDecCrossNormBias)
	o := g.value("cross_out", w)
	cw := []Binding{}
	for _, r := range [][2]jlm.Role{{jlm.RoleDecCrossQ, jlm.RoleDecCrossQBias}, {jlm.RoleDecCrossK, jlm.RoleDecCrossKBias},
		{jlm.RoleDecCrossV, jlm.RoleDecCrossVBias}, {jlm.RoleDecCrossOut, jlm.RoleDecCrossOutBias}} {
		cw = append(cw, Binding{Role: r[0], K: w, Rows: w}, vec(r[1]))
	}
	g.op(Op{Kind: OpCrossAttend, In: []ValueID{h, mem}, Out: []ValueID{o}, Weight: cw, Attr: Attrs{NHead: b.Heads, HeadDim: hd}})
	x = add("x_cross", x, o)
	h = ln("ffn_in", x, jlm.RoleDecFFNNorm, jlm.RoleDecFFNNormBias)
	up := mv("up", h, jlm.RoleDecFFNUp, jlm.RoleDecFFNUpBias, w, b.NFFN)
	act := g.value("act", b.NFFN)
	g.op(Op{Kind: OpAct, In: []ValueID{up}, Out: []ValueID{act}, Attr: Attrs{Act: ActGELUErf}})
	x = add("x_out", x, mv("down", act, jlm.RoleDecFFNDown, jlm.RoleDecFFNDownBias, b.NFFN, w))
	g.Out = []ValueID{x}
	return g
}

// invSqrt is 1/sqrt(n), an attention's scale over heads of n.
func invSqrt(n int) float64 { return 1 / math.Sqrt(float64(n)) }
