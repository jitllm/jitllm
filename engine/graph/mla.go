package graph

import (
	"errors"
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
)

// MLA describes multi-head latent attention (DeepSeek-V2's): the block's
// attention sublayer when it caches one compressed latent row per position
// instead of per-head keys and values. Dense.MLA, where set, replaces q, k, v
// and the attention with it; the norms, the residual adds and the FFN (dense,
// or a mixture) are the block's as before.
//
// The query is RoleAttnQ over the normed row, or through a latent of
// QLoraRank (RoleAttnQA, an RMSNorm, RoleAttnQB); each head of HeadDim splits
// into HeadDim-NRot unrotated and NRot rotated elements. RoleAttnKVA turns the
// row into the KVLoraRank-wide latent, RMSNormed, and one NRot-wide rotary key
// every head shares. The graph states the reference form (OpAttendLatent):
// the up-projections RoleAttnKB and RoleAttnVB turn the cached latent into
// each head's key and value. NoRope is a block that turns neither part
// (Kimi-Linear's NoPE MLA).
type MLA struct {
	NHead, HeadDim, NRot int
	KVLoraRank           int
	QLoraRank            int
	HeadDimV             int
	LatentEps            float64
	NoRope               bool
	// Index, where set, is the block's indexer (OpIndex): the attention
	// reads only the positions it selects.
	Index *Index
}

// Index is an attention indexer's geometry: Heads heads of HeadDim, the
// TopK best positions kept.
type Index struct {
	Heads, HeadDim, TopK int
}

// bindings are the indexer's weights over a row of nEmbd and a query
// latent of qLora.
func (x *Index) bindings(nEmbd, qLora int) []Binding {
	return []Binding{{Role: jlm.RoleIdxQB, K: qLora, Rows: x.Heads * x.HeadDim}, {Role: jlm.RoleIdxK, K: nEmbd, Rows: x.HeadDim},
		{Role: jlm.RoleIdxProj, K: nEmbd, Rows: x.Heads}, {Role: jlm.RoleIdxKNorm, K: x.HeadDim}, {Role: jlm.RoleIdxKNormB, K: x.HeadDim}}
}

// emit writes the attention sublayer over the normed row h into g, from the
// block d, and returns the attention's row before the output projection
// (NHead values of HeadDimV).
func (a *MLA) emit(g *Graph, d *Dense, h ValueID) ValueID {
	vec := func(r jlm.Role, n int) Binding { return Binding{Role: r, K: n} }
	mv := func(name string, in ValueID, w jlm.Role, rows int) ValueID {
		out := g.value(name, rows)
		g.op(Op{Kind: OpMatVec, In: []ValueID{in}, Out: []ValueID{out},
			Weight: []Binding{{Role: w, K: g.Values[in].Width, Rows: rows}}})
		return out
	}
	norm := func(name string, in ValueID, w jlm.Role) ValueID {
		out := g.value(name, g.Values[in].Width)
		g.op(Op{Kind: OpNorm, In: []ValueID{in}, Out: []ValueID{out}, Weight: []Binding{vec(w, g.Values[in].Width)},
			Attr: Attrs{Eps: a.LatentEps}})
		return out
	}
	slice := func(name string, in ValueID, headDim, off, width int) ValueID {
		n := 1
		if headDim > 0 {
			n = g.Values[in].Width / headDim
		}
		out := g.value(name, n*width)
		g.op(Op{Kind: OpSlice, In: []ValueID{in}, Out: []ValueID{out}, Attr: Attrs{HeadDim: headDim, Offset: off, Width: width}})
		return out
	}
	rope := func(name string, in ValueID) ValueID {
		if a.NoRope {
			return in
		}
		out := g.value(name, g.Values[in].Width)
		g.op(Op{Kind: OpRope, In: []ValueID{in}, Out: []ValueID{out},
			Attr: Attrs{HeadDim: a.NRot, NRot: a.NRot, Base: d.Base, Layout: d.Layout, Freqs: d.RopeFreqs, Mscale: d.Mscale}})
		return out
	}
	qd, nope, lat := a.NHead*a.HeadDim, a.HeadDim-a.NRot, a.KVLoraRank
	var q, qa ValueID
	if a.QLoraRank != 0 {
		qa = norm("q_latent", mv("q_a", h, jlm.RoleAttnQA, a.QLoraRank), jlm.RoleAttnQANorm)
		q = mv("q", qa, jlm.RoleAttnQB, qd)
	} else {
		q = mv("q", h, jlm.RoleAttnQ, qd)
	}
	qn := slice("q_nope", q, a.HeadDim, 0, nope)
	qp := rope("q_rot", slice("q_pe", q, a.HeadDim, nope, a.NRot))
	kva := mv("kv_a", h, jlm.RoleAttnKVA, lat+a.NRot)
	kl := norm("kv_latent", slice("kv_a_latent", kva, 0, 0, lat), jlm.RoleAttnKVANorm)
	kp := rope("k_rot", slice("k_pe", kva, 0, lat, a.NRot))
	g.op(Op{Kind: OpLatentAppend, In: []ValueID{kl, kp}, Reads: []StateRef{{"kv", 0}}, Writes: []StateRef{{"kv", 1}}})
	in := []ValueID{qn, qp}
	if x := a.Index; x != nil {
		sel := g.value("idx_sel", x.TopK)
		g.op(Op{Kind: OpIndex, In: []ValueID{h, qa}, Out: []ValueID{sel}, Weight: x.bindings(g.Values[h].Width, a.QLoraRank),
			Reads: []StateRef{{"idx", 0}}, Writes: []StateRef{{"idx", 1}}, Attr: Attrs{NHead: x.Heads, HeadDim: x.HeadDim, TopK: x.TopK}})
		in = append(in, sel)
	}
	out := g.value("attn", a.NHead*a.HeadDimV)
	g.op(Op{Kind: OpAttendLatent, In: in, Out: []ValueID{out}, Reads: []StateRef{{"kv", 1}},
		Weight: []Binding{{Role: jlm.RoleAttnKB, K: nope, Rows: lat, Sheets: a.NHead},
			{Role: jlm.RoleAttnVB, K: lat, Rows: a.HeadDimV, Sheets: a.NHead}},
		Attr: Attrs{NHead: a.NHead, HeadDim: a.HeadDim, NRot: a.NRot, Scale: d.Scale}})
	return out
}

// verifyMLAOp holds an MLA op to the block's facts.
func verifyMLAOp(o Op, at string, c *Dense, width func(ValueID) int, ins, outs func(int) bool, bad func(string, ...any)) {
	a := c.MLA
	if a == nil && o.Kind != OpSlice {
		bad("%s: a latent attention op in a graph whose facts state no MLA", at)
		return
	}
	switch o.Kind {
	case OpSlice:
		// A slice is the op language's own, not MLA's alone (an output
		// gate's heads too): its shape is all it is held to.
		if !ins(1) || !outs(1) {
			return
		}
		at2, in := o.Attr, width(o.In[0])
		n, span := 1, in
		if at2.HeadDim > 0 {
			n, span = in/at2.HeadDim, at2.HeadDim
		}
		if at2.Width <= 0 || at2.Offset < 0 || at2.Offset+at2.Width > span || n*span != in || width(o.Out[0]) != n*at2.Width {
			bad("%s: %d from %d of heads of %d over a row of %d into %d", at, at2.Width, at2.Offset, at2.HeadDim, in, width(o.Out[0]))
		}
	case OpLatentAppend:
		if !ins(2) || !outs(0) {
			return
		}
		if len(o.Writes) != 1 || o.Writes[0].Name != "kv" {
			bad("%s: writes no kv version", at)
		}
		if width(o.In[0]) != a.KVLoraRank || width(o.In[1]) != a.NRot {
			bad("%s: a latent of %d and a rotary key of %d; the configuration states %d and %d",
				at, width(o.In[0]), width(o.In[1]), a.KVLoraRank, a.NRot)
		}
	case OpIndex:
		x := a.Index
		if x == nil || a.QLoraRank == 0 {
			bad("%s: an indexer in a graph whose facts state none, or no query latent", at)
			return
		}
		if !ins(2) || !outs(1) {
			return
		}
		at2 := o.Attr
		if at2.NHead != x.Heads || at2.HeadDim != x.HeadDim || at2.TopK != x.TopK || width(o.Out[0]) != x.TopK ||
			width(o.In[1]) != a.QLoraRank || len(o.Weight) != 5 {
			bad("%s: %d heads of %d keeping %d over a latent of %d; the configuration states %d of %d keeping %d over %d",
				at, at2.NHead, at2.HeadDim, at2.TopK, width(o.In[1]), x.Heads, x.HeadDim, x.TopK, a.QLoraRank)
		}
		if len(o.Reads) != 1 || o.Reads[0].Name != "idx" || len(o.Writes) != 1 || o.Writes[0].Name != "idx" {
			bad("%s: reads and writes no idx version", at)
		}
	case OpAttendLatent:
		k := 2
		if a.Index != nil {
			k = 3
		}
		if !ins(k) || !outs(1) || len(o.Weight) != 2 {
			bad("%s: wants two query parts, one output and the two up-projections", at)
			return
		}
		at2 := o.Attr
		if len(o.Reads) != 1 || o.Reads[0].Name != "kv" || o.Reads[0].Version == 0 {
			bad("%s: reads no kv version this block wrote", at)
		}
		if at2.NHead != a.NHead || at2.HeadDim != a.HeadDim || at2.NRot != a.NRot || at2.Scale != c.Scale {
			bad("%s: %d heads of %d (%d rotated), scale %g; the configuration states %d of %d (%d), %g",
				at, at2.NHead, at2.HeadDim, at2.NRot, at2.Scale, a.NHead, a.HeadDim, a.NRot, c.Scale)
		}
		kb, vb := o.Weight[0], o.Weight[1]
		nope := a.HeadDim - a.NRot
		if kb.Role != jlm.RoleAttnKB || kb.Sheets != a.NHead || kb.K != nope || kb.Rows != a.KVLoraRank ||
			vb.Role != jlm.RoleAttnVB || vb.Sheets != a.NHead || vb.K != a.KVLoraRank || vb.Rows != a.HeadDimV {
			bad("%s: up-projections %v %dx%dx%d and %v %dx%dx%d", at, kb.Role, kb.K, kb.Rows, kb.Sheets,
				vb.Role, vb.K, vb.Rows, vb.Sheets)
		}
		if width(o.In[0]) != a.NHead*nope || width(o.In[1]) != a.NHead*a.NRot || width(o.Out[0]) != a.NHead*a.HeadDimV {
			bad("%s: queries of %d and %d into %d", at, width(o.In[0]), width(o.In[1]), width(o.Out[0]))
		}
	}
}

var errAbsorbNoLatent = errors.New("the attention reads no latent cache: there is nothing to absorb into")

// MLAAbsorb is the absorbed latent attention: W_kb's sheet h folds into head
// h's query (q_abs = W_kb_h q_nope, KVLoraRank wide, the rotary part after
// it), every head attends over the cached latent rows as one shared key head
// with the value the row's latent prefix, and W_vb's sheet h turns head h's
// KVLoraRank-wide result into HeadDimV after the softmax. Neither a key nor a
// value is materialised per head. It is exact algebra (associativity), so its
// contract is the f32 band of a reordered sum.
//
// The reference form decompresses every cached row per head per step: at a
// decode it reads NHead*(nope+HeadDimV)*KVLoraRank multiply-adds per cached
// position where the absorbed form spends them once on the query and the
// output. Every tier lowers only the absorbed form; whether the reference
// form pays for a compute-bound prompt chunk (where the decompressed keys are
// reused by every row) is open, and it has no lowering on any tier. The
// static guard decides it at compile time, at every row class.
var MLAAbsorb = &Rewrite{
	Name: "mla-absorb",
	Match: func(g *Graph) []Region {
		var out []Region
		for i, o := range g.Ops {
			if o.Kind == OpAttendLatent {
				out = append(out, Region{Ops: []int{i}})
			}
		}
		return out
	},
	Legal: func(g *Graph, r Region) error {
		if len(r.Ops) != 1 || g.Ops[r.Ops[0]].Kind != OpAttendLatent {
			return fmt.Errorf("the region is not one latent attention")
		}
		o := g.Ops[r.Ops[0]]
		if len(o.Reads) != 1 || o.Reads[0].Name != "kv" {
			return errAbsorbNoLatent
		}
		return nil
	},
	Contract: map[string]Contract{
		"host":  {Band: 1e-4, Gate: "model.TestMLABatchMatchesDecode, model.TestGreedyMatchesLlamaCpp (DeepSeek-V2-Lite, Moonlight)"},
		"ptx":   {Band: 1e-2, Gate: "model.TestMLAOnEveryDevice"},
		"spirv": {Band: 1e-2, Gate: "model.TestMLAOnEveryDevice"},
		"msl":   {Band: 1e-2, Gate: "model.TestMLAOnEveryDevice"},
	},
	Keys:    func(*Graph, Region) string { return "" },
	When:    WhenCompile,
	Chooser: "the static guard: the absorbed form is the one form a tier lowers; the reference form for a prompt chunk is open and unlowered",
}
