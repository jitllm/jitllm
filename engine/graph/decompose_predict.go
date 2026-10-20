package graph

import "github.com/jitllm/jitllm/format/jlm"

// predict writes a prediction block's prologue as primitives. The op reads
// In[0], the trunk's hidden row (S streams of d, stream-major, S >= 1), and
// In[1], the next token's embedding row (d); its Out[0] is S rows of d, one
// per stream, each
//
//	eh_proj([ enorm(e) ; hnorm_s(h_s) ])        the embedding half first
//
// where enorm is one scale of d applied to the embedding (computed once, the
// same row for every stream) and hnorm is S scales of d, stream s normed
// against its own. S is the quotient of the hidden row's width by the
// embedding's, never a name. Weights, by role: the embedding's norm
// (RoleNextnENorm, d), the hidden row's (RoleNextnHNorm, S*d) and the
// projection (RoleNextnEHProj, 2d to d). The op's other bindings (the
// block's own head norm, head, embedding table and stream mix) are not read
// by what Out[0] states and are not part of the prologue.
//
// In order: the embedding's norm; the hidden row's norm (one weighted norm
// while S is 1; for S > 1 each stream's weightless norm, then the stacked
// scales as one multiply, since a norm's single scale is shared by every
// head); then per stream the slice of the normed hidden, the join with the
// normed embedding and the projection; the stream outputs joined stream-
// major. It stays whole where a binding is missing or repeated, a width does
// not chain, or a value is not f32.
//
// The bindings the draft reads afterwards stay reachable through the source
// graph: Decompose returns a new graph and leaves its argument, whose op
// still carries every binding, untouched.
func (d *Graph) predict(o Op) bool {
	if len(o.In) != 2 || len(o.Out) != 1 {
		return false
	}
	var enorm, hnorm, eh *Binding
	// pick files w under its slot, refusing a second binding of one role.
	pick := func(slot **Binding, w *Binding) bool {
		if *slot != nil {
			return false
		}
		*slot = w
		return true
	}
	for i := range o.Weight {
		w, ok := &o.Weight[i], true
		switch w.Role {
		case jlm.RoleNextnENorm:
			ok = pick(&enorm, w)
		case jlm.RoleNextnHNorm:
			ok = pick(&hnorm, w)
		case jlm.RoleNextnEHProj:
			ok = pick(&eh, w)
		}
		if !ok {
			return false
		}
	}
	h, e, out := o.In[0], o.In[1], o.Out[0]
	hd, w := d.Values[h].Width, d.Values[e].Width
	if enorm == nil || hnorm == nil || eh == nil || w < 1 || hd < w || hd%w != 0 || d.Values[out].Width != hd ||
		d.Values[h].Elem != ElemF32 || d.Values[e].Elem != ElemF32 || d.Values[out].Elem != ElemF32 ||
		enorm.K != w || enorm.Rows != 0 || hnorm.K != hd || hnorm.Rows != 0 || eh.K != 2*w || eh.Rows != w {
		return false
	}
	streams := hd / w
	p := prims{d: d}
	en := p.out("predict.enorm", w, Op{Kind: OpNorm, In: []ValueID{e}, Weight: []Binding{*enorm}, Attr: Attrs{Eps: o.Attr.Eps}})
	if streams == 1 {
		hn := p.out("predict.hnorm", w, Op{Kind: OpNorm, In: []ValueID{h}, Weight: []Binding{*hnorm}, Attr: Attrs{Eps: o.Attr.Eps}})
		cat := p.out("predict.cat", 2*w, Op{Kind: OpConcat, In: []ValueID{en, hn}})
		d.op(Op{Kind: OpMatVec, In: []ValueID{cat}, Out: []ValueID{out}, Weight: []Binding{*eh}})
		return true
	}
	hh := p.out("predict.hhead", hd, Op{Kind: OpHeadNorm, In: []ValueID{h}, Attr: Attrs{Eps: o.Attr.Eps, HeadDim: w}})
	hn := p.mapw("predict.hnorm", hd, MapMul, *hnorm, hh)
	rows := make([]ValueID, streams)
	for s := range rows {
		hs := p.out("predict.hstream", w, Op{Kind: OpSlice, In: []ValueID{hn}, Attr: Attrs{Offset: s * w, Width: w}})
		cat := p.out("predict.cat", 2*w, Op{Kind: OpConcat, In: []ValueID{en, hs}})
		rows[s] = p.out("predict.row", w, Op{Kind: OpMatVec, In: []ValueID{cat}, Weight: []Binding{*eh}})
	}
	d.op(Op{Kind: OpConcat, In: rows, Out: []ValueID{out}})
	return true
}
