package graph

import (
	"strconv"

	"github.com/jitllm/jitllm/format/jlm"
)

// MatVec is the graph y = W x: one OpMatVec of a rows x k weight, the
// request stating its format under role.
func MatVec(role jlm.Role, rows, k int) *Graph {
	b := New()
	b.Out(b.MatVec(b.In("x", k), role, rows))
	return b.G
}

// Builder builds a graph op by op: its inputs, the ops over them, and the
// values it hands back. Every op is a generic one with its numbers; nothing
// in it names a model.
type Builder struct {
	G *Graph
}

// New is a model-wide graph (the head's, the input rows', a lone op's) in
// this build's op language.
func New() *Builder { return &Builder{G: &Graph{Version: Version, Block: -1}} }

// In is an input row of width elements.
func (s *Builder) In(name string, width int) ValueID {
	v := s.G.value(name, width)
	s.G.In = append(s.G.In, v)
	return v
}

// Out marks the values the graph hands back.
func (s *Builder) Out(vs ...ValueID) { s.G.Out = append(s.G.Out, vs...) }

func (s *Builder) width(v ValueID) int { return s.G.Values[v].Width }

// value is a new value named apart from every other: the composer names its
// own buffers after a value's name.
func (s *Builder) value(name string, width int) ValueID {
	return s.G.value(name+"."+strconv.Itoa(len(s.G.Values)), width)
}

// MatVec is W x for role's rows x width(x) weight, plus each bias role's
// vector.
func (s *Builder) MatVec(x ValueID, role jlm.Role, rows int, bias ...jlm.Role) ValueID {
	k := s.width(x)
	w := []Binding{{Role: role, K: k, Rows: rows}}
	for _, b := range bias {
		w = append(w, Binding{Role: b, K: rows})
	}
	y := s.value(role.String(), rows)
	s.G.op(Op{Kind: OpMatVec, In: []ValueID{x}, Out: []ValueID{y}, Weight: w})
	return y
}

// Norm is RMSNorm of x with role's scale.
func (s *Builder) Norm(x ValueID, role jlm.Role, eps float64) ValueID {
	y := s.value(role.String(), s.width(x))
	s.G.op(Op{Kind: OpNorm, In: []ValueID{x}, Out: []ValueID{y}, Weight: []Binding{{Role: role, K: s.width(x)}}, Attr: Attrs{Eps: eps}})
	return y
}

// LayerNorm is the mean-centred norm of x with role's scale and each bias
// role's vector.
func (s *Builder) LayerNorm(x ValueID, role jlm.Role, eps float64, bias ...jlm.Role) ValueID {
	k := s.width(x)
	w := []Binding{{Role: role, K: k}}
	for _, b := range bias {
		w = append(w, Binding{Role: b, K: k})
	}
	y := s.value(role.String(), k)
	s.G.op(Op{Kind: OpLayerNorm, In: []ValueID{x}, Out: []ValueID{y}, Weight: w, Attr: Attrs{Eps: eps}})
	return y
}

// Add is x + y.
func (s *Builder) Add(x, y ValueID) ValueID {
	z := s.value("sum", s.width(x))
	s.G.op(Op{Kind: OpAdd, In: []ValueID{x, y}, Out: []ValueID{z}})
	return z
}

// Expr is the elementwise function terms states over in and the bound
// vectors w, into a row of width.
func (s *Builder) Expr(name string, width int, terms []Term, w []Binding, in ...ValueID) ValueID {
	z := s.value(name, width)
	s.G.op(Op{Kind: OpExpr, In: in, Out: []ValueID{z}, Weight: w, Attr: Attrs{Expr: terms}})
	return z
}

// HeadNorm is RMSNorm over each head of headDim within x, role's scale of
// headDim shared by every head.
func (s *Builder) HeadNorm(x ValueID, role jlm.Role, headDim int, eps float64) ValueID {
	y := s.value(role.String(), s.width(x))
	s.G.op(Op{Kind: OpHeadNorm, In: []ValueID{x}, Out: []ValueID{y}, Weight: []Binding{{Role: role, K: headDim}},
		Attr: Attrs{Eps: eps, HeadDim: headDim}})
	return y
}

// KVAppend writes k and v as the step's rows of the block's history: it
// reads the history's version n and writes n+1, which it returns.
func (s *Builder) KVAppend(k, v ValueID, n int) int {
	s.G.op(Op{Kind: OpKVAppend, In: []ValueID{k, v}, Reads: []StateRef{{Name: "kv", Version: n}}, Writes: []StateRef{{Name: "kv", Version: n + 1}}})
	return n + 1
}

// Attend is q's attention over the history's version n, of a's heads,
// scale, window and softcap, with the per-head sinks of sinks where given.
func (s *Builder) Attend(q ValueID, n int, a Attrs, sinks ...jlm.Role) ValueID {
	return s.AttendOf(q, StateRef{Name: "kv", Version: n}, a, sinks...)
}

// AttendOf is Attend over the history h: the block's own ("kv"), or block
// N's as that block left it (SharedHistory), which a block writing no
// history of its own reads.
func (s *Builder) AttendOf(q ValueID, h StateRef, a Attrs, sinks ...jlm.Role) ValueID {
	var w []Binding
	for _, r := range sinks {
		w = append(w, Binding{Role: r, K: a.NHead})
	}
	y := s.value("attend", s.width(q))
	s.G.op(Op{Kind: OpAttend, In: []ValueID{q}, Out: []ValueID{y}, Weight: w, Reads: []StateRef{h}, Attr: a})
	return y
}

// Select is OpSelect's selection over the history h, of q's heads and
// a's groups, block, budget and key, scaling each head's score by w's
// element where w is given.
func (s *Builder) Select(q ValueID, h StateRef, a Attrs, w ...ValueID) ValueID {
	y := s.value("select", a.NKVHead)
	s.G.op(Op{Kind: OpSelect, In: append([]ValueID{q}, w...), Out: []ValueID{y}, Reads: []StateRef{h}, Attr: a})
	return y
}

// EntAppend writes k as the row's entry of the entry history ("ent") at
// version n, the slot its descriptor names, and returns the next version.
func (s *Builder) EntAppend(k ValueID, n int) int {
	s.G.op(Op{Kind: OpKVAppend, In: []ValueID{k, k}, Reads: []StateRef{{Name: EntHistory, Version: n}}, Writes: []StateRef{{Name: EntHistory, Version: n + 1}}})
	return n + 1
}

// ColPool is OpColPool over the history h: a's window of Span rows, its
// value and gate HeadDim wide at Offset and GateAt of rows Width wide.
func (s *Builder) ColPool(h StateRef, a Attrs) ValueID {
	y := s.value("colpool", a.HeadDim)
	s.G.op(Op{Kind: OpColPool, Out: []ValueID{y}, Reads: []StateRef{h}, Attr: a})
	return y
}

// SelectPooled is Select over pooled block keys (Attrs.Pooled), each normed
// by role norm's weight.
func (s *Builder) SelectPooled(q ValueID, h StateRef, a Attrs, norm jlm.Role) ValueID {
	a.Pooled = true
	y := s.value("select", a.NKVHead)
	s.G.op(Op{Kind: OpSelect, In: []ValueID{q}, Out: []ValueID{y}, Weight: []Binding{{Role: norm, K: a.HeadDim}}, Reads: []StateRef{h}, Attr: a})
	return y
}

// AttendTwo is q's attention over the window of the history h and the
// entries of the entry history e under one softmax (OpAttend of two reads),
// over the entries selection sel keeps where it is given.
func (s *Builder) AttendTwo(q ValueID, h, e StateRef, sel *ValueID, a Attrs, sinks ...jlm.Role) ValueID {
	var w []Binding
	for _, r := range sinks {
		w = append(w, Binding{Role: r, K: a.NHead})
	}
	in := []ValueID{q}
	if sel != nil {
		in = append(in, *sel)
	}
	y := s.value("attend", s.width(q))
	s.G.op(Op{Kind: OpAttend, In: in, Out: []ValueID{y}, Weight: w, Reads: []StateRef{h, e}, Attr: a})
	return y
}

// AttendSelected is AttendOf over the positions selection sel keeps.
func (s *Builder) AttendSelected(q ValueID, h StateRef, sel ValueID, a Attrs, sinks ...jlm.Role) ValueID {
	y := s.AttendOf(q, h, a, sinks...)
	o := &s.G.Ops[len(s.G.Ops)-1]
	o.In = append(o.In, sel)
	return y
}

// CrossAttend is q's cross-attention over history h as other rows left
// it, h a version this graph does not append to: each row attends, both
// ways, the span of h its row's descriptor names (a resampler's learned
// queries over a picture's patches).
func (s *Builder) CrossAttend(q ValueID, h StateRef, a Attrs) ValueID {
	a.Bidir = true
	return s.AttendOf(q, h, a)
}

// EntHistory is a block's entry history: a second paged history, one row an
// entry its compressor folds, its values its keys.
const EntHistory = "ent"

// SharedHistory is block n's history as another block reads it.
func SharedHistory(n int) StateRef { return StateRef{Name: "kv@" + strconv.Itoa(n)} }

// NewBlock is block li's graph in this build's op language.
func NewBlock(li int) *Builder { return &Builder{G: &Graph{Version: Version, Block: li}} }

// Result is one output an op writes: a new value of Width elements of Elem,
// named after Name and apart from every other.
type Result struct {
	Name  string
	Width int
	Elem  Elem
	// Rows is Value.Rows.
	Rows int
}

// Emit is one op of kind over in, binding w, with attributes a, reading and
// writing the state versions given: any op of the language, its outputs new
// values as outs state them.
func (s *Builder) Emit(kind OpKind, in []ValueID, outs []Result, w []Binding, a Attrs, reads, writes []StateRef) []ValueID {
	ids := make([]ValueID, len(outs))
	for i, r := range outs {
		ids[i] = s.value(r.Name, r.Width)
		s.G.Values[ids[i]].Elem, s.G.Values[ids[i]].Rows = r.Elem, r.Rows
	}
	s.G.op(Op{Kind: kind, In: in, Out: ids, Weight: w, Attr: a, Reads: reads, Writes: writes})
	return ids
}

// Int8Codes is x quantized to int8 per row by its largest magnitude: the
// row's amax over 127 is its scale d, and each element times 1/d (0 where d
// is 0) rounded half away from zero and clamped to int8's range is its code.
// A matrix product over the codes, times d, is the product over x's int8
// form. Its divisions are exact: the codes are the reference's only where d
// and 1/d are correctly rounded.
func (s *Builder) Int8Codes(x ValueID) (codes, scale ValueID) {
	w := s.width(x)
	m := func(name string, fn MapFn, k float64, width int, in ...ValueID) ValueID {
		return s.Emit(OpMap, in, []Result{{Name: name, Width: width}}, nil, Attrs{Fn: fn, Const: k, Exact: true}, nil, nil)[0]
	}
	amax := s.Emit(OpReduce, []ValueID{m("q8.abs", MapAbs, 0, w, x)}, []Result{{Name: "q8.amax", Width: 1}}, nil,
		Attrs{Fn: MapMax}, nil, nil)[0]
	scale = m("q8.scale", MapDiv, 127, 1, amax)
	inv := m("q8.inv", MapWhere, 0, 1, scale, m("q8.recip", MapRecip, 0, 1, scale))
	r := m("q8.round", MapRound, 0, w, m("q8.t", MapMul, 0, w, x, inv))
	hi := m("q8.hi", MapMin, 127, w, r)
	codes = s.Emit(OpMap, []ValueID{hi}, []Result{{Name: "q8.codes", Width: w, Elem: ElemI8}}, nil,
		Attrs{Fn: MapMax, Const: -128}, nil, nil)[0]
	return codes, scale
}
