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

// ActMul is act(a) * b.
func (s *Builder) ActMul(act Act, a, b ValueID) ValueID {
	z := s.value("actmul", s.width(a))
	s.G.op(Op{Kind: OpActMul, In: []ValueID{a, b}, Out: []ValueID{z}, Attr: Attrs{Act: act}})
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
	var w []Binding
	for _, r := range sinks {
		w = append(w, Binding{Role: r, K: a.NHead})
	}
	y := s.value("attend", s.width(q))
	s.G.op(Op{Kind: OpAttend, In: []ValueID{q}, Out: []ValueID{y}, Weight: w, Reads: []StateRef{{Name: "kv", Version: n}}, Attr: a})
	return y
}

// NewBlock is block li's graph in this build's op language.
func NewBlock(li int) *Builder { return &Builder{G: &Graph{Version: Version, Block: li}} }

// Result is one output an op writes: a new value of Width elements of Elem,
// named after Name and apart from every other.
type Result struct {
	Name  string
	Width int
	Elem  Elem
}

// Emit is one op of kind over in, binding w, with attributes a, reading and
// writing the state versions given: any op of the language, its outputs new
// values as outs state them.
func (s *Builder) Emit(kind OpKind, in []ValueID, outs []Result, w []Binding, a Attrs, reads, writes []StateRef) []ValueID {
	ids := make([]ValueID, len(outs))
	for i, r := range outs {
		ids[i] = s.value(r.Name, r.Width)
		s.G.Values[ids[i]].Elem = r.Elem
	}
	s.G.op(Op{Kind: kind, In: in, Out: ids, Weight: w, Attr: a, Reads: reads, Writes: writes})
	return ids
}
