package graph

import (
	"strconv"

	"github.com/jitllm/jitllm/format/jlm"
)

// MatVec is the slice y = W x: one OpMatVec of a rows x k weight, the
// request stating its format under role.
func MatVec(role jlm.Role, rows, k int) *Graph {
	b := NewSlice()
	b.Out(b.MatVec(b.In("x", k), role, rows))
	return b.G
}

// Slice builds a call site's slice of a block op by op: its inputs, the ops
// over them, and the values it hands back. Every op is a generic one with
// its numbers; nothing in it names a model.
type Slice struct {
	G *Graph
}

func NewSlice() *Slice { return &Slice{G: &Graph{}} }

// In is an input row of width elements.
func (s *Slice) In(name string, width int) ValueID {
	v := s.G.value(name, width)
	s.G.In = append(s.G.In, v)
	return v
}

// Out marks the values the slice hands back.
func (s *Slice) Out(vs ...ValueID) { s.G.Out = append(s.G.Out, vs...) }

func (s *Slice) width(v ValueID) int { return s.G.Values[v].Width }

// value is a new value named apart from every other: the composer names its
// own buffers after a value's name.
func (s *Slice) value(name string, width int) ValueID {
	return s.G.value(name+"."+strconv.Itoa(len(s.G.Values)), width)
}

// MatVec is W x for role's rows x width(x) weight, plus each bias role's
// vector.
func (s *Slice) MatVec(x ValueID, role jlm.Role, rows int, bias ...jlm.Role) ValueID {
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
func (s *Slice) Norm(x ValueID, role jlm.Role, eps float64) ValueID {
	y := s.value(role.String(), s.width(x))
	s.G.op(Op{Kind: OpNorm, In: []ValueID{x}, Out: []ValueID{y}, Weight: []Binding{{Role: role, K: s.width(x)}}, Attr: Attrs{Eps: eps}})
	return y
}

// Add is x + y.
func (s *Slice) Add(x, y ValueID) ValueID {
	z := s.value("sum", s.width(x))
	s.G.op(Op{Kind: OpAdd, In: []ValueID{x, y}, Out: []ValueID{z}})
	return z
}

// ActMul is act(a) * b.
func (s *Slice) ActMul(act Act, a, b ValueID) ValueID {
	z := s.value("actmul", s.width(a))
	s.G.op(Op{Kind: OpActMul, In: []ValueID{a, b}, Out: []ValueID{z}, Attr: Attrs{Act: act}})
	return z
}

// HeadNorm is RMSNorm over each head of headDim within x, role's scale of
// headDim shared by every head.
func (s *Slice) HeadNorm(x ValueID, role jlm.Role, headDim int, eps float64) ValueID {
	y := s.value(role.String(), s.width(x))
	s.G.op(Op{Kind: OpHeadNorm, In: []ValueID{x}, Out: []ValueID{y}, Weight: []Binding{{Role: role, K: headDim}},
		Attr: Attrs{Eps: eps, HeadDim: headDim}})
	return y
}

// Rope turns each head of headDim's first nRot elements by the step's
// rotary table, in pairs of layout, at base.
func (s *Slice) Rope(x ValueID, headDim, nRot int, base float64, layout RopeLayout) ValueID {
	y := s.value("rope", s.width(x))
	s.G.op(Op{Kind: OpRope, In: []ValueID{x}, Out: []ValueID{y}, Attr: Attrs{HeadDim: headDim, NRot: nRot, Base: base, Layout: layout}})
	return y
}
