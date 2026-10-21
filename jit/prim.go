package jit

import (
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/ir"
)

type PrimKind uint8

const (
	PrimMap PrimKind = iota
	PrimReduce
	PrimContract
	PrimGather
	PrimScatter
	PrimScan
	PrimLayout
)

type OperandKind uint8

const (
	OperandValue OperandKind = iota
	// OperandWeight is a matrix's payload, words whatever its format.
	OperandWeight
	// OperandVector is a vector weight, floats.
	OperandVector
	OperandScratch
	// OperandTable is one of the step's tables: the rotary's angles, the
	// rows' history descriptors and page table, the history's pools.
	OperandTable
)

// Operand is a tensor a primitive reads or writes: a graph value, a weight
// by role, or the composer's own scratch, its elements of Elem (floats
// where unset).
type Operand struct {
	Kind   OperandKind
	Value  graph.ValueID
	Role   jlm.Role
	Format jlm.Type
	Name   string
	Elem   ir.Type
	// Temp is a value inside a fused chain, never stored past it.
	Temp bool
}

// Access is where an index reaches an operand: (i*Mul/Div)%Mod + Off, Mul
// and Div 0 as one, Mod 0 as none.
type Access struct {
	Mul, Div, Mod, Off int
}

var atElement = Access{}

// Start is how a reduction begins: from a constant, or from its first
// element.
type Start uint8

const (
	StartConst Start = iota
	StartFirst
)

// Expr is one node of a primitive's scalar function: an operation on earlier
// nodes, the In'th operand's value, or a constant.
type Expr struct {
	Op      Op
	Type    ir.Type
	A, B, C int
	In      int
	Imm     int64
	// Ax and Ext are an ExprAt's coordinates and the extents they index.
	Ax, Ext []int
}

// ExprIn reads operand In at its access; ExprLoad reads it at node A;
// ExprElem is the element's index; ExprCoord is coordinate A of the
// primitive's Shape; ExprAt reads operand In at the element of coordinates
// Ax in a space of extents Ext; ExprLive is the first position past the
// attended Range.
const (
	ExprIn    Op = 255
	ExprLoad  Op = 254
	ExprElem  Op = 253
	ExprCoord Op = 252
	ExprAt    Op = 251
	ExprLive  Op = 250
)

// PrimStore is one store a primitive's element makes: operand Out at
// expression node Index, the value node Value.
type PrimStore struct {
	Out          Operand
	Index, Value int
}

// Prim is one primitive. Where Shape is set, a map's elements and a
// reduction's terms are its coordinates, a reduction's over the last.
//   - Contract: Out[r] = sum over k of In[0][r*K + k] * In[1][k], Rows rows.
//   - Map: Out[OutAt(e)] = Expr of In[j][At[j](e)], for the N elements e.
//   - Reduce: Out[OutAt(o)] = Combine over k of Expr of In[j][At[j](o*K + k)],
//     for the N outputs o, begun at Init (StartConst) or the first term.
type Prim struct {
	Kind    PrimKind
	In      []Operand
	At      []Access
	Out     Operand
	OutAt   Access
	N, K    int
	Rows    int
	Shape   []int
	Expr    []Expr
	Combine Op
	Start   Start
	Init    int64
	Type    ir.Type
	// Stores, where set, are a map's stores in place of Out at OutAt;
	// Index, where set, is where a reduction stores its output: an
	// expression of the output (ExprElem).
	Stores []PrimStore
	Index  []Expr
	// Range, a fact for passes, is the positions axis Range.Axis attends;
	// Part, a range split, its partition axis and the positions in one.
	Range *Range
	Part  *Part
	// Span, a fact for passes, is how many of the positions along the
	// Range's axis are live from its start: an expression of the
	// coordinates; past it each term is the start or the element unread.
	Span []Expr
	// Heads, a fact for passes, is the primitive's space as attention's:
	// its rows, query heads, key heads and head width.
	Heads *Heads
}

// Range is an axis of positions, coordinate k at First rounded down to
// Align plus k, the attended ones First up to End: First and End expressions
// of the coordinates.
type Range struct {
	Axis       int
	Align      int
	First, End []Expr
}

// Part is a split range: coordinate Split its partition, Axis the
// position in it.
type Part struct {
	Axis, Split int
}

// outShape is a primitive's output space: a map's elements, a reduction's
// terms less the reduced axis.
func (p *Prim) outShape() []int {
	if p.Kind == PrimReduce {
		return p.Shape[:len(p.Shape)-1]
	}
	return p.Shape
}

type Heads struct {
	Rows, Query, Key, Dim int
}
