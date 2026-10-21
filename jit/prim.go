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
	// OperandTable is one of the step's tables.
	OperandTable
)

// Operand is a tensor a primitive reads or writes: a graph value, a weight
// by role, a table of the step or the composer's own scratch, its elements
// of Elem (floats where unset).
type Operand struct {
	Kind   OperandKind
	Value  graph.ValueID
	Role   jlm.Role
	Format jlm.Type
	Name   string
	Elem   ir.Type
	// Temp is a value a decomposition passes between its own primitives,
	// read by nothing past them.
	Temp bool
}

// Expr is one node of a primitive's scalar function: an operation on earlier
// nodes, or a leaf.
type Expr struct {
	Op      Op
	Type    ir.Type
	A, B, C int
	In      int
	Imm     int64
	Ax, Ext []int
}

// The leaves: ExprCoord is coordinate A of the primitive's Shape; ExprAt
// reads operand In at the coordinates Ax of a space of extents Ext; ExprLoad
// reads it at the flat index node A; ExprLive is the first position past
// the attended range; ExprPos is the position at the range's coordinate.
const (
	ExprCoord Op = 255 - iota
	ExprAt
	ExprLoad
	ExprLive
	ExprPos
)

// PrimStore is one store a map's element makes: operand Out at the
// coordinates Ax of a space of extents Ext, or at the flat index node Index
// where Ax is nil.
type PrimStore struct {
	Out          Operand
	Ax, Ext      []int
	Index, Value int
}

// Prim is one primitive over the coordinates of Shape:
//   - Map: an element of Shape is Expr, stored to Out at its coordinates or
//     by Stores.
//   - Reduce: the last axis of Shape is combined by Combine from Init, the
//     rest is the output.
//   - Contract: a Reduce of products, each term's product summed.
//
// A reduction begins from Init, or from its first term where Start says;
// it stores at Index, an expression of its output's coordinates, where set.
type Prim struct {
	Kind    PrimKind
	Shape   []int
	In      []Operand
	Out     Operand
	Expr    []Expr
	Combine Op
	Init    int64
	Start   Start
	Type    ir.Type
	Stores  []PrimStore
	Index   []Expr
	Facts   Facts
}

type Start uint8

const (
	StartConst Start = iota
	StartFirst
)

// Facts are what a decomposition knows that a pass needs, carried as data.
type Facts struct {
	// Range is an axis of attended positions.
	Range *Range
	// Heads is the primitive's space as attention's.
	Heads *Heads
	// Rows is how many of the leading axis's rows are live, an expression
	// the step's tables give at run time.
	Rows []Expr
}

// Range is axis Axis of the primitive's Shape as positions: coordinate k at
// First plus k, attended from First up to End, both expressions of the
// coordinates.
type Range struct {
	Axis       int
	First, End []Expr
}

// Heads is attention's geometry: rows, query heads, key heads, head width.
type Heads struct {
	Rows, Query, Key, Dim int
}

// outShape is a primitive's output space.
func (p *Prim) outShape() []int {
	if p.Kind == PrimReduce || p.Kind == PrimContract {
		return p.Shape[:len(p.Shape)-1]
	}
	return p.Shape
}
