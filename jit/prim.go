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
}

// Access is where an index reaches an operand: (i/Div)%Mod + Off, Div 0 as
// one and Mod 0 as none.
type Access struct {
	Div, Mod, Off int
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
}

// ExprIn marks an Expr that reads an operand.
const ExprIn Op = 255

// Prim is one primitive.
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
	Expr    []Expr
	Combine Op
	Start   Start
	Init    int64
	Type    ir.Type
}
