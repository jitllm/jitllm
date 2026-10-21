package jit

import (
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
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
	OperandWeight
	OperandScratch
)

// Operand is a tensor a primitive reads or writes: a graph value, a weight
// by role, or the composer's own scratch.
type Operand struct {
	Kind   OperandKind
	Value  graph.ValueID
	Role   jlm.Role
	Format jlm.Type
	Name   string
}

// Prim is one primitive. Contract is Out[i] = sum over k of A[i, k] * B[k],
// A row-major over K, over Rows output elements. Map is Out[i] = Fn of the
// inputs' elements i, over Rows elements.
type Prim struct {
	Kind    PrimKind
	In      []Operand
	Out     Operand
	Rows, K int
	Fn      Op
}
