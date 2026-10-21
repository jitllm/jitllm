package jit

import (
	"errors"
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/ir"
)

func decompose(b Block) ([]Prim, error) {
	var prims []Prim
	var errs []error
	for i, o := range b.Graph.Ops {
		ps, err := decomposeOp(b, o)
		if err != nil {
			errs = append(errs, fmt.Errorf("op %d (%s): %w", i, o.Kind, err))
			continue
		}
		prims = append(prims, ps...)
	}
	return prims, errors.Join(errs...)
}

func decomposeOp(b Block, o graph.Op) ([]Prim, error) {
	value := func(v graph.ValueID) Operand {
		return Operand{Kind: OperandValue, Value: v, Name: b.Graph.Values[v].Name}
	}
	switch o.Kind {
	case graph.OpMatVec:
		if len(o.Weight) != 1 || len(o.In) != 1 || len(o.Out) != 1 {
			return nil, fmt.Errorf("a matvec of %d weights", len(o.Weight))
		}
		w := o.Weight[0]
		f, ok := b.Formats[w.Role]
		if !ok {
			return nil, fmt.Errorf("no format for %s", w.Role)
		}
		return []Prim{{Kind: PrimContract, Rows: w.Rows, K: w.K,
			In:  []Operand{{Kind: OperandWeight, Role: w.Role, Format: f, Name: w.Role.String()}, value(o.In[0])},
			Out: value(o.Out[0])}}, nil
	case graph.OpAdd:
		if len(o.In) != 2 || len(o.Out) != 1 || o.Attr.Alpha != 0 && o.Attr.Alpha != 1 || o.Attr.Scale != 0 {
			return nil, fmt.Errorf("an add of %d operands, alpha %v, scale %v", len(o.In), o.Attr.Alpha, o.Attr.Scale)
		}
		w := b.Graph.Values[o.Out[0]].Width
		return []Prim{{Kind: PrimMap, N: b.Rows * w, In: []Operand{value(o.In[0]), value(o.In[1])},
			At: []Access{atElement, atElement}, Out: value(o.Out[0]),
			Expr: []Expr{{Op: ExprIn, In: 0}, {Op: ExprIn, In: 1}, {Op: OpAdd, Type: ir.F32, A: 0, B: 1}}}}, nil
	case graph.OpNorm:
		if len(o.Weight) != 1 || len(o.In) != 1 || len(o.Out) != 1 {
			return nil, fmt.Errorf("a norm of %d weights", len(o.Weight))
		}
		w := b.Graph.Values[o.Out[0]].Width
		sum := Operand{Kind: OperandScratch, Name: b.Graph.Values[o.Out[0]].Name + ".sumsq"}
		wt := Operand{Kind: OperandVector, Role: o.Weight[0].Role, Name: o.Weight[0].Role.String()}
		f := func(x float32) int64 { return int64(math.Float32bits(x)) }
		return []Prim{
			{Kind: PrimReduce, N: b.Rows, K: w, In: []Operand{value(o.In[0])}, At: []Access{atElement}, Out: sum, Combine: OpAdd,
				Expr: []Expr{{Op: ExprIn, In: 0}, {Op: OpMul, Type: ir.F32, A: 0, B: 0}}},
			{Kind: PrimMap, N: b.Rows * w, In: []Operand{value(o.In[0]), sum, wt}, At: []Access{atElement, {Div: w}, {Mod: w}},
				Out: value(o.Out[0]), Expr: []Expr{
					{Op: ExprIn, In: 1},
					{Op: OpConst, Type: ir.F32, Imm: f(1 / float32(w))},
					{Op: OpMul, Type: ir.F32, A: 0, B: 1},
					{Op: OpConst, Type: ir.F32, Imm: f(float32(o.Attr.Eps))},
					{Op: OpAdd, Type: ir.F32, A: 2, B: 3},
					{Op: OpConst, Type: ir.F32, Imm: f(1)},
					{Op: OpSqrt, Type: ir.F32, A: 4},
					{Op: OpDiv, Type: ir.F32, A: 5, B: 6},
					{Op: ExprIn, In: 0},
					{Op: OpMul, Type: ir.F32, A: 8, B: 7},
					{Op: ExprIn, In: 2},
					{Op: OpMul, Type: ir.F32, A: 9, B: 10},
				}},
		}, nil
	}
	return nil, errNoDecomposition
}

var errNoDecomposition = errors.New("no decomposition into primitives yet")
