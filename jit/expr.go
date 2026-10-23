package jit

import (
	"math"

	"github.com/jitllm/jitllm/jit/ir"
)

// exprs builds a primitive's function node by node.
type exprs struct{ e []Expr }

func (x *exprs) add(e Expr) int { x.e = append(x.e, e); return len(x.e) - 1 }

func (x *exprs) u(v uint32) int { return x.add(Expr{Op: OpConst, Type: ir.U32, Imm: int64(v)}) }

func (x *exprs) i(v int32) int { return x.add(Expr{Op: OpConst, Type: ir.I32, Imm: int64(v)}) }

func (x *exprs) f(v float32) int {
	return x.add(Expr{Op: OpConst, Type: ir.F32, Imm: int64(math.Float32bits(v))})
}

func (x *exprs) op(op Op, t ir.Type, a, b int) int { return x.add(Expr{Op: op, Type: t, A: a, B: b}) }

func (x *exprs) un(op Op, t ir.Type, a int) int { return x.add(Expr{Op: op, Type: t, A: a}) }

func (x *exprs) sel(t ir.Type, p, a, b int) int {
	return x.add(Expr{Op: OpSelect, Type: t, A: p, B: a, C: b})
}

func (x *exprs) coord(a int) int { return x.add(Expr{Op: ExprCoord, A: a}) }

// at is operand j at coordinates ax of a space of extents ext.
func (x *exprs) at(j int, ext []int, ax ...int) int {
	return x.add(Expr{Op: ExprAt, In: j, Ax: ax, Ext: ext})
}

// load is operand j at flat index ix.
func (x *exprs) load(j, ix int) int { return x.add(Expr{Op: ExprLoad, In: j, A: ix}) }

func (x *exprs) live() int { return x.add(Expr{Op: ExprLive}) }

func (x *exprs) pos() int { return x.add(Expr{Op: ExprPos}) }
