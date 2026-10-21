package jit

import (
	"errors"
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/ir"
)

func decompose(b Block, p Profile) ([]Prim, error) {
	var prims []Prim
	var errs []error
	quantized := map[graph.ValueID]bool{}
	rowLocal := map[graph.ValueID]bool{}
	for i, o := range b.Graph.Ops {
		if o.Kind == graph.OpMatVec && p.QuantizeActivations && len(o.In) == 1 && rowLocal[o.In[0]] && !quantized[o.In[0]] {
			v := o.In[0]
			quantized[v] = true
			name := b.Graph.Values[v].Name
			prims = append(prims, quantize(Operand{Kind: OperandValue, Value: v, Name: name}, b.Graph.Values[v].Width, b.Rows,
				Operand{Kind: OperandScratch, Name: name + ".q8a"}, Operand{Kind: OperandScratch, Name: name + ".q8x"})...)
		}
		if o.Kind == graph.OpNorm || o.Kind == graph.OpActMul {
			for _, v := range o.Out {
				rowLocal[v] = true
			}
		}
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
	case graph.OpRope:
		a := o.Attr
		w := b.Graph.Values[o.In[0]].Width
		if len(o.In) != 1 || len(o.Out) != 1 || a.HeadDim <= 0 || w%a.HeadDim != 0 || a.NRot != a.HeadDim || a.Freqs ||
			a.Sections != [4]int{} || a.Mscale != 0 || a.Layout != graph.RopeNorm && a.Layout != graph.RopeNeox {
			return nil, fmt.Errorf("a rotary of %d of %d, layout %d", a.NRot, a.HeadDim, a.Layout)
		}
		return []Prim{rope(value(o.In[0]), value(o.Out[0]), b.Rows, w, a.HeadDim, a.NRot, a.Layout == graph.RopeNeox, b.queryRope(o))}, nil
	case graph.OpKVAppend:
		if len(o.In) != 2 || b.Page <= 0 {
			return nil, fmt.Errorf("a history append of %d values over pages of %d", len(o.In), b.Page)
		}
		return []Prim{
			appendKey(value(o.In[0]), b.Rows, b.Graph.Values[o.In[0]].Width, b.Page),
			appendValue(value(o.In[1]), b.Rows, b.Graph.Values[o.In[1]].Width, b.Page),
		}, nil
	case graph.OpActMul:
		if len(o.In) != 2 || len(o.Out) != 1 {
			return nil, fmt.Errorf("an act_mul of %d operands", len(o.In))
		}
		var x exprs
		g := x.in(0)
		switch o.Attr.Act {
		case graph.ActSiLU:
			g = x.op(OpDiv, ir.F32, g, x.op(OpAdd, ir.F32, x.f(1), x.un(OpExp, ir.F32, x.op(OpSub, ir.F32, x.f(0), g))))
		default:
			return nil, fmt.Errorf("no formula for activation %d yet", o.Attr.Act)
		}
		x.op(OpMul, ir.F32, g, x.in(1))
		w := b.Graph.Values[o.Out[0]].Width
		return []Prim{{Kind: PrimMap, N: b.Rows * w, In: []Operand{value(o.In[0]), value(o.In[1])},
			At: []Access{atElement, atElement}, Out: value(o.Out[0]), Expr: x.e}}, nil
	case graph.OpNorm:
		if len(o.Weight) != 1 || len(o.In) != 1 || len(o.Out) != 1 {
			return nil, fmt.Errorf("a norm of %d weights", len(o.Weight))
		}
		w := b.Graph.Values[o.Out[0]].Width
		sum := Operand{Kind: OperandScratch, Name: b.Graph.Values[o.Out[0]].Name + ".sumsq", Temp: true}
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

// queryRope reports whether rope o turns what attention reads as its query:
// those rows land at the step's q offset.
func (b Block) queryRope(o graph.Op) bool {
	for _, a := range b.Graph.Ops {
		if a.Kind == graph.OpAttend && len(a.In) > 0 && a.In[0] == o.Out[0] {
			return true
		}
	}
	return false
}
