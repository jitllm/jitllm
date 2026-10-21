package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/ir"
)

// decompositions is every op kind the composer decomposes, each into
// primitives over coordinates: a value of the step is [rows][width].
func decompositions() map[graph.OpKind]Decomposition {
	return map[graph.OpKind]Decomposition{
		graph.OpNorm:      decomposeNorm,
		graph.OpMatVec:    decomposeMatVec,
		graph.OpAdd:       decomposeAdd,
		graph.OpActMul:    decomposeActMul,
		graph.OpRope:      decomposeRope,
		graph.OpKVAppend:  decomposeKVAppend,
		graph.OpAttend:    decomposeAttend,
		graph.OpRopeTable: decomposeRopeTable,
	}
}

func (c *Context) value(v graph.ValueID) Operand {
	return Operand{Kind: OperandValue, Value: v, Name: c.Graph.Values[v].Name}
}

func (c *Context) width(v graph.ValueID) int { return c.Graph.Values[v].Width }

func (c *Context) rows(v graph.ValueID) []int { return []int{c.Step.Rows, c.width(v)} }

func (c *Context) weight(w graph.Binding) (Operand, error) {
	f, ok := c.Formats[w.Role]
	if !ok {
		return Operand{}, fmt.Errorf("no format for %s", w.Role)
	}
	return Operand{Kind: OperandWeight, Role: w.Role, Format: f, Name: w.Role.String()}, nil
}

// decomposeNorm is an RMS norm: each row's sum of squares, then each element
// scaled by the row's reciprocal root mean square and its weight.
func decomposeNorm(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.Weight) != 1 || len(o.In) != 1 || len(o.Out) != 1 {
		return nil, fmt.Errorf("a norm of %d weights", len(o.Weight))
	}
	F := ir.F32
	sh := c.rows(o.In[0])
	R, W := sh[0], sh[1]
	sum := Operand{Kind: OperandScratch, Name: c.Graph.Values[o.Out[0]].Name + ".sumsq"}
	wt := Operand{Kind: OperandVector, Role: o.Weight[0].Role, Name: o.Weight[0].Role.String()}

	var s exprs
	xs := s.at(0, sh, s.coord(0), s.coord(1))
	s.op(OpMul, F, xs, xs)

	var m exprs
	r, i := m.coord(0), m.coord(1)
	ms := m.op(OpMul, F, m.at(1, []int{R}, r), m.f(1/float32(W)))
	inv := m.op(OpDiv, F, m.f(1), m.un(OpSqrt, F, m.op(OpAdd, F, ms, m.f(float32(o.Attr.Eps)))))
	m.op(OpMul, F, m.op(OpMul, F, m.at(0, sh, r, i), inv), m.at(2, []int{W}, i))

	return []Prim{
		{Kind: PrimReduce, Shape: sh, In: []Operand{c.value(o.In[0])}, Out: sum, Combine: OpAdd, Expr: s.e},
		{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0]), sum, wt}, Out: c.value(o.Out[0]), Expr: m.e},
	}, nil
}

// decomposeMatVec is each output row's sum over K of the weight's row times
// the input.
func decomposeMatVec(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.Weight) != 1 || len(o.In) != 1 || len(o.Out) != 1 {
		return nil, fmt.Errorf("a matvec of %d weights", len(o.Weight))
	}
	w := o.Weight[0]
	wo, err := c.weight(w)
	if err != nil {
		return nil, err
	}
	if wo.Format != jlm.TypeF32 {
		return nil, fmt.Errorf("a matvec over %s weights has no decomposition yet", wo.Format)
	}
	var x exprs
	r, row, k := x.coord(0), x.coord(1), x.coord(2)
	x.op(OpMul, ir.F32, x.at(0, []int{w.Rows, w.K}, row, k), x.at(1, c.rows(o.In[0]), r, k))
	var prims []Prim
	if v := o.In[0]; c.Int8 != nil && c.Part == PartBlock && !c.formed[v] && c.producedByRow(v) && c.width(v)%c.Int8.Window == 0 {
		c.formed[v] = true
		prims = c.int8(v)
	}
	return append(prims, Prim{Kind: PrimContract, Shape: []int{c.Step.Rows, w.Rows, w.K}, In: []Operand{wo, c.value(o.In[0])},
		Out: c.value(o.Out[0]), Expr: x.e}), nil
}

// producedByRow reports whether value v is the output of an op computing a
// row from the row alone: a norm or a gated activation.
func (c *Context) producedByRow(v graph.ValueID) bool {
	for _, o := range c.Graph.Ops {
		for _, out := range o.Out {
			if out == v {
				return o.Kind == graph.OpNorm || o.Kind == graph.OpActMul
			}
		}
	}
	return false
}

func decomposeAdd(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 2 || len(o.Out) != 1 || o.Attr.Alpha != 0 && o.Attr.Alpha != 1 || o.Attr.Scale != 0 {
		return nil, fmt.Errorf("an add of %d operands, alpha %v, scale %v", len(o.In), o.Attr.Alpha, o.Attr.Scale)
	}
	sh := c.rows(o.Out[0])
	var x exprs
	r, i := x.coord(0), x.coord(1)
	x.op(OpAdd, ir.F32, x.at(0, sh, r, i), x.at(1, sh, r, i))
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0]), c.value(o.In[1])}, Out: c.value(o.Out[0]), Expr: x.e}}, nil
}

// decomposeActMul is the gated activation: act(gate) times up.
func decomposeActMul(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 2 || len(o.Out) != 1 {
		return nil, fmt.Errorf("an act_mul of %d operands", len(o.In))
	}
	F := ir.F32
	sh := c.rows(o.Out[0])
	var x exprs
	r, i := x.coord(0), x.coord(1)
	g := x.at(0, sh, r, i)
	switch o.Attr.Act {
	case graph.ActSiLU:
		g = x.op(OpDiv, F, g, x.op(OpAdd, F, x.f(1), x.un(OpExp, F, x.op(OpSub, F, x.f(0), g))))
	default:
		return nil, fmt.Errorf("no formula for activation %d yet", o.Attr.Act)
	}
	x.op(OpMul, F, g, x.at(1, sh, r, i))
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0]), c.value(o.In[1])}, Out: c.value(o.Out[0]), Expr: x.e}}, nil
}
