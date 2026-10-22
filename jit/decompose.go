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
		graph.OpNorm:         decomposeNorm,
		graph.OpHeadNorm:     decomposeHeadNorm,
		graph.OpMatVec:       decomposeMatVec,
		graph.OpAdd:          decomposeAdd,
		graph.OpActMul:       decomposeActMul,
		graph.OpAct:          decomposeAct,
		graph.OpMap:          decomposeMap,
		graph.OpLayerNorm:    decomposeLayerNorm,
		graph.OpRope:         decomposeRope,
		graph.OpKVAppend:     decomposeKVAppend,
		graph.OpAttend:       decomposeAttend,
		graph.OpRopeTable:    decomposeRopeTable,
		graph.OpTopK:         decomposeTopK,
		graph.OpReduce:       decomposeReduce,
		graph.OpTake:         decomposeTake,
		graph.OpGather:       decomposeGather,
		graph.OpCombine:      decomposeCombine,
		graph.OpExpertMatVec: decomposeExpertMatVec,
		graph.OpRank:         decomposeRank,
		graph.OpClamp:        decomposeClamp,
		graph.OpScale:        decomposeScale,
		graph.OpStateConv:    decomposeStateConv,
		graph.OpStateScan:    decomposeStateScan,
		graph.OpStateDelta:   decomposeStateDelta,
		graph.OpLatentAppend: decomposeLatentAppend,
		graph.OpAttendLatent: decomposeAttendLatent,
		graph.OpSlice:        decomposeSlice,
		graph.OpTile:         decomposeTile,
		graph.OpConcat:       decomposeConcat,
		graph.OpEmbed:        decomposeEmbed,
	}
}

func (c *Context) value(v graph.ValueID) Operand {
	o := Operand{Kind: OperandValue, Value: v, Name: c.Graph.Values[v].Name}
	if c.Graph.Values[v].Elem == graph.ElemU32 {
		o.Elem = ir.U32
	}
	return o
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

// decomposeHeadNorm is an RMS norm over each head of a row: the row seen as
// heads of HeadDim, each head's sum of squares, then each element scaled by
// its head's reciprocal root mean square and the weight every head shares,
// ones where there is none.
func decomposeHeadNorm(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.Weight) > 1 || len(o.In) != 1 || len(o.Out) != 1 {
		return nil, fmt.Errorf("a head norm of %d weights", len(o.Weight))
	}
	F := ir.F32
	rows := c.rows(o.In[0])
	R, W, D := rows[0], rows[1], o.Attr.HeadDim
	if D <= 0 || W%D != 0 {
		return nil, fmt.Errorf("a head norm of heads of %d over a row of %d", D, W)
	}
	sh := []int{R, W / D, D}
	sum := Operand{Kind: OperandScratch, Name: c.Graph.Values[o.Out[0]].Name + ".sumsq"}

	var s exprs
	xs := s.at(0, sh, s.coord(0), s.coord(1), s.coord(2))
	s.op(OpMul, F, xs, xs)

	var m exprs
	r, h, i := m.coord(0), m.coord(1), m.coord(2)
	ms := m.op(OpMul, F, m.at(1, sh[:2], r, h), m.f(1/float32(D)))
	inv := m.op(OpDiv, F, m.f(1), m.un(OpSqrt, F, m.op(OpAdd, F, ms, m.f(float32(o.Attr.Eps)))))
	y := m.op(OpMul, F, m.at(0, sh, r, h, i), inv)
	in := []Operand{c.value(o.In[0]), sum}
	if len(o.Weight) == 1 {
		m.op(OpMul, F, y, m.at(2, []int{D}, i))
		in = append(in, Operand{Kind: OperandVector, Role: o.Weight[0].Role, Name: o.Weight[0].Role.String()})
	}
	return []Prim{
		{Kind: PrimReduce, Shape: sh, In: []Operand{c.value(o.In[0])}, Out: sum, Combine: OpAdd, Expr: s.e},
		{Kind: PrimMap, Shape: sh, In: in, Out: c.value(o.Out[0]), Expr: m.e},
	}, nil
}

// decomposeMatVec is each output row's sum over K of the weight's row times
// the input.
func decomposeMatVec(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.Weight) == 2 && len(o.In) == 1 && len(o.Out) == 1 {
		return decomposeBiased(o, c)
	}
	if len(o.Weight) != 1 || len(o.In) != 1 || len(o.Out) != 1 {
		return nil, fmt.Errorf("a matvec of %d weights", len(o.Weight))
	}
	w := o.Weight[0]
	wo, err := c.weight(w)
	if err != nil {
		return nil, err
	}
	switch wo.Format {
	case jlm.TypeF32:
	case jlm.TypeQ8:
		return decomposeMatVecQ8(o, c, wo)
	default:
		if l, ok := jlm.Layout(wo.Format); ok {
			if prims, ok := decomposeMatVecInt8(o, c, wo, l); ok {
				return prims, nil
			}
		}
		p, err := decomposeMatVecDecoded(w.Rows, w.K, wo, c.value(o.In[0]), c.value(o.Out[0]), c.Step.Rows)
		if err != nil {
			return nil, fmt.Errorf("a matvec over %s weights: %w", wo.Format, err)
		}
		return []Prim{p}, nil
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

// decomposeBiased is a matvec with a bias: the product, then each output
// element plus the bias's.
func decomposeBiased(o graph.Op, c *Context) ([]Prim, error) {
	b := o.Weight[1]
	if f := c.Formats[b.Role]; f != jlm.TypeF32 {
		return nil, fmt.Errorf("a bias %s stored as %s", b.Role, f)
	}
	inner := o
	inner.Weight = o.Weight[:1]
	prims, err := decomposeMatVec(inner, c)
	if err != nil {
		return nil, err
	}
	out := c.value(o.Out[0])
	raw := Operand{Kind: OperandScratch, Name: out.Name + ".raw"}
	for i := range prims {
		if prims[i].Out == out {
			prims[i].Out = raw
		}
	}
	sh := c.rows(o.Out[0])
	var x exprs
	r, i := x.coord(0), x.coord(1)
	x.op(OpAdd, ir.F32, x.at(0, sh, r, i), x.at(1, []int{sh[1]}, i))
	bias := Operand{Kind: OperandVector, Role: b.Role, Name: b.Role.String()}
	return append(prims, Prim{Kind: PrimMap, Shape: sh, In: []Operand{raw, bias}, Out: out, Expr: x.e}), nil
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
	if len(o.In) != 2 || len(o.Out) != 1 {
		return nil, fmt.Errorf("an add of %d operands", len(o.In))
	}
	F := ir.F32
	sh := c.rows(o.Out[0])
	var x exprs
	r, i := x.coord(0), x.coord(1)
	a, b := x.at(0, sh, r, i), x.at(1, sh, r, i)
	if al := o.Attr.Alpha; al != 0 && al != 1 {
		// Granite's residual scale: x + Alpha*y.
		b = x.op(OpMul, F, b, x.f(float32(al)))
	}
	sum := x.op(OpAdd, F, a, b)
	if o.Attr.Scale != 0 {
		// A scaled add, (x + y) * Scale.
		x.op(OpMul, F, sum, x.f(float32(o.Attr.Scale)))
	}
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
	if o.Attr.Act == graph.ActSiLU {
		g := x.at(0, sh, r, i)
		g = x.op(OpDiv, F, g, x.op(OpAdd, F, x.f(1), x.un(OpExp, F, x.op(OpSub, F, x.f(0), g))))
		x.op(OpMul, F, g, x.at(1, sh, r, i))
	} else if _, err := x.actMul(o.Attr.Act, x.at(0, sh, r, i), x.at(1, sh, r, i)); err != nil {
		return nil, err
	}
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0]), c.value(o.In[1])}, Out: c.value(o.Out[0]), Expr: x.e}}, nil
}
