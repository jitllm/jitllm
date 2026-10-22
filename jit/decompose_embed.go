package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/ir"
)

// tableTokens is the step's tokens, a word a row, which the engine writes.
var tableTokens = Operand{Kind: OperandTable, Name: "tokens", Elem: ir.U32}

// decomposeEmbed is each row's token's row of the embedding, every element
// decoded from the weight's format.
func decomposeEmbed(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.Weight) != 1 || len(o.In) != 0 || len(o.Out) != 1 {
		return nil, fmt.Errorf("an embedding of %d weights and %d inputs", len(o.Weight), len(o.In))
	}
	w := o.Weight[0]
	wo, err := c.weight(w)
	if err != nil {
		return nil, err
	}
	U := ir.U32
	R, W := c.Step.Rows, w.K
	var x exprs
	r, i := x.coord(0), x.coord(1)
	tok := x.load(0, r)
	ext := map[int]int{0: R}
	in := []Operand{tableTokens}
	switch wo.Format {
	case jlm.TypeF32:
		in = append(in, wo)
		x.un(OpBitcast, ir.F32, x.load(1, x.op(OpAdd, U, x.op(OpMul, U, tok, x.u(uint32(W))), i)))
		ext[1] = w.Rows * W
	case jlm.TypeQ8:
		qs, ds := wo, wo
		ds.Region = 1
		in = append(in, qs, ds)
		q8Element(&x, 1, 2, tok, i, w.Rows)
		ext[1], ext[2] = W/4*w.Rows, W/q8Block*((w.Rows+1)/2)
	default:
		return nil, fmt.Errorf("an embedding of %s weights has no decomposition yet", wo.Format)
	}
	return []Prim{{Kind: PrimMap, Shape: []int{R, W}, In: in, Out: c.value(o.Out[0]), Expr: x.e,
		Facts: Facts{Extents: ext}}}, nil
}
