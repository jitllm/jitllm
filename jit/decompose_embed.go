package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
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
	R, W := c.Step.Rows, w.K
	var x exprs
	r, i := x.coord(0), x.coord(1)
	tok := x.load(0, r)
	in := []Operand{tableTokens}
	facts := Facts{Extents: map[int]int{0: R}}
	if _, err := weightElement(&x, wo, w.Rows, W, &in, &facts, tok, i); err != nil {
		return nil, fmt.Errorf("an embedding of %s weights: %w", wo.Format, err)
	}
	return []Prim{{Kind: PrimMap, Shape: []int{R, W}, In: in, Out: c.value(o.Out[0]), Expr: x.e, Facts: facts}}, nil
}
