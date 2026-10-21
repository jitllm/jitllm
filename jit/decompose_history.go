package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/ir"
)

// A row's history descriptor, the words the engine writes for each row of a
// step.
const (
	rowTab   = iota // where the row's page table starts
	rowStart        // the first position it attends
	rowEnd          // one past the last
	rowWrite        // where its new key and value go
	rowWords
)

// The step's tables: the rotary's cos and sin a row, the rows' descriptors,
// the page table, the history's key and value pools.
var (
	tableAngles = Operand{Kind: OperandTable, Name: "rope.cs"}
	tableQOff   = Operand{Kind: OperandTable, Name: "q.off", Elem: ir.U32}
	tableRows   = Operand{Kind: OperandTable, Name: "rows", Elem: ir.U32}
	tablePages  = Operand{Kind: OperandTable, Name: "pages", Elem: ir.U32}
	tableKeys   = Operand{Kind: OperandTable, Name: "kv.k", PerPart: true}
	tableValues = Operand{Kind: OperandTable, Name: "kv.v", PerPart: true}
)

// decomposeRope turns each head's first NRot elements in pairs by the
// step's angles: an element a pair, both turned values stored.
func decomposeRope(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	if len(o.In) != 1 || len(o.Out) != 1 || a.HeadDim <= 0 || c.width(o.In[0])%a.HeadDim != 0 || a.NRot != a.HeadDim || a.Freqs ||
		a.Sections != [4]int{} || a.Mscale != 0 || a.Layout != graph.RopeNorm && a.Layout != graph.RopeNeox {
		return nil, fmt.Errorf("a rotary of %d of %d, layout %d", a.NRot, a.HeadDim, a.Layout)
	}
	U, F := ir.U32, ir.F32
	sh := c.rows(o.In[0])
	R, W, hd := sh[0], sh[1], a.HeadDim
	pairs := a.NRot / 2
	var x exprs
	r, h, p := x.coord(0), x.coord(1), x.coord(2)
	first, step := x.op(OpMul, U, p, x.u(2)), 1
	if a.Layout == graph.RopeNeox {
		first, step = p, pairs
	}
	at := x.op(OpAdd, U, x.op(OpMul, U, h, x.u(uint32(hd))), first)
	at2 := x.op(OpAdd, U, at, x.u(uint32(step)))
	x0, x1 := x.at(0, sh, r, at), x.at(0, sh, r, at2)
	cs := []int{R, a.NRot}
	co, si := x.at(1, cs, r, x.op(OpMul, U, p, x.u(2))), x.at(1, cs, r, x.op(OpAdd, U, x.op(OpMul, U, p, x.u(2)), x.u(1)))
	y0 := x.op(OpSub, F, x.op(OpMul, F, x0, co), x.op(OpMul, F, x1, si))
	y1 := x.op(OpAdd, F, x.op(OpMul, F, x0, si), x.op(OpMul, F, x1, co))
	out := c.value(o.Out[0])
	return []Prim{{Kind: PrimMap, Shape: []int{R, W / hd, pairs}, In: []Operand{c.value(o.In[0]), tableAngles}, Expr: x.e,
		Stores: []PrimStore{{Out: out, Ax: []int{r, at}, Ext: sh, Value: y0}, {Out: out, Ax: []int{r, at2}, Ext: sh, Value: y1}}}}, nil
}

// decomposeKVAppend writes each row's key and value into its page of the key
// and value pools, each a page of positions of the width.
func decomposeKVAppend(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 2 || c.Step.Page <= 0 {
		return nil, fmt.Errorf("a history append of %d values over pages of %d", len(o.In), c.Step.Page)
	}
	U := ir.U32
	page := c.Step.Page
	slot := func(x *exprs) (pid, off int) {
		row := func(w int) int {
			return x.load(1, x.op(OpAdd, U, x.op(OpMul, U, x.coord(0), x.u(rowWords)), x.u(uint32(w))))
		}
		tab, pos := row(rowTab), row(rowWrite)
		pid = x.load(2, x.op(OpAdd, U, tab, x.op(OpDiv, U, pos, x.u(uint32(page)))))
		return pid, x.op(OpRem, U, pos, x.u(uint32(page)))
	}
	var prims []Prim
	for i, pool := range []Operand{tableKeys, tableValues} {
		v := o.In[i]
		sh := c.rows(v)
		w := sh[1]
		var x exprs
		d := x.coord(1)
		pid, off := slot(&x)
		val := x.at(0, sh, x.coord(0), d)
		prims = append(prims, Prim{Kind: PrimMap, Shape: sh, In: []Operand{c.value(v), tableRows, tablePages}, Expr: x.e,
			Stores: []PrimStore{{Out: pool, Ax: []int{pid, off, d}, Ext: []int{c.pages(), page, w}, Value: val}}})
	}
	return prims, nil
}

// pages bounds the pages of the history a step reads: every row's whole
// history at the step's longest.
func (c *Context) pages() int {
	return c.Step.Rows * ((c.Step.Keys + c.Step.Page - 1) / c.Step.Page)
}

// StepWord is word w of a table the engine writes for a step, at step s,
// where the step decides it: the history's and the rotary table's.
func StepWord(o Operand, w int, s Step) (float64, bool) {
	if v, ok := historyWord(o, w, s); ok {
		return v, true
	}
	return ropeWord(o, w, s)
}

// historyWord is a word of the history's tables at step s: a decode row
// attends its s.Keys positions from the first, its page table from the
// table's start, and the table holds the pages in order.
func historyWord(o Operand, w int, s Step) (float64, bool) {
	if o.Kind == tablePages.Kind && o.Name == tablePages.Name {
		return float64(w), true
	}
	if o.Kind != tableRows.Kind || o.Name != tableRows.Name {
		return 0, false
	}
	switch w % rowWords {
	case rowTab, rowStart:
		return 0, true
	case rowEnd:
		return float64(s.Keys), true
	}
	return float64(max(0, s.Keys-1)), true
}
