package composer

import (
	"fmt"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/graph"
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

// rotaryTable is the step's table a rotary op turns by: the model's one
// where it has one, else the rotary its width and base name, its factors
// telling two apart. A table turns its pairs by the position axes the op
// states, or it is not the op's.
func (c *Context) rotaryTable(a graph.Attrs) (Operand, error) {
	axes := func(r graph.Rotary) bool {
		return r.Sections == a.Sections && r.Interleaved == a.Interleaved && r.Cycle == a.Cycle && r.CycleAxes == a.CycleAxes
	}
	if len(c.Rotaries) <= 1 {
		if len(c.Rotaries) == 1 && !axes(c.Rotaries[0]) {
			return Operand{}, fmt.Errorf("a rotary over the axes of sections %v (interleaved %v), where the model's table turns by sections %v (interleaved %v)",
				a.Sections, a.Interleaved, c.Rotaries[0].Sections, c.Rotaries[0].Interleaved)
		}
		return TableAngles.of(c.RotaryBase), nil
	}
	// A split rotary's table holds a pair an element.
	nrot := a.NRot
	if a.Layout == graph.RopeSplit {
		nrot *= 2
	}
	found := -1
	for i, r := range c.Rotaries {
		if r.NRot == nrot && r.Base == a.Base && axes(r) && (found < 0 || r.Freqs == a.Freqs) {
			found = i
		}
	}
	if found < 0 {
		return Operand{}, fmt.Errorf("a rotary of %d at base %v over sections %v the model states no table of", a.NRot, a.Base, a.Sections)
	}
	return TableAngles.of(c.RotaryBase + found), nil
}

// decomposeRope turns each head's first NRot elements in pairs by the
// step's angles: an element a pair, both turned values stored.
func decomposeRope(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	// The angles are the step's table, built from the engine's rotary: its
	// per-pair factors (Freqs), its cos/sin scale (Mscale) and, over several
	// position axes (Sections), each pair's axis are in it. Each pair turns
	// by one angle either way, so the rotation is the same.
	if len(o.In) != 1 && len(o.In) != 2 || len(o.Out) != 1 || a.HeadDim <= 0 || c.width(o.In[0])%a.HeadDim != 0 ||
		a.NRot <= 0 || a.NRot%2 != 0 || a.Offset < 0 || a.Offset+a.NRot > a.HeadDim ||
		a.Layout != graph.RopeNorm && a.Layout != graph.RopeNeox && a.Layout != graph.RopeSplit {
		return nil, fmt.Errorf("a rotary of %d of %d, layout %d", a.NRot, a.HeadDim, a.Layout)
	}
	U, F := ir.U32, ir.F32
	sh := c.rows(o.In[0])
	R, W, hd := sh[0], sh[1], a.HeadDim
	pairs := a.NRot / 2
	var x exprs
	r, h, p := x.coord(0), x.coord(1), x.coord(2)
	first, step := x.op(OpMul, U, p, x.u(2)), 1
	if a.Layout != graph.RopeNorm {
		first, step = p, pairs
	}
	if a.Offset > 0 {
		first = x.op(OpAdd, U, first, x.u(uint32(a.Offset)))
	}
	at := x.op(OpAdd, U, x.op(OpMul, U, h, x.u(uint32(hd))), first)
	at2 := x.op(OpAdd, U, at, x.u(uint32(step)))
	x0, x1 := x.at(0, sh, r, at), x.at(0, sh, r, at2)
	// A split rotary's table holds a pair an element: the second element
	// of pair p turns by the table's pair p + pairs.
	cs := []int{R, a.NRot}
	if a.Layout == graph.RopeSplit {
		cs[1] = 2 * a.NRot
	}
	angle := func(q int) (co, si int) {
		e := x.op(OpMul, U, q, x.u(2))
		return x.at(1, cs, r, e), x.at(1, cs, r, x.op(OpAdd, U, e, x.u(1)))
	}
	co, si := angle(p)
	co1, si1 := co, si
	if a.Layout == graph.RopeSplit {
		co1, si1 = angle(x.op(OpAdd, U, p, x.u(uint32(pairs))))
	}
	if a.Inverse {
		si, si1 = x.op(OpSub, F, x.f(0), si), x.op(OpSub, F, x.f(0), si1)
	}
	y0 := x.op(OpSub, F, x.op(OpMul, F, x0, co), x.op(OpMul, F, x1, si))
	y1 := x.op(OpAdd, F, x.op(OpMul, F, x0, si1), x.op(OpMul, F, x1, co1))
	out := c.value(o.Out[0])
	table, err := c.rotaryTable(a)
	if err != nil {
		return nil, err
	}
	if len(o.In) == 2 {
		if c.width(o.In[1]) != cs[1] {
			return nil, fmt.Errorf("a rotary of %d turning by a table of %d", cs[1], c.width(o.In[1]))
		}
		table = c.value(o.In[1])
	}
	in := []Operand{c.value(o.In[0]), table}
	prims := []Prim{{Kind: PrimMap, Shape: []int{R, W / hd, pairs}, In: in, Expr: x.e,
		Stores: []PrimStore{{Out: out, Ax: []int{r, at}, Ext: sh, Value: y0}, {Out: out, Ax: []int{r, at2}, Ext: sh, Value: y1}}}}
	// A partial rotary turns NRot of each head from Offset; the rest of
	// the head, before and after, is the input's.
	for _, keep := range [][2]int{{0, a.Offset}, {a.Offset + a.NRot, hd - a.Offset - a.NRot}} {
		if keep[1] <= 0 {
			continue
		}
		var t exprs
		tr, th, tj := t.coord(0), t.coord(1), t.coord(2)
		ta := t.op(OpAdd, U, t.op(OpAdd, U, t.op(OpMul, U, th, t.u(uint32(hd))), t.u(uint32(keep[0]))), tj)
		v := t.at(0, sh, tr, ta)
		prims = append(prims, Prim{Kind: PrimMap, Shape: []int{R, W / hd, keep[1]}, In: []Operand{c.value(o.In[0])}, Expr: t.e,
			Stores: []PrimStore{{Out: out, Ax: []int{tr, ta}, Ext: sh, Value: v}}})
	}
	return prims, nil
}

// decomposeKVAppend writes each row's key and value into its page of the key
// and value pools, each a page of positions of the width: the key alone
// where the values are the keys' rows.
func decomposeKVAppend(o graph.Op, c *Context) ([]Prim, error) {
	undo, err := c.onHistory(o)
	if err != nil {
		return nil, err
	}
	defer undo()
	if len(o.In) != 2 || c.Step.Page <= 0 {
		return nil, fmt.Errorf("a history append of %d values over pages of %d", len(o.In), c.Step.Page)
	}
	pools := []Operand{c.tab(TableKeys), c.tab(TableValues)}
	layouts := [2]Pool{c.History.Keys, c.History.Values}
	rows, pages := TableRows.op(), c.tab(TablePages)
	if c.History.ValuesInKeys {
		if o.In[0] != o.In[1] {
			return nil, fmt.Errorf("a history append of a value other than its key, into a pool whose values are its keys")
		}
		pools = pools[:1]
	}
	U := ir.U32
	page, P := c.Step.Page, max(1, o.Attr.Span)
	if P != max(1, c.History.Period) {
		return nil, fmt.Errorf("an append of an entry each %d positions into a history of one each %d", P, c.History.Period)
	}
	slots := c.slots()
	// A row closing no window writes zeros to the page's last slot, which
	// nothing reads: the IR stores unconditionally.
	open := -1
	slot := func(x *exprs) (pid, off int) {
		row := func(w int) int {
			return x.load(1, x.op(OpAdd, U, x.op(OpMul, U, x.coord(0), x.u(rowWords)), x.u(uint32(w))))
		}
		tab, pos := row(rowTab), row(rowWrite)
		pid = x.load(2, x.op(OpAdd, U, tab, x.op(OpDiv, U, pos, x.u(uint32(page)))))
		off = x.op(OpRem, U, pos, x.u(uint32(page)))
		if P > 1 {
			open = x.op(OpLess, ir.Pred, x.op(OpRem, U, pos, x.u(uint32(P))), x.u(uint32(P-1)))
			off = x.sel(U, open, x.u(uint32(slots-1)), x.op(OpDiv, U, off, x.u(uint32(P))))
		}
		return pid, off
	}
	var prims []Prim
	for i, pool := range pools {
		l := layouts[i]
		v := o.In[i]
		sh := c.rows(v)
		w := sh[1]
		if err := l.check(w); err != nil {
			return nil, err
		}
		var x exprs
		d := x.coord(1)
		pid, off := slot(&x)
		at := []int{pid, off, d}
		ext := []int{c.pages(), slots, w}
		shape := sh
		src := c.value(v)
		var val int
		switch {
		case l.Row > 0:
			// The rows quantized whole, a head's row at a time, then a
			// word a thread into the position's slot.
			q := Operand{Kind: OperandScratch, Name: src.Name + ".q8", Elem: U, Temp: true}
			rw := l.words(w)
			heads := []int{sh[0] * (w / l.Row), l.Row}
			if l.Block != 0 {
				prims = append(prims, blockRows(c, src, heads, q, l.Block)...)
			} else {
				prims = append(prims, q8Rows(c, src, heads, q)...)
			}
			src = q
			val = x.at(0, []int{sh[0], rw}, x.coord(0), d)
			if open >= 0 {
				val = x.sel(U, open, x.u(0), val)
			}
			shape, ext = []int{sh[0], rw}, []int{c.pages(), slots, rw}
		case l.Half:
			// A word a thread: its position's two halves, the lower element
			// low, so no two threads write one word.
			U := ir.U32
			lo := x.at(0, sh, x.coord(0), x.op(OpMul, U, d, x.u(2)))
			hi := x.at(0, sh, x.coord(0), x.op(OpAdd, U, x.op(OpMul, U, d, x.u(2)), x.u(1)))
			if l.BF16 {
				val = x.op(OpOr, U, x.bf16Bits(lo), x.op(OpShl, U, x.bf16Bits(hi), x.u(16)))
			} else {
				val = x.op(OpOr, U, x.un(OpHalf, U, lo), x.op(OpShl, U, x.un(OpHalf, U, hi), x.u(16)))
			}
			if open >= 0 {
				val = x.sel(U, open, x.u(0), val)
			}
			shape, ext = []int{sh[0], w / 2}, []int{c.pages(), slots, w / 2}
		case l.ByDim:
			val = x.at(0, sh, x.coord(0), d)
			if open >= 0 {
				val = x.sel(ir.F32, open, x.f(0), val)
			}
			at, ext = []int{pid, d, off}, []int{c.pages(), w, slots}
		default:
			val = x.at(0, sh, x.coord(0), d)
			if open >= 0 {
				val = x.sel(ir.F32, open, x.f(0), val)
			}
		}
		prims = append(prims, Prim{Kind: PrimMap, Shape: shape, In: []Operand{src, rows, pages}, Expr: x.e,
			Facts:  Facts{Extents: map[int]int{1: c.Step.Rows * rowWords, 2: c.pages()}},
			Stores: []PrimStore{{Out: l.operand(pool), Ax: at, Ext: ext, Value: val}}})
	}
	return prims, nil
}

// decomposePosition is each row's history position, the word its key is
// written at, plus Offset, divided into whole Spans where Span > 1, as a
// float.
func decomposePosition(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 0 || len(o.Out) != 1 || c.width(o.Out[0]) != 1 {
		return nil, fmt.Errorf("a position of %d inputs into %d outputs", len(o.In), len(o.Out))
	}
	U := ir.U32
	R := c.Step.Rows
	var x exprs
	p := x.load(0, x.op(OpAdd, U, x.op(OpMul, U, x.coord(0), x.u(rowWords)), x.u(rowWrite)))
	if o.Attr.Offset != 0 {
		p = x.op(OpAdd, U, p, x.u(uint32(o.Attr.Offset)))
	}
	if o.Attr.Span > 1 {
		p = x.op(OpDiv, U, p, x.u(uint32(o.Attr.Span)))
	}
	x.un(OpCvtF32, ir.F32, p)
	return []Prim{{Kind: PrimMap, Shape: []int{R, 1}, In: []Operand{TableRows.op()}, Out: c.value(o.Out[0]), Expr: x.e,
		Facts: Facts{Extents: map[int]int{0: R * rowWords}}}}, nil
}

// pages bounds the pages of the history a step reads: every history's
// whole, at the step's longest.
func (c *Context) pages() int {
	return c.Step.Histories() * ((c.Step.Keys + c.Step.Page - 1) / c.Step.Page)
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
// table's start, and the table holds the pages in order. The windows'
// descriptors are laid out as the rows', and a step states no window, so a
// windowed attention is priced over its whole history: an upper bound.
func historyWord(o Operand, w int, s Step) (float64, bool) {
	if o.Kind == OperandTable && o.Table == TablePages {
		return float64(w), true
	}
	if o.Kind != TableRows.op().Kind || o.Name != TableRows.op().Name && o.Name != TableWindows.op().Name {
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
