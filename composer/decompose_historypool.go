package composer

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/graph"
)

// decomposeHistoryPool is OpHistoryPool: for each row and channel, over
// every segment's Span positions of the window holding the row's position
// (or the one before it), the scores gate plus bias -- a position before
// the history's first or past the row's own no term, read at the row's own
// position so every read is of a page the row holds -- their maximum, the
// weights exp(score - maximum) and their sum, the weighted values, and the
// channel their sum over the weights' sum.
func decomposeHistoryPool(o graph.Op, c *Context) ([]Prim, error) {
	undo, err := c.onHistory(o)
	if err != nil {
		return nil, err
	}
	defer undo()
	a := o.Attr
	if len(o.Out) != 1 || len(o.Weight) != 1 || a.Span < 1 || len(a.Words) < 4 || len(a.Words)%4 != 0 || c.Step.Page <= 0 {
		return nil, fmt.Errorf("a history pool of windows of %d over %d segment words", a.Span, len(a.Words))
	}
	if c.History.Period > 1 {
		return nil, fmt.Errorf("a history pool over entries of a history of one each %d positions", c.History.Period)
	}
	if err := c.History.Keys.check(a.Width); err != nil {
		return nil, err
	}
	U, F := ir.U32, ir.F32
	R, W, P, S := c.Step.Rows, c.width(o.Out[0]), a.Span, len(a.Words)/4
	shape, rw := []int{R, W, S * P}, []int{R, W}
	minus := math.Float32frombits(0xff7fffff)
	// The bias is a block's vector (the reader copies it out), Span rows of
	// the binding's K.
	bw := o.Weight[0].K
	bias := Operand{Kind: OperandVector, Role: o.Weight[0].Role, Name: o.Weight[0].Role.String()}
	hs := history{pageSize: c.Step.Page}
	pool := c.History.Keys.operand(c.tab(TableKeys))
	tmp := func(name string) Operand {
		return Operand{Kind: OperandScratch, Name: c.value(o.Out[0]).Name + "." + name, Temp: true}
	}
	scores, maxes, weights, sums, num := tmp("scores"), tmp("max"), tmp("weights"), tmp("sum"), tmp("num")
	// word is word w of the term's segment, a constant the segment chooses.
	word := func(x *exprs, w int) int {
		seg := x.op(OpDiv, U, x.coord(2), x.u(uint32(P)))
		v := x.u(uint32(a.Words[4*(S-1)+w]))
		for s := S - 2; s >= 0; s-- {
			v = x.sel(U, x.op(OpLess, ir.Pred, seg, x.u(uint32(s+1))), x.u(uint32(a.Words[4*s+w])), v)
		}
		return v
	}
	// term is the term's position, read at the row's own where it is none,
	// whether it is one, and its offset in its window.
	term := func(x *exprs) (pos, ok, j int) {
		t := hs.row(x, rowWrite)
		j = x.op(OpRem, U, x.coord(2), x.u(uint32(P)))
		back := x.op(OpMul, U, word(x, 0), x.u(uint32(P)))
		ws := x.op(OpMul, U, x.op(OpDiv, U, t, x.u(uint32(P))), x.u(uint32(P)))
		p := x.op(OpAdd, U, x.op(OpSub, U, ws, back), j)
		one := x.sel(U, x.op(OpLess, ir.Pred, ws, back), x.u(0), x.sel(U, x.op(OpLess, ir.Pred, t, p), x.u(0), x.u(1)))
		ok = x.op(OpLess, ir.Pred, x.u(0), one)
		return x.sel(U, ok, p, t), ok, j
	}
	facts := Facts{Extents: map[int]int{1: R * rowWords, 2: c.pages()}}

	var sx exprs
	pos, ok, j := term(&sx)
	pid, off := hs.page(&sx, pos)
	gate := c.poolRead(&sx, 0, c.History.Keys, a.Width, pid, off, sx.op(OpAdd, U, word(&sx, 2), sx.coord(1)))
	b := sx.at(3, []int{P * bw}, sx.op(OpAdd, U, sx.op(OpMul, U, j, sx.u(uint32(bw))), sx.op(OpAdd, U, word(&sx, 3), sx.coord(1))))
	sx.sel(F, ok, sx.op(OpAdd, F, gate, b), sx.f(minus))

	var mx exprs
	mx.at(0, shape, mx.coord(0), mx.coord(1), mx.coord(2))

	var wx exprs
	wx.un(OpExp, F, wx.op(OpSub, F, wx.at(0, shape, wx.coord(0), wx.coord(1), wx.coord(2)), wx.at(1, rw, wx.coord(0), wx.coord(1))))

	var lx exprs
	lx.at(0, shape, lx.coord(0), lx.coord(1), lx.coord(2))

	var vx exprs
	vpos, _, _ := term(&vx)
	vpid, voff := hs.page(&vx, vpos)
	val := c.poolRead(&vx, 3, c.History.Keys, a.Width, vpid, voff, vx.op(OpAdd, U, word(&vx, 1), vx.coord(1)))
	vx.op(OpMul, F, vx.at(0, shape, vx.coord(0), vx.coord(1), vx.coord(2)), val)

	var ox exprs
	ox.op(OpDiv, F, ox.at(0, rw, ox.coord(0), ox.coord(1)), ox.at(1, rw, ox.coord(0), ox.coord(1)))

	return []Prim{
		{Kind: PrimMap, Shape: shape, In: []Operand{pool, TableRows.op(), c.tab(TablePages), bias}, Out: scores, Expr: sx.e, Facts: facts},
		{Kind: PrimReduce, Shape: shape, In: []Operand{scores}, Out: maxes, Combine: OpMax, Init: int64(math.Float32bits(minus)), Expr: mx.e},
		{Kind: PrimMap, Shape: shape, In: []Operand{scores, maxes}, Out: weights, Expr: wx.e},
		{Kind: PrimReduce, Shape: shape, In: []Operand{weights}, Out: sums, Combine: OpAdd, Expr: lx.e},
		{Kind: PrimContract, Shape: shape, In: []Operand{weights, TableRows.op(), c.tab(TablePages), pool}, Out: num, Expr: vx.e, Facts: facts},
		{Kind: PrimMap, Shape: rw, In: []Operand{num, sums}, Out: c.value(o.Out[0]), Expr: ox.e},
	}, nil
}
