package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/ir"
)

// Pool is how one history pool lays out a page: ByDim keeps a dimension's
// positions together, [width][page], rather than a position's row,
// [page][width]; Half stores binary16, two halves to a 32-bit word, the
// lower element in the low half.
type Pool struct {
	ByDim, Half bool
}

// History is the layout of the pools a part's history lives in: the call
// site's storage, stated as numbers, which the composer addresses as it is.
type History struct {
	Keys, Values Pool
}

func (l Pool) check(width int) error {
	switch {
	case l.Half && l.ByDim:
		return fmt.Errorf("a binary16 pool laid out by dimension: a word would hold two positions")
	case l.Half && width%2 != 0:
		return fmt.Errorf("a binary16 pool of odd width %d: a word would cross a position", width)
	}
	return nil
}

// operand is pool o as l lays it out: a binary16 pool is words.
func (l Pool) operand(o Operand) Operand {
	if l.Half {
		o.Elem = ir.U32
	}
	return o
}

// poolRead is the float at position off of page pid, element d of a row of
// width w, of the pool operand j lays out as l.
func (c *Context) poolRead(x *exprs, j int, l Pool, w, pid, off, d int) int {
	U, pg := ir.U32, c.Step.Page
	switch {
	case l.Half:
		word := x.at(j, []int{c.pages(), pg, w / 2}, pid, off, x.op(OpShr, U, d, x.u(1)))
		shift := x.op(OpShl, U, x.op(OpAnd, U, d, x.u(1)), x.u(4))
		return x.un(OpWidenHalf, ir.F32, x.op(OpShr, U, word, shift))
	case l.ByDim:
		return x.at(j, []int{c.pages(), w, pg}, pid, d, off)
	}
	return x.at(j, []int{c.pages(), pg, w}, pid, off, d)
}
