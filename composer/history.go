package composer

import (
	"fmt"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/graph"
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
// ValuesInKeys is one pool for both: a position's value row is its key row,
// held once in the keys' pool, and an append whose value is not its key is
// refused.
type History struct {
	Keys, Values Pool
	ValuesInKeys bool
	// Passes reads the history in passes over the rows' descriptors, which
	// the call site clips to each pass: the attention folds each pass into
	// the running partial "run.in" -> "run.out" and finishes from the last.
	Passes bool
	// Period, past 1, keeps one entry each Period positions (an append's
	// Span): a page holds the entries its positions close, entry j at slot
	// ((j+1)*Period-1)%page/Period of the page of position (j+1)*Period-1,
	// and one slot past them that a row closing none writes zeros into.
	Period int
}

// slots is a page's slots of a history laid out as h, its pages of page
// positions.
func (h History) slots(page int) int {
	if h.Period <= 1 {
		return page
	}
	return (page+h.Period-1)/h.Period + 1
}

// slots is a page's slots of the history the op being decomposed reads.
func (c *Context) slots() int { return c.History.slots(c.Step.Page) }

// values is the pool history n's values are read from, its operand and its
// layout.
func (h History) values(n uint8) (Operand, Pool) {
	if h.ValuesInKeys {
		return h.Keys.operand(TableKeys.of(int(n))), h.Keys
	}
	return h.Values.operand(TableValues.of(int(n))), h.Values
}

// historyOf is the history an op reads or writes (its first, a block's ops
// touch one history each) and that history's layout.
func (c *Context) historyOf(o graph.Op) (uint8, History, error) {
	var h uint8
	switch {
	case len(o.Reads) > 0:
		h = o.Reads[0].Hist
	case len(o.Writes) > 0:
		h = o.Writes[0].Hist
	}
	l, err := c.layoutOf(h)
	return h, l, err
}

// layoutOf is the layout of the block's history h: the request's History
// for 0, More[h-1] past it.
func (c *Context) layoutOf(h uint8) (History, error) {
	switch {
	case h == 0 && c.hist == 0:
		return c.History, nil
	case h == 0:
		return c.base, nil
	case int(h) > len(c.More):
		return History{}, fmt.Errorf("history %d of a request laying out %d", h, 1+len(c.More))
	}
	return c.More[h-1], nil
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
	return c.poolReadIn(x, j, l, c.slots(), w, pid, off, d)
}

// poolReadIn is poolRead of a pool of pg slots a page.
func (c *Context) poolReadIn(x *exprs, j int, l Pool, pg, w, pid, off, d int) int {
	U := ir.U32
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

// onHistory makes o's history the one the context reads -- its layout as
// c.History, its tables at its instance (tab) -- until undo restores the
// block's.
func (c *Context) onHistory(o graph.Op) (undo func(), err error) {
	h, l, err := c.historyOf(o)
	if err != nil {
		return nil, err
	}
	was, wasH := c.History, c.hist
	if c.hist == 0 {
		c.base = c.History
	}
	c.History, c.hist = l, h
	return func() { c.History, c.hist = was, wasH }, nil
}

// tab is table t of the history the op being decomposed reads.
func (c *Context) tab(t Table) Operand { return t.of(int(c.hist)) }
