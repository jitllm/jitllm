package composer

import (
	"fmt"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/graph"
)

// gatherWord is the request word each of an index's request operands reads.
func gatherWord(k int) (uint32, bool) {
	switch k {
	case graph.GatherGridW:
		return RequestGrid, true
	case graph.GatherGridH:
		return RequestGridH, true
	case graph.GatherMark:
		return RequestMark, true
	case graph.GatherMarkN:
		return RequestMarkN, true
	case graph.GatherTrail:
		return RequestTrail, true
	case graph.GatherTrailN:
		return RequestTrailN, true
	case graph.GatherSamples:
		return RequestPicW, true
	}
	return 0, false
}

// indexer is an integer expression's terms (graph.Index) as a primitive's
// expressions over its inputs: the step's tables appended as an operand
// reads them, each term once, in order.
type indexer struct {
	x        *exprs
	in       *[]Operand
	facts    *Facts
	terms    []graph.Term
	vals     []int
	preds    []int
	done     int
	pos, req int
	reach    int
	// row and col are the element's row and column, -1 where an operand
	// may not read them; outRows and inRows the output's and the input's.
	row, col        int
	outRows, inRows int
}

func newIndexer(x *exprs, in *[]Operand, facts *Facts, terms []graph.Term, reach int) *indexer {
	return &indexer{x: x, in: in, facts: facts, terms: terms, vals: make([]int, len(terms)), preds: make([]int, len(terms)),
		pos: -1, req: -1, reach: reach, row: -1, col: -1}
}

func (ix *indexer) table(at *int, op Operand) int {
	if *at < 0 {
		*at = len(*ix.in)
		*ix.in = append(*ix.in, op)
	}
	return *at
}

func (ix *indexer) operand(k int) (int, error) {
	x := ix.x
	u := func(v int) int { return x.u(uint32(v)) }
	switch k {
	case graph.GatherRow, graph.GatherCol:
		v := ix.row
		if k == graph.GatherCol {
			v = ix.col
		}
		if v < 0 {
			return 0, fmt.Errorf("an index reads operand %d, which it has none of here", k)
		}
		return v, nil
	case graph.GatherOutRows:
		return u(ix.outRows), nil
	case graph.GatherInRows:
		return u(ix.inRows), nil
	case graph.GatherLive, graph.GatherPos:
		j := ix.table(&ix.pos, TablePos.op())
		ix.facts.Extents[j] = ix.reach + 1
		if k == graph.GatherLive {
			return x.load(j, u(0)), nil
		}
		if ix.row < 0 {
			return 0, fmt.Errorf("an index reads a row's position with no row")
		}
		return x.load(j, x.op(OpAdd, ir.U32, ix.row, u(1))), nil
	}
	if w, ok := gatherWord(k); ok {
		j := ix.table(&ix.req, TableRequest.op())
		ix.facts.Extents[j] = max(ix.facts.Extents[j], int(w)+1)
		return x.load(j, u(int(w))), nil
	}
	return 0, fmt.Errorf("an index reads operand %d", k)
}

// eval is the terms up to upto, each once; a Less is kept as its predicate
// too, for a Where it decides.
func (ix *indexer) eval(upto int) error {
	x, U := ix.x, ir.U32
	u := func(v int) int { return x.u(uint32(v)) }
	for ; ix.done < upto; ix.done++ {
		k, t := ix.done, ix.terms[ix.done]
		var args [3]int
		for j := range t.Fn.Arity() {
			switch g := t.Args[j]; g.Kind {
			case graph.ArgIn:
				v, err := ix.operand(g.Index)
				if err != nil {
					return err
				}
				args[j] = v
			case graph.ArgConst:
				args[j] = u(int(g.Const))
			case graph.ArgTerm:
				if g.Index < 0 || g.Index >= k {
					return fmt.Errorf("an index's term %d reads term %d", k, g.Index)
				}
				args[j] = ix.vals[g.Index]
			default:
				return fmt.Errorf("an index's term %d has operand kind %d", k, g.Kind)
			}
		}
		ix.preds[k] = -1
		switch t.Fn {
		case graph.MapAdd:
			ix.vals[k] = x.op(OpAdd, U, args[0], args[1])
		case graph.MapSub:
			ix.vals[k] = x.op(OpSub, U, args[0], args[1])
		case graph.MapMul:
			ix.vals[k] = x.op(OpMul, U, args[0], args[1])
		case graph.MapDiv:
			ix.vals[k] = x.op(OpDiv, U, args[0], args[1])
		case graph.MapMax:
			ix.vals[k] = x.op(OpMax, U, args[0], args[1])
		case graph.MapMin:
			ix.vals[k] = x.op(OpMin, U, args[0], args[1])
		case graph.MapLess:
			ix.preds[k] = x.op(OpLess, ir.Pred, args[0], args[1])
			ix.vals[k] = x.sel(U, ix.preds[k], u(1), u(0))
		case graph.MapWhere:
			p := -1
			if g := t.Args[0]; g.Kind == graph.ArgTerm {
				p = ix.preds[g.Index]
			}
			if p < 0 {
				p = x.op(OpLess, ir.Pred, u(0), args[0])
			}
			ix.vals[k] = x.sel(U, p, args[1], args[2])
		default:
			return fmt.Errorf("an index's term %d is %v, not an index's function", k, t.Fn)
		}
	}
	return nil
}

// last is the expression's value: its last term's.
func (ix *indexer) last() (int, error) {
	if err := ix.eval(len(ix.terms)); err != nil {
		return 0, err
	}
	return ix.vals[len(ix.terms)-1], nil
}

func factsOf(f Facts) Facts {
	if len(f.Extents) == 0 {
		return Facts{}
	}
	return f
}

// decomposeGatherRows is each output row the row of its sources -- the
// input's rows, then each table's -- the op's index expression names, zero
// past them (graph.OpGatherRows): one map over the output's rows and width.
// Where the op bounds the rows written, a row past the bound is the last,
// recomputed and stored again.
func decomposeGatherRows(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	if len(o.In) > 1 || len(o.Out) != 1 || len(a.Expr) == 0 || a.Span < 0 || a.Span > len(a.Expr) {
		return nil, fmt.Errorf("a row gather of %d operands, %d terms bounded at %d", len(o.In), len(a.Expr), a.Span)
	}
	U, F := ir.U32, ir.F32
	sh := c.rows(o.Out[0])
	R, E := sh[0], sh[1]
	var x exprs
	u := func(v int) int { return x.u(uint32(v)) }
	var in []Operand
	facts := Facts{Extents: map[int]int{}}
	nIn := 0
	if len(o.In) == 1 {
		if c.width(o.In[0]) != E {
			return nil, fmt.Errorf("a row gather of rows %d wide into rows %d wide", c.width(o.In[0]), E)
		}
		nIn = c.rows(o.In[0])[0]
		in = append(in, c.value(o.In[0]))
	}
	tables := 0
	for _, w := range o.Weight {
		tables += max(w.Rows, 1)
	}
	if err := graph.GatherReads(a, R, nIn, tables); err != nil {
		return nil, fmt.Errorf("a row gather %w", err)
	}
	ix := newIndexer(&x, &in, &facts, a.Expr, max(R, nIn, c.Step.Rows))
	ix.outRows, ix.inRows = R, nIn
	r, col := x.coord(0), x.coord(1)
	row := r
	if a.Span > 0 {
		if err := ix.eval(a.Span); err != nil {
			return nil, err
		}
		row = x.op(OpMin, U, r, x.op(OpMin, U, ix.vals[a.Span-1], u(R-1)))
	}
	ix.row = row
	s, err := ix.last()
	if err != nil {
		return nil, err
	}
	// The sources end to end: each its rows' value where s falls in them,
	// the next's past, zero past the last.
	type source struct {
		lo, n int
		read  func(at int) (int, error)
	}
	var srcs []source
	if nIn > 0 {
		srcs = append(srcs, source{0, nIn, func(at int) (int, error) { return x.at(0, []int{nIn, E}, at, col), nil }})
	}
	lo := nIn
	for _, w := range o.Weight {
		if w.K != E {
			return nil, fmt.Errorf("a row gather of rows %d wide from a table %d wide", E, w.K)
		}
		n := max(w.Rows, 1)
		var read func(int) (int, error)
		if w.Rows == 0 || w.Role.Expanded() {
			j := len(in)
			in = append(in, Operand{Kind: OperandVector, Role: w.Role, Name: w.Role.String()})
			read = func(at int) (int, error) { return x.at(j, []int{n, E}, at, col), nil }
		} else {
			wo, err := c.weight(w)
			if err != nil {
				return nil, err
			}
			read = func(at int) (int, error) {
				v, err := weightElement(&x, wo, n, E, &in, &facts, at, col)
				if err != nil {
					return 0, fmt.Errorf("a gathered table of %s: %w", wo.Format, err)
				}
				return v, nil
			}
		}
		srcs = append(srcs, source{lo, n, read})
		lo += n
	}
	v := x.f(0)
	for k := len(srcs) - 1; k >= 0; k-- {
		sc := srcs[k]
		at := x.op(OpMin, U, x.op(OpSub, U, x.op(OpMax, U, s, u(sc.lo)), u(sc.lo)), u(sc.n-1))
		if sc.lo == 0 {
			at = x.op(OpMin, U, s, u(sc.n-1))
		}
		rv, err := sc.read(at)
		if err != nil {
			return nil, err
		}
		v = x.sel(F, x.op(OpLess, ir.Pred, s, u(sc.lo+sc.n)), rv, v)
	}
	out := c.value(o.Out[0])
	p := Prim{Kind: PrimMap, Shape: sh, In: in, Out: out, Expr: x.e, Facts: factsOf(facts)}
	if a.Span > 0 {
		p.Stores = []PrimStore{{Out: out, Ax: []int{row, col}, Ext: sh, Value: v}}
	}
	return []Prim{p}, nil
}

// decomposeGatherElems is each output element the element of the input,
// its rows end to end, the op's index expression names, zero past them
// (graph.OpGatherElems): one map over the output's rows and width.
func decomposeGatherElems(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 || len(o.Attr.Expr) == 0 {
		return nil, fmt.Errorf("an element gather of %d operands, %d terms", len(o.In), len(o.Attr.Expr))
	}
	U, F := ir.U32, ir.F32
	sh, ish := c.rows(o.Out[0]), c.rows(o.In[0])
	n := ish[0] * ish[1]
	var x exprs
	in := []Operand{c.value(o.In[0])}
	facts := Facts{Extents: map[int]int{0: n}}
	ix := newIndexer(&x, &in, &facts, o.Attr.Expr, max(sh[0], ish[0], c.Step.Rows))
	ix.outRows, ix.inRows = sh[0], ish[0]
	ix.row, ix.col = x.coord(0), x.coord(1)
	s, err := ix.last()
	if err != nil {
		return nil, err
	}
	v := x.load(0, x.op(OpMin, U, s, x.u(uint32(n-1))))
	x.sel(F, x.op(OpLess, ir.Pred, s, x.u(uint32(n))), v, x.f(0))
	return []Prim{{Kind: PrimMap, Shape: sh, In: in, Out: c.value(o.Out[0]), Expr: x.e, Facts: facts}}, nil
}

// decomposeMeanRows is each column's sum over the input's first n rows,
// n the op's expression, over max(n - Const, 1) (graph.OpMeanRows): the sum
// over every row, the rows from n on none, then the quotient.
func decomposeMeanRows(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	if len(o.In) != 1 || len(o.Out) != 1 || len(a.Expr) == 0 || a.Const < 0 || c.width(o.In[0]) != c.width(o.Out[0]) {
		return nil, fmt.Errorf("a mean over rows of %d operands, %d terms", len(o.In), len(a.Expr))
	}
	U, F := ir.U32, ir.F32
	sh := c.rows(o.In[0])
	R, W := sh[0], sh[1]
	sum := Operand{Kind: OperandScratch, Name: c.value(o.Out[0]).Name + ".sum", Temp: true}
	count := func(x *exprs, in *[]Operand, facts *Facts) (int, error) {
		ix := newIndexer(x, in, facts, a.Expr, max(R, c.Step.Rows))
		ix.outRows, ix.inRows = 1, R
		return ix.last()
	}
	var sx exprs
	sin := []Operand{c.value(o.In[0])}
	sf := Facts{Extents: map[int]int{}}
	{
		col, r := sx.coord(0), sx.coord(1)
		v := sx.at(0, sh, r, col)
		n, err := count(&sx, &sin, &sf)
		if err != nil {
			return nil, err
		}
		sx.sel(F, sx.op(OpLess, ir.Pred, r, n), v, sx.f(0))
	}
	var mx exprs
	mIn := []Operand{sum}
	mf := Facts{Extents: map[int]int{}}
	{
		col := mx.coord(1)
		s := mx.at(0, []int{W}, col)
		n, err := count(&mx, &mIn, &mf)
		if err != nil {
			return nil, err
		}
		d := mx.op(OpMax, U, mx.op(OpSub, U, mx.op(OpMax, U, n, mx.u(uint32(a.Const))), mx.u(uint32(a.Const))), mx.u(1))
		mx.op(OpDiv, F, s, mx.un(OpCvtF32, F, mx.un(OpBitcast, ir.I32, d)))
	}
	return []Prim{
		{Kind: PrimReduce, Shape: []int{W, R}, In: sin, Out: sum, Combine: OpAdd, Expr: sx.e, Facts: factsOf(sf)},
		{Kind: PrimMap, Shape: []int{1, W}, In: mIn, Out: c.value(o.Out[0]), Expr: mx.e, Facts: factsOf(mf)},
	}, nil
}
