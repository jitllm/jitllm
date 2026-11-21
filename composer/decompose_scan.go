package composer

import (
	"fmt"
	"math/bits"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/graph"
)

// A part's recurrent state is a pool of slots, a sequence's state a slot:
// the scan's ("rec.state", for each head its rows by its columns, head
// after head) and the convolution's window ("rec.conv"). A step reads a
// slot of each and writes a slot of its ".out" (the same storage, or the
// pool's other half), the slots the step's recurrent descriptor names
// ("rec.rows": the rows, then the slot read and the slot written).

// slotLen is a scan's state floats a sequence: its n, and the tail past
// them where the op states one (graph.OpStateScan's Span).
func slotLen(o graph.Op, n int) int { return max(n, o.Attr.Span) }

// recSlot is the first element of the slot the step's descriptor (operand
// j) names at word w, of slots n elements long.
func (x *exprs) recSlot(j, w, n int) int {
	return x.op(OpMul, ir.U32, x.load(j, x.u(uint32(w))), x.u(uint32(n)))
}

// groupHead is head h's group of H heads over KH groups: h over the heads
// a group serves.
func (x *exprs) groupHead(h, H, KH int) int {
	U := ir.U32
	if KH == 1 {
		return x.u(0)
	}
	if rep := H / KH; rep&(rep-1) == 0 {
		return x.op(OpShr, U, h, x.u(uint32(bits.TrailingZeros(uint(rep)))))
	}
	return x.op(OpDiv, U, h, x.u(uint32(H/KH)))
}

// cycleHead is head h's place in a cycle of KH: h mod KH.
func (x *exprs) cycleHead(h, KH int) int {
	U := ir.U32
	if KH == 1 {
		return x.u(0)
	}
	if KH&(KH-1) == 0 {
		return x.op(OpAnd, U, h, x.u(uint32(KH-1)))
	}
	return x.op(OpRem, U, h, x.u(uint32(KH)))
}

// heads is a scan's head h's group and cycle, each formed when first read.
type heads struct {
	x               *exprs
	h, H, KH        int
	group, cycle    int
	grouped, cycled bool
}

func (k *heads) of(m graph.ScanAxis) int {
	if m&graph.ScanCycle != 0 {
		if !k.cycled {
			k.cycle, k.cycled = k.x.cycleHead(k.h, k.KH), true
		}
		return k.cycle
	}
	if !k.grouped {
		k.group, k.grouped = k.x.groupHead(k.h, k.H, k.KH), true
	}
	return k.group
}

// scanShape is a scan's state: H heads in KH groups, each Dv rows by Dk
// columns, N floats.
type scanShape struct{ H, KH, Dv, Dk, N int }

// operand is input j, along axes m, at row r of rows and state element
// (h, v, i), h's group or cycle as kh gives it.
func (s scanShape) operand(x *exprs, j int, m graph.ScanAxis, rows, r, h, v, i int, kh *heads) int {
	ext, ax := []int{rows}, []int{r}
	if m&graph.ScanHead != 0 {
		ext, ax = append(ext, s.H), append(ax, h)
	}
	if m&(graph.ScanGroup|graph.ScanCycle) != 0 {
		ext, ax = append(ext, s.KH), append(ax, kh.of(m))
	}
	if m&graph.ScanRow != 0 {
		ext, ax = append(ext, s.Dv), append(ax, v)
	}
	if m&graph.ScanCol != 0 {
		ext, ax = append(ext, s.Dk), append(ax, i)
	}
	return x.at(j, ext, ax...)
}

// scanTerms evaluates a scan's terms into x, each once: operand k by in,
// the state before the row by state, a term summed over the state's row by
// sum, and a term another primitive holds by held.
type scanTerms struct {
	x     *exprs
	o     graph.Op
	in    func(k int) int
	state func() int
	sum   func(t int) (int, error)
	held  func(t int) (int, bool)
	vals  map[int]int
}

func (e *scanTerms) term(t int) (int, error) {
	if v, ok := e.vals[t]; ok {
		return v, nil
	}
	if v, ok := e.held(t); ok {
		e.vals[t] = v
		return v, nil
	}
	tm := e.o.Attr.Expr[t]
	args := make([]int, tm.Fn.Arity())
	for a := range args {
		g := tm.Args[a]
		var err error
		switch g.Kind {
		case graph.ArgIn:
			args[a] = e.in(g.Index)
		case graph.ArgConst:
			args[a] = e.x.f(float32(g.Const))
		case graph.ArgTerm:
			args[a], err = e.term(g.Index)
		case graph.ArgState:
			args[a] = e.state()
		case graph.ArgSum:
			args[a], err = e.sum(g.Index)
		default:
			return 0, fmt.Errorf("a scan term reading an operand of kind %d", g.Kind)
		}
		if err != nil {
			return 0, err
		}
	}
	v, err := e.x.mapFn(tm.Fn, args, e.o.Attr.Exact)
	if err != nil {
		return 0, err
	}
	e.vals[t] = v
	return v, nil
}

// decomposeStateScan is the op's recurrence over the part's state: a scan
// over a step of several rows (or one whose states are kept), and over one
// row its terms staged.
func decomposeStateScan(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	if len(o.Out) != 1 || len(o.In) != len(a.Words) || len(a.Expr) < 2 || a.Update < 0 || a.Update >= len(a.Expr)-1 || c.Step.Rows < 1 {
		return nil, fmt.Errorf("a scan of %d operands, %d terms, over %d rows", len(o.In), len(a.Expr), c.Step.Rows)
	}
	s := scanShape{H: a.NHead, KH: a.NKVHead, Dk: a.HeadDim}
	if s.H < 1 || s.KH < 1 || s.Dk < 1 || s.H%s.KH != 0 || c.width(o.Out[0])%s.H != 0 {
		return nil, fmt.Errorf("a scan of %d heads over %d groups of %d", s.H, s.KH, s.Dk)
	}
	s.Dv = c.width(o.Out[0]) / s.H
	s.N = s.H * s.Dv * s.Dk
	for k, m := range a.Words {
		if w, ok := graph.ScanWidth(graph.ScanAxis(m), s.H, s.KH, s.Dv, s.Dk); !ok || w != c.width(o.In[k]) {
			return nil, fmt.Errorf("a scan's operand %d of %d along axes %#x", k, c.width(o.In[k]), m)
		}
	}
	if c.Step.Rows > 1 || a.Keeps {
		return scanRows(o, c, s)
	}
	return scanRow(o, c, s)
}

// scanRow is the scan at one row, its terms staged: each term summed over
// the state's row a reduction into a scratch row, the state after the row
// a map into a scratch state, the readout a reduction of it, then the
// scratch into the slot written, so no kernel reads the state while
// writing it.
func scanRow(o graph.Op, c *Context, s scanShape) ([]Prim, error) {
	U := ir.U32
	a := o.Attr
	sh := []int{1, s.H, s.Dv, s.Dk}
	name := c.value(o.Out[0]).Name
	next := Operand{Kind: OperandScratch, Name: name + ".state"}
	sums := map[int]Operand{}
	held := false

	// stage is term t's expression over the state's elements, its
	// operands and what it knows of their extents.
	stage := func(t int) (exprs, []Operand, Facts, error) {
		var x exprs
		var in []Operand
		at := map[string]int{}
		use := func(key string, op Operand) int {
			if j, ok := at[key]; ok {
				return j
			}
			at[key] = len(in)
			in = append(in, op)
			return len(in) - 1
		}
		r, h, j, i := x.coord(0), x.coord(1), x.coord(2), x.coord(3)
		group := &heads{x: &x, h: h, H: s.H, KH: s.KH}
		var f Facts
		st := -1
		e := &scanTerms{x: &x, o: o, vals: map[int]int{}}
		e.in = func(k int) int {
			return s.operand(&x, use(fmt.Sprint("in", k), c.value(o.In[k])), graph.ScanAxis(a.Words[k]), 1, r, h, j, i, group)
		}
		e.state = func() int {
			if st < 0 {
				tab, rows := use("state", TableRecState.op()), use("rows", TableRecRows.op())
				f = Facts{Extents: map[int]int{tab: s.N, rows: 3}}
				flat := x.op(OpAdd, U, x.op(OpMul, U, x.op(OpAdd, U, x.op(OpMul, U, h, x.u(uint32(s.Dv))), j), x.u(uint32(s.Dk))), i)
				st = x.load(tab, x.op(OpAdd, U, x.recSlot(rows, 1, slotLen(o, s.N)), flat))
			}
			return st
		}
		e.sum = func(u int) (int, error) {
			op, ok := sums[u]
			if !ok {
				return 0, fmt.Errorf("a scan's term %d summed before its reduction", u)
			}
			return x.at(use(fmt.Sprint("sum", u), op), []int{1, s.H, s.Dv}, r, h, j), nil
		}
		e.held = func(u int) (int, bool) {
			if u != a.Update || !held {
				return 0, false
			}
			return x.at(use("next", next), sh, r, h, j, i), true
		}
		if _, err := e.term(t); err != nil {
			return exprs{}, nil, Facts{}, err
		}
		return x, in, f, nil
	}

	summed := map[int]bool{}
	for _, t := range a.Expr {
		for _, g := range t.Args {
			if g.Kind == graph.ArgSum {
				summed[g.Index] = true
			}
		}
	}
	var ps []Prim
	for t := range len(a.Expr) - 1 {
		if summed[t] {
			x, in, f, err := stage(t)
			if err != nil {
				return nil, err
			}
			op := Operand{Kind: OperandScratch, Name: fmt.Sprintf("%s.sum%d", name, t)}
			ps = append(ps, Prim{Kind: PrimReduce, Shape: sh, In: in, Out: op, Combine: OpAdd, Expr: x.e, Facts: f})
			sums[t] = op
		}
		if t == a.Update {
			x, in, f, err := stage(t)
			if err != nil {
				return nil, err
			}
			ps = append(ps, Prim{Kind: PrimMap, Shape: sh, In: in, Out: next, Expr: x.e, Facts: f})
			held = true
		}
	}
	x, in, f, err := stage(len(a.Expr) - 1)
	if err != nil {
		return nil, err
	}
	ps = append(ps, Prim{Kind: PrimReduce, Shape: sh, In: in, Out: c.value(o.Out[0]), Combine: OpAdd, Expr: x.e, Facts: f})

	var w exprs
	ix := w.op(OpAdd, U, w.op(OpMul, U, w.coord(0), w.u(uint32(s.Dk))), w.coord(1))
	val := w.load(0, ix)
	to := w.op(OpAdd, U, w.recSlot(1, 2, slotLen(o, s.N)), ix)
	whole := Facts{Extents: map[int]int{0: s.N, 1: 3}, Whole: true}
	return append(ps, Prim{Kind: PrimMap, Shape: []int{s.H * s.Dv, s.Dk}, In: []Operand{next, TableRecRows.op()}, Expr: w.e, Facts: whole,
		Stores: []PrimStore{{Out: TableRecStateOut.op(), Index: to, Value: val, Extent: s.N}}}), nil
}

// scanRows is the scan over a step of several rows: its state rows the
// heads' rows, each stepped through its run's rows in order -- a prompt
// chunk's rows one run, a ragged step's sequences a run each -- its run's
// slot read at the start and its slot written after the last row. A term
// summed over the state's row is a group sum across the state's columns.
func scanRows(o graph.Op, c *Context, s scanShape) ([]Prim, error) {
	U, F := ir.U32, ir.F32
	a := o.Attr
	R := c.Step.Rows
	n := len(o.In)
	in := make([]Operand, 0, n+2)
	for _, v := range o.In {
		in = append(in, c.value(v))
	}
	tab, desc := n, n+1
	if c.scanLoops {
		// The state is read where it is written, each run's start in its
		// slot read of that table.
		in = append(in, TableRecStateOut.op(), TableRecRows.op())
	} else {
		in = append(in, TableRecState.op(), TableRecRows.op())
	}
	var x exprs
	h, j, i, r, run := x.coord(0), x.coord(1), x.coord(2), x.coord(3), x.coord(4)
	group := &heads{x: &x, h: h, H: s.H, KH: s.KH}
	start := x.op(OpMul, U, x.op(OpAdd, U, x.op(OpMul, U, h, x.u(uint32(s.Dv))), j), x.u(uint32(s.Dk)))
	place := x.op(OpAdd, U, start, i)
	slot := func(w int) int {
		at := x.op(OpAdd, U, x.op(OpAdd, U, run, run), x.u(uint32(w)))
		base := x.op(OpMul, U, x.load(desc, at), x.u(uint32(slotLen(o, s.N))))
		if c.scanLoops {
			// The component added last: a loop of the components walks
			// consecutive elements (Vectorize).
			return x.op(OpAdd, U, x.op(OpAdd, U, base, start), i)
		}
		return x.op(OpAdd, U, base, place)
	}
	from := slot(1)
	init := x.load(tab, from)
	final := slot(2)
	st := -1
	sums := map[int]int{}
	e := &scanTerms{x: &x, o: o, vals: map[int]int{}, held: func(int) (int, bool) { return 0, false }}
	e.in = func(k int) int {
		return s.operand(&x, k, graph.ScanAxis(a.Words[k]), R, r, h, j, i, group)
	}
	e.state = func() int {
		if st < 0 {
			st = x.add(Expr{Op: ExprState, Type: F})
		}
		return st
	}
	e.sum = func(u int) (int, error) {
		if v, ok := sums[u]; ok {
			return v, nil
		}
		v, err := e.term(u)
		if err != nil {
			return 0, err
		}
		sums[u] = x.un(OpGroupSum, F, v)
		return sums[u], nil
	}
	next, err := e.term(a.Update)
	if err != nil {
		return nil, err
	}
	last, err := e.term(len(a.Expr) - 1)
	if err != nil {
		return nil, err
	}
	y := x.un(OpGroupSum, F, last)
	sc := &Scan{Desc: desc, Init: init, Next: next, Final: TableRecStateOut.op(), Index: final, Extent: s.N, Loops: c.scanLoops, From: from}
	if a.Keeps {
		sc.Keep, sc.KeepIx, sc.Stride = TableRecKeep.op(), place, slotLen(o, s.N)
	}
	return []Prim{{Kind: PrimScan, Shape: []int{s.H, s.Dv, s.Dk}, In: in, Expr: x.e,
		Stores: []PrimStore{{Out: c.value(o.Out[0]), Ax: []int{r, h, j}, Ext: []int{R, s.H, s.Dv}, Value: y}},
		Scan:   sc,
		Facts:  Facts{Extents: map[int]int{tab: s.N, desc: RunDescWords(R)}}}}, nil
}

// RunDescWords is how many words a run descriptor of rows rows holds: the
// row count, each run's slots, its first entry and rows, the order list,
// each row's entry and run (Scan).
func RunDescWords(rows int) int { return 1 + 7*rows }
