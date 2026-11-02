package composer

import (
	"fmt"
	"math/bits"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/engine/graph"
)

// A part's recurrent state is a pool of slots, a sequence's state a slot:
// the delta rule's ("rec.state", for each value head its rows of the
// value's width by the key's, value head after value head) and the
// convolution's window ("rec.conv"). A step reads a slot of each and writes
// a slot of its ".out" (the same storage, or the pool's other half), the
// slots the step's recurrent descriptor names ("rec.rows": the rows, then
// the slot read and the slot written).
var (
	tableRecState    = partTable(TableRecState)
	tableRecStateOut = partTable(TableRecStateOut)
	tableRecConv     = partTable(TableRecConv)
	tableRecConvOut  = partTable(TableRecConvOut)
	tableRecRows     = tableOp(TableRecRows, 0, ir.U32)
)

// slotLen is a delta rule's state floats a sequence: its n, and the tail
// past them where the op states one (graph.OpDeltaRule's Span).
func slotLen(o graph.Op, n int) int { return max(n, o.Attr.Span) }

// recSlot is the first element of the slot the step's descriptor (operand
// j) names at word w, of slots n elements long.
func (x *exprs) recSlot(j, w, n int) int {
	return x.op(OpMul, ir.U32, x.load(j, x.u(uint32(w))), x.u(uint32(n)))
}

// keyHead is value head h's key head of H value heads over KH key heads:
// the one where there is one, h mod KH where the pairing is tiled, else h
// over the value heads a key head serves.
func (x *exprs) keyHead(h, H, KH int, tiled bool) int {
	U := ir.U32
	if KH == 1 {
		return x.u(0)
	}
	if tiled {
		if KH&(KH-1) == 0 {
			return x.op(OpAnd, U, h, x.u(uint32(KH-1)))
		}
		return x.op(OpSub, U, h, x.op(OpMul, U, x.op(OpDiv, U, h, x.u(uint32(KH))), x.u(uint32(KH))))
	}
	if rep := H / KH; rep&(rep-1) == 0 {
		return x.op(OpShr, U, h, x.u(uint32(bits.TrailingZeros(uint(rep)))))
	}
	return x.op(OpDiv, U, h, x.u(uint32(H/KH)))
}

// decomposeDeltaRule is the delta rule over the part's state at one row:
// each state row's decayed dot with its key head's k, the row's (v - that
// dot) times beta (v times beta where the rule is additive), then the decayed state plus k times it into a scratch
// state, its dot with q the output, and the scratch into the slot written,
// so no kernel reads the state while writing it.
func decomposeDeltaRule(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	if len(o.In) != 5 || len(o.Out) != 1 || c.Step.Rows < 1 {
		return nil, fmt.Errorf("a delta rule of %d operands over %d rows", len(o.In), c.Step.Rows)
	}
	U, F := ir.U32, ir.F32
	H, KH, Dk := a.NHead, a.NKVHead, a.HeadDim
	if H < 1 || KH < 1 || Dk < 1 || H%KH != 0 || c.width(o.In[2])%H != 0 {
		return nil, fmt.Errorf("a delta rule of %d value heads over %d key heads of %d", H, KH, Dk)
	}
	Dv := c.width(o.In[2]) / H
	chanDecay := c.width(o.In[3]) == H*Dk
	N := H * Dv * Dk
	if c.Step.Rows > 1 {
		return deltaScan(o, c, H, KH, Dk, Dv, chanDecay)
	}
	sh := []int{1, H, Dv, Dk}
	name := c.value(o.Out[0]).Name
	sk := Operand{Kind: OperandScratch, Name: name + ".sk"}
	delta := Operand{Kind: OperandScratch, Name: name + ".delta"}
	next := Operand{Kind: OperandScratch, Name: name + ".state"}
	q, k, v, decay, beta := c.value(o.In[0]), c.value(o.In[1]), c.value(o.In[2]), c.value(o.In[3]), c.value(o.In[4])

	// keyHead is value head h's key head; flat is the state's element at
	// (h, j, i) of the slot read; g is the decay on the element at (h, i).
	keyHead := func(x *exprs, h int) int { return x.keyHead(h, H, KH, a.Tiled) }
	flat := func(x *exprs, desc, h, j, i int) int {
		at := x.op(OpAdd, U, x.op(OpMul, U, x.op(OpAdd, U, x.op(OpMul, U, h, x.u(uint32(Dv))), j), x.u(uint32(Dk))), i)
		return x.op(OpAdd, U, x.recSlot(desc, 1, slotLen(o, N)), at)
	}
	g := func(x *exprs, at, r, h, i int) int {
		if chanDecay {
			return x.at(at, []int{1, H, Dk}, r, h, i)
		}
		return x.at(at, []int{1, H}, r, h)
	}

	var s exprs
	r, h, j, i := s.coord(0), s.coord(1), s.coord(2), s.coord(3)
	kh := keyHead(&s, h)
	s.op(OpMul, F, s.op(OpMul, F, s.load(0, flat(&s, 3, h, j, i)), g(&s, 2, r, h, i)), s.at(1, []int{1, KH, Dk}, r, kh, i))

	var e exprs
	r, h, j = e.coord(0), e.coord(1), e.coord(2)
	if a.Additive {
		e.op(OpMul, F, e.at(0, []int{1, H, Dv}, r, h, j), e.at(1, []int{1, H}, r, h))
	} else {
		e.op(OpMul, F, e.op(OpSub, F, e.at(0, []int{1, H, Dv}, r, h, j), e.at(1, []int{1, H, Dv}, r, h, j)), e.at(2, []int{1, H}, r, h))
	}

	var u exprs
	r, h, j, i = u.coord(0), u.coord(1), u.coord(2), u.coord(3)
	kh = keyHead(&u, h)
	d := u.at(2, []int{1, H, Dv}, r, h, j)
	u.op(OpAdd, F, u.op(OpMul, F, u.load(0, flat(&u, 4, h, j, i)), g(&u, 3, r, h, i)), u.op(OpMul, F, u.at(1, []int{1, KH, Dk}, r, kh, i), d))

	var y exprs
	r, h, j, i = y.coord(0), y.coord(1), y.coord(2), y.coord(3)
	kh = keyHead(&y, h)
	y.op(OpMul, F, y.at(0, sh, r, h, j, i), y.at(1, []int{1, KH, Dk}, r, kh, i))

	var w exprs
	ix := w.op(OpAdd, U, w.op(OpMul, U, w.coord(0), w.u(uint32(Dk))), w.coord(1))
	val := w.load(0, ix)
	to := w.op(OpAdd, U, w.recSlot(1, 2, slotLen(o, N)), ix)
	desc := func(j int) Facts { return Facts{Extents: map[int]int{0: N, j: 3}} }
	whole := desc(1)
	whole.Whole = true
	var ps []Prim
	if a.Additive {
		ps = append(ps, Prim{Kind: PrimMap, Shape: sh[:3], In: []Operand{v, beta}, Out: delta, Expr: e.e})
	} else {
		ps = append(ps,
			Prim{Kind: PrimReduce, Shape: sh, In: []Operand{tableRecState, k, decay, tableRecRows}, Out: sk, Combine: OpAdd, Expr: s.e, Facts: desc(3)},
			Prim{Kind: PrimMap, Shape: sh[:3], In: []Operand{v, sk, beta}, Out: delta, Expr: e.e})
	}
	return append(ps,
		Prim{Kind: PrimMap, Shape: sh, In: []Operand{tableRecState, k, delta, decay, tableRecRows}, Out: next, Expr: u.e, Facts: desc(4)},
		Prim{Kind: PrimReduce, Shape: sh, In: []Operand{next, q}, Out: c.value(o.Out[0]), Combine: OpAdd, Expr: y.e},
		Prim{Kind: PrimMap, Shape: []int{H * Dv, Dk}, In: []Operand{next, tableRecRows}, Expr: w.e, Facts: whole,
			Stores: []PrimStore{{Out: tableRecStateOut, Index: to, Value: val, Extent: N}}},
	), nil
}

// deltaScan is the delta rule over a step of several rows: a scan whose
// state rows are the value heads' rows, each stepped through its run's
// rows in order -- a prompt chunk's rows one run, a ragged step's
// sequences a run each -- its run's slot read at the start and its slot
// written after the last row. A row's step is the one-row rule's: the
// decayed state's dot with k, (v - that dot) times beta along k, and the
// new state's dot with q.
func deltaScan(o graph.Op, c *Context, H, KH, Dk, Dv int, chanDecay bool) ([]Prim, error) {
	U, F := ir.U32, ir.F32
	R := c.Step.Rows
	N := H * Dv * Dk
	q, k, v, decay, beta := c.value(o.In[0]), c.value(o.In[1]), c.value(o.In[2]), c.value(o.In[3]), c.value(o.In[4])
	in := []Operand{q, k, v, decay, beta, tableRecState, tableRecRows}
	if c.scanLoops {
		// The state is read where it is written, each run's start in its
		// slot read of that table.
		in[5] = tableRecStateOut
	}
	const desc = 6
	var x exprs
	h, j, i, r, run := x.coord(0), x.coord(1), x.coord(2), x.coord(3), x.coord(4)
	kh := x.keyHead(h, H, KH, o.Attr.Tiled)
	start := x.op(OpMul, U, x.op(OpAdd, U, x.op(OpMul, U, h, x.u(uint32(Dv))), j), x.u(uint32(Dk)))
	place := x.op(OpAdd, U, start, i)
	slot := func(w int) int {
		at := x.op(OpAdd, U, x.op(OpAdd, U, run, run), x.u(uint32(w)))
		base := x.op(OpMul, U, x.load(desc, at), x.u(uint32(slotLen(o, N))))
		if c.scanLoops {
			// The component added last: a loop of the components walks
			// consecutive elements (Vectorize).
			return x.op(OpAdd, U, x.op(OpAdd, U, base, start), i)
		}
		return x.op(OpAdd, U, base, place)
	}
	from := slot(1)
	init := x.load(5, from)
	final := slot(2)
	g := x.at(3, []int{R, H}, r, h)
	if chanDecay {
		g = x.at(3, []int{R, H, Dk}, r, h, i)
	}
	kk := x.at(1, []int{R, KH, Dk}, r, kh, i)
	sb := x.op(OpMul, F, x.add(Expr{Op: ExprState, Type: F}), g)
	vj := x.at(2, []int{R, H, Dv}, r, h, j)
	if !o.Attr.Additive {
		vj = x.op(OpSub, F, vj, x.un(OpGroupSum, F, x.op(OpMul, F, sb, kk)))
	}
	d := x.op(OpMul, F, vj, x.at(4, []int{R, H}, r, h))
	next := x.op(OpAdd, F, sb, x.op(OpMul, F, kk, d))
	y := x.un(OpGroupSum, F, x.op(OpMul, F, next, x.at(0, []int{R, KH, Dk}, r, kh, i)))
	return []Prim{{Kind: PrimScan, Shape: []int{H, Dv, Dk}, In: in, Expr: x.e,
		Stores: []PrimStore{{Out: c.value(o.Out[0]), Ax: []int{r, h, j}, Ext: []int{R, H, Dv}, Value: y}},
		Scan:   &Scan{Desc: desc, Init: init, Next: next, Final: tableRecStateOut, Index: final, Extent: N, Loops: c.scanLoops, From: from},
		Facts:  Facts{Extents: map[int]int{5: N, desc: recDescWords(R)}}}}, nil
}

// recDescWords is how many words a run descriptor of rows rows holds
// (kernels.RunDescWords).
func recDescWords(rows int) int { return 1 + 7*rows }
