package jit

import (
	"fmt"
	"math/bits"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/ir"
)

// A part's recurrent state is a pool of slots, a sequence's state a slot:
// the delta rule's ("rec.state", for each value head its rows of the
// value's width by the key's, value head after value head) and the
// convolution's window ("rec.conv"). A step reads a slot of each and writes
// a slot of its ".out" (the same storage, or the pool's other half), the
// slots the step's recurrent descriptor names ("rec.rows": the rows, then
// the slot read and the slot written).
var (
	tableRecState    = Operand{Kind: OperandTable, Name: "rec.state", PerPart: true}
	tableRecStateOut = Operand{Kind: OperandTable, Name: "rec.state.out", PerPart: true}
	tableRecConv     = Operand{Kind: OperandTable, Name: "rec.conv", PerPart: true}
	tableRecConvOut  = Operand{Kind: OperandTable, Name: "rec.conv.out", PerPart: true}
	tableRecRows     = Operand{Kind: OperandTable, Name: "rec.rows", Elem: ir.U32}
)

// recSlot is the first element of the slot the step's descriptor (operand
// j) names at word w, of slots n elements long.
func (x *exprs) recSlot(j, w, n int) int {
	return x.op(OpMul, ir.U32, x.load(j, x.u(uint32(w))), x.u(uint32(n)))
}

// decomposeDeltaRule is the delta rule over the part's state at one row:
// each state row's decayed dot with its key head's k, the row's (v - that
// dot) times beta, then the decayed state plus k times it into a scratch
// state, its dot with q the output, and the scratch into the slot written,
// so no kernel reads the state while writing it.
func decomposeDeltaRule(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	if len(o.In) != 5 || len(o.Out) != 1 || c.Step.Rows != 1 {
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
	sh := []int{1, H, Dv, Dk}
	name := c.value(o.Out[0]).Name
	sk := Operand{Kind: OperandScratch, Name: name + ".sk"}
	delta := Operand{Kind: OperandScratch, Name: name + ".delta"}
	next := Operand{Kind: OperandScratch, Name: name + ".state"}
	q, k, v, decay, beta := c.value(o.In[0]), c.value(o.In[1]), c.value(o.In[2]), c.value(o.In[3]), c.value(o.In[4])

	// keyHead is value head h's key head; flat is the state's element at
	// (h, j, i) of the slot read; g is the decay on the element at (h, i).
	keyHead := func(x *exprs, h int) int {
		if a.Tiled {
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
	flat := func(x *exprs, desc, h, j, i int) int {
		at := x.op(OpAdd, U, x.op(OpMul, U, x.op(OpAdd, U, x.op(OpMul, U, h, x.u(uint32(Dv))), j), x.u(uint32(Dk))), i)
		return x.op(OpAdd, U, x.recSlot(desc, 1, N), at)
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
	e.op(OpMul, F, e.op(OpSub, F, e.at(0, []int{1, H, Dv}, r, h, j), e.at(1, []int{1, H, Dv}, r, h, j)), e.at(2, []int{1, H}, r, h))

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
	to := w.op(OpAdd, U, w.recSlot(1, 2, N), ix)
	desc := func(j int) Facts { return Facts{Extents: map[int]int{0: N, j: 3}} }
	return []Prim{
		{Kind: PrimReduce, Shape: sh, In: []Operand{tableRecState, k, decay, tableRecRows}, Out: sk, Combine: OpAdd, Expr: s.e, Facts: desc(3)},
		{Kind: PrimMap, Shape: sh[:3], In: []Operand{v, sk, beta}, Out: delta, Expr: e.e},
		{Kind: PrimMap, Shape: sh, In: []Operand{tableRecState, k, delta, decay, tableRecRows}, Out: next, Expr: u.e, Facts: desc(4)},
		{Kind: PrimReduce, Shape: sh, In: []Operand{next, q}, Out: c.value(o.Out[0]), Combine: OpAdd, Expr: y.e},
		{Kind: PrimMap, Shape: []int{H * Dv, Dk}, In: []Operand{next, tableRecRows}, Expr: w.e, Facts: desc(1),
			Stores: []PrimStore{{Out: tableRecStateOut, Index: to, Value: val, Extent: N}}},
	}, nil
}
