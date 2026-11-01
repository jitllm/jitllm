package jit

import (
	"cmp"
	"errors"
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/ir"
)

// erfP and erfA are erf's rational approximation, Abramowitz and Stegun
// 7.1.26: erf(z) = 1 - (a1 t + ... + a5 t^5) e^(-z^2), t = 1/(1 + p z), for
// z >= 0, odd in z.
const erfP = float32(0.3275911)

var erfA = [5]float32{0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429}

func (x *exprs) fma(a, b, c int) int { return x.add(Expr{Op: OpFma, Type: ir.F32, A: a, B: b, C: c}) }

func (x *exprs) neg(a int) int { return x.op(OpSub, ir.F32, x.f(0), a) }

func (x *exprs) sigmoid(v int) int {
	return x.op(OpDiv, ir.F32, x.f(1), x.op(OpAdd, ir.F32, x.f(1), x.un(OpExp, ir.F32, x.neg(v))))
}

// tanh is 1 - 2/(e^2a + 1), which saturates to +-1 where e^2a overflows or
// vanishes.
func (x *exprs) tanh(a int) int {
	F := ir.F32
	e := x.un(OpExp, F, x.op(OpMul, F, x.f(2), a))
	return x.op(OpSub, F, x.f(1), x.op(OpDiv, F, x.f(2), x.op(OpAdd, F, e, x.f(1))))
}

// erf is erfA's approximation at |a|, given a's sign.
func (x *exprs) erf(a int) int {
	F := ir.F32
	z := x.op(OpMax, F, a, x.neg(a))
	t := x.op(OpDiv, F, x.f(1), x.fma(x.f(erfP), z, x.f(1)))
	p := x.f(erfA[4])
	for i := 3; i >= 0; i-- {
		p = x.fma(p, t, x.f(erfA[i]))
	}
	y := x.op(OpSub, F, x.f(1), x.op(OpMul, F, x.op(OpMul, F, p, t), x.un(OpExp, F, x.neg(x.op(OpMul, F, z, z)))))
	return x.sel(F, x.op(OpLess, ir.Pred, a, x.f(0)), x.neg(y), y)
}

// decomposeExpr is the op's terms, element by element: its inputs read as
// an OpMap's are (an operand narrower than the row broadcast), its bound
// vectors' elements as scalars at fixed places. One map holds them all,
// unless the device's registers refuse that map (ErrRegisterPressure): then
// each term is a map of its own, the terms between them scratch rows.
func decomposeExpr(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.Out) != 1 || len(o.Attr.Expr) == 0 {
		return nil, fmt.Errorf("an expression of %d outputs, %d terms", len(o.Out), len(o.Attr.Expr))
	}
	if e := c.Graph.Values[o.Out[0]].Elem; e != graph.ElemF32 {
		return nil, fmt.Errorf("an expression into %s values", c.Graph.Values[o.Out[0]].Name)
	}
	for _, v := range o.In {
		if n, w := c.width(v), c.width(o.Out[0]); n <= 0 || w%n != 0 {
			return nil, fmt.Errorf("an operand of %d elements under an expression of %d", n, w)
		}
	}
	for _, w := range o.Weight {
		if w.Rows > 1 {
			return nil, fmt.Errorf("an expression over a matrix")
		}
	}
	all := make([]int, len(o.Attr.Expr))
	for k := range all {
		all[k] = k
	}
	whole, err := exprPrim(o, c, all, nil)
	if err != nil || len(o.Attr.Expr) == 1 || c.split == nil {
		return []Prim{whole}, err
	}
	if at := (exprAt{c.Of, o.Out[0]}); !c.split.terms[at] {
		c.split.whole[c.Of] = append(c.split.whole[c.Of], exprSize{at, len(whole.Expr)})
		return []Prim{whole}, nil
	}
	out := c.value(o.Out[0])
	rows := make([]Operand, len(o.Attr.Expr))
	prims := make([]Prim, len(o.Attr.Expr))
	for k := range o.Attr.Expr {
		rows[k] = Operand{Kind: OperandScratch, Name: fmt.Sprintf("%s.t%d", out.Name, k)}
		if prims[k], err = exprPrim(o, c, []int{k}, rows); err != nil {
			return nil, err
		}
		if k < len(rows)-1 {
			prims[k].Out = rows[k]
		}
	}
	return prims, nil
}

// exprAt is an expression of a run: its part and its value.
type exprAt struct {
	of  int
	out graph.ValueID
}

type exprSize struct {
	at    exprAt
	nodes int
}

// exprSplit is the expressions of a run formed a map a term, and those the
// last decomposition formed whole, by part, each with its map's nodes.
type exprSplit struct {
	terms map[exprAt]bool
	whole map[int][]exprSize
	// unfused is the parts formed with their kernels joined to none,
	// unjammed the times a part's jammed form was refused, each halving
	// its rows a call (RowJam),
	// unpaired those formed with each head a call of its own (HeadPairs),
	// unwide those formed with no loop's rows across lanes (Jam),
	// uncursored those formed indexing their buffers (Cursors), unsplit
	// those formed with their dots' runs whole (HalfSplit), unstreamed
	// those formed walking each row group through every sub-block (Streams),
	// unvector those formed with no loop across lanes (Vectorize).
	unfused, unpaired, unwide, uncursored, unsplit, unvector map[int]bool
	unstreamed, crowded                                      map[KernelAt]bool
	unjammed                                                 map[int]int
}

func newExprSplit() *exprSplit {
	return &exprSplit{terms: map[exprAt]bool{}, whole: map[int][]exprSize{}, unfused: map[int]bool{}, unjammed: map[int]int{}, unpaired: map[int]bool{}, unwide: map[int]bool{}, uncursored: map[int]bool{}, unsplit: map[int]bool{}, unvector: map[int]bool{}, unstreamed: map[KernelAt]bool{}, crowded: map[KernelAt]bool{}}
}

// splitRefused forms a part whose refused kernel walked a weight's planes
// along the rows (Streams) without that walk; else one whose refused kernel
// ran several rows a call with half the rows a call, and nothing else; else
// one whose refused kernel ran two heads a call with a head a call, and one
// whose refused kernel jammed a loop's rows across lanes with none; else one
// whose refused kernel walked its addresses
// indexing them; else a map a term the largest expression formed whole
// in each part of the kernels refused, or a part with none left unfused,
// reporting whether it changed a part.
func (sp *exprSplit) splitRefused(err error) bool {
	if err == nil {
		return false
	}
	var pe *PressureError
	if !errors.As(err, &pe) {
		return false
	}
	more := false
	for _, at := range pe.Crowded {
		if !sp.crowded[at] {
			sp.crowded[at], more = true, true
		}
	}
	for _, at := range pe.Streamed {
		if !sp.unstreamed[at] {
			sp.unstreamed[at], more = true, true
		}
	}
	if more {
		return true
	}
	for _, of := range pe.Split {
		if !sp.unsplit[of] {
			sp.unsplit[of], more = true, true
		}
	}
	if more {
		return true
	}
	for _, of := range pe.Jammed {
		sp.unjammed[of]++
		more = true
	}
	if more {
		return true
	}
	for _, of := range pe.Paired {
		if !sp.unpaired[of] {
			sp.unpaired[of], more = true, true
		}
	}
	for _, of := range pe.Wide {
		if !sp.unwide[of] {
			sp.unwide[of], more = true, true
		}
	}
	if more {
		return true
	}
	for _, of := range pe.Cursored {
		if !sp.uncursored[of] {
			sp.uncursored[of], more = true, true
		}
	}
	if more {
		return true
	}
	for _, of := range pe.Parts {
		split := false
		if ws := sp.whole[of]; len(ws) > 0 {
			big := slices.MaxFunc(ws, func(a, b exprSize) int { return cmp.Compare(a.nodes, b.nodes) })
			split = !sp.terms[big.at]
			sp.terms[big.at] = true
		}
		if !split && !sp.unfused[of] {
			sp.unfused[of], split = true, true
		}
		more = more || split
	}
	clear(sp.whole)
	if more {
		return true
	}
	// Nothing left to take apart: the part's kernels formed with no loop
	// across lanes.
	for _, of := range pe.Parts {
		if !sp.unvector[of] {
			sp.unvector[of], more = true, true
		}
	}
	return more
}

// exprPrim is the map of terms ks of o, the last its value: an earlier term
// is one of ks or, where rows is given, the scratch row rows names for it.
func exprPrim(o graph.Op, c *Context, ks []int, rows []Operand) (Prim, error) {
	sh := c.rows(o.Out[0])
	R, W := sh[0], sh[1]
	var x exprs
	r, i := x.coord(0), x.coord(1)
	var in []Operand
	slot := map[string]int{}
	use := func(op Operand) int {
		key := fmt.Sprintf("%d/%d/%v/%s", op.Kind, op.Value, op.Role, op.Name)
		j, ok := slot[key]
		if !ok {
			j, slot[key], in = len(in), len(in), append(in, op)
		}
		return j
	}
	operand := func(v graph.ValueID) int {
		n := c.width(v)
		e := i
		switch {
		case n == 1:
			e = x.u(0)
		case n != W:
			e = x.op(OpDiv, ir.U32, i, x.u(uint32(W/n)))
		}
		a := x.at(use(c.value(v)), []int{R, n}, r, e)
		if c.Graph.Values[v].Elem == graph.ElemU32 {
			a = x.un(OpCvtF32, ir.F32, x.un(OpBitcast, ir.I32, a))
		}
		return a
	}
	vals := map[int]int{}
	for _, k := range ks {
		t := o.Attr.Expr[k]
		args := make([]int, t.Fn.Arity())
		for j := range args {
			switch a := t.Args[j]; a.Kind {
			case graph.ArgIn:
				if a.Index < 0 || a.Index >= len(o.In) {
					return Prim{}, fmt.Errorf("term %d reads operand %d of %d", k, a.Index, len(o.In))
				}
				args[j] = operand(o.In[a.Index])
			case graph.ArgWeight:
				if a.Index < 0 || a.Index >= len(o.Weight) || a.Elem < 0 || a.Elem >= o.Weight[a.Index].K {
					return Prim{}, fmt.Errorf("term %d reads element %d of bound vector %d", k, a.Elem, a.Index)
				}
				w := o.Weight[a.Index]
				args[j] = x.at(use(Operand{Kind: OperandVector, Role: w.Role, Name: w.Role.String()}), []int{w.K}, x.u(uint32(a.Elem)))
			case graph.ArgConst:
				args[j] = x.f(float32(a.Const))
			case graph.ArgTerm:
				v, ok := vals[a.Index]
				switch {
				case a.Index < 0 || a.Index >= k:
					return Prim{}, fmt.Errorf("term %d reads term %d", k, a.Index)
				case !ok && rows == nil:
					return Prim{}, fmt.Errorf("term %d reads term %d, which is not made", k, a.Index)
				case !ok:
					v = x.at(use(rows[a.Index]), []int{R, W}, r, i)
				}
				args[j] = v
			default:
				return Prim{}, fmt.Errorf("term %d's operand %d is of kind %d", k, j, a.Kind)
			}
		}
		v, err := x.mapFn(t.Fn, args, c.exactDiv || !o.Attr.Exact)
		if err != nil {
			return Prim{}, err
		}
		vals[k] = v
	}
	return Prim{Kind: PrimMap, Shape: sh, In: in, Out: c.value(o.Out[0]), Expr: x.e}, nil
}
