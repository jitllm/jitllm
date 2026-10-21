package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/ir"
)

func lower(p Prim) ([]*Nest, error) {
	switch p.Kind {
	case PrimContract:
		return []*Nest{lowerContract(p)}, nil
	case PrimMap:
		return []*Nest{lowerMap(p)}, nil
	case PrimReduce:
		return []*Nest{lowerReduce(p)}, nil
	}
	return nil, fmt.Errorf("jit: no lowering of primitive %d", p.Kind)
}

// lowerContract is a thread an output element and a loop over K.
func lowerContract(p Prim) *Nest {
	n := &Nest{Name: "contract", Space: []int{p.Rows}, Group: defaultGroup}
	a := n.buf(Buf{Kind: BufIn, Elem: storageElem(p.In[0]), Operand: p.In[0]})
	b := n.buf(Buf{Kind: BufIn, Elem: ir.F32, Operand: p.In[1]})
	out := n.buf(Buf{Kind: BufOut, Elem: ir.F32, Operand: p.Out})
	r := n.add(Node{Op: OpCoord, A: 0, Type: ir.U32})
	c := n.loop(p.K, -1)
	k := n.add(Node{Op: OpIter, At: c, Type: ir.U32})
	acc := n.add(Node{Op: OpFold, At: c, Type: ir.F32, B: n.f32(0)})
	w := n.load(a, n.bin(OpAdd, ir.U32, n.bin(OpMul, ir.U32, r, n.u32(p.K)), k))
	x := n.load(b, k)
	n.Nodes[acc].A = n.add(Node{Op: OpFma, Type: ir.F32, A: w, B: x, C: acc})
	n.Stores = append(n.Stores, Store{Buf: out, Ix: r, Val: acc, At: -1})
	return n
}

const defaultGroup = 128

// storageElem is the word type an operand's storage holds: a weight's
// payload is words, whatever its format.
func storageElem(o Operand) ir.Type {
	if o.Kind == OperandWeight {
		return ir.U32
	}
	return elemOf(o.Elem)
}

// load is buffer b's element at ix as the value it stores: a word of a
// verbatim f32 weight is its float.
func (n *Nest) load(b, ix int) int {
	bf := n.Bufs[b]
	v := n.add(Node{Op: OpLoad, Type: bf.Elem, Buf: b, A: ix})
	if bf.Elem == ir.U32 && bf.Operand.Kind == OperandWeight && bf.Operand.Format == jlm.TypeF32 {
		v = n.add(Node{Op: OpBitcast, Type: ir.F32, A: v})
	}
	return v
}

// access is index i reaching an operand through a.
func (n *Nest) access(i int, a Access) int {
	if a.Mul > 1 {
		i = n.bin(OpMul, ir.U32, i, n.u32(a.Mul))
	}
	if a.Div > 1 {
		i = n.bin(OpDiv, ir.U32, i, n.u32(a.Div))
	}
	if a.Mod > 0 {
		i = n.bin(OpRem, ir.U32, i, n.u32(a.Mod))
	}
	if a.Off != 0 {
		i = n.bin(OpAdd, ir.U32, i, n.u32(a.Off))
	}
	return i
}

func elemOf(t ir.Type) ir.Type {
	if t == ir.Void {
		return ir.F32
	}
	return t
}

// lowerMap is a thread an element.
func lowerMap(p Prim) *Nest {
	n := &Nest{Name: "map", Space: []int{p.N}, Group: defaultGroup}
	e := n.add(Node{Op: OpCoord, A: 0, Type: ir.U32})
	var vals []int
	if p.Shape != nil {
		vals = n.instance(p, p.Expr, n.scope(p, e, n.decode(e, p.Shape)))
	} else {
		vals = n.exprs(p, e, func(j int) int { return n.access(e, p.At[j]) })
	}
	if len(p.Stores) == 0 {
		out := n.buf(Buf{Kind: BufOut, Elem: elemOf(p.Out.Elem), Operand: p.Out})
		n.Stores = append(n.Stores, Store{Buf: out, Ix: n.access(e, p.OutAt), Val: vals[len(vals)-1], At: -1})
		return n
	}
	for _, s := range p.Stores {
		out := n.buf(Buf{Kind: BufOut, Elem: elemOf(s.Out.Elem), Operand: s.Out})
		n.Stores = append(n.Stores, Store{Buf: out, Ix: vals[s.Index], Val: vals[s.Value], At: -1})
	}
	return n
}

// lowerReduce is a thread an output and a loop over its K terms.
func lowerReduce(p Prim) *Nest {
	n := &Nest{Name: "reduce", Space: []int{p.N}, Group: defaultGroup}
	o := n.add(Node{Op: OpCoord, A: 0, Type: ir.U32})
	t := elemOf(p.Type)
	var outer []int
	if p.Shape != nil {
		outer = n.decode(o, p.outShape())
	}
	term := func(k int) int {
		el := n.bin(OpAdd, ir.U32, n.bin(OpMul, ir.U32, o, n.u32(p.K)), k)
		var vals []int
		if p.Shape != nil {
			vals = n.instance(p, p.Expr, n.scope(p, el, append(append([]int(nil), outer...), k)))
		} else {
			vals = n.exprs(p, el, func(j int) int { return n.access(el, p.At[j]) })
		}
		return vals[len(vals)-1]
	}
	trips, first := p.K, 0
	init := n.add(Node{Op: OpConst, Type: t, Imm: p.Init})
	if p.Start == StartFirst {
		init, trips, first = term(n.u32(0)), p.K-1, 1
	}
	c := n.loop(trips, -1)
	k := n.add(Node{Op: OpIter, At: c, Type: ir.U32})
	if first > 0 {
		k = n.bin(OpAdd, ir.U32, k, n.u32(first))
	}
	acc := n.add(Node{Op: OpFold, At: c, Type: t, B: init})
	n.Nodes[acc].A = n.bin(p.Combine, t, acc, term(k))
	out := n.buf(Buf{Kind: BufOut, Elem: elemOf(p.Out.Elem), Operand: p.Out})
	ix := n.access(o, p.OutAt)
	if len(p.Index) > 0 {
		var vals []int
		if p.Shape != nil {
			vals = n.instance(p, p.Index, n.scope(p, o, outer))
		} else {
			vals = n.exprs(Prim{Expr: p.Index}, o, nil)
		}
		ix = vals[len(vals)-1]
	}
	n.Stores = append(n.Stores, Store{Buf: out, Ix: ix, Val: acc, At: -1})
	return n
}

// expr is p's function as nodes, its last node's value.
func (n *Nest) expr(p Prim, index func(j int) int) int {
	vals := n.exprs(p, -1, index)
	return vals[len(vals)-1]
}

// exprs is p's function as nodes at element e, operand j at index(j) or at
// a node, each operand one buffer however often it is read.
func (n *Nest) exprs(p Prim, e int, index func(j int) int) []int {
	bufs := map[Operand]int{}
	bufOf := func(o Operand) int {
		b, ok := bufs[o]
		if !ok {
			b = n.buf(Buf{Kind: BufIn, Elem: storageElem(o), Operand: o})
			bufs[o] = b
		}
		return b
	}
	vals := make([]int, len(p.Expr))
	for i, x := range p.Expr {
		switch x.Op {
		case ExprIn:
			vals[i] = n.load(bufOf(p.In[x.In]), index(x.In))
		case ExprLoad:
			vals[i] = n.load(bufOf(p.In[x.In]), vals[x.A])
		case ExprElem:
			vals[i] = e
		case OpConst:
			vals[i] = n.add(Node{Op: OpConst, Type: x.Type, Imm: x.Imm})
		default:
			vals[i] = n.add(Node{Op: x.Op, Type: x.Type, A: vals[x.A], B: vals[x.B], C: vals[x.C]})
		}
	}
	return vals
}

// decode is flat index e's coordinates in a space of extents ext, the last
// fastest.
func (n *Nest) decode(e int, ext []int) []int {
	c := make([]int, len(ext))
	for d := len(ext) - 1; d > 0; d-- {
		c[d] = n.bin(OpRem, ir.U32, e, n.u32(ext[d]))
		e = n.bin(OpDiv, ir.U32, e, n.u32(ext[d]))
	}
	if len(ext) > 0 {
		c[0] = e
	}
	return c
}

// flatten is coordinates c's flat index in a space of extents ext.
func (n *Nest) flatten(c, ext []int) int {
	ix := c[0]
	for d := 1; d < len(ext); d++ {
		ix = n.bin(OpAdd, ir.U32, n.bin(OpMul, ir.U32, ix, n.u32(ext[d])), c[d])
	}
	return ix
}

// scope is where a primitive's function is instanced: its element and
// coordinates, and how its operands, its range and its reads at
// coordinates are reached.
type scope struct {
	elem   int
	coords []int
	load   func(j, ix int) int
	at     func(j int, ax, ext []int) int
	live   func() int
}

// scope is p's function's default scope at element e of coordinates c:
// every operand a buffer, a read at coordinates at its flat index.
func (n *Nest) scope(p Prim, e int, c []int) scope {
	bufs := map[Operand]int{}
	var s scope
	s.elem, s.coords = e, c
	s.load = func(j, ix int) int {
		o := p.In[j]
		b, ok := bufs[o]
		if !ok {
			b = n.buf(Buf{Kind: BufIn, Elem: storageElem(o), Operand: o})
			bufs[o] = b
		}
		return n.load(b, ix)
	}
	s.at = func(j int, ax, ext []int) int { return s.load(j, n.flatten(ax, ext)) }
	s.live = func() int {
		vals := n.instance(p, p.Range.End, s)
		return vals[len(vals)-1]
	}
	return s
}

// instance is expressions es of p as nodes in scope s.
func (n *Nest) instance(p Prim, es []Expr, s scope) []int {
	vals := make([]int, len(es))
	for i, x := range es {
		switch x.Op {
		case ExprIn:
			vals[i] = s.load(x.In, s.elem)
		case ExprLoad:
			vals[i] = s.load(x.In, vals[x.A])
		case ExprElem:
			vals[i] = s.elem
		case ExprCoord:
			vals[i] = s.coords[x.A]
		case ExprAt:
			ax := make([]int, len(x.Ax))
			for k, a := range x.Ax {
				ax[k] = vals[a]
			}
			vals[i] = s.at(x.In, ax, x.Ext)
		case ExprLive:
			vals[i] = s.live()
		case OpConst:
			vals[i] = n.add(Node{Op: OpConst, Type: x.Type, Imm: x.Imm})
		default:
			vals[i] = n.add(Node{Op: x.Op, Type: x.Type, A: vals[x.A], B: vals[x.B], C: vals[x.C]})
		}
	}
	return vals
}
