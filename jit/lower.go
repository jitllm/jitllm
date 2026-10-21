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
	v := n.expr(p, func(j int) int { return n.access(e, p.At[j]) })
	out := n.buf(Buf{Kind: BufOut, Elem: elemOf(p.Out.Elem), Operand: p.Out})
	n.Stores = append(n.Stores, Store{Buf: out, Ix: n.access(e, p.OutAt), Val: v, At: -1})
	return n
}

// lowerReduce is a thread an output and a loop over its K terms.
func lowerReduce(p Prim) *Nest {
	n := &Nest{Name: "reduce", Space: []int{p.N}, Group: defaultGroup}
	o := n.add(Node{Op: OpCoord, A: 0, Type: ir.U32})
	t := elemOf(p.Type)
	term := func(k int) int {
		el := n.bin(OpAdd, ir.U32, n.bin(OpMul, ir.U32, o, n.u32(p.K)), k)
		return n.expr(p, func(j int) int { return n.access(el, p.At[j]) })
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
	n.Stores = append(n.Stores, Store{Buf: out, Ix: n.access(o, p.OutAt), Val: acc, At: -1})
	return n
}

// expr is p's function as nodes, operand j loaded at index(j), each operand
// one buffer however often it is read.
func (n *Nest) expr(p Prim, index func(j int) int) int {
	bufs := map[int]int{}
	vals := make([]int, len(p.Expr))
	for i, x := range p.Expr {
		switch x.Op {
		case ExprIn:
			b, ok := bufs[x.In]
			if !ok {
				o := p.In[x.In]
				b = n.buf(Buf{Kind: BufIn, Elem: storageElem(o), Operand: o})
				bufs[x.In] = b
			}
			vals[i] = n.load(b, index(x.In))
		case OpConst:
			vals[i] = n.add(Node{Op: OpConst, Type: x.Type, Imm: x.Imm})
		default:
			vals[i] = n.add(Node{Op: x.Op, Type: x.Type, A: vals[x.A], B: vals[x.B], C: vals[x.C]})
		}
	}
	return vals[len(vals)-1]
}
