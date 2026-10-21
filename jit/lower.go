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
	return ir.F32
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

// lowerMap is a thread an element.
func lowerMap(p Prim) *Nest {
	n := &Nest{Name: "map", Space: []int{p.Rows}, Group: defaultGroup}
	var ins []int
	for _, o := range p.In {
		ins = append(ins, n.buf(Buf{Kind: BufIn, Elem: ir.F32, Operand: o}))
	}
	out := n.buf(Buf{Kind: BufOut, Elem: ir.F32, Operand: p.Out})
	e := n.add(Node{Op: OpCoord, A: 0, Type: ir.U32})
	v := n.load(ins[0], e)
	for _, b := range ins[1:] {
		v = n.bin(p.Fn, ir.F32, v, n.load(b, e))
	}
	n.Stores = append(n.Stores, Store{Buf: out, Ix: e, Val: v, At: -1})
	return n
}
