package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/ir"
)

// generate is primitive p's naive IR: an element of its output space each,
// a reduction's terms a loop, nothing fused and nothing bound.
func generate(p Prim) (*Kernel, error) {
	switch p.Kind {
	case PrimMap, PrimReduce, PrimContract:
	default:
		return nil, fmt.Errorf("jit: no naive form of primitive %d", p.Kind)
	}
	out := p.outShape()
	k := &Kernel{Name: "map", Kind: p.Kind, Space: append([]int(nil), out...), Heads: p.Facts.Heads}
	coords := make([]int, len(out))
	for i := range coords {
		coords[i] = k.add(Node{Op: OpCoord, A: i, Type: ir.U32})
	}
	terms, c := coords, -1
	first := p.Kind != PrimMap && p.Start == StartFirst
	if p.Kind != PrimMap {
		k.Name = "reduce"
		K := p.Shape[len(p.Shape)-1]
		if first {
			K--
		}
		c = k.loop(K, -1)
		it := k.add(Node{Op: OpIter, At: c, Type: ir.U32})
		if first {
			it = k.bin(OpAdd, ir.U32, it, k.u32(1))
		}
		terms = append(append([]int(nil), coords...), it)
	}
	g := &generator{k: k, p: &p, flat: map[Operand]int{}, shaped: map[Operand]int{}}
	if r := p.Facts.Range; r != nil {
		pos := &Positions{Axis: r.Axis, Loop: -1, Align: 1}
		if p.Kind != PrimMap && r.Axis == len(p.Shape)-1 {
			pos.Axis, pos.Loop = -1, c
		}
		pos.First, pos.End = g.value(r.First, terms), g.value(r.End, terms)
		g.pos = pos
		k.Positions = pos
	}
	if len(p.Facts.Rows) > 0 {
		k.Rows = g.value(p.Facts.Rows, terms) + 1
	}
	vals := g.values(p.Expr, terms)
	if p.Kind == PrimMap {
		if len(p.Stores) == 0 {
			b := k.buf(Buf{Kind: BufOut, Elem: elemOf(p.Out.Elem), Operand: p.Out, Shape: append([]int(nil), out...)})
			k.Stores = append(k.Stores, Store{Buf: b, Val: vals[len(vals)-1], At: -1, Ax: append([]int(nil), coords...)})
		}
		for _, s := range p.Stores {
			b := k.buf(Buf{Kind: BufOut, Elem: elemOf(s.Out.Elem), Operand: s.Out, Shape: append([]int(nil), s.Ext...)})
			st := Store{Buf: b, Val: vals[s.Value], At: -1}
			if s.Ax != nil {
				for _, a := range s.Ax {
					st.Ax = append(st.Ax, vals[a])
				}
			} else {
				st.Ix = vals[s.Index]
			}
			k.Stores = append(k.Stores, st)
		}
		return k, nil
	}
	t := elemOf(p.Type)
	init := k.add(Node{Op: OpConst, Type: t, Imm: p.Init})
	if first {
		init = g.value(p.Expr, append(append([]int(nil), coords...), k.u32(0)))
	}
	acc := k.add(Node{Op: OpFold, At: c, Type: t, B: init})
	term := vals[len(vals)-1]
	if p.Kind == PrimContract {
		m := k.Nodes[term]
		if m.Op != OpMul {
			return nil, fmt.Errorf("jit: a contraction's term is not a product")
		}
		k.Nodes[acc].A = k.add(Node{Op: OpFma, Type: t, A: m.A, B: m.B, C: acc})
	} else {
		k.Nodes[acc].A = k.bin(p.Combine, t, acc, term)
	}
	if len(p.Index) > 0 {
		b := k.buf(Buf{Kind: BufOut, Elem: elemOf(p.Out.Elem), Operand: p.Out})
		k.Stores = append(k.Stores, Store{Buf: b, Ix: g.value(p.Index, coords), Val: acc, At: -1})
		return k, nil
	}
	b := k.buf(Buf{Kind: BufOut, Elem: elemOf(p.Out.Elem), Operand: p.Out, Shape: append([]int(nil), out...)})
	k.Stores = append(k.Stores, Store{Buf: b, Val: acc, At: -1, Ax: append([]int(nil), coords...)})
	return k, nil
}

// generator instances a primitive's expressions as IR nodes, each operand
// one buffer: one read by flat index, one read at coordinates of its shape.
type generator struct {
	k            *Kernel
	p            *Prim
	flat, shaped map[Operand]int
	pos          *Positions
}

func (g *generator) value(es []Expr, coords []int) int {
	vals := g.values(es, coords)
	return vals[len(vals)-1]
}

func (g *generator) values(es []Expr, coords []int) []int {
	k := g.k
	vals := make([]int, len(es))
	for i, e := range es {
		switch e.Op {
		case ExprCoord:
			vals[i] = coords[e.A]
		case ExprLoad:
			vals[i] = k.load(g.buffer(g.flat, e.In, nil), vals[e.A])
		case ExprAt:
			ax := make([]int, len(e.Ax))
			for j, a := range e.Ax {
				ax[j] = vals[a]
			}
			b := g.buffer(g.shaped, e.In, e.Ext)
			vals[i] = k.add(Node{Op: OpLoad, Type: k.Bufs[b].Elem, Buf: b, Ax: ax})
			if o := k.Bufs[b].Operand; o.Kind == OperandWeight && o.Format == jlm.TypeF32 {
				vals[i] = k.add(Node{Op: OpBitcast, Type: ir.F32, A: vals[i]})
			}
		case ExprPos:
			r := g.p.Facts.Range
			if r == nil {
				vals[i] = k.u32(0)
				continue
			}
			vals[i] = k.bin(OpAdd, ir.U32, g.value(r.First, coords), coords[r.Axis])
			if g.pos != nil {
				g.pos.Pos = append(g.pos.Pos, vals[i])
			}
		case ExprLive:
			if g.p.Facts.Range == nil {
				vals[i] = k.u32(0)
				continue
			}
			vals[i] = g.value(g.p.Facts.Range.End, coords)
			if g.pos != nil {
				g.pos.Live = append(g.pos.Live, vals[i])
			}
		case OpConst:
			vals[i] = k.add(Node{Op: OpConst, Type: e.Type, Imm: e.Imm})
		default:
			vals[i] = k.add(Node{Op: e.Op, Type: e.Type, A: vals[e.A], B: vals[e.B], C: vals[e.C]})
		}
	}
	return vals
}

func (g *generator) buffer(m map[Operand]int, in int, ext []int) int {
	o := g.p.In[in]
	if b, ok := m[o]; ok {
		return b
	}
	b := g.k.buf(Buf{Kind: BufIn, Elem: storageElem(o), Operand: o, Shape: append([]int(nil), ext...)})
	m[o] = b
	return b
}

// storageElem is the word type an operand's storage holds: a weight's
// payload is words, whatever its format.
func storageElem(o Operand) ir.Type {
	if o.Kind == OperandWeight {
		return ir.U32
	}
	return elemOf(o.Elem)
}

func elemOf(t ir.Type) ir.Type {
	if t == ir.Void {
		return ir.F32
	}
	return t
}

// load is buffer b's element at ix as the value it stores: a word of a
// verbatim f32 weight is its float.
func (k *Kernel) load(b, ix int) int {
	bf := k.Bufs[b]
	v := k.add(Node{Op: OpLoad, Type: bf.Elem, Buf: b, A: ix})
	if bf.Elem == ir.U32 && bf.Operand.Kind == OperandWeight && bf.Operand.Format == jlm.TypeF32 {
		v = k.add(Node{Op: OpBitcast, Type: ir.F32, A: v})
	}
	return v
}
