package jit

import "github.com/jitllm/jitllm/jit/ir"

// Split cuts a sum over a loop into contiguous segments, each element's
// segments threads of their own writing partial planes, and a combine that
// sums the planes. The segments double while they stay under Max, the
// threads under twice Slots, and the trips divide into Block-wide blocks a
// segment.
type Split struct {
	Slots, Max, Block int
	Kinds             []PrimKind
}

func (Split) Name() string { return "split" }

func (s Split) Apply(prog []*Nest, t Target) ([]*Nest, error) {
	return eachNest(s.Name(), prog, s.apply)
}

func (s Split) apply(n *Nest) ([]*Nest, bool, error) {
	c, f, term, st, ok := splittable(n)
	if !ok || !covers(s.Kinds, n.Kind) {
		return nil, false, nil
	}
	T, E := n.Loops[c].Trip, n.elements()
	S := 1
	for S < s.Max && E*S < 2*s.Slots && T%(s.Block*S*2) == 0 {
		S *= 2
	}
	if S == 1 {
		return nil, false, nil
	}
	p := n.clone()
	p.Name = n.Name + ".part"
	p.Space = append([]int{S}, n.Space...)
	for i := range p.Nodes {
		if p.Nodes[i].Op == OpCoord {
			p.Nodes[i].A++
		}
	}
	seg := p.add(Node{Op: OpCoord, A: 0, Type: ir.U32})
	p.Loops[c].Trip = T / S
	if it := p.iterOf(c); it >= 0 {
		k := p.bin(OpAdd, ir.U32, p.bin(OpMul, ir.U32, seg, p.u32(T/S)), it)
		p.redirect(it, k, k)
	}
	planes := p.buf(Buf{Kind: BufOut, Elem: ir.F32, Operand: Operand{Kind: OperandScratch, Name: "planes"}})
	at := p.bin(OpAdd, ir.U32, p.bin(OpMul, ir.U32, seg, p.u32(E)), p.flat(1))
	val := f
	if term >= 0 {
		first := p.add(Node{Op: OpLess, Type: ir.Pred, A: seg, B: p.u32(1)})
		val = p.bin(OpAdd, ir.F32, f, p.add(Node{Op: OpSelect, Type: ir.F32, A: first, B: term, C: p.f32(0)}))
	}
	p.Stores[st] = Store{Buf: planes, Ix: at, Val: val, At: -1}

	m := &Nest{Name: n.Name + ".combine", Space: append([]int(nil), n.Space...), Group: n.Group}
	in := m.buf(Buf{Kind: BufIn, Elem: ir.F32, Operand: Operand{Kind: OperandScratch, Name: "planes"}})
	out := m.buf(n.Bufs[n.Stores[st].Buf])
	m.Bufs[out].Kind = BufOut
	l := m.loop(S, -1)
	it := m.add(Node{Op: OpIter, At: l, Type: ir.U32})
	acc := m.add(Node{Op: OpFold, At: l, Type: n.Nodes[f].Type, B: m.f32(0)})
	x := m.load(in, m.bin(OpAdd, ir.U32, m.bin(OpMul, ir.U32, it, m.u32(E)), m.flat(0)))
	m.Nodes[acc].A = m.bin(OpAdd, ir.F32, acc, x)
	m.Stores = append(m.Stores, Store{Buf: out, Ix: m.flat(0), Val: acc, At: -1})
	return []*Nest{p, m}, true, nil
}

// splittable finds the element's sum: a top-level loop with one fold, a sum,
// which the element's one store writes at its own place, alone or with a
// term beside it the loop does not change (-1 for none).
func splittable(n *Nest) (c, f, term, st int, ok bool) {
	if len(n.Stores) != 1 || n.Stores[0].At != -1 || !n.atElement(n.Stores[0].Ix) {
		return 0, 0, 0, 0, false
	}
	f, term = n.Stores[0].Val, -1
	if v := n.Nodes[f]; v.Op == OpAdd {
		switch {
		case n.Nodes[v.A].Op == OpFold:
			f, term = v.A, v.B
		case n.Nodes[v.B].Op == OpFold:
			f, term = v.B, v.A
		}
	}
	nd := n.Nodes[f]
	if nd.Op != OpFold || n.Loops[nd.At].Parent != -1 || len(n.foldsOf(nd.At)) != 1 || !n.sumFold(f) ||
		term >= 0 && n.varies(term, nd.At) {
		return 0, 0, 0, 0, false
	}
	return nd.At, f, term, 0, true
}
