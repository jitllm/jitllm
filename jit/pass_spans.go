package jit

import "github.com/jitllm/jitllm/jit/ir"

// Spans runs a loop over a range's positions that pages the history one
// span at a time, a call for each run of positions inside one page: the
// span's first position and length are the caller's, its page is the
// call's alone, and a position's place in its page is the span's first
// place plus the trip.
type Spans struct {
	Page int
}

func (Spans) Name() string { return "spans" }

func (p Spans) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		pos := k.Positions
		if pos == nil || pos.Loop < 0 || p.Page <= 0 || !k.pages(pos.Loop, p.Page) {
			return nil, false, nil
		}
		m := k.clone()
		U := ir.U32
		c := pos.Loop
		it := m.iterOf(c)
		start := m.add(Node{Op: OpSpan, Type: U})
		length := m.add(Node{Op: OpSpan, Type: U, Imm: 1})
		first := m.Positions.First
		at := m.add(Node{Op: OpIter, At: c, Type: U})
		global := m.bin(OpAdd, U, m.bin(OpSub, U, start, first), at)
		if it >= 0 {
			m.redirectAll(it, global)
			m.Nodes[global].B = at
		}
		m.Loops[c].Count = length + 1
		g := newAlg(m)
		for i := range m.Nodes {
			nd := m.Nodes[i]
			if (nd.Op != OpDiv && nd.Op != OpRem) || !m.isConst(nd.B, int64(p.Page)) {
				continue
			}
			l := g.of(nd.A)
			w, rest := g.coef(l, g.id(atom{node: at}))
			if one, ok := w.constant(); !ok || one != 1 || !g.still(rest) {
				continue
			}
			s := g.build(rest)
			if nd.Op == OpDiv {
				m.Nodes[i] = Node{Op: OpDiv, Type: U, A: s, B: nd.B}
			} else {
				m.Nodes[i] = Node{Op: OpAdd, Type: U, A: m.bin(OpRem, U, s, nd.B), B: at}
			}
		}
		return []*Kernel{m}, true, nil
	})
}

// pages reports whether loop c reads its positions' pages: a division of a
// position by the page.
func (k *Kernel) pages(c, page int) bool {
	for _, nd := range k.Nodes {
		if nd.Op == OpDiv && k.isConst(nd.B, int64(page)) && k.varies(nd.A, c) {
			return true
		}
	}
	return false
}
