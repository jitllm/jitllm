package jit

import "github.com/jitllm/jitllm/jit/ir"

// Layout stores a table of the step so the axis its readers spread over
// threads runs fastest: a pool of positions of a width that a contraction
// reads summing over the width is stored width by width, its positions
// fastest. Every access of the table, its writers' too, takes the address
// of that layout, each coordinate times its stride.
type Layout struct{}

func (Layout) Name() string { return "layout" }

func (p Layout) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	flip := map[Operand]bool{}
	for _, k := range ks {
		if k.Kind != PrimContract {
			continue
		}
		for _, fd := range k.Nodes {
			if fd.Op != OpFold {
				continue
			}
			c := fd.At
			for _, nd := range k.Nodes {
				b := k.Bufs[nd.Buf]
				if nd.Op == OpLoad && len(nd.Ax) == 3 && b.Operand.Kind == OperandTable && k.varies(nd.Ax[2], c) && !k.varies(nd.Ax[1], c) {
					flip[b.Operand] = true
				}
			}
		}
	}
	if len(flip) == 0 {
		return ks, nil
	}
	out := make([]*Kernel, len(ks))
	for i, k := range ks {
		out[i] = k
		touched := false
		for _, b := range k.Bufs {
			touched = touched || flip[b.Operand]
		}
		if !touched {
			continue
		}
		m := k.clone()
		// address is the element at ax of a pool laid out [pages][width][positions].
		address := func(b int, ax []int) int {
			sh := m.Bufs[b].Shape
			pg, w := sh[1], sh[2]
			ix := m.bin(OpMul, ir.U32, ax[0], m.u32(pg*w))
			ix = m.bin(OpAdd, ir.U32, ix, m.bin(OpMul, ir.U32, ax[2], m.u32(pg)))
			return m.bin(OpAdd, ir.U32, ix, ax[1])
		}
		for j := range m.Nodes {
			if nd := m.Nodes[j]; nd.Op == OpLoad && len(nd.Ax) == 3 && flip[m.Bufs[nd.Buf].Operand] {
				ix := address(nd.Buf, nd.Ax)
				m.Nodes[j].A, m.Nodes[j].Ax = ix, nil
			}
		}
		for j := range m.Stores {
			if s := m.Stores[j]; len(s.Ax) == 3 && flip[m.Bufs[s.Buf].Operand] {
				ix := address(s.Buf, s.Ax)
				m.Stores[j].Ix, m.Stores[j].Ax = ix, nil
			}
		}
		for b := range m.Bufs {
			if flip[m.Bufs[b].Operand] {
				m.Bufs[b].Shape = nil
			}
		}
		m.Passes = append(m.Passes, p.Name())
		out[i] = m
	}
	return out, nil
}
