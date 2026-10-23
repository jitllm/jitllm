package jit

// BitSelect writes a select as bits, for a device whose vectors have no
// select: B where A's bits are set, C where they are clear, as (B and A) or
// (C and not A). Regs, where set, are the registers of the mask and of the
// halves: a constant only the select reads takes the first half's and
// is cleared in place, the other half the second; else the set half takes
// the first and the clear half the second. The or keeps the select's
// register.
type BitSelect struct {
	Regs []int
}

func (BitSelect) Name() string { return "bitselect" }

func (p BitSelect) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		var m *Kernel
		for i, nd := range k.Nodes {
			if nd.Op != OpSelect {
				continue
			}
			if m == nil {
				m = k.clone()
			}
			set := m.bin(OpAnd, nd.Type, nd.B, nd.A)
			clear := m.add(Node{Op: OpAndNot, Type: nd.Type, A: nd.C, B: nd.A})
			m.Nodes[i] = Node{Op: OpOr, Type: nd.Type, A: set, B: clear, Hint: nd.Hint, Share: nd.Share}
			if len(p.Regs) == 3 {
				hint := func(n, r int) {
					if m.Nodes[n].Hint == 0 {
						m.Nodes[n].Hint = r + 1
					}
				}
				hint(nd.A, p.Regs[0])
				lone := func(x int) bool { return m.Nodes[x].Op == OpConst && k.readCount(x) == 1 }
				switch {
				case lone(nd.C):
					hint(nd.C, p.Regs[1])
					hint(clear, p.Regs[1])
					hint(set, p.Regs[2])
				case lone(nd.B):
					hint(nd.B, p.Regs[1])
					hint(set, p.Regs[1])
					hint(clear, p.Regs[2])
				default:
					hint(set, p.Regs[1])
					hint(clear, p.Regs[2])
				}
			}
		}
		if m == nil {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}
