package jit

// BitSelect writes a select as bits, for a device whose vectors have no
// select: B where A's bits are set, C where they are clear, as (B and A) or
// (C and not A).
type BitSelect struct{}

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
			m.Nodes[i] = Node{Op: OpOr, Type: nd.Type, A: set, B: clear}
		}
		if m == nil {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}
