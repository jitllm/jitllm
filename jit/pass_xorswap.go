package jit

// XorSwap writes two selects that swap one pair of values under one
// predicate, A ? B : C and A ? C : B, as one masked difference of the pair
// taken into each: d = (C xor B) and A, then C xor d and B xor d. Reg,
// where set, is 1 + the register of the difference.
type XorSwap struct {
	Reg int
}

func (XorSwap) Name() string { return "xorswap" }

func (p XorSwap) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		var m *Kernel
		for i, a := range k.Nodes {
			if a.Op != OpSelect {
				continue
			}
			for j := i + 1; j < len(k.Nodes); j++ {
				b := k.Nodes[j]
				if b.Op != OpSelect || b.A != a.A || b.B != a.C || b.C != a.B || b.Type != a.Type {
					continue
				}
				if m == nil {
					m = k.clone()
				}
				x := m.bin(OpXor, a.Type, a.C, a.B)
				diff := m.bin(OpAnd, a.Type, a.A, x)
				m.Nodes[x].Hint, m.Nodes[diff].Hint = p.Reg, p.Reg
				m.Nodes[i] = Node{Op: OpXor, Type: a.Type, A: a.C, B: diff}
				m.Nodes[j] = Node{Op: OpXor, Type: a.Type, A: a.B, B: diff}
				break
			}
		}
		if m == nil {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}
