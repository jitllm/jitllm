package jit

// FMA fuses a sum's step that adds a product, acc + a*b, into one
// multiply-add, fma(a, b, acc).
type FMA struct{}

func (FMA) Name() string { return "fma" }

func (p FMA) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		var m *Kernel
		for i, nd := range k.Nodes {
			if nd.Op != OpFold {
				continue
			}
			s := k.Nodes[nd.A]
			if s.Op != OpAdd || s.Type != nd.Type {
				continue
			}
			prod := s.B
			if s.B == i {
				prod = s.A
			} else if s.A != i {
				continue
			}
			if pr := k.Nodes[prod]; pr.Op != OpMul || pr.Type != nd.Type {
				continue
			}
			if m == nil {
				m = k.clone()
			}
			pr := m.Nodes[prod]
			m.Nodes[nd.A] = Node{Op: OpFma, Type: nd.Type, A: pr.A, B: pr.B, C: i}
		}
		if m == nil {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}
