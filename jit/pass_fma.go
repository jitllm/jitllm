package jit

import "github.com/jitllm/jitllm/jit/ir"

// FMA fuses a sum's step that adds a product nothing else reads, acc + a*b,
// into one multiply-add, fma(a, b, acc); where All is set, any float add of
// such a product.
type FMA struct {
	All bool
}

func (FMA) Name() string { return "fma" }

func (p FMA) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		var m *Kernel
		if p.All {
			for i, nd := range k.Nodes {
				if nd.Op != OpAdd || nd.Type != ir.F32 {
					continue
				}
				acc, prod := nd.A, nd.B
				if k.Nodes[prod].Op != OpMul {
					acc, prod = prod, acc
				}
				if pr := k.Nodes[prod]; pr.Op != OpMul || pr.Type != ir.F32 || k.uses(prod) != 1 {
					continue
				}
				if m == nil {
					m = k.clone()
				}
				pr := m.Nodes[prod]
				m.Nodes[i] = Node{Op: OpFma, Type: ir.F32, A: pr.A, B: pr.B, C: acc}
			}
		}
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
			if pr := k.Nodes[prod]; pr.Op != OpMul || pr.Type != nd.Type || k.uses(prod) != 1 {
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
