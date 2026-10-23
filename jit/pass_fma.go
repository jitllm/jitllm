package jit

import "github.com/jitllm/jitllm/jit/ir"

// FMA fuses a float sum's step that adds a product nothing else reads,
// acc + a*b, into one multiply-add, fma(a, b, acc); where All is set, any
// float add of such a product. OpFma is a float operation: an integer sum
// keeps its multiply and add. An integer add of a byte dot from zero
// nothing else reads, acc + dot4(a, b, 0), is the dot from acc,
// dot4(a, b, acc): integers, so exact.
type FMA struct {
	All bool
}

func (FMA) Name() string { return "fma" }

func (p FMA) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		var m *Kernel
		for i, nd := range k.Nodes {
			if nd.Op != OpAdd || nd.Type != ir.I32 {
				continue
			}
			acc, dot := nd.A, nd.B
			if k.Nodes[dot].Op != OpDot4 {
				acc, dot = dot, acc
			}
			dn := k.Nodes[dot]
			if dn.Op != OpDot4 || k.uses(dot) != 1 {
				continue
			}
			if z := k.Nodes[dn.C]; z.Op != OpConst || z.Imm != 0 {
				continue
			}
			if m == nil {
				m = k.clone()
			}
			m.Nodes[i] = Node{Op: OpDot4, Type: dn.Type, A: dn.A, B: dn.B, C: acc}
		}
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
			if nd.Type != ir.F32 || s.Op != OpAdd || s.Type != nd.Type {
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
