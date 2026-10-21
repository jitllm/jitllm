package jit

// Chains reassociates a workgroup thread's chain of multiply-adds from a
// constant, n of them, into N chains interleaved -- term t into chain t mod
// N, each from the constant -- summed pairwise.
type Chains struct {
	N int
}

func (Chains) Name() string { return "chains" }

func (p Chains) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if !k.Work || p.N < 2 {
			return nil, false, nil
		}
		m := k.clone()
		changed := false
		for i := range m.Stores {
			s := &m.Stores[i]
			if s.Barrier || s.Run {
				continue
			}
			if v, ok := m.reassociate(s.Val, p.N); ok {
				s.Val = v
				changed = true
			}
		}
		if !changed {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

// reassociate is the chain ending at node v as n interleaved chains summed
// pairwise.
func (k *Kernel) reassociate(v, n int) (int, bool) {
	var terms [][2]int
	x := v
	for k.Nodes[x].Op == OpFma {
		nd := k.Nodes[x]
		terms = append(terms, [2]int{nd.A, nd.B})
		x = nd.C
	}
	start := k.Nodes[x]
	if start.Op != OpConst || len(terms) < 2*n || len(terms)%n != 0 {
		return 0, false
	}
	t := k.Nodes[v].Type
	acc := make([]int, n)
	for c := range acc {
		acc[c] = x
	}
	for j := len(terms) - 1; j >= 0; j-- {
		c := (len(terms) - 1 - j) % n
		acc[c] = k.add(Node{Op: OpFma, Type: t, A: terms[j][0], B: terms[j][1], C: acc[c]})
	}
	for len(acc) > 1 {
		var next []int
		for c := 0; c+1 < len(acc); c += 2 {
			next = append(next, k.bin(OpAdd, t, acc[c], acc[c+1]))
		}
		if len(acc)%2 == 1 {
			next = append(next, acc[len(acc)-1])
		}
		acc = next
	}
	return acc[0], true
}
