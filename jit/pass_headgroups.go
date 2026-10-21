package jit

import "github.com/jitllm/jitllm/jit/ir"

// HeadGroups tiles a workgroup's heads by the query heads that share a key
// head, as many as the workgroup geometry groups: the workgroup a group of
// heads, each phase's work done for every head of the group within the
// same loops, a value of its own a head.
type HeadGroups struct {
	Work Workgroups
}

func (HeadGroups) Name() string { return "headgroups" }

func (p HeadGroups) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if !k.Work || k.Heads == nil || len(k.Space) != 3 {
			return nil, false, nil
		}
		G := p.Work.geometry(*k.Heads, d.Lanes).group
		if G < 2 || k.Space[1]%G != 0 {
			return nil, false, nil
		}
		m := k.clone()
		m.Space[1] /= G
		head := -1
		for i, nd := range m.Nodes {
			if nd.Op == OpCoord && nd.A == 1 {
				head = i
			}
		}
		if head < 0 {
			return nil, false, nil
		}
		group := m.add(Node{Op: OpCoord, A: 1, Type: ir.U32})
		base := m.bin(OpMul, ir.U32, group, m.u32(G))
		stores := m.Stores
		m.Stores = nil
		reps := make([]*replica, G)
		for hl := range reps {
			reps[hl] = &replica{k: m, sub: map[int]int{head: m.bin(OpAdd, ir.U32, base, m.u32(hl))}, memo: map[int]int{}}
		}
		for _, s := range stores {
			if s.Barrier || s.Run {
				m.Stores = append(m.Stores, s)
				continue
			}
			for _, r := range reps {
				ns := s
				ns.Val = r.copy(s.Val)
				if s.Ax != nil {
					ns.Ax = nil
					for _, a := range s.Ax {
						ns.Ax = append(ns.Ax, r.copy(a))
					}
				} else {
					ns.Ix = r.copy(s.Ix)
				}
				m.Stores = append(m.Stores, ns)
			}
		}
		m.Heads = k.Heads
		return []*Kernel{m}, true, nil
	})
}

// replica copies nodes with sub's replaced, a fold that reads a replaced
// node a fold of its own.
type replica struct {
	k    *Kernel
	sub  map[int]int
	memo map[int]int
}

func (r *replica) copy(i int) int {
	if v, ok := r.sub[i]; ok {
		return v
	}
	if v, ok := r.memo[i]; ok {
		return v
	}
	if !r.reaches(i, map[int]bool{}) {
		r.memo[i] = i
		return i
	}
	nd := r.k.Nodes[i]
	v := r.k.add(Node{})
	r.memo[i] = v
	for _, a := range r.k.args(&nd) {
		*a = r.copy(*a)
	}
	r.k.Nodes[v] = nd
	return v
}

// reaches reports whether node i reads a replaced node.
func (r *replica) reaches(i int, seen map[int]bool) bool {
	if _, ok := r.sub[i]; ok {
		return true
	}
	if seen[i] {
		return false
	}
	seen[i] = true
	nd := r.k.Nodes[i]
	for _, a := range r.k.args(&nd) {
		if r.reaches(*a, seen) {
			return true
		}
	}
	return false
}
