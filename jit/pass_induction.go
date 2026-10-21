package jit

import "github.com/jitllm/jitllm/jit/ir"

// Induction carries a loop's iteration as a value where the code reads it
// as data rather than as an address: a sum of the loop from 0, one more
// each trip.
type Induction struct{}

func (Induction) Name() string { return "induction" }

func (p Induction) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		var m *Kernel
		live := k.live()
		for c := range k.Loops {
			it := k.iterOf(c)
			if it < 0 || !k.readByLive(it, live) || k.addresses(it) {
				continue
			}
			if m == nil {
				m = k.clone()
			}
			f := m.add(Node{Op: OpFold, At: c, Type: ir.I32, B: m.add(Node{Op: OpConst, Type: ir.I32})})
			m.redirect(it, f, f)
			m.Nodes[f].A = m.bin(OpAdd, ir.I32, f, m.add(Node{Op: OpConst, Type: ir.I32, Imm: 1}))
		}
		if m == nil {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

// addresses reports whether node i feeds an index a load or a store reads.
func (k *Kernel) addresses(i int) bool {
	for _, nd := range k.Nodes {
		if nd.Op == OpLoad {
			for _, a := range k.args(&nd) {
				if k.readsNode(*a, i) {
					return true
				}
			}
		}
	}
	for _, s := range k.Stores {
		if !s.Barrier && !s.Run && (k.readsNode(s.Ix, i) || s.Ax != nil && func() bool {
			for _, a := range s.Ax {
				if k.readsNode(a, i) {
					return true
				}
			}
			return false
		}()) {
			return true
		}
	}
	return false
}

// readsNode reports whether node i is computed from node x.
func (k *Kernel) readsNode(i, x int) bool {
	if i == x {
		return true
	}
	nd := k.Nodes[i]
	if nd.Op == OpFold {
		return false
	}
	for _, a := range k.args(&nd) {
		if k.readsNode(*a, x) {
			return true
		}
	}
	return false
}

// readByLive reports whether a node the code computes reads node i.
func (k *Kernel) readByLive(i int, live map[int]bool) bool {
	for j, nd := range k.Nodes {
		if !live[j] {
			continue
		}
		for _, a := range k.args(&nd) {
			if *a == i {
				return true
			}
		}
	}
	return false
}
