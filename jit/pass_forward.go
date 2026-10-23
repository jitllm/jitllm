package jit

// Forward replaces a workgroup kernel's read of what it stored earlier at
// the element's end, at the same place of the same buffer, with the stored
// value.
type Forward struct{}

func (Forward) Name() string { return "forward" }

func (p Forward) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if !k.Work {
			return nil, false, nil
		}
		m := k.clone()
		changed := false
		for i, s := range m.Stores {
			if s.Barrier || s.Run || s.At != -1 {
				continue
			}
			for j := i + 1; j < len(m.Stores); j++ {
				t := m.Stores[j]
				if t.Barrier || t.Run {
					continue
				}
				o := m.Bufs[s.Buf].Operand
				if m.Bufs[t.Buf].Operand == o && m.samePlace(s, t) {
					break
				}
				for _, root := range m.storeArgs(&t) {
					for _, l := range m.loadsFrom(*root, o) {
						if m.placeOf(l, s) {
							m.redirect(l, s.Val, s.Val)
							changed = true
						}
					}
				}
			}
		}
		if !changed {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

// loadsFrom is the loads of operand o node root reads.
func (k *Kernel) loadsFrom(root int, o Operand) []int {
	var out []int
	seen := map[int]bool{}
	var walk func(i int)
	walk = func(i int) {
		if seen[i] {
			return
		}
		seen[i] = true
		nd := k.Nodes[i]
		if nd.Op == OpLoad && k.Bufs[nd.Buf].Operand == o {
			out = append(out, i)
		}
		for _, a := range k.Operands(&nd) {
			walk(*a)
		}
	}
	walk(root)
	return out
}

// placeOf reports whether load l reads where store s writes.
func (k *Kernel) placeOf(l int, s Store) bool {
	nd := k.Nodes[l]
	if s.Ax != nil {
		if len(nd.Ax) != len(s.Ax) {
			return false
		}
		for i := range s.Ax {
			if !k.same(nd.Ax[i], s.Ax[i]) {
				return false
			}
		}
		return true
	}
	return nd.Ax == nil && k.same(nd.A, s.Ix)
}

func (k *Kernel) samePlace(s, t Store) bool {
	if (s.Ax == nil) != (t.Ax == nil) || len(s.Ax) != len(t.Ax) {
		return false
	}
	if s.Ax == nil {
		return k.same(s.Ix, t.Ix)
	}
	for i := range s.Ax {
		if !k.same(s.Ax[i], t.Ax[i]) {
			return false
		}
	}
	return true
}

// same reports whether nodes a and b of k compute the same value: the
// same node, or the same operation of the same operands.
func (k *Kernel) same(a, b int) bool {
	if a == b {
		return true
	}
	x, y := k.Nodes[a], k.Nodes[b]
	if x.Op != y.Op || x.Type != y.Type || x.Imm != y.Imm || x.At != y.At || x.Buf != y.Buf ||
		x.Op == OpFold || x.Op == OpLoad || x.Op == OpExchange || x.Op == OpGroupSum || x.Op == OpGroupMax {
		return false
	}
	if x.Op == OpConst || x.Op == OpCoord || x.Op == OpIter || x.Op == OpThread {
		return x.A == y.A
	}
	xa, ya := k.Operands(&x), k.Operands(&y)
	if len(xa) != len(ya) {
		return false
	}
	for i := range xa {
		if !k.same(*xa[i], *ya[i]) {
			return false
		}
	}
	return true
}
