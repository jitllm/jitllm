package jit

// Guards marks the loops the caller never runs for no trips: a scalar loop
// whose count is an argument word it alone reads, the call's length rather
// than a count made from one (a tail's may be none), and that carries a sum
// or stores one: the rows of a reduction and the positions of a history.
type Guards struct{}

func (Guards) Name() string { return "guards" }

func (p Guards) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		var m *Kernel
		for c, l := range k.Loops {
			if l.Nonempty || l.Lanes > 0 || l.Count == 0 || k.Nodes[l.Count-1].Op != OpParam || k.countReaders(l.Count-1) != 1 || !k.lengthWord(l.Count-1) {
				continue
			}
			if !k.reduces(c) {
				continue
			}
			if m == nil {
				m = k.clone()
			}
			m.Loops[c].Nonempty = true
		}
		if m == nil {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

// lengthWord reports whether argument word p is a length the call states,
// not one the call computes from it.
func (k *Kernel) lengthWord(p int) bool {
	for _, a := range k.Args {
		if a.Buf < 0 && a.Field == int(k.Nodes[p].Imm) && a.Node > 0 {
			return k.Nodes[a.Node-1].Op == OpLength
		}
	}
	return true
}

func (k *Kernel) countReaders(i int) int {
	n := 0
	for _, l := range k.Loops {
		if l.Count == i+1 {
			n++
		}
	}
	for j, nd := range k.Nodes {
		if j == i {
			continue
		}
		for _, a := range k.Operands(&nd) {
			if *a == i {
				n++
			}
		}
	}
	return n
}

// reduces reports whether loop c, or a loop in it, carries a sum, or c
// stores what a sum or its lanes' sum made.
func (k *Kernel) reduces(c int) bool {
	for d := range k.Loops {
		if k.within(d, c) && len(k.foldsOf(d)) > 0 {
			return true
		}
	}
	for _, s := range k.Stores {
		if s.At == c && !s.Barrier && !s.Run && k.computedFrom(s.Val, OpHSum) {
			return true
		}
	}
	return false
}
