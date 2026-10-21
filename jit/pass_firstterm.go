package jit

// FirstTerm starts a largest or least fold that starts from a constant from
// its first term, its loop left whole since taking a term twice changes
// neither: where no store reads the fold and every reader runs in a loop of
// the fold's count, so a loop of no trips reads nothing it made.
type FirstTerm struct{}

func (FirstTerm) Name() string { return "firstterm" }

func (p FirstTerm) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		var m *Kernel
		hs := k.Homes()
		for f, nd := range k.Nodes {
			if nd.Op != OpFold || k.Nodes[nd.B].Op != OpConst || k.extremeTerm(f) < 0 || len(k.Loops[nd.At].Cursors) > 0 {
				continue
			}
			for _, g := range k.foldsOf(nd.At) {
				if k.carriedBy(g) == f {
					f = -1
					break
				}
			}
			if f < 0 || !k.readAlongside(f, hs) {
				continue
			}
			it := k.iterOf(nd.At)
			if it < 0 {
				continue
			}
			if m == nil {
				m = k.clone()
			}
			m.Nodes[f].B = m.copyOf(m.extremeTerm(f), map[int]int{it: m.u32(0)})
		}
		if m == nil {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

// readAlongside reports whether fold f is read by no store and only in loops
// that run its loop's count.
func (k *Kernel) readAlongside(f int, hs []int) bool {
	for _, s := range k.Stores {
		for _, a := range k.storeArgs(&s) {
			if *a == f {
				return false
			}
		}
	}
	l := k.Loops[k.Nodes[f].At]
	for i, nd := range k.Nodes {
		if i == k.Nodes[f].A {
			continue
		}
		for _, a := range k.Operands(&nd) {
			if *a != f {
				continue
			}
			if hs[i] < 0 {
				return false
			}
			r := k.Loops[hs[i]]
			if r.Count != l.Count || r.Trip != l.Trip || hs[i] == k.Nodes[f].At {
				return false
			}
		}
	}
	return true
}
