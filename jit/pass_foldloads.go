package jit

// FoldLoads takes a load its one reader can read from memory as that
// reader's operand: the second multiplicand of a sum's step in a vector
// loop, or of a map's operation in a vector loop no scalar tail follows.
type FoldLoads struct{}

func (FoldLoads) Name() string { return "foldloads" }

func (p FoldLoads) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		var m *Kernel
		hs := k.homes()
		for i, nd := range k.Nodes {
			var b int
			switch nd.Op {
			case OpFma, OpMul, OpAdd:
				b = nd.B
			default:
				continue
			}
			ld := k.Nodes[b]
			if ld.Op != OpLoad || ld.Mem || k.uses(b) != 1 {
				continue
			}
			c := hs[i]
			if c < 0 || k.Loops[c].Lanes == 0 {
				continue
			}
			sum := nd.Op == OpFma && k.Nodes[nd.C].Op == OpFold
			if !sum && (nd.Op == OpFma || k.hasTail(c)) {
				continue
			}
			if m == nil {
				m = k.clone()
			}
			m.Nodes[b].Mem = true
		}
		if m == nil {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

// uses is how many nodes and stores read node i.
func (k *Kernel) uses(i int) int {
	n := 0
	for _, nd := range k.Nodes {
		for _, a := range k.args(&nd) {
			if *a == i {
				n++
			}
		}
	}
	for _, s := range k.Stores {
		if !s.Barrier && !s.Run {
			for _, a := range k.storeArgs(&s) {
				if *a == i {
					n++
				}
			}
		}
	}
	return n
}

// hasTail reports whether a scalar loop takes vector loop c's last trips.
func (k *Kernel) hasTail(c int) bool {
	t := k.tailOf(c)
	return t >= 0 && k.Loops[t].Lanes == 0
}
