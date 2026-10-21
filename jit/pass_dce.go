package jit

// DeadStores drops the stores of a decomposition's temporaries that no
// kernel of the part reads.
type DeadStores struct{}

func (DeadStores) Name() string { return "deadstores" }

func (p DeadStores) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	read := map[Operand]bool{}
	for _, k := range ks {
		for i := range k.Live() {
			if nd := k.Nodes[i]; nd.Op == OpLoad {
				read[k.Bufs[nd.Buf].Operand] = true
			}
		}
	}
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		var keep []Store
		for _, s := range k.Stores {
			if !s.Barrier && !s.Run {
				if o := k.Bufs[s.Buf].Operand; o.Temp && !read[o] && k.Bufs[s.Buf].Kind != BufShared {
					continue
				}
			}
			keep = append(keep, s)
		}
		if len(keep) == len(k.Stores) {
			return nil, false, nil
		}
		m := k.clone()
		m.Stores = keep
		return []*Kernel{m}, true, nil
	})
}

// Live is the nodes k's stores and loop counts reach.
func (k *Kernel) Live() map[int]bool {
	seen := map[int]bool{}
	var walk func(i int)
	walk = func(i int) {
		if seen[i] {
			return
		}
		seen[i] = true
		nd := k.Nodes[i]
		for _, a := range k.Operands(&nd) {
			walk(*a)
		}
	}
	for _, s := range k.Stores {
		if !s.Barrier && !s.Run {
			for _, a := range k.storeArgs(&s) {
				walk(*a)
			}
		}
	}
	for _, l := range k.Loops {
		if l.Count > 0 {
			walk(l.Count - 1)
		}
	}
	return seen
}
