package jit

// LoopFuse fuses, within a workgroup, a loop into the loop before it where
// both run over the same positions the same number of trips and the later
// reads what the earlier wrote only at the position of its own trip: the
// later's work done in the earlier's trips, its reads of the earlier's
// values those values.
type LoopFuse struct{}

func (LoopFuse) Name() string { return "loopfuse" }

func (p LoopFuse) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if !k.Work {
			return nil, false, nil
		}
		m := k.clone()
		changed := false
		for {
			a, b, ok := m.fusable()
			if !ok {
				break
			}
			m.fuseLoops(a, b)
			changed = true
		}
		if !changed {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

// topOrder is the top-level loops in the order the stores reach them, a
// loop whose folds a store reads before the store's own, a barrier -2.
func (k *Kernel) topOrder() []int {
	var out []int
	seen := map[int]bool{}
	top := func(l int) {
		for k.Loops[l].Parent >= 0 {
			l = k.Loops[l].Parent
		}
		if !seen[l] {
			seen[l] = true
			out = append(out, l)
		}
	}
	for _, s := range k.Stores {
		if s.Barrier {
			out = append(out, -2)
			continue
		}
		for _, f := range k.foldsRead(s) {
			top(k.Nodes[f].At)
		}
		if s.At >= 0 {
			top(s.At)
		}
	}
	return out
}

// foldsRead is the folds a store at the element's end reads.
func (k *Kernel) foldsRead(s Store) []int {
	var out []int
	seen := map[int]bool{}
	var walk func(i int)
	walk = func(i int) {
		if seen[i] {
			return
		}
		seen[i] = true
		nd := k.Nodes[i]
		if nd.Op == OpFold {
			out = append(out, i)
			return
		}
		for _, x := range k.args(&nd) {
			walk(*x)
		}
	}
	for _, x := range k.storeArgs(&s) {
		walk(*x)
	}
	return out
}

// fusable finds two loops over the same positions, the second after the
// first with at most a barrier between, the second's reads of what the
// first stores at its own trip's position.
func (k *Kernel) fusable() (a, b int, ok bool) {
	order := k.topOrder()
	for i := 0; i < len(order); i++ {
		a = order[i]
		if a < 0 || !k.Loops[a].Range || k.hasInner(a) {
			continue
		}
		j := i + 1
		for j < len(order) && order[j] == -2 {
			j++
		}
		if j >= len(order) || order[j] < 0 {
			continue
		}
		b = order[j]
		if !k.Loops[b].Range || k.Loops[b].Trip != k.Loops[a].Trip || k.hasInner(b) || k.storesAt(b) {
			continue
		}
		if k.readsOnlyAtTrip(a, b) {
			return a, b, true
		}
	}
	return 0, 0, false
}

func (k *Kernel) hasInner(c int) bool {
	for _, l := range k.Loops {
		if l.Parent == c {
			return true
		}
	}
	return false
}

func (k *Kernel) storesAt(c int) bool {
	for _, s := range k.Stores {
		if s.At == c && !s.Barrier && !s.Run {
			return true
		}
	}
	return false
}

// readsOnlyAtTrip reports whether loop b reads loop a's stores, each read
// at the position of b's own trip where one of them writes.
func (k *Kernel) readsOnlyAtTrip(a, b int) bool {
	ia, ib := k.iterOf(a), k.iterOf(b)
	if ia < 0 || ib < 0 {
		return false
	}
	written := map[Operand]bool{}
	for _, s := range k.Stores {
		if s.At == a && !s.Barrier && !s.Run {
			written[k.Bufs[s.Buf].Operand] = true
		}
	}
	any := false
	for _, f := range k.foldsOf(b) {
		for o := range written {
			for _, l := range k.loadsFrom(k.Nodes[f].A, o) {
				if _, ok := k.storeAtTrip(a, l, ib, ia); !ok {
					return false
				}
				any = true
			}
		}
	}
	return any
}

// storeAtTrip is loop a's store load l, at loop b's iteration ib, reads.
func (k *Kernel) storeAtTrip(a, l, ib, ia int) (Store, bool) {
	for _, s := range k.Stores {
		if s.At == a && !s.Barrier && !s.Run && k.Bufs[s.Buf].Operand == k.Bufs[k.Nodes[l].Buf].Operand && k.atTrip(l, ib, s, ia) {
			return s, true
		}
	}
	return Store{}, false
}

// atTrip reports whether load l, at loop b's iteration ib, reads where
// store s writes at loop a's iteration ia.
func (k *Kernel) atTrip(l, ib int, s Store, ia int) bool {
	nd := k.Nodes[l]
	if len(nd.Ax) != len(s.Ax) || s.Ax == nil {
		return false
	}
	for i := range s.Ax {
		x, y := nd.Ax[i], s.Ax[i]
		if xi, yi := k.Nodes[x], k.Nodes[y]; xi.Op == OpIter && yi.Op == OpIter && xi.At == k.Nodes[ib].At && yi.At == k.Nodes[ia].At {
			continue
		}
		if !k.same(x, y) {
			return false
		}
	}
	return true
}

// fuseLoops does loop b's work in loop a's trips.
func (k *Kernel) fuseLoops(a, b int) {
	ia, ib := k.iterOf(a), k.iterOf(b)
	for _, f := range k.foldsOf(b) {
		for _, s := range k.Stores {
			if s.At != a || s.Barrier || s.Run {
				continue
			}
			for _, l := range k.loadsFrom(k.Nodes[f].A, k.Bufs[s.Buf].Operand) {
				if st, ok := k.storeAtTrip(a, l, ib, ia); ok {
					k.redirect(l, st.Val, st.Val)
				}
			}
		}
		k.Nodes[f].At = a
	}
	for i, nd := range k.Nodes {
		if nd.Op == OpIter && nd.At == b {
			k.redirect(i, ia, ia)
		}
	}
	for i := range k.Stores {
		if k.Stores[i].At == b {
			k.Stores[i].At = a
		}
	}
	var keep []Store
	for _, s := range k.Stores {
		if !(s.Run && s.At == a && len(keep) > 0 && runOf(keep, a)) {
			keep = append(keep, s)
		}
	}
	k.Stores = keep
	last := -1
	for i, st := range k.Stores {
		if st.At == a {
			last = i
		}
	}
	for j := last + 1; last >= 0 && j < len(k.Stores); j++ {
		if k.Stores[j].Barrier {
			k.Stores = append(k.Stores[:j], k.Stores[j+1:]...)
			return
		}
		if !k.Stores[j].Run {
			return
		}
	}
}

// runOf reports whether stores already run loop a.
func runOf(ss []Store, a int) bool {
	for _, s := range ss {
		if s.At == a && !s.Barrier {
			return true
		}
	}
	return false
}
