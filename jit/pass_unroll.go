package jit

import "github.com/jitllm/jitllm/jit/ir"

// Unroll unrolls a workgroup's loops: a loop nested in a phase of at most
// Full trips with one fold and no stores of its own, all of its trips; a
// loop over positions bound as Over, Factor trips a trip. Where Vectors is
// set, a vector loop that carries a sum and stores nothing, or that runs a
// fixed count of at least Vectors trips, runs Vectors vectors a trip, the
// vectors left in a loop of one a trip after it; a sum's vector loop of
// fewer fixed trips runs them all in one.
type Unroll struct {
	Full    int
	Factor  int
	Over    Bound
	Vectors int
}

func (Unroll) Name() string { return "unroll" }

func (p Unroll) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if !k.Work && p.Vectors > 1 {
			m := k.clone()
			changed := false
			for c := 0; c < len(m.Loops); c++ {
				l := m.Loops[c]
				sum := len(m.foldsOf(c)) > 0 && !m.storesAt(c)
				fixed := l.Count == 0 && l.Trip >= p.Vectors && len(m.foldsOf(c)) == 0
				if l.Lanes > 0 && l.Copies == 0 && !m.hasInner(c) && sum && l.Count == 0 && l.Trip > 1 && l.Trip < p.Vectors {
					m.unrollShort(c)
					changed = true
					continue
				}
				if l.Lanes > 0 && l.Copies == 0 && !m.hasInner(c) && (sum || fixed) {
					m.unrollVectors(c, p.Vectors)
					changed = true
					c++
				}
			}
			if !changed {
				return nil, false, nil
			}
			return []*Kernel{m}, true, nil
		}
		if !k.Work {
			return nil, false, nil
		}
		m := k.clone()
		changed := false
		for c := range m.Loops {
			l := m.Loops[c]
			folds := m.foldsOf(c)
			switch {
			case p.Full > 0 && l.Parent >= 0 && l.Count == 0 && l.Trip > 0 && l.Trip <= p.Full && len(folds) > 0 && !m.storesAt(c) &&
				!m.hasInner(c) && len(m.itersOf(c)) > 0:
				for _, f := range folds {
					m.unrollFull(c, f)
				}
				m.Loops[c].Trip = 0
				changed = true
			case p.Factor > 1 && l.Bound == p.Over && l.Bound != Unbound && l.Count == 0 && l.Trip > 0 && !m.storesAt(c) && len(folds) > 0:
				m.unrollBy(c, p.Factor)
				changed = true
			}
		}
		if !changed {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

// unrollFull is loop c's fold f as a chain of its trips from the start, the
// loop gone.
func (k *Kernel) unrollFull(c, f int) {
	its := k.itersOf(c)
	step := k.Nodes[f].A
	acc := k.Nodes[f].B
	for j := 0; j < k.Loops[c].Trip; j++ {
		sub := map[int]int{f: acc}
		at := k.u32(j)
		for _, it := range its {
			sub[it] = at
		}
		acc = k.copyFolded(step, sub)
	}
	k.redirect(f, acc, acc)
}

// unrollBy runs loop c u trips a trip: each fold stepped once for each, in
// order, the trip's position it*u+j.
func (k *Kernel) unrollBy(c, u int) {
	its := k.itersOf(c)
	it := k.add(Node{Op: OpIter, At: c, Type: ir.U32})
	folds := k.foldsOf(c)
	steps := make([]int, len(folds))
	for i, f := range folds {
		steps[i] = k.Nodes[f].A
	}
	last := append([]int(nil), folds...)
	for j := 0; j < u; j++ {
		at := k.bin(OpAdd, ir.U32, k.bin(OpMul, ir.U32, it, k.u32(u)), k.u32(j))
		sub := map[int]int{}
		for _, o := range its {
			sub[o] = at
		}
		for i, f := range folds {
			sub[f] = last[i]
		}
		for i := range folds {
			last[i] = k.copyFolded(steps[i], sub)
		}
	}
	for i, f := range folds {
		k.Nodes[f].A = last[i]
	}
	l := &k.Loops[c]
	l.Trip = (l.Trip + u - 1) / u
	l.Unit *= u
}

// copyFolded is node i with sub's nodes replaced, the replaced nodes' readers
// copied, the rest shared; a fold is replaced or shared, never copied.
func (k *Kernel) copyFolded(i int, sub map[int]int) int {
	return k.copyOf(i, sub)
}

// unrollVectors runs vector loop c n vectors a trip, its body copied n
// times, copy j at iteration j*Lanes on, and the vectors left in a loop
// after it that carries its sums on.
func (k *Kernel) unrollVectors(c, n int) {
	U := ir.U32
	l := k.Loops[c]
	L := l.Lanes
	it := k.iterOf(c)
	var rem Loop
	var done int
	rem.Parent, rem.Lanes, rem.Range, rem.Of = l.Parent, L, l.Range, c+1
	if l.Of > 0 {
		rem.Of = l.Of
	}
	if l.Count > 0 {
		v := l.Count - 1
		k.Loops[c].Count = k.bin(OpShr, U, v, k.u32(bitsOf(n))) + 1
		rem.Count = k.bin(OpAnd, U, v, k.u32(n-1)) + 1
		done = k.bin(OpMul, U, k.Loops[c].Count-1, k.u32(n*L))
	} else {
		rem.Trip = l.Trip % n
		k.Loops[c].Trip = l.Trip / n
		done = k.u32(l.Trip / n * n * L)
	}
	k.Loops[c].Copies = n
	r := k.insertLoop(c+1, rem)
	ri := k.add(Node{Op: OpIter, At: r, Type: U})
	if k.storesAt(c) {
		k.unrollStores(c, r, n, done)
	}
	for _, f := range k.foldsOf(c) {
		step := k.Nodes[f].A
		acc := f
		for j := 0; j < n; j++ {
			at := it
			if j > 0 {
				at = k.bin(OpAdd, U, it, k.u32(j*L))
			}
			acc = k.copyOf(step, map[int]int{f: acc, it: at})
		}
		k.Nodes[f].A = acc
		fr := k.add(Node{Op: OpFold, At: r, Type: k.Nodes[f].Type, B: f})
		k.Nodes[fr].A = k.copyOf(step, map[int]int{f: fr, it: k.bin(OpAdd, U, done, ri)})
		hs := k.homes()
		for i := range k.Nodes {
			if i == fr || i < len(hs) && hs[i] >= 0 && (k.within(hs[i], c) || k.within(hs[i], r)) {
				continue
			}
			for _, a := range k.args(&k.Nodes[i]) {
				if *a == f {
					*a = fr
				}
			}
		}
		for i := range k.Stores {
			if k.Stores[i].Val == f {
				k.Stores[i].Val = fr
			}
		}
	}
}

func bitsOf(n int) int {
	b := 0
	for 1<<b < n {
		b++
	}
	return b
}

// unrollStores writes loop c's stores once for each of n copies, copy j
// at iteration j*Lanes on, and the remainder loop r's once more.
func (k *Kernel) unrollStores(c, r, n, done int) {
	U := ir.U32
	L := k.Loops[c].Lanes
	it := k.iterOf(c)
	ri := k.iterOf(r)
	var out []Store
	for _, s := range k.Stores {
		if s.At != c || s.Barrier || s.Run {
			out = append(out, s)
			continue
		}
		for j := 0; j < n; j++ {
			at := it
			if j > 0 {
				at = k.bin(OpAdd, U, it, k.u32(j*L))
			}
			out = append(out, k.storeCopy(s, map[int]int{it: at}, c))
		}
	}
	for _, s := range k.Stores {
		if s.At == c && !s.Barrier && !s.Run {
			if ri < 0 {
				ri = k.add(Node{Op: OpIter, At: r, Type: U})
			}
			out = append(out, k.storeCopy(s, map[int]int{it: k.bin(OpAdd, U, done, ri)}, r))
		}
	}
	k.Stores = out
}

// storeCopy is store s with its nodes copied under sub, run in loop at.
func (k *Kernel) storeCopy(s Store, sub map[int]int, at int) Store {
	n := s
	n.At = at
	n.Val = k.copyOf(s.Val, sub)
	if s.Ax != nil {
		n.Ax = make([]int, len(s.Ax))
		for j, a := range s.Ax {
			n.Ax[j] = k.copyOf(a, sub)
		}
	} else {
		n.Ix = k.copyOf(s.Ix, sub)
	}
	return n
}

// unrollShort runs vector loop c's fixed trips, fewer than a trip of the
// unrolled form, as copies in one trip.
func (k *Kernel) unrollShort(c int) {
	n := k.Loops[c].Trip
	L := k.Loops[c].Lanes
	it := k.iterOf(c)
	for _, f := range k.foldsOf(c) {
		step := k.Nodes[f].A
		acc := f
		for j := 0; j < n; j++ {
			at := it
			if j > 0 {
				at = k.bin(OpAdd, ir.U32, it, k.u32(j*L))
			}
			acc = k.copyOf(step, map[int]int{f: acc, it: at})
		}
		k.Nodes[f].A = acc
	}
	k.Loops[c].Copies, k.Loops[c].Trip = n, 1
}
