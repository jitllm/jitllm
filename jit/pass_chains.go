package jit

// Chains reassociates a workgroup thread's chain of multiply-adds from a
// constant, n of them, into N chains interleaved -- term t into chain t mod
// N, each from the constant -- summed pairwise. In a loop whose trip adds N
// copies of its body into one sum, the sum becomes N sums, copy j into sum
// j, the first carried on by the loops that continue it, and the N summed
// pairwise where the last of them is read.
type Chains struct {
	N int
}

func (Chains) Name() string { return "chains" }

func (p Chains) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if !k.Work && p.N > 1 {
			m := k.clone()
			changed := false
			for c, l := range m.Loops {
				if l.Copies == p.N {
					for _, f := range m.foldsOf(c) {
						if m.chainFold(c, f, p.N) {
							changed = true
						}
					}
				}
			}
			if !changed {
				return nil, false, nil
			}
			return []*Kernel{m}, true, nil
		}
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

// chainFold splits fold f of loop c, a chain of n adds into it a trip, into
// n folds.
func (k *Kernel) chainFold(c, f, n int) bool {
	var links []int
	x := k.Nodes[f].A
	for len(links) < n {
		nd := k.Nodes[x]
		var prev int
		switch {
		case nd.Op == OpFma:
			prev = nd.C
		case nd.Op == OpAdd:
			prev = nd.A
		default:
			return false
		}
		links = append(links, x)
		x = prev
	}
	if x != f {
		return false
	}
	t, init := k.Nodes[f].Type, k.Nodes[f].B
	sums := make([]int, n)
	for j := 0; j < n; j++ {
		link := links[n-1-j]
		if j == 0 {
			sums[0] = f
		} else {
			sums[j] = k.add(Node{Op: OpFold, At: c, Type: t, B: init})
		}
		nd := k.Nodes[link]
		if nd.Op == OpFma {
			nd.C = sums[j]
		} else {
			nd.A = sums[j]
		}
		k.Nodes[link] = nd
		if j > 0 {
			k.Nodes[sums[j]].A = link
		}
	}
	k.Nodes[f].A = links[n-1]
	last := f
	family := map[int]bool{c: true}
	for {
		next := -1
		for i, nd := range k.Nodes {
			if nd.Op == OpFold && nd.B == last && i != last {
				next = i
			}
		}
		if next < 0 {
			break
		}
		last = next
		family[k.Nodes[last].At] = true
	}
	hs := k.homes()
	acc := append([]int{last}, sums[1:]...)
	for len(acc) > 1 {
		var nx []int
		for j := 0; j+1 < len(acc); j += 2 {
			nx = append(nx, k.bin(OpAdd, t, acc[j], acc[j+1]))
		}
		if len(acc)%2 == 1 {
			nx = append(nx, acc[len(acc)-1])
		}
		acc = nx
	}
	total := acc[0]
	for i := range k.Nodes {
		nd := &k.Nodes[i]
		if nd.Op == OpFold || i == total || i < len(hs) && hs[i] >= 0 && family[hs[i]] {
			continue
		}
		if k.isCombine(i, last, sums) {
			continue
		}
		for _, a := range k.args(nd) {
			if *a == last {
				*a = total
			}
		}
	}
	return true
}

// isCombine reports whether node i is one of the adds summing the chains.
func (k *Kernel) isCombine(i, last int, sums []int) bool {
	nd := k.Nodes[i]
	if nd.Op != OpAdd {
		return false
	}
	in := func(x int) bool {
		if x == last {
			return true
		}
		for _, s := range sums {
			if s == x {
				return true
			}
		}
		return k.Nodes[x].Op == OpAdd && k.isCombine(x, last, sums)
	}
	return in(nd.A) && in(nd.B)
}
