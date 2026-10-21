package jit

import "github.com/jitllm/jitllm/jit/ir"

// Jam runs an outer loop of a fixed count of whole vectors whose one inner
// loop holds its sums as vectors inside that loop: each vector of the outer
// trips its copy of the inner loop's body and sums, the inner loop's loads
// that the outer trips move consecutive and the rest broadcast, and the
// sums stored a vector each. At most Sums sums are made. With Walk set the
// inner loop is a tile's pass (Loop.Walk); Spread, where set, are the
// registers the broadcast loads take in turn and Rows the one the loads
// every vector's copy reads share, sum j of fold f taking register
// f*Sums/folds + j. A step multiplies by its broadcast last.
type Jam struct {
	Lanes, Sums int
	Walk        bool
	Spread      []int
	Rows        int
}

func (Jam) Name() string { return "jam" }

func (p Jam) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		for o, l := range k.Loops {
			in := k.innerLoops(o)
			if l.Count > 0 || l.Lanes > 0 || l.Wide > 0 || l.Trip%p.Lanes != 0 || len(in) != 1 || len(k.innerLoops(in[0])) > 0 {
				continue
			}
			i := in[0]
			n := l.Trip / p.Lanes
			folds := k.foldsOf(i)
			if len(folds) == 0 || n*len(folds) > p.Sums || len(k.foldsOf(o)) > 0 || !k.jammable(o, i) {
				continue
			}
			return []*Kernel{k.jam(o, i, p)}, true, nil
		}
		return nil, false, nil
	})
}

// jammable reports whether outer loop o's iteration moves the loads of
// inner loop i and o's stores only as their last coordinate, by one, and
// o stores only the sums i makes.
func (k *Kernel) jammable(o, i int) bool {
	it := k.iterOf(o)
	if it < 0 {
		return false
	}
	hs := k.Homes()
	consecutive := func(ax []int) bool {
		if ax == nil {
			return false
		}
		for _, a := range ax[:len(ax)-1] {
			if k.readsNode(a, it) {
				return false
			}
		}
		last := ax[len(ax)-1]
		nd := k.Nodes[last]
		return last == it || nd.Op == OpAdd && (nd.B == it && !k.readsNode(nd.A, it) || nd.A == it && !k.readsNode(nd.B, it))
	}
	for x, nd := range k.Nodes {
		if nd.Op != OpLoad || x >= len(hs) || hs[x] != i {
			continue
		}
		if k.readsNode(x, it) && !consecutive(nd.Ax) {
			return false
		}
	}
	for _, s := range k.Stores {
		if s.Barrier || s.Run {
			continue
		}
		if s.At != o || k.Nodes[s.Val].Op != OpFold || k.Nodes[s.Val].At != i || !consecutive(s.Ax) {
			return false
		}
	}
	return true
}

// jam is k with loop o's trips, lanes L a vector, jammed into inner loop i.
func (k *Kernel) jam(o, i int, p Jam) *Kernel {
	L := p.Lanes
	m := k.clone()
	it := m.iterOf(o)
	n := m.Loops[o].Trip / L
	hs := m.Homes()
	var lanes []int
	for x, nd := range m.Nodes {
		if nd.Op == OpLoad && x < len(hs) && hs[x] == i && !m.readsNode(x, it) {
			lanes = append(lanes, x)
		}
	}
	bcast := map[int]int{}
	for j, x := range lanes {
		bcast[x] = m.add(Node{Op: OpBroadcast, Type: m.Nodes[x].Type, A: x})
		if j < len(p.Spread) {
			m.Nodes[bcast[x]].Hint = p.Spread[j] + 1
		}
	}
	made := len(m.Nodes)
	folds := m.foldsOf(i)
	of := make([]map[int]int, n)
	for j := range n {
		sub := map[int]int{it: m.u32(j * L)}
		for x, b := range bcast {
			sub[x] = b
		}
		of[j] = map[int]int{}
		for fi, f := range folds {
			g := m.add(Node{Op: OpFold, At: i, Type: m.Nodes[f].Type, B: m.Nodes[f].B})
			if p.Spread != nil {
				m.Nodes[g].Hint = fi*p.Sums/len(folds) + j + 1
			}
			of[j][f] = g
			sub[f] = g
		}
		for _, f := range folds {
			g := of[j][f]
			m.Nodes[g].A = m.copyKeep(m.Nodes[f].A, sub)
			if st := &m.Nodes[m.Nodes[g].A]; st.Op == OpFma && m.Nodes[st.A].Op == OpBroadcast {
				st.A, st.B = st.B, st.A
			}
		}
		of[j][it] = sub[it]
	}
	var stores []Store
	for _, s := range m.Stores {
		if s.Barrier || s.Run || s.At != o {
			stores = append(stores, s)
			continue
		}
		for j := range n {
			ns := s
			ns.Val = of[j][s.Val]
			ns.Ax = make([]int, len(s.Ax))
			for x, a := range s.Ax {
				ns.Ax[x] = m.copyKeep(a, map[int]int{it: of[j][it]})
			}
			stores = append(stores, ns)
		}
	}
	m.Stores = stores
	if p.Spread != nil {
		share := 1
		for _, nd := range m.Nodes {
			share = max(share, nd.Share+1)
		}
		for x := made; x < len(m.Nodes); x++ {
			if nd := &m.Nodes[x]; nd.Op == OpLoad && nd.Type == ir.F32 {
				nd.Hint, nd.Share = p.Rows+1, share
			}
		}
	}
	m.Loops[o].Trip, m.Loops[o].Wide = 1, L
	m.Loops[i].Wide, m.Loops[i].Walk = L, p.Walk
	m.compact()
	return m
}
