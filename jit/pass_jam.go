package jit

import (
	"math/bits"

	"github.com/jitllm/jitllm/jit/ir"
)

// Jam runs an outer loop of whole vectors whose one inner loop holds its
// sums as vectors inside that loop: each vector of the outer trips its copy
// of the inner loop's body and sums, the inner loop's loads that the outer
// trips move consecutive and the rest broadcast, and the sums stored a
// vector each. At most Sums sums are made, less the registers an op of the
// inner loop needs besides its own (Temps); an outer loop counted at run
// time, or whose vectors' sums would be more, runs a vector a trip, the
// trips that do not fill one after it in a copy of the loops as they were.
// With Walk set the
// inner loop is a tile's pass (Loop.Walk); Spread, where set, are the
// registers the broadcast loads take in turn and Rows the one the loads
// every vector's copy reads share, sum j of fold f taking register
// f*Sums/folds + j. A step multiplies by its broadcast last.
type Jam struct {
	Lanes, Sums int
	Temps       map[Op]int
	// Strip is the most vectors a strip-mined trip runs, a power of two;
	// 0 is one.
	Strip  int
	Walk   bool
	Spread []int
	Rows   int
}

func (Jam) Name() string { return "jam" }

func (p Jam) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		for o, l := range k.Loops {
			in := k.innerLoops(o)
			if l.Lanes > 0 || l.Wide > 0 || l.Count == 0 && l.Trip%p.Lanes != 0 || len(in) != 1 || len(k.innerLoops(in[0])) > 0 {
				continue
			}
			i := in[0]
			folds := k.foldsOf(i)
			budget := p.Sums - k.temps(i, p.Temps)
			if len(folds) == 0 {
				continue
			}
			n, strip := l.Trip/p.Lanes, l.Count > 0
			if strip || n*len(folds) > budget {
				// As many vectors a trip as the budget holds, a power of two
				// so the trips that do not fill them are a mask.
				n, strip = 1, true
				for 2*n <= max(1, p.Strip) && 2*n*len(folds) <= budget && (l.Count > 0 || 2*n*p.Lanes <= l.Trip) {
					n *= 2
				}
			}
			if n*len(folds) > budget || len(k.foldsOf(o)) > 0 || !k.jammable(o, i) {
				continue
			}
			return []*Kernel{k.jam(o, i, p, strip, n)}, true, nil
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
	g := newAlg(k)
	a := g.id(atom{node: it})
	// step reports whether index x moves by one a trip of o: one times the
	// iteration and nothing else that reads it.
	step := func(x int) bool {
		w, rest := g.coef(g.of(x), a)
		if c, ok := w.constant(); !ok || c != 1 {
			return false
		}
		for m := range rest {
			for _, id := range monoIDs(m) {
				for _, nd := range g.nodes(id) {
					if k.readsNode(nd, it) {
						return false
					}
				}
			}
		}
		return true
	}
	consecutive := func(ax []int, ix int) bool {
		if ax == nil {
			return step(ix)
		}
		for _, a := range ax[:len(ax)-1] {
			if k.readsNode(a, it) {
				return false
			}
		}
		return step(ax[len(ax)-1])
	}
	for x, nd := range k.Nodes {
		if nd.Op != OpLoad || x >= len(hs) || hs[x] != i {
			continue
		}
		if k.readsNode(x, it) && !consecutive(nd.Ax, nd.A) {
			return false
		}
	}
	for _, s := range k.Stores {
		if s.Barrier || s.Run {
			continue
		}
		if s.At != o || k.Nodes[s.Val].Op != OpFold || k.Nodes[s.Val].At != i || !consecutive(s.Ax, s.Ix) {
			return false
		}
	}
	return true
}

// temps is the most registers besides its own an op of loop i needs.
func (k *Kernel) temps(i int, need map[Op]int) int {
	hs := k.Homes()
	t := 0
	for x, nd := range k.Nodes {
		if x < len(hs) && hs[x] >= 0 && k.within(hs[x], i) {
			t = max(t, need[nd.Op])
		}
	}
	return t
}

// jamCopier copies nodes into a jam's vector: the replacements of sub
// standing for theirs, iteration it the vector's first iteration where a
// load's address reads it and row, each lane's own, where a value does. A
// node read both ways is copied for each.
func (k *Kernel) jamCopier(it int, sub map[int]int, row int) func(int) int {
	memo := [2]map[int]int{{}, {}}
	var walk func(i int, addr bool) int
	walk = func(i int, addr bool) int {
		ctx := 0
		if addr {
			ctx = 1
		}
		if v, ok := memo[ctx][i]; ok {
			return v
		}
		v := i
		nd := k.Nodes[i]
		switch r, ok := sub[i]; {
		case i == it && !addr:
			v = row
		case ok && (i == it || nd.Op == OpFold || !addr):
			v = r
		case nd.Op == OpFold:
		default:
			changed := false
			ops := k.Operands(&nd)
			for _, p := range ops {
				if w := walk(*p, addr || nd.Op == OpLoad); w != *p {
					*p, changed = w, true
				}
			}
			if changed {
				v = k.add(nd)
			}
		}
		memo[ctx][i] = v
		return v
	}
	return func(i int) int { return walk(i, false) }
}

// stripTail runs outer loop o a vector of L trips a trip: o counts the
// vectors, and a copy of o and its inner loop i after it runs the trips
// that do not fill one, at the iterations after them; a fixed count of
// whole vectors has none.
func (k *Kernel) stripTail(o, i, L int) {
	U := ir.U32
	sh := uint32(bits.TrailingZeros(uint(L)))
	lo, li := k.Loops[o], k.Loops[i]
	tail := Loop{Parent: lo.Parent, Range: lo.Range, Of: o + 1}
	var vec, done int
	if lo.Count > 0 {
		n := lo.Count - 1
		vec = k.bin(OpShr, U, n, k.u32(int(sh)))
		done = k.bin(OpShl, U, vec, k.u32(int(sh)))
		tail.Count = k.bin(OpAnd, U, n, k.u32(L-1)) + 1
	} else {
		if lo.Trip%L == 0 {
			return
		}
		done = k.u32(lo.Trip / L * L)
		tail.Trip = lo.Trip % L
	}
	to := len(k.Loops)
	k.Loops = append(k.Loops, tail)
	inner := li
	inner.Parent, inner.Cursors, inner.Reg = to, nil, 0
	ti := len(k.Loops)
	k.Loops = append(k.Loops, inner)
	sub := map[int]int{}
	sub[k.iterOf(o)] = k.bin(OpAdd, U, done, k.add(Node{Op: OpIter, At: to, Type: U}))
	if x := k.iterOf(i); x >= 0 {
		sub[x] = k.add(Node{Op: OpIter, At: ti, Type: U})
	}
	folds := k.foldsOf(i)
	for _, f := range folds {
		sub[f] = k.add(Node{Op: OpFold, At: ti, Type: k.Nodes[f].Type})
	}
	for _, f := range folds {
		b := k.copyKeep(k.Nodes[f].B, sub)
		a := k.copyKeep(k.Nodes[f].A, sub)
		k.Nodes[sub[f]].A, k.Nodes[sub[f]].B = a, b
	}
	var copied []Store
	for _, s := range k.Stores {
		if s.Barrier || s.Run || s.At != o {
			continue
		}
		copied = append(copied, k.storeCopy(s, sub, to))
	}
	k.Stores = append(k.Stores, copied...)
	if lo.Count > 0 {
		k.Loops[o].Count = vec + 1
	}
}

// jam is k with loop o's trips, lanes L a vector, jammed into inner loop i.
func (k *Kernel) jam(o, i int, p Jam, runtime bool, n int) *Kernel {
	L := p.Lanes
	m := k.clone()
	it := m.iterOf(o)
	if runtime {
		m.stripTail(o, i, n*L)
	}
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
		if runtime {
			sub[it] = m.bin(OpAdd, ir.U32, m.bin(OpMul, ir.U32, it, m.u32(n*L)), m.u32(j*L))
		}
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
		// Inside an address the iteration is the vector's first; in a value
		// each lane's own.
		row := m.bin(OpAdd, ir.U32, sub[it], m.add(Node{Op: OpLanes, Type: ir.U32, Imm: int64(L)}))
		cp := m.jamCopier(it, sub, row)
		for _, f := range folds {
			g := of[j][f]
			m.Nodes[g].A = cp(m.Nodes[f].A)
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
	if runtime {
		m.Loops[o].Trip, m.Loops[o].Wide = k.Loops[o].Trip/(n*L), L
	} else {
		m.Loops[o].Trip, m.Loops[o].Wide = 1, L
	}
	m.Loops[i].Wide, m.Loops[i].Walk = L, p.Walk
	m.compact()
	return m
}
