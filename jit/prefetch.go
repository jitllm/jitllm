package jit

import "github.com/jitllm/jitllm/jit/ir"

// Prefetch carries an indirection's table reads from trip to trip: a load a
// loop's trip changes, of the trip alone, that another load's index reads
// is loaded for the first trip before the loop and for the next one in each
// trip.
type Prefetch struct{}

func (Prefetch) Name() string { return "prefetch" }

func (p Prefetch) Apply(prog []*Nest, t Target) ([]*Nest, error) {
	return eachNest(p.Name(), prog, func(n *Nest) ([]*Nest, bool, error) {
		m := n.clone()
		if !m.prefetch() {
			return nil, false, nil
		}
		return []*Nest{m}, true, nil
	})
}

func (n *Nest) prefetch() bool {
	feeds := map[int]bool{}
	for _, nd := range n.Nodes {
		if nd.Op == OpLoad {
			n.mark(nd.A, feeds)
		}
	}
	changed := false
	for c := range n.Loops {
		it := n.iterOf(c)
		if it < 0 {
			continue
		}
		for i := range n.Nodes {
			nd := n.Nodes[i]
			if nd.Op != OpLoad || !feeds[i] || !n.varies(i, c) || !n.ofIteration(nd.A, c) {
				continue
			}
			first := n.copyOf(i, map[int]int{it: n.u32(0)})
			next := n.copyOf(i, map[int]int{it: n.bin(OpAdd, ir.U32, it, n.u32(1))})
			f := n.add(Node{Op: OpFold, At: c, Type: nd.Type, A: next, B: first})
			n.redirect(i, f, f)
			changed = true
		}
	}
	return changed
}

// mark marks node i and what it reads, through loads' indices.
func (n *Nest) mark(i int, seen map[int]bool) {
	if seen[i] {
		return
	}
	seen[i] = true
	nd := n.Nodes[i]
	if nd.Op == OpFold {
		return
	}
	for _, a := range n.args(&nd) {
		n.mark(*a, seen)
	}
}

// ofIteration reports whether node i reads no fold of loop c.
func (n *Nest) ofIteration(i, c int) bool {
	nd := n.Nodes[i]
	if nd.Op == OpFold {
		return !n.within(nd.At, c)
	}
	for _, a := range n.args(&nd) {
		if !n.ofIteration(*a, c) {
			return false
		}
	}
	return true
}
