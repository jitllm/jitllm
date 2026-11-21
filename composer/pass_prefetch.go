package composer

import (
	"slices"

	"github.com/jitllm/jitllm/composer/ir"
)

// Prefetch carries an indirection's table reads from trip to trip: a load a
// loop's trip changes, of the trip alone, that another load's index reads
// is loaded for the first trip before the loop and for the next one in each
// trip; a kernel of a kind in Except is left as it is. With Staged set it
// carries instead, in a kernel of a kind in Staged alone, every load of
// device memory the trip of a loop every thread takes together reads, the
// last trip's next its own: what a trip stages into workgroup memory is in
// registers before the trip ahead of it multiplies. A sum over a weight
// matrix a split can still cut is left to it: a carried load is a second
// fold of its loop.
type Prefetch struct {
	Staged, Except []PrimKind
}

func (Prefetch) Name() string { return "prefetch" }

func (p Prefetch) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(n *Kernel) ([]*Kernel, bool, error) {
		m := n.clone()
		ok := false
		switch {
		case len(p.Staged) == 0 && !slices.Contains(p.Except, n.Kind) && !cuttable(n):
			ok = m.prefetch()
		case slices.Contains(p.Staged, n.Kind):
			ok = m.prefetchStaged()
		}
		if !ok {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

func cuttable(n *Kernel) bool {
	_, _, ok := sums(n)
	return ok && !n.Work && readsMatrix(n)
}

func (n *Kernel) prefetch() bool {
	feeds := map[int]bool{}
	for _, nd := range n.Nodes {
		if nd.Op == OpLoad {
			for _, a := range n.Operands(&nd) {
				n.mark(*a, feeds)
			}
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
			if nd.Op != OpLoad || !feeds[i] || !n.Varies(i, c) || !n.addressOfIteration(&nd, c) {
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

func (n *Kernel) prefetchStaged() bool {
	changed := false
	for c, l := range n.Loops {
		it := n.iterOf(c)
		if it < 0 || !l.Uniform || l.Spread || l.Trip < 2 {
			continue
		}
		ahead := n.bin(OpMin, ir.U32, n.bin(OpAdd, ir.U32, it, n.u32(1)), n.u32(l.Trip-1))
		for i := range n.Nodes {
			nd := n.Nodes[i]
			if nd.Op != OpLoad || n.Bufs[nd.Buf].Kind == BufShared || !n.Varies(i, c) || !n.addressOfIteration(&nd, c) {
				continue
			}
			first := n.copyOf(i, map[int]int{it: n.u32(0)})
			next := n.copyOf(i, map[int]int{it: ahead})
			f := n.add(Node{Op: OpFold, At: c, Type: nd.Type, A: next, B: first})
			n.redirect(i, f, f)
			changed = true
		}
	}
	return changed
}

// mark marks node i and what it reads, through loads' indices.
func (n *Kernel) mark(i int, seen map[int]bool) {
	if seen[i] {
		return
	}
	seen[i] = true
	nd := n.Nodes[i]
	if nd.Op == OpFold {
		return
	}
	for _, a := range n.Operands(&nd) {
		n.mark(*a, seen)
	}
}

// ofIteration reports whether node i reads no fold of loop c and no
// iteration but c's and those of the loops around it: what a copy of it
// before loop c can read.
func (n *Kernel) ofIteration(i, c int) bool {
	nd := n.Nodes[i]
	switch nd.Op {
	case OpFold:
		return !n.within(nd.At, c)
	case OpIter:
		return n.within(c, nd.At)
	}
	for _, a := range n.Operands(&nd) {
		if !n.ofIteration(*a, c) {
			return false
		}
	}
	return true
}

// addressOfIteration reports whether load nd's address reads no fold of
// loop c.
func (n *Kernel) addressOfIteration(nd *Node, c int) bool {
	for _, a := range n.Operands(nd) {
		if !n.ofIteration(*a, c) {
			return false
		}
	}
	return true
}
