package jit

import (
	"math/bits"

	"github.com/jitllm/jitllm/jit/ir"
)

// Vectorize runs an innermost loop that carries no value and reads and
// writes consecutive elements of its iteration Lanes iterations a trip, one
// in each lane, and the trips that do not fill a vector after it in a loop
// of one a trip.
type Vectorize struct {
	Lanes int
}

func (Vectorize) Name() string { return "vectorize" }

func (p Vectorize) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		m := k.clone()
		changed := false
		for c := range m.Loops {
			if m.vectorizable(c) {
				m.vectorize(c, p.Lanes)
				changed = true
			}
		}
		if !changed {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

func (k *Kernel) vectorizable(c int) bool {
	l := k.Loops[c]
	if l.Lanes > 0 || len(k.innerLoops(c)) > 0 || len(k.foldsOf(c)) > 0 || k.iterOf(c) < 0 {
		return false
	}
	it := k.iterOf(c)
	stores := 0
	for _, s := range k.Stores {
		if s.At != c || s.Barrier || s.Run {
			continue
		}
		stores++
		if !k.consecutive(s.Buf, s.Ix, s.Ax, it, c) {
			return false
		}
	}
	if stores == 0 {
		return false
	}
	for i, nd := range k.Nodes {
		if nd.Op == OpLoad && k.varies(i, c) && !k.consecutive(nd.Buf, nd.A, nd.Ax, it, c) {
			return false
		}
	}
	return true
}

// consecutive reports whether an access of buffer b at flat index ix, or at
// coordinates ax of its shape, takes consecutive elements over loop c's
// iterations it: the last coordinate the iteration, the others fixed in c.
func (k *Kernel) consecutive(b, ix int, ax []int, it, c int) bool {
	if ax != nil {
		for _, a := range ax[:len(ax)-1] {
			if k.varies(a, c) {
				return false
			}
		}
		return ax[len(ax)-1] == it
	}
	if ix == it {
		return true
	}
	nd := k.Nodes[ix]
	return nd.Op == OpAdd && (nd.B == it && !k.varies(nd.A, c) || nd.A == it && !k.varies(nd.B, c))
}

func (k *Kernel) vectorize(c, L int) {
	l := k.Loops[c]
	U := ir.U32
	shift := bits.TrailingZeros(uint(L))
	var tailCount, done int
	tail := Loop{Parent: l.Parent, Range: l.Range}
	if l.Count > 0 {
		n := l.Count - 1
		vec := k.bin(OpShr, U, n, k.u32(shift))
		k.Loops[c].Count = vec + 1
		tailCount = k.bin(OpAnd, U, n, k.u32(L-1))
		tail.Count = tailCount + 1
		done = k.bin(OpShl, U, vec, k.u32(shift))
	} else {
		tail.Trip = l.Trip % L
		k.Loops[c].Trip = l.Trip / L
		done = k.u32(l.Trip / L * L)
	}
	k.Loops[c].Lanes = L
	t := len(k.Loops)
	k.Loops = append(k.Loops, tail)
	ti := k.add(Node{Op: OpIter, At: t, Type: U})
	sub := map[int]int{k.iterOf(c): k.bin(OpAdd, U, done, ti)}
	var stores []Store
	last := -1
	for i, s := range k.Stores {
		if s.At != c || s.Barrier || s.Run {
			continue
		}
		last = i
		n := s
		n.At = t
		n.Val = k.copyOf(s.Val, sub)
		if s.Ax != nil {
			n.Ax = make([]int, len(s.Ax))
			for j, a := range s.Ax {
				n.Ax[j] = k.copyOf(a, sub)
			}
		} else {
			n.Ix = k.copyOf(s.Ix, sub)
		}
		stores = append(stores, n)
	}
	k.Stores = append(k.Stores[:last+1:last+1], append(stores, k.Stores[last+1:]...)...)
}
