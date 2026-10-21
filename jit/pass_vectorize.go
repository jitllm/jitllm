package jit

import (
	"math/bits"

	"github.com/jitllm/jitllm/jit/ir"
)

// Vectorize runs an innermost loop that reads and writes consecutive
// elements of its iteration Lanes iterations a trip, one in each lane, and
// the trips that do not fill a vector after it in a loop of one a trip. A
// sum the loop carries becomes a sum in each lane, the tail adding into its
// first lane, and the lanes are summed where the sum is read.
type Vectorize struct {
	Lanes int
}

func (Vectorize) Name() string { return "vectorize" }

func (p Vectorize) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		m := k.clone()
		changed := false
		for c := 0; c < len(m.Loops); c++ {
			if m.vectorizable(c) {
				m.vectorize(c, p.Lanes)
				changed = true
				c++
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
	if l.Lanes > 0 || len(k.innerLoops(c)) > 0 || k.iterOf(c) < 0 {
		return false
	}
	folds := k.foldsOf(c)
	for _, f := range folds {
		if !k.sumFold(f) || k.Nodes[f].Type != ir.F32 {
			return false
		}
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
	if stores == 0 && len(folds) == 0 {
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
		last := ax[len(ax)-1]
		if last == it {
			return true
		}
		nd := k.Nodes[last]
		return nd.Op == OpAdd && (nd.B == it && !k.varies(nd.A, c) || nd.A == it && !k.varies(nd.B, c))
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
	tail := Loop{Parent: l.Parent, Range: l.Range, Of: c + 1}
	if l.Of > 0 {
		tail.Of = l.Of
	}
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
	k.broadcastInto(c)
	t := k.insertLoop(c+1, tail)
	ti := k.add(Node{Op: OpIter, At: t, Type: U})
	sub := map[int]int{k.iterOf(c): k.bin(OpAdd, U, done, ti)}
	for _, f := range k.foldsOf(c) {
		ft := k.add(Node{Op: OpFold, At: t, Type: k.Nodes[f].Type, B: f})
		step := k.Nodes[f].A
		sub[f] = ft
		k.Nodes[ft].A = k.copyOf(step, sub)
		h := k.add(Node{Op: OpHSum, Type: k.Nodes[f].Type, A: ft})
		hs := k.homes()
		for i := range k.Nodes {
			if i == step || i == ft || i == h || i < len(hs) && hs[i] >= 0 && (k.within(hs[i], c) || k.within(hs[i], t)) {
				continue
			}
			for _, a := range k.args(&k.Nodes[i]) {
				if *a == f {
					*a = h
				}
			}
		}
		for i := range k.Stores {
			if k.Stores[i].Val == f {
				k.Stores[i].Val = h
			}
		}
	}
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

// broadcastInto puts each value made from a lane sum that vector loop c
// reads, made outside it, in every lane once before the loop.
func (k *Kernel) broadcastInto(c int) {
	hs := k.homes()
	made := map[int]int{}
	for i := range k.Nodes {
		if hs[i] >= 0 && k.within(hs[i], c) || k.Nodes[i].Op == OpBroadcast {
			for _, a := range k.args(&k.Nodes[i]) {
				x := *a
				if x >= len(hs) || hs[x] >= 0 && k.within(hs[x], c) || k.Nodes[x].Type != ir.F32 || !k.computedFrom(x, OpHSum) {
					continue
				}
				b, ok := made[x]
				if !ok {
					b = k.add(Node{Op: OpBroadcast, Type: ir.F32, A: x})
					made[x] = b
				}
				*a = b
			}
		}
	}
}
