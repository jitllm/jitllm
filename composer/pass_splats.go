package composer

import "github.com/jitllm/jitllm/composer/ir"

// Splats keeps an address in a general register where a vector also reads
// it: a node of an access's index that the vector words would take (a
// compare of an index beside a word of data, a jammed loop's base index
// beside each lane's), and an index, a length or an argument word a vector
// word reads, is read by its vector readers through a broadcast of it, so
// the word stays where an address needs it and the lanes get it in every
// lane.
type Splats struct{}

func (Splats) Name() string { return "splats" }

func (p Splats) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		m := k.clone()
		splat := map[int]int{}
		splats := func(take func(i, a int, vw, cone map[int]bool) bool) {
			vw, cone := m.VectorWords(), m.addressCone()
			var uses [][2]int
			for i := range m.Nodes {
				if m.Nodes[i].Op == OpLoad || m.Nodes[i].Op == OpBroadcast {
					continue
				}
				nd := m.Nodes[i]
				for s, a := range m.Operands(&nd) {
					if take(i, *a, vw, cone) {
						uses = append(uses, [2]int{i, s})
					}
				}
			}
			for _, u := range uses {
				a := *m.Operands(&m.Nodes[u[0]])[u[1]]
				b, ok := splat[a]
				if !ok {
					b = m.add(Node{Op: OpBroadcast, Type: m.Nodes[a].Type, A: a})
					splat[a] = b
				}
				*m.Operands(&m.Nodes[u[0]])[u[1]] = b
			}
		}
		// The index's words its vector readers would take first: with them
		// cut, the vector words are the ones the code computes in lanes.
		splats(func(i, a int, vw, cone map[int]bool) bool { return !cone[i] && cone[a] && vw[a] })
		// Then the words those read from general registers: an index's node
		// among them where a vector mask picks it.
		splats(func(i, a int, vw, cone map[int]bool) bool {
			if cone[i] && !vw[i] {
				return false
			}
			t := m.Nodes[a].Type
			return vw[i] && t != ir.F32 && t != ir.Pred && m.counter(a) || (m.Nodes[i].Type == ir.F32 || vw[i]) && m.gprMask(a, vw)
		})
		if len(splat) == 0 {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

// gprMask reports whether node i is a compare of words general registers
// hold, its mask in one (the lowering's lessGPR): a vector reading it reads
// a broadcast.
func (k *Kernel) gprMask(i int, vw map[int]bool) bool {
	nd := k.Nodes[i]
	return nd.Op == OpLess && nd.Type == ir.Pred && k.Nodes[nd.A].Type != ir.F32 && !vw[i]
}

// counter reports whether node i is a word the code keeps in a general
// register whatever reads it: an index, a length, a coordinate or an
// argument word.
func (k *Kernel) counter(i int) bool {
	switch k.Nodes[i].Op {
	case OpIter, OpLength, OpSpan, OpCoord, OpParam:
		return true
	}
	return false
}

// addressCone is the integer nodes an access's index or a loop's count is
// made of: each index of a load or a store of a buffer other than a
// constant block, each loop's count, and the integer data they are
// computed from.
func (k *Kernel) addressCone() map[int]bool {
	cone := map[int]bool{}
	var walk func(i int)
	walk = func(i int) {
		nd := k.Nodes[i]
		if cone[i] || nd.Type == ir.F32 || nd.Type == ir.Pred || nd.Op == OpConst || nd.Op == OpLoad && nd.Mem {
			return
		}
		cone[i] = true
		if nd.Op == OpLoad || nd.Op == OpFold {
			return
		}
		for _, a := range k.Operands(&nd) {
			walk(*a)
		}
	}
	index := func(b, ix int, ax []int) {
		if k.Bufs[b].Operand.Kind == OperandConsts {
			return
		}
		if ax != nil {
			for _, a := range ax {
				walk(a)
			}
			return
		}
		walk(ix)
	}
	for _, nd := range k.Nodes {
		if nd.Op == OpLoad {
			index(nd.Buf, nd.A, nd.Ax)
		}
	}
	for _, s := range k.Stores {
		if !s.Barrier && !s.Run {
			index(s.Buf, s.Ix, s.Ax)
		}
	}
	// A loop's count is a general register's, as an address is: a vector
	// loop's tail counts what its lanes leave.
	for _, l := range k.Loops {
		if l.Count > 0 {
			walk(l.Count - 1)
		}
	}
	return cone
}
