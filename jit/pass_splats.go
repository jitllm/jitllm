package jit

import "github.com/jitllm/jitllm/jit/ir"

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
		vw := k.VectorWords()
		cone := k.addressCone()
		type use struct{ node, slot int }
		var uses []use
		for i := range k.Nodes {
			if cone[i] || k.Nodes[i].Op == OpLoad || k.Nodes[i].Op == OpBroadcast {
				continue
			}
			nd := k.Nodes[i]
			for s, a := range k.Operands(&nd) {
				if cone[*a] && vw[*a] || vw[i] && k.Nodes[*a].Type != ir.F32 && k.Nodes[*a].Type != ir.Pred && k.counter(*a) {
					uses = append(uses, use{i, s})
				}
			}
		}
		if len(uses) == 0 {
			return nil, false, nil
		}
		m := k.clone()
		splat := map[int]int{}
		for _, u := range uses {
			a := *m.Operands(&m.Nodes[u.node])[u.slot]
			b, ok := splat[a]
			if !ok {
				b = m.add(Node{Op: OpBroadcast, Type: m.Nodes[a].Type, A: a})
				splat[a] = b
			}
			*m.Operands(&m.Nodes[u.node])[u.slot] = b
		}
		return []*Kernel{m}, true, nil
	})
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

// addressCone is the integer nodes an access's index is made of: each
// index of a load or a store of a buffer other than a constant block, and
// the integer data it is computed from.
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
	return cone
}
