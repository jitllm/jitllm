package jit

import "github.com/jitllm/jitllm/jit/ir"

// Induction carries a loop's iteration as a value where the code reads it
// as data rather than as an address: a sum of the loop from 0, one more
// each trip; in a vector loop from each lane's index, the lanes more each
// trip; and in the loop that takes a vector loop's remaining trips, where
// it reads the trips done and its own, on from the vector loop's. Tail,
// where set, is 1 + the register of that loop's step.
type Induction struct {
	Tail int
}

func (Induction) Name() string { return "induction" }

func (p Induction) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		var m *Kernel
		live, addr := k.Live(), k.addressNodes()
		vec := map[int]int{}
		for c := range k.Loops {
			it := k.iterOf(c)
			if it < 0 {
				continue
			}
			l := k.Loops[c]
			from, data := it, k.dataReaders(it, live, addr)
			v, cont := vec[l.Of-1]
			if l.Lanes == 0 && l.Of > 0 && cont {
				for d, nd := range k.Nodes {
					if nd.Op == OpAdd && nd.B == it && live[d] {
						if dr := k.dataReaders(d, live, addr); len(dr) > 0 {
							from, data = d, dr
						}
					}
				}
			}
			if len(data) == 0 {
				continue
			}
			if m == nil {
				m = k.clone()
			}
			start, step := m.add(Node{Op: OpConst, Type: ir.I32}), int64(1)
			switch {
			case l.Lanes > 0:
				start, step = m.add(Node{Op: OpLanes, Type: ir.I32, Imm: int64(l.Lanes)}), int64(l.Lanes)
			case from != it:
				start = v
			}
			f := m.add(Node{Op: OpFold, At: c, Type: ir.I32, B: start})
			for _, r := range data {
				for _, a := range m.Operands(&m.Nodes[r]) {
					if *a == from {
						*a = f
					}
				}
			}
			st := m.add(Node{Op: OpConst, Type: ir.I32, Imm: step})
			if from != it {
				m.Nodes[st].Hint = p.Tail
			}
			m.Nodes[f].A = m.bin(OpAdd, ir.I32, f, st)
			if l.Lanes > 0 {
				vec[c] = f
			}
		}
		if m == nil {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

// addressNodes is the nodes that feed an index a load or a store reads:
// every node an index is computed from, a fold's reads not followed, as
// readsNode walks them.
func (k *Kernel) addressNodes() map[int]bool {
	seen := map[int]bool{}
	var stack []int
	push := func(i int) {
		if !seen[i] {
			seen[i] = true
			stack = append(stack, i)
		}
	}
	for _, nd := range k.Nodes {
		if nd.Op == OpLoad {
			for _, a := range k.Operands(&nd) {
				push(*a)
			}
		}
	}
	for _, s := range k.Stores {
		if s.Barrier || s.Run {
			continue
		}
		push(s.Ix)
		for _, a := range s.Ax {
			push(a)
		}
	}
	for len(stack) > 0 {
		i := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if nd := k.Nodes[i]; nd.Op != OpFold {
			for _, a := range k.Operands(&nd) {
				push(*a)
			}
		}
	}
	return seen
}

// readsNode reports whether node i is computed from node x.
func (k *Kernel) readsNode(i, x int) bool {
	if i == x {
		return true
	}
	nd := k.Nodes[i]
	if nd.Op == OpFold {
		return false
	}
	for _, a := range k.Operands(&nd) {
		if k.readsNode(*a, x) {
			return true
		}
	}
	return false
}

// dataReaders is the live nodes that read node i as data: not a load, and
// feeding no address.
func (k *Kernel) dataReaders(i int, live, addr map[int]bool) []int {
	var out []int
	for j, nd := range k.Nodes {
		if !live[j] || nd.Op == OpLoad || addr[j] {
			continue
		}
		for _, a := range k.Operands(&nd) {
			if *a == i {
				out = append(out, j)
				break
			}
		}
	}
	return out
}
