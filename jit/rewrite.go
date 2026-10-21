package jit

import "github.com/jitllm/jitllm/jit/ir"

// redirect points every reader of node from, but node keep, at node to.
func (n *Nest) redirect(from, to, keep int) {
	for i := range n.Nodes {
		if i == keep {
			continue
		}
		nd := &n.Nodes[i]
		for _, p := range n.args(nd) {
			if *p == from {
				*p = to
			}
		}
	}
	for i := range n.Stores {
		if n.Stores[i].Barrier {
			continue
		}
		if n.Stores[i].Ix == from {
			n.Stores[i].Ix = to
		}
		if n.Stores[i].Val == from {
			n.Stores[i].Val = to
		}
	}
}

// args is node nd's operand fields, a fold's start and step among them.
func (n *Nest) args(nd *Node) []*int {
	if nd.Op == OpExchange {
		return exchangeArgs(nd)
	}
	switch arity(nd.Op) {
	case 0:
		return nil
	case 1:
		return []*int{&nd.A}
	case 3:
		return []*int{&nd.A, &nd.B, &nd.C}
	}
	return []*int{&nd.A, &nd.B}
}

// arity is how many operands an op reads, a fold its step and its start.
func arity(op Op) int {
	switch op {
	case OpConst, OpCoord, OpIter, OpThread, OpFlat:
		return 0
	case OpLoad, OpBitcast, OpSqrt, OpExp, OpCvtF32, OpCvtI32, OpGroupSum, OpGroupMax, OpLocal:
		return 1
	case OpFma, OpSelect:
		return 3
	}
	return 2
}

// exchangeArgs is an exchange's operand fields, its start where it has one.
func exchangeArgs(nd *Node) []*int {
	if nd.C < 0 {
		return []*int{&nd.A, &nd.B}
	}
	return []*int{&nd.A, &nd.B, &nd.C}
}

// varies reports whether node i changes from trip to trip of loop c.
func (n *Nest) varies(i, c int) bool {
	seen := map[int]bool{}
	var walk func(i int) bool
	walk = func(i int) bool {
		if seen[i] {
			return false
		}
		seen[i] = true
		nd := n.Nodes[i]
		if (nd.Op == OpIter || nd.Op == OpFold) && n.within(nd.At, c) {
			return true
		}
		if nd.Op == OpFold {
			return false
		}
		for _, a := range n.args(&nd) {
			if walk(*a) {
				return true
			}
		}
		return false
	}
	return walk(i)
}

// within reports whether loop h is c or sits in it.
func (n *Nest) within(h, c int) bool {
	for ; h >= 0; h = n.Loops[h].Parent {
		if h == c {
			return true
		}
	}
	return false
}

// copyOf is node i with each node in sub replaced: the nodes that read a
// replaced one copied, the rest shared.
func (n *Nest) copyOf(i int, sub map[int]int) int {
	memo := map[int]int{}
	var walk func(i int) int
	walk = func(i int) int {
		if v, ok := sub[i]; ok {
			return v
		}
		if v, ok := memo[i]; ok {
			return v
		}
		nd := n.Nodes[i]
		if nd.Op == OpFold {
			return i
		}
		changed := false
		for _, p := range n.args(&nd) {
			if v := walk(*p); v != *p {
				*p, changed = v, true
			}
		}
		v := i
		if changed {
			v = n.add(nd)
		}
		memo[i] = v
		return v
	}
	return walk(i)
}

// sumFold reports whether fold f's step adds into it: an fma onto it or an
// add of it.
func (n *Nest) sumFold(f int) bool {
	st := n.Nodes[n.Nodes[f].A]
	return st.Op == OpFma && st.C == f || st.Op == OpAdd && st.A == f
}

// foldsOf is loop c's folds.
func (n *Nest) foldsOf(c int) []int {
	var out []int
	for i, nd := range n.Nodes {
		if nd.Op == OpFold && nd.At == c {
			out = append(out, i)
		}
	}
	return out
}

// iterOf is loop c's iteration node, -1 where nothing reads it.
func (n *Nest) iterOf(c int) int {
	for i, nd := range n.Nodes {
		if nd.Op == OpIter && nd.At == c {
			return i
		}
	}
	return -1
}

// flat is the element's index in the space from its coordinates, the last
// fastest, coordinates from first on.
func (n *Nest) flat(first int) int {
	ix := -1
	for d := first; d < len(n.Space); d++ {
		c := n.add(Node{Op: OpCoord, A: d, Type: ir.U32})
		if ix < 0 {
			ix = c
			continue
		}
		ix = n.bin(OpAdd, ir.U32, n.bin(OpMul, ir.U32, ix, n.u32(n.Space[d])), c)
	}
	if ix < 0 {
		return n.u32(0)
	}
	return ix
}
