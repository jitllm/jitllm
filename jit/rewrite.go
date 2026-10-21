package jit

import "github.com/jitllm/jitllm/jit/ir"

// redirect points every reader of node from, but node keep, at node to.
func (n *Kernel) redirect(from, to, keep int) {
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
		if n.Stores[i].Barrier || n.Stores[i].Run {
			continue
		}
		for _, p := range n.storeArgs(&n.Stores[i]) {
			if *p == from {
				*p = to
			}
		}
	}
}

// storeArgs is a store's node fields.
func (n *Kernel) storeArgs(s *Store) []*int {
	out := []*int{&s.Ix, &s.Val}
	for i := range s.Ax {
		out = append(out, &s.Ax[i])
	}
	return out
}

// args is node nd's operand fields, a fold's start and step among them.
func (n *Kernel) args(nd *Node) []*int {
	if nd.Op == OpExchange {
		return exchangeArgs(nd)
	}
	if nd.Op == OpLoad && nd.Ax != nil {
		nd.Ax = append([]int(nil), nd.Ax...)
		out := make([]*int, len(nd.Ax))
		for i := range nd.Ax {
			out[i] = &nd.Ax[i]
		}
		return out
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
	case OpConst, OpCoord, OpIter, OpThread, OpFlat, OpParam, OpLength, OpSpan, OpLanes:
		return 0
	case OpLoad, OpBitcast, OpSqrt, OpExp, OpCvtF32, OpCvtI32, OpGroupSum, OpGroupMax, OpLocal, OpShuffle, OpBroadcast, OpHSum, OpHMax, OpHMin, OpHalf, OpRoundI32, OpDupEven, OpDupOdd, OpSwapPairs, OpCopy:
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
func (n *Kernel) varies(i, c int) bool {
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
		if nd.Op == OpLoad && nd.At > 0 && n.within(nd.At-1, c) {
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
func (n *Kernel) within(h, c int) bool {
	for ; h >= 0; h = n.Loops[h].Parent {
		if h == c {
			return true
		}
	}
	return false
}

// copyOf is node i with each node in sub replaced: the nodes that read a
// replaced one copied, the rest shared.
func (n *Kernel) copyOf(i int, sub map[int]int) int {
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

// copyKeep is copyOf with each copy kept in sub, so a later copy of the
// same nodes reuses it.
func (n *Kernel) copyKeep(i int, sub map[int]int) int {
	var walk func(i int) int
	walk = func(i int) int {
		if v, ok := sub[i]; ok {
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
		if !changed {
			return i
		}
		v := n.add(nd)
		sub[i] = v
		return v
	}
	return walk(i)
}

// sumFold reports whether fold f's step adds into it: an fma onto it or an
// add of it.
func (n *Kernel) sumFold(f int) bool {
	st := n.Nodes[n.Nodes[f].A]
	return st.Op == OpFma && st.C == f || st.Op == OpAdd && st.A == f
}

// extremeTerm is the term fold f's step takes the largest or least of with
// it, -1 where the step is not such a choice.
func (n *Kernel) extremeTerm(f int) int {
	st := n.Nodes[n.Nodes[f].A]
	if (st.Op != OpMax && st.Op != OpMin) || st.Type != ir.F32 {
		return -1
	}
	switch f {
	case st.A:
		return st.B
	case st.B:
		return st.A
	}
	return -1
}

// carriedBy is the fold whose choice fold f carries a value with: f's step
// selects a new value where that fold's term is above it, as a select or
// as its bits, -1 where f is not carried.
func (n *Kernel) carriedBy(f int) int {
	st := n.Nodes[n.Nodes[f].A]
	pred := -1
	switch {
	case st.Op == OpSelect && st.C == f:
		pred = st.A
	case st.Op == OpOr:
		set, clear := n.Nodes[st.A], n.Nodes[st.B]
		if set.Op == OpAnd && clear.Op == OpAndNot && clear.A == f && clear.B == set.B {
			pred = set.B
		}
	}
	if pred < 0 {
		return -1
	}
	won := n.Nodes[pred]
	if won.Op != OpLess || n.Nodes[won.A].Op != OpFold || n.extremeTerm(won.A) != won.B {
		return -1
	}
	return won.A
}

// foldsOf is loop c's folds.
func (n *Kernel) foldsOf(c int) []int {
	var out []int
	for i, nd := range n.Nodes {
		if nd.Op == OpFold && nd.At == c {
			out = append(out, i)
		}
	}
	return out
}

// iterOf is loop c's iteration node, -1 where nothing reads it.
func (n *Kernel) iterOf(c int) int {
	for i, nd := range n.Nodes {
		if nd.Op == OpIter && nd.At == c {
			return i
		}
	}
	return -1
}

// flat is the element's index in the space from its coordinates, the last
// fastest, coordinates from first on.
func (n *Kernel) flat(first int) int {
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

// copyInto is src's node root as nodes of dst: each node src reaches from
// root copied, its operands first, a load into dst's buffer of the same
// operand, kind and shape; leaf, where it answers, stands for a node,
// given the node with its operands already copied.
func copyInto(dst, src *Kernel, root int, leaf func(i int, nd Node) (int, bool)) int {
	memo := map[int]int{}
	var walk func(i int) int
	walk = func(i int) int {
		if v, ok := memo[i]; ok {
			return v
		}
		nd := src.Nodes[i]
		if nd.Op == OpFold {
			if v, ok := leaf(i, nd); ok {
				memo[i] = v
				return v
			}
			v := dst.add(Node{})
			memo[i] = v
			for _, a := range src.args(&nd) {
				*a = walk(*a)
			}
			dst.Nodes[v] = nd
			return v
		}
		args := src.args(&nd)
		for _, a := range args {
			*a = walk(*a)
		}
		if v, ok := leaf(i, nd); ok {
			memo[i] = v
			return v
		}
		if nd.Op == OpLoad {
			nd.Buf = dst.bufLike(src.Bufs[nd.Buf])
		}
		v := dst.add(nd)
		memo[i] = v
		return v
	}
	return walk(root)
}

// bufLike is k's buffer of b's operand, kind and shape, added where k has
// none.
func (k *Kernel) bufLike(b Buf) int {
	for i, x := range k.Bufs {
		if x.Operand == b.Operand && x.Kind == b.Kind && sameShape(x.Shape, b.Shape) {
			return i
		}
	}
	return k.buf(b)
}

// compact drops the nodes nothing reaches: neither a store, nor a loop's
// count, live span or element, nor a fact.
func (k *Kernel) compact() {
	live := map[int]bool{}
	var walk func(i int)
	walk = func(i int) {
		if i < 0 || i >= len(k.Nodes) || live[i] {
			return
		}
		live[i] = true
		nd := k.Nodes[i]
		for _, a := range k.args(&nd) {
			walk(*a)
		}
	}
	for i := range k.Stores {
		s := k.Stores[i]
		if !s.Barrier && !s.Run {
			for _, a := range k.storeArgs(&s) {
				walk(*a)
			}
		}
	}
	for _, l := range k.Loops {
		for _, x := range []int{l.Count, l.Span, l.Elem, l.End} {
			if x > 0 {
				walk(x - 1)
			}
		}
	}
	for _, x := range []int{k.Rows, k.Clamp} {
		if x > 0 {
			walk(x - 1)
		}
	}
	for _, b := range k.Bufs {
		for _, x := range append([]int{b.Base, b.Offset}, b.Dims...) {
			if x > 0 {
				walk(x - 1)
			}
		}
	}
	for _, a := range k.Args {
		if a.Node > 0 {
			walk(a.Node - 1)
		}
	}
	if p := k.Positions; p != nil {
		for _, x := range append(append([]int{p.First, p.End}, p.Live...), p.Pos...) {
			walk(x)
		}
		for _, x := range []int{p.Span, p.Part} {
			if x > 0 {
				walk(x - 1)
			}
		}
	}
	if len(live) == len(k.Nodes) {
		return
	}
	at := make([]int, len(k.Nodes))
	var nodes []Node
	for i, nd := range k.Nodes {
		at[i] = -1
		if live[i] {
			at[i] = len(nodes)
			nodes = append(nodes, nd)
		}
	}
	for i := range nodes {
		for _, a := range k.args(&nodes[i]) {
			*a = at[*a]
		}
	}
	k.Nodes = nodes
	for i := range k.Stores {
		s := &k.Stores[i]
		if s.Barrier || s.Run {
			continue
		}
		for _, a := range k.storeArgs(s) {
			*a = at[*a]
		}
	}
	re := func(x int) int {
		if x > 0 {
			return at[x-1] + 1
		}
		return x
	}
	for i := range k.Loops {
		l := &k.Loops[i]
		l.Count, l.Span, l.Elem, l.End = re(l.Count), re(l.Span), re(l.Elem), re(l.End)
	}
	k.Rows, k.Clamp = re(k.Rows), re(k.Clamp)
	for i := range k.Bufs {
		k.Bufs[i].Base, k.Bufs[i].Offset = re(k.Bufs[i].Base), re(k.Bufs[i].Offset)
		for j := range k.Bufs[i].Dims {
			k.Bufs[i].Dims[j] = re(k.Bufs[i].Dims[j])
		}
	}
	for i := range k.Args {
		k.Args[i].Node = re(k.Args[i].Node)
	}
	if k.Seq != nil {
		var seq []Entry
		for _, st := range k.Seq {
			if st.Kind == EntryNode {
				if at[st.I] < 0 {
					continue
				}
				st.I = at[st.I]
			}
			seq = append(seq, st)
		}
		k.Seq = seq
	}
	if q := k.Quarter; q != nil {
		if at[q.Sum] < 0 || at[q.Rest] < 0 || at[q.Turn] < 0 || at[q.Part] < 0 {
			k.Quarter = nil
		} else {
			k.Quarter = &Quarter{Sum: at[q.Sum], Rest: at[q.Rest], Turn: at[q.Turn], Part: at[q.Part], Bound: q.Bound}
		}
	}
	if p := k.Positions; p != nil {
		q := *p
		q.First, q.End = at[p.First], at[p.End]
		q.Live, q.Pos = nil, nil
		for _, x := range p.Live {
			q.Live = append(q.Live, at[x])
		}
		for _, x := range p.Pos {
			q.Pos = append(q.Pos, at[x])
		}
		q.Span, q.Part = re(p.Span), re(p.Part)
		k.Positions = &q
	}
}

// loopOf is the innermost loop node i varies in, -1 for none.
func (n *Kernel) loopOf(i int) int {
	best := -1
	for c := range n.Loops {
		if n.varies(i, c) && (best < 0 || n.within(c, best)) {
			best = c
		}
	}
	return best
}

// insertLoop puts l at index at, the loops from at on one later.
func (n *Kernel) insertLoop(at int, l Loop) int {
	sh := func(x int) int {
		if x >= at {
			return x + 1
		}
		return x
	}
	for i := range n.Loops {
		n.Loops[i].Parent = sh(n.Loops[i].Parent)
		if n.Loops[i].Of > 0 {
			n.Loops[i].Of = sh(n.Loops[i].Of-1) + 1
		}
	}
	l.Parent = sh(l.Parent)
	if l.Of > 0 {
		l.Of = sh(l.Of-1) + 1
	}
	n.Loops = append(n.Loops[:at:at], append([]Loop{l}, n.Loops[at:]...)...)
	for i := range n.Nodes {
		nd := &n.Nodes[i]
		switch {
		case nd.Op == OpIter || nd.Op == OpFold:
			nd.At = sh(nd.At)
		case nd.Op == OpLoad && nd.At > 0:
			nd.At = sh(nd.At-1) + 1
		}
	}
	for i := range n.Stores {
		if n.Stores[i].At >= 0 {
			n.Stores[i].At = sh(n.Stores[i].At)
		}
	}
	if p := n.Positions; p != nil && p.Loop >= 0 {
		q := *p
		q.Loop = sh(p.Loop)
		n.Positions = &q
	}
	for i := range n.Seq {
		if e := &n.Seq[i]; e.Kind == EntryLoop || e.Kind == EntryEnd {
			e.I = sh(e.I)
		}
	}
	return at
}

// homes is, for each node, the innermost loop whose trips change it: an
// iteration's loop, a cursor's load's loop, a sum's loop for the nodes of
// the sum's step, and for a sum read after its loop the loop around it.
// -1 for a node no loop changes.
func (n *Kernel) homes() []int {
	step := map[int]int{}
	for f, nd := range n.Nodes {
		if nd.Op != OpFold {
			continue
		}
		var walk func(i int)
		walk = func(i int) {
			if i == f {
				return
			}
			if _, ok := step[i]; ok {
				return
			}
			x := n.Nodes[i]
			if x.Op == OpFold || !n.reachesFold(i, f) {
				return
			}
			step[i] = nd.At
			for _, a := range n.args(&x) {
				walk(*a)
			}
		}
		walk(nd.A)
	}
	out := make([]int, len(n.Nodes))
	done := make([]bool, len(n.Nodes))
	inner := func(a, b int) int {
		switch {
		case a < 0:
			return b
		case b < 0:
			return a
		case n.within(a, b):
			return a
		}
		return b
	}
	var home func(i int) int
	home = func(i int) int {
		if done[i] {
			return out[i]
		}
		done[i], out[i] = true, -1
		nd := n.Nodes[i]
		h := -1
		switch {
		case nd.Op == OpIter:
			h = nd.At
		case nd.Op == OpFold:
			h = n.Loops[nd.At].Parent
		case nd.Op == OpLoad && nd.At > 0:
			h = nd.At - 1
		}
		if c, ok := step[i]; ok {
			h = inner(h, c)
		}
		if nd.Op != OpFold {
			for _, a := range n.args(&nd) {
				h = inner(h, home(*a))
			}
		}
		out[i] = h
		return h
	}
	for i := range n.Nodes {
		home(i)
	}
	return out
}

// reachesFold reports whether node i reads fold f.
func (n *Kernel) reachesFold(i, f int) bool {
	seen := map[int]bool{}
	var walk func(i int) bool
	walk = func(i int) bool {
		if i == f {
			return true
		}
		if seen[i] {
			return false
		}
		seen[i] = true
		nd := n.Nodes[i]
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

// depth is the length of the longest chain of operations node i is made
// from, a sum and a leaf 0.
func (n *Kernel) depth(i int, memo map[int]int) int {
	if d, ok := memo[i]; ok {
		return d
	}
	memo[i] = 0
	nd := n.Nodes[i]
	if nd.Op == OpFold || arity(nd.Op) == 0 {
		return 0
	}
	d := 0
	for _, a := range n.args(&nd) {
		d = max(d, n.depth(*a, memo)+1)
	}
	memo[i] = d
	return d
}

// evalArgs is node nd's operands in the order its code evaluates them: the
// deeper first, the earlier of two alike, an fma's addend the earliest.
func (n *Kernel) evalArgs(nd *Node, memo map[int]int) []*int {
	args := n.args(nd)
	if nd.Op == OpFma {
		args = []*int{&nd.C, &nd.A, &nd.B}
	}
	out := append([]*int(nil), args...)
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && n.depth(*out[j], memo) > n.depth(*out[j-1], memo); j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out
}
