package compose

import "fmt"

// SplitFold is the split schedule as a rewrite of g: g's one store reads a
// fold over loop 0 (a sum: the step adds to the fold), and its value is
// formed in two groups. The first runs loop 0 split between ways threads
// and stores each segment's partial into its plane (Schedule.Split); an
// additive term beside the fold that reads no part of the loop, fold + t,
// joins the first segment's partial alone, so the planes sum to it. The
// second sums the planes (SumPlanes) and computes the rest of the store's
// value from the sum.
//
// The groups' buffers are g's, renumbered: bufs1 and bufs2 name each
// group's buffers by g's index, -1 for the planes (written by the first as
// BufOut, read by the second as BufRow).
func (g *Row) SplitFold(ways int) (first, second *Row, bufs1, bufs2 []int, err error) {
	if len(g.Stores) != 1 || len(g.Loops) == 0 || g.Q8 != nil {
		return nil, nil, nil, nil, fmt.Errorf("compose: a split of a group of %d stores, %d loops", len(g.Stores), len(g.Loops))
	}
	st := g.Stores[0]
	vary := g.varies(0)
	// The fold the store reads, through additions.
	fold, term, sum := -1, -1, -1
	var find func(n int) bool
	find = func(n int) bool {
		nd := g.Nodes[n]
		if nd.Kind == NFold && nd.At == 0 {
			fold = n
			return true
		}
		if nd.Kind == NBinary && nd.Op == OpAdd {
			for side, x := range []int{nd.A, nd.B} {
				other := []int{nd.B, nd.A}[side]
				if g.Nodes[x].Kind == NFold && g.Nodes[x].At == 0 && !vary[other] {
					fold, term, sum = x, other, n
					return true
				}
			}
		}
		for _, o := range g.operands(n) {
			if find(o) {
				return true
			}
		}
		return false
	}
	if !find(st.Val) || st.Indexed {
		return nil, nil, nil, nil, fmt.Errorf("compose: a split of a store that reads no fold of loop 0")
	}
	// The first group: g's nodes, its store the fold (+ the term in segment 0).
	first = &Row{Width: g.Width, Rows: g.Rows, Bufs: append([]Buf(nil), g.Bufs...), Nodes: append([]Node(nil), g.Nodes...),
		Loops: append([]int(nil), g.Loops...)}
	bufs1 = make([]int, len(g.Bufs))
	for i := range bufs1 {
		bufs1[i] = i
	}
	v := fold
	if term >= 0 {
		seg := first.Add(Node{Kind: NSegment})
		in0 := first.Less(seg, first.UConst(1))
		v = first.Bin(OpAdd, fold, first.Select(in0, term, first.Const(0)))
	}
	planes := first.Buf(BufOut)
	bufs1 = append(bufs1, -1)
	first.Store(planes, v)
	// The original output stays a parameter of the first group, unwritten:
	// the launch binds it as before.
	// The second: the planes' sum, then the rest of the store's value.
	second = &Row{Width: g.Width, Rows: g.Rows}
	in := second.Buf(BufRow)
	bufs2 = []int{-1}
	s := second.SumPlanes(in, ways)
	bufOf := map[int]int{}
	memo := map[int]int{}
	var clone func(n int) (int, error)
	clone = func(n int) (int, error) {
		if x, ok := memo[n]; ok {
			return x, nil
		}
		if n == fold || n == sum {
			return s, nil
		}
		if vary[n] {
			return 0, fmt.Errorf("compose: a split store reads loop 0 other than through its fold (node %d)", n)
		}
		nd := g.Nodes[n]
		if nd.Kind == NLoad {
			b, ok := bufOf[nd.Buf]
			if !ok {
				b = second.Buf(g.Bufs[nd.Buf].Kind)
				bufs2 = append(bufs2, nd.Buf)
				bufOf[nd.Buf] = b
			}
			nd.Buf = b
		}
		var err error
		for _, p := range []*int{&nd.A, &nd.B, &nd.C, &nd.Ix} {
			if !usesOperand(nd, p, &nd) {
				continue
			}
			if *p, err = clone(*p); err != nil {
				return 0, err
			}
		}
		x := second.Add(nd)
		memo[n] = x
		return x, nil
	}
	sv, err := clone(st.Val)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	out := second.Buf(g.Bufs[st.Buf].Kind)
	bufs2 = append(bufs2, st.Buf)
	second.Store(out, sv)
	return first, second, bufs1, bufs2, nil
}

// usesOperand reports whether field p of node nd is an operand it reads.
func usesOperand(nd Node, p *int, base *Node) bool {
	switch p {
	case &base.A:
		switch nd.Kind {
		case NUnary, NBinary, NUBinary, NLess, NSelect, NReduce, NWindow, NDot4, NAt, NShuffle:
			return true
		case NFold:
			return nd.A >= 0
		}
	case &base.B:
		switch nd.Kind {
		case NBinary, NUBinary, NLess, NSelect, NDot4, NFold:
			return true
		}
	case &base.C:
		return nd.Kind == NSelect || nd.Kind == NDot4 || nd.ternary()
	case &base.Ix:
		return nd.Kind == NLoad && nd.Indexed || nd.Kind == NAt
	}
	return false
}

// GroupFold is the in-group split (Schedule.GroupSplit) as a rewrite of g:
// g's one store reads a fold over loop 0, each thread folds its segment,
// every row's partial goes into a workgroup array of group places a row at
// the thread's own place, and after the barrier the thread sums its row's
// segments, rows-per-group places apart, and computes the rest of the
// store's value from the sum.
func (g *Row) GroupFold(ways, group int) (*Row, error) {
	if len(g.Stores) != 1 || len(g.Loops) == 0 || g.Q8 != nil || ways < 2 || group%ways != 0 {
		return nil, fmt.Errorf("compose: an in-group split of %d stores, %d loops, %d ways", len(g.Stores), len(g.Loops), ways)
	}
	st := g.Stores[0]
	fold := -1
	var find func(n int) bool
	find = func(n int) bool {
		nd := g.Nodes[n]
		if nd.Kind == NFold && nd.At == 0 {
			fold = n
			return true
		}
		for _, o := range g.operands(n) {
			if find(o) {
				return true
			}
		}
		return false
	}
	if st.Indexed || !find(st.Val) {
		return nil, fmt.Errorf("compose: an in-group split of a store that reads no fold of loop 0")
	}
	r := g.clone()
	r.Stores = nil
	u, ub := r.UConst, r.UBin
	rpg := group / ways
	local := r.Add(Node{Kind: NLocal})
	at := func(n int) int { return ub(OpAdd, n, ub(OpMul, r.RowIdx(), u(group))) }
	sh := r.Shared(group*g.Rows, false)
	r.StoreAt(sh, fold, at(local))
	read := sh
	own := ub(OpRem, local, u(rpg))
	sum := r.LoadAt(read, at(own))
	for sg := 1; sg < ways; sg++ {
		sum = r.Bin(OpAdd, sum, r.LoadAt(read, ub(OpAdd, at(own), u(sg*rpg))))
	}
	memo := map[int]int{fold: sum}
	var redo func(n int) int
	redo = func(n int) int {
		if x, ok := memo[n]; ok {
			return x
		}
		nd := r.Nodes[n]
		changed := false
		for _, p := range []*int{&nd.A, &nd.B, &nd.C, &nd.Ix} {
			if !usesOperand(nd, p, &nd) {
				continue
			}
			if x := redo(*p); x != *p {
				*p, changed = x, true
			}
		}
		x := n
		if changed {
			x = r.Add(nd)
		}
		memo[n] = x
		return x
	}
	r.Store(st.Buf, redo(st.Val))
	return r, nil
}
