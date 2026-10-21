package compose

import (
	"fmt"
	"math/bits"
)

// laneLoops spreads a row of one element over schedule s's Lanes lanes
// (LaneLoops): each loop's trips dealt across the lanes, lane l taking
// trips l, l+L, ...; a fold read after its loop combined across the lanes
// by its step's own operation (a maximum or a sum). The row's element is
// then the lane, and what reads the loops' results reads the lanes'
// combination.
func (g *Row) laneLoops(s Schedule) (*Row, error) {
	L := s.Lanes
	if !s.LaneLoops || L < 2 || L&(L-1) != 0 || g.Width != 1 && g.Rows != 1 {
		return g, nil
	}
	r := g.clone()
	if r.Width > 1 {
		// A flat row: each element its own row first.
		for i, nd := range r.Nodes {
			if nd.Kind == NColumn {
				r.Nodes[i] = Node{Kind: NRowIdx}
			}
		}
		r.Rows, r.Width = r.Width, 1
	}
	r.Width = L
	for i, nd := range r.Nodes {
		if nd.Kind == NColumn {
			r.Nodes[i] = Node{Kind: NUConst}
		}
	}
	lane := r.Add(Node{Kind: NColumn})
	u, ub := r.UConst, r.UBin
	body := make([][]bool, len(g.Loops))
	for c := range g.Loops {
		body[c] = g.bodyOf(c)
	}
	for c, trip := range g.Loops {
		var iter int
		ragged := false
		switch {
		case trip > 0:
			// Trips past a whole number of lanes: the last trip's lanes past
			// the end read the last element and add nothing.
			ragged = trip%L != 0
			r.Loops[c] = (trip + L - 1) / L
		default:
			n := g.Trip[c]
			r.Trip[c] = ub(OpShr, ub(OpAdd, ub(OpSub, ub(OpMax, n, lane), lane), u(L-1)), u(bits.TrailingZeros(uint(L))))
		}
		iter = r.Add(Node{Kind: NIter, At: c})
		for i, nd := range g.Nodes {
			if nd.Kind != NIter || nd.At != c {
				continue
			}
			if g.compared(i) {
				// An index a comparison reads is carried: started at the
				// lane, stepped by the lanes a trip.
				r.Nodes[i] = Node{Kind: NFold, At: c, B: lane}
				r.Nodes[i].A = ub(OpAdd, i, u(L))
				continue
			}
			if !ragged {
				r.Nodes[i] = Node{Kind: NUBinary, Op: OpAdd, A: lane, B: ub(OpMul, iter, u(L))}
				continue
			}
			x := ub(OpAdd, lane, ub(OpMul, iter, u(L)))
			r.Nodes[i] = Node{Kind: NUBinary, Op: OpMin, A: x, B: u(trip - 1)}
			in := r.Less(x, u(trip))
			for f, nd := range g.Nodes {
				if nd.Kind != NFold || nd.At != c {
					continue
				}
				// A product step's second factor selected to zero: the one
				// a tile's heads share, so selected once.
				st := r.Nodes[nd.A]
				if st.Kind != NBinary || st.Op != OpFma || st.C != f {
					return nil, fmt.Errorf("compose: a fold of %d trips over %d lanes whose step is not a product's sum", trip, L)
				}
				st.B = r.Select(in, st.B, r.Const(0))
				r.Nodes[nd.A] = st
			}
		}
		for f, nd := range g.Nodes {
			if nd.Kind != NFold || nd.At != c {
				continue
			}
			red, ok := g.foldRed(f)
			if !ok {
				if g.readAfter(f, body[c]) {
					return nil, fmt.Errorf("compose: a fold of loop %d over lanes whose step is not a maximum or a sum", c)
				}
				continue
			}
			comb := r.Add(Node{Kind: NReduce, Red: red, A: f, At: 1})
			for k := range g.Nodes {
				if k == comb || body[c][k] {
					continue
				}
				x := &r.Nodes[k]
				if x.Kind == NFold && x.At == c {
					continue
				}
				x.replaceOperand(f, comb)
			}
			for j, st := range r.Stores {
				if st.Val == f && g.storeLoop(g.Stores[j]) != c {
					r.Stores[j].Val = comb
				}
			}
		}
	}
	return r, nil
}

// bodyOf is which nodes loop c's trips compute: what its folds' steps and
// the stores it runs read, back to what does not change from trip to trip.
func (g *Row) bodyOf(c int) []bool {
	vary := g.varies(c)
	in := make([]bool, len(g.Nodes))
	var walk func(n int)
	walk = func(n int) {
		if in[n] || !vary[n] {
			return
		}
		in[n] = true
		if nd := g.Nodes[n]; nd.Kind == NFold && nd.At == c {
			return
		}
		for _, o := range g.loopOperands(n) {
			walk(o)
		}
	}
	for _, nd := range g.Nodes {
		if nd.Kind == NFold && nd.At == c && nd.A >= 0 {
			walk(nd.A)
		}
	}
	for _, st := range g.Stores {
		if g.storeLoop(st) == c {
			walk(st.Val)
			if st.Indexed {
				walk(st.Ix)
			}
		}
	}
	return in
}

// foldRed is fold f's combine: the operation its step applies to itself.
func (g *Row) foldRed(f int) (Red, bool) {
	st := g.Nodes[g.Nodes[f].A]
	if st.Kind == NBinary && st.Op == OpFma && st.C == f {
		return RedSum, true // f + a*b
	}
	if st.Kind != NBinary || st.A != f && st.B != f {
		return 0, false
	}
	switch st.Op {
	case OpMax:
		return RedMax, true
	case OpAdd:
		return RedSum, true
	}
	return 0, false
}

// readAfter reports whether a node outside body, or a store, reads f.
func (g *Row) readAfter(f int, body []bool) bool {
	for k := range g.Nodes {
		if !body[k] && k != f && slicesContains(g.operands(k), f) {
			return true
		}
	}
	for _, st := range g.Stores {
		if st.Val == f {
			return true
		}
	}
	return false
}

func slicesContains(s []int, x int) bool {
	for _, v := range s {
		if v == x {
			return true
		}
	}
	return false
}

// replaceOperand points every operand of n that is from at to.
func (n *Node) replaceOperand(from, to int) {
	switch n.Kind {
	case NUnary, NWindow, NShuffle, NHint, NReduce:
		if n.A == from {
			n.A = to
		}
	case NBinary, NUBinary, NLess, NSelect, NDot4:
		if n.A == from {
			n.A = to
		}
		if n.B == from {
			n.B = to
		}
		if (n.Kind == NSelect || n.Kind == NDot4 || n.ternary()) && n.C == from {
			n.C = to
		}
	case NAt:
		if n.A == from {
			n.A = to
		}
		if n.Ix == from {
			n.Ix = to
		}
	case NLoad:
		if n.Indexed && n.Ix == from {
			n.Ix = to
		}
	case NFold:
		if n.B == from {
			n.B = to
		}
		if n.A == from {
			n.A = to
		}
	}
}

// compared reports whether a comparison reads node n.
func (g *Row) compared(n int) bool {
	for i, nd := range g.Nodes {
		if nd.Kind == NLess && g.readsNode(i, n) {
			return true
		}
	}
	return false
}
