package compose

import (
	"maps"
	"slices"
)

// divmods rewrites each quotient and remainder of an index by a constant d
// whose index is a sum of terms: the terms whose coefficients d divides go
// to the quotient divided, and the rest, where they provably lie in [0, d),
// are the remainder. A remainder by m lies in [0, m), a tile and its step
// in their counts under schedule s.
func (g *Row) divmods(s Schedule) *Row {
	tr, tc := max(s.Tile, 1), max(s.TileCols, 1)
	r := g
	cloned := false
	n0 := len(g.Nodes)
	for i := 0; i < n0; i++ {
		nd := r.Nodes[i]
		if nd.Kind != NUBinary || nd.Op != OpDiv && nd.Op != OpRem || r.Nodes[nd.B].Kind != NUConst || r.Nodes[nd.B].At <= 0 {
			continue
		}
		d := int64(r.Nodes[nd.B].At)
		l := r.linear(nd.A)
		var q, rest linSum
		q.terms, rest.terms = map[int]int64{}, map[int]int64{}
		q.c, rest.c = l.c/d, l.c%d
		if l.c < 0 {
			continue
		}
		hi, ok := rest.c, true
		for a, k := range l.terms {
			if k%d == 0 {
				q.terms[a] = k / d
				continue
			}
			lo, top, known := r.rangeOf(a, tr, tc)
			if k < 0 || lo < 0 || !known {
				ok = false
				break
			}
			rest.terms[a] = k
			hi += k * top
		}
		if !ok || hi >= d || len(q.terms) == 0 && q.c == 0 && nd.Op == OpRem {
			continue
		}
		if !cloned {
			r, cloned = g.clone(), true
		}
		if nd.Op == OpDiv {
			r.Nodes[i] = r.Nodes[r.build(q)]
		} else {
			r.Nodes[i] = r.Nodes[r.build(rest)]
		}
	}
	return r
}

type linSum struct {
	terms map[int]int64
	c     int64
}

// linear is index n as a sum of terms of atoms and a constant.
func (g *Row) linear(n int) linSum {
	nd := g.Nodes[n]
	switch {
	case nd.Kind == NUConst:
		return linSum{terms: map[int]int64{}, c: int64(nd.At)}
	case nd.Kind == NUBinary && (nd.Op == OpAdd || nd.Op == OpSub):
		a, b := g.linear(nd.A), g.linear(nd.B)
		sign := int64(1)
		if nd.Op == OpSub {
			sign = -1
		}
		for k, v := range b.terms {
			a.terms[k] += sign * v
			if a.terms[k] == 0 {
				delete(a.terms, k)
			}
		}
		a.c += sign * b.c
		return a
	case nd.Kind == NUBinary && nd.Op == OpMul:
		a, b := g.linear(nd.A), g.linear(nd.B)
		if len(a.terms) == 0 {
			a, b = b, a
		}
		if len(b.terms) == 0 {
			for k := range a.terms {
				a.terms[k] *= b.c
				if a.terms[k] == 0 {
					delete(a.terms, k)
				}
			}
			a.c *= b.c
			return a
		}
	}
	return linSum{terms: map[int]int64{n: 1}}
}

// rangeOf is the least and greatest value of atom n, known where it is
// bounded.
func (g *Row) rangeOf(n, tr, tc int) (lo, hi int64, known bool) {
	nd := g.Nodes[n]
	switch {
	case nd.Kind == NUBinary && nd.Op == OpRem && g.Nodes[nd.B].Kind == NUConst:
		return 0, int64(g.Nodes[nd.B].At) - 1, true
	case nd.Kind == NUBinary && nd.Op == OpDiv && g.Nodes[nd.B].Kind == NUConst && g.Nodes[nd.B].At > 0:
		if l, h, ok := g.linRange(g.linear(nd.A), tr, tc); ok && l >= 0 {
			d := int64(g.Nodes[nd.B].At)
			return l / d, h / d, true
		}
	case nd.Kind == NRowOff:
		return 0, int64(tr - 1), true
	case nd.Kind == NColOff:
		return 0, int64(tc - 1), true
	case nd.Kind == NRowTile:
		return 0, int64(g.Rows/tr - 1), true
	case nd.Kind == NColTile:
		return 0, int64(g.Width/tc - 1), true
	}
	return 0, 0, false
}

func (g *Row) linRange(l linSum, tr, tc int) (lo, hi int64, known bool) {
	lo, hi = l.c, l.c
	for a, k := range l.terms {
		al, ah, ok := g.rangeOf(a, tr, tc)
		if !ok {
			return 0, 0, false
		}
		if k >= 0 {
			lo, hi = lo+k*al, hi+k*ah
		} else {
			lo, hi = lo+k*ah, hi+k*al
		}
	}
	return lo, hi, true
}

// build is l as nodes: its terms in node order, then its constant.
func (g *Row) build(l linSum) int {
	u, ub := g.UConst, g.UBin
	sum := -1
	for _, a := range slices.Sorted(maps.Keys(l.terms)) {
		t := a
		if k := l.terms[a]; k != 1 {
			t = ub(OpMul, a, u(int(k)))
		}
		if sum < 0 {
			sum = t
		} else {
			sum = ub(OpAdd, sum, t)
		}
	}
	switch {
	case sum < 0:
		return u(int(l.c))
	case l.c != 0:
		return ub(OpAdd, sum, u(int(l.c)))
	}
	return sum
}
