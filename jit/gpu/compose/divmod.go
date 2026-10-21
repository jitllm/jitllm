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
	r := g.clone()
	changed := false
	// The nodes the pass began with, then once over those it made: a
	// rewrite's own quotients are rewritten, its rewrites' are not.
	end := len(r.Nodes)
	for pass := 0; pass < 2; pass++ {
		start := 0
		if pass == 1 {
			start, end = end, len(r.Nodes)
		}
		for i := start; i < end; i++ {
			if n, ok := r.divmod(i, tr, tc); ok {
				r.Nodes[i], changed = r.Nodes[n], true
			}
		}
	}
	if !changed {
		return g
	}
	return r
}

// divmod is node i rewritten, where it is a quotient or remainder by a
// constant that divmods simplifies.
func (r *Row) divmod(i, tr, tc int) (int, bool) {
	nd := r.Nodes[i]
	if n, ok := r.maskCarry(nd); ok {
		return n, true
	}
	if nd.Kind != NUBinary || nd.Op != OpDiv && nd.Op != OpRem || r.Nodes[nd.B].Kind != NUConst || r.Nodes[nd.B].At <= 0 {
		return 0, false
	}
	d := int64(r.Nodes[nd.B].At)
	if n, ok := r.nested(nd, d); ok {
		return n, true
	}
	// (x + c) by d where x is a multiple of g dividing d and 0 <= c < g:
	// x mod d is a multiple of g at most d-g, so c rides past the division.
	if a := r.Nodes[nd.A]; a.Kind == NUBinary && a.Op == OpAdd && r.Nodes[a.B].Kind == NUConst {
		c := int64(r.Nodes[a.B].At)
		if g := gcd(r.multipleOf(a.A), d); c > 0 && c < g {
			q := r.UBin(nd.Op, a.A, nd.B)
			if nd.Op == OpDiv {
				return q, true
			}
			return r.UBin(OpAdd, q, a.B), true
		}
	}
	l := r.linear(nd.A)
	if l.c < 0 {
		return r.byCommon(nd, l, d)
	}
	q := linSum{terms: map[int]int64{}, c: l.c / d}
	rest := linSum{terms: map[int]int64{}, c: l.c % d}
	hi, bounded := rest.c, true
	for a, k := range l.terms {
		if k%d == 0 {
			q.terms[a] = k / d
			continue
		}
		lo, top, known := r.rangeOf(a, tr, tc)
		if k < 0 || lo < 0 {
			return 0, false
		}
		rest.terms[a] = k
		bounded = bounded && known
		hi += k * top
	}
	small := bounded && hi < d
	switch {
	case nd.Op == OpRem && small:
		return r.build(rest), true
	case nd.Op == OpDiv && small:
		return r.build(q), true
	}
	if n, ok := r.byCommon(nd, l, d); ok {
		return n, true
	}
	switch {
	case len(q.terms) == 0 && q.c == 0:
		return 0, false
	case nd.Op == OpRem:
		return r.UBin(OpRem, r.build(rest), nd.B), true
	}
	// The remainder's terms keep their own quotient: floor((M + x)/d) is
	// M/d + floor(x/d) for M a multiple of d and x >= 0.
	q.terms[r.UBin(OpDiv, r.build(rest), nd.B)] = 1
	return r.build(q), true
}

// nested is a quotient or remainder by d of a remainder by a, d dividing a:
// (x%a)%d is x%d and (x%a)/d is (x/d)%(a/d).
func (g *Row) nested(nd Node, d int64) (int, bool) {
	in := g.Nodes[nd.A]
	if in.Kind != NUBinary || in.Op != OpRem || g.Nodes[in.B].Kind != NUConst {
		return 0, false
	}
	a := int64(g.Nodes[in.B].At)
	if a <= 0 || a%d != 0 {
		return 0, false
	}
	if nd.Op == OpRem {
		return g.UBin(OpRem, in.A, nd.B), true
	}
	return g.UBin(OpRem, g.UBin(OpDiv, in.A, nd.B), g.UConst(int(a/d))), true
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

// termsGCD is the greatest common divisor of d and every term of l, each a
// coefficient times what its atom is known a multiple of.
func (r *Row) termsGCD(l linSum, d int64) int64 {
	g := d
	for a, k := range l.terms {
		g = gcd(g, k*r.multipleOf(a))
	}
	return g
}

// multipleOf is a number index n is known a multiple of (1 where none is).
func (r *Row) multipleOf(n int) int64 {
	nd := r.Nodes[n]
	switch {
	case nd.Kind == NHint && nd.Hint == HintMultiple:
		m := r.multipleOf(nd.A)
		return m / gcd(m, int64(nd.At)) * int64(nd.At)
	case nd.Kind == NHint:
		return r.multipleOf(nd.A)
	case nd.Kind == NUConst:
		return int64(nd.At)
	case nd.Kind == NUBinary && nd.Op == OpMul:
		return r.multipleOf(nd.A) * r.multipleOf(nd.B)
	case nd.Kind == NUBinary && (nd.Op == OpAdd || nd.Op == OpSub):
		return gcd(r.multipleOf(nd.A), r.multipleOf(nd.B))
	}
	return 1
}

func gcd(a, b int64) int64 {
	if a < 0 {
		a = -a
	}
	if b < 0 {
		b = -b
	}
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// byCommon is a quotient or remainder by d of l whose every term is a
// multiple of m, m dividing d: with c the constant's remainder by m, x%d is
// (x-c)%d + c and x/d is (x-c)/d.
func (r *Row) byCommon(nd Node, l linSum, d int64) (int, bool) {
	m := r.termsGCD(l, d)
	if m <= 1 || len(l.terms) == 0 {
		return 0, false
	}
	c := (l.c%m + m) % m
	if c == 0 {
		return 0, false
	}
	l.c -= c
	in := r.UBin(nd.Op, r.build(l), nd.B)
	if nd.Op == OpRem {
		return r.UBin(OpAdd, in, r.UConst(int(c))), true
	}
	return in, true
}

// maskCarry is (x + c) & (d-1) as (x & (d-1)) + c, d a power of two, where x
// is a multiple of g dividing d and 0 <= c < g: the mask form of divmod's
// carry rule.
func (r *Row) maskCarry(nd Node) (int, bool) {
	if nd.Kind != NUBinary || nd.Op != OpAnd || r.Nodes[nd.B].Kind != NUConst {
		return 0, false
	}
	d := int64(r.Nodes[nd.B].At) + 1
	if d < 2 || d&(d-1) != 0 {
		return 0, false
	}
	a := r.Nodes[nd.A]
	if a.Kind != NUBinary || a.Op != OpAdd || r.Nodes[a.B].Kind != NUConst {
		return 0, false
	}
	c := int64(r.Nodes[a.B].At)
	if g := gcd(r.multipleOf(a.A), d); c <= 0 || c >= g {
		return 0, false
	}
	return r.UBin(OpAdd, r.UBin(OpAnd, a.A, nd.B), a.B), true
}
