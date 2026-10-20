package compose

import (
	"slices"
	"sort"
)

// affNode is an index as a sum of terms over nodes: coef*node each, plus c.
type affNode struct {
	terms map[int]int64
	c     int64
}

// inductions is strength reduction as a rewrite of g under schedule s: an
// index a load reads that is affine in loop c's iterator and reads a loop
// nested in c too becomes a fold of c -- started at its part invariant in c,
// stepped by its stride a trip -- plus its nested part. Indices of one
// invariant part and stride share the fold. Loops are taken innermost first;
// g is copied before it is changed.
func (g *Row) inductions(s Schedule) *Row {
	out := g
	for c := len(g.Loops) - 1; c >= 0; c-- {
		out = out.induct(c, s)
	}
	return out
}

func (g *Row) induct(c int, s Schedule) *Row {
	vary := g.varies(c)
	iter := -1
	for i, nd := range g.Nodes {
		if nd.Kind == NIter && nd.At == c {
			iter = i
		}
	}
	if iter < 0 {
		return g
	}
	// Loop c2 is nested in c where a fold of c2 starts or steps by c's
	// iterator; a fold of c reading c2's result does not make it so.
	readsIter := map[int]bool{}
	var reads func(n int) bool
	reads = func(n int) bool {
		if v, ok := readsIter[n]; ok {
			return v
		}
		readsIter[n] = false
		v := n == iter
		for _, o := range g.operands(n) {
			v = v || reads(o)
		}
		readsIter[n] = v
		return v
	}
	nested := make([]bool, len(g.Nodes))
	for c2 := range g.Loops {
		if c2 == c {
			continue
		}
		in := false
		for _, nd := range g.Nodes {
			if nd.Kind == NFold && nd.At == c2 && (reads(nd.B) || nd.A >= 0 && reads(nd.A)) {
				in = true
			}
		}
		if !in {
			continue
		}
		v2 := g.varies(c2)
		for i := range nested {
			nested[i] = nested[i] || v2[i]
		}
	}
	memo := map[int]*affNode{}
	var aff func(n int) *affNode
	aff = func(n int) *affNode {
		if a, ok := memo[n]; ok {
			return a
		}
		nd := g.Nodes[n]
		var r *affNode
		switch {
		case n == iter || nested[n] && !vary[n] || !vary[n] && !nested[n]:
			if nd.Kind == NUConst {
				r = &affNode{terms: map[int]int64{}, c: int64(nd.At)}
			} else {
				r = &affNode{terms: map[int]int64{n: 1}}
			}
		case nd.Kind == NUBinary && (nd.Op == OpAdd || nd.Op == OpSub):
			x, y := aff(nd.A), aff(nd.B)
			if x == nil || y == nil {
				break
			}
			sign := int64(1)
			if nd.Op == OpSub {
				sign = -1
			}
			r = &affNode{terms: map[int]int64{}, c: x.c + sign*y.c}
			for k, v := range x.terms {
				r.terms[k] += v
			}
			for k, v := range y.terms {
				r.terms[k] += sign * v
			}
		case nd.Kind == NUBinary && nd.Op == OpMul:
			x, y := aff(nd.A), aff(nd.B)
			if x == nil || y == nil {
				break
			}
			if len(y.terms) != 0 {
				x, y = y, x
			}
			if len(y.terms) != 0 {
				break
			}
			r = &affNode{terms: map[int]int64{}, c: x.c * y.c}
			for k, v := range x.terms {
				r.terms[k] = v * y.c
			}
		}
		memo[n] = r
		return r
	}
	type fam struct {
		stride int64
		base   []int // the invariant atoms, sorted
		coefs  []int64
		c      int64
		uses   []int
	}
	var fams []*fam
	seen := map[int]bool{}
	for _, nd := range g.Nodes {
		if nd.Kind != NLoad || !nd.Indexed || seen[nd.Ix] || !vary[nd.Ix] || !nested[nd.Ix] {
			continue
		}
		seen[nd.Ix] = true
		a := aff(nd.Ix)
		if a == nil || a.terms[iter] == 0 {
			continue
		}
		var base []int
		for k, v := range a.terms {
			if k != iter && v != 0 && !nested[k] {
				base = append(base, k)
			}
		}
		sort.Ints(base)
		coefs := make([]int64, len(base))
		for i, k := range base {
			coefs[i] = a.terms[k]
		}
		var f *fam
		for _, x := range fams {
			if x.stride == a.terms[iter] && slices.Equal(x.base, base) && slices.Equal(x.coefs, coefs) {
				f = x
			}
		}
		if f == nil {
			f = &fam{stride: a.terms[iter], base: base, coefs: coefs, c: a.c}
			fams = append(fams, f)
		}
		f.uses = append(f.uses, nd.Ix)
	}
	if len(fams) == 0 {
		return g
	}
	r := g.clone()
	u := r.UConst
	scaled := func(n int, k int64) int {
		if k == 1 {
			return n
		}
		return r.UBin(OpMul, n, u(int(k)))
	}
	for _, f := range fams {
		start := -1
		add := func(v int) {
			if start < 0 {
				start = v
			} else {
				start = r.UBin(OpAdd, start, v)
			}
		}
		for i, k := range f.base {
			add(scaled(k, f.coefs[i]))
		}
		if f.c != 0 || start < 0 {
			add(u(int(f.c)))
		}
		if c == 0 && s.Split > 1 && len(r.Loops) > 0 {
			// Under a split loop 0 starts at the segment's first trip.
			seg := r.Add(Node{Kind: NSegment})
			add(scaled(scaled(seg, int64(r.Loops[0]/s.Split)), f.stride))
		}
		fold := r.Fold(c, start)
		r.SetFold(fold, r.UBin(OpAdd, fold, u(int(f.stride))))
		for _, n := range f.uses {
			a := aff(n)
			rest := -1
			for k, v := range a.terms {
				if nested[k] && k != iter && v != 0 {
					t := scaled(k, v)
					if rest < 0 {
						rest = t
					} else {
						rest = r.UBin(OpAdd, rest, t)
					}
				}
			}
			if d := a.c - f.c; d != 0 {
				if rest < 0 {
					rest = u(int(d))
				} else {
					rest = r.UBin(OpAdd, rest, u(int(d)))
				}
			}
			if rest < 0 {
				r.Nodes[n] = Node{Kind: NUBinary, Op: OpAdd, A: fold, B: u(0)}
			} else {
				r.Nodes[n] = Node{Kind: NUBinary, Op: OpAdd, A: fold, B: rest}
			}
		}
	}
	return r
}

// clone is a copy of g its rewrites may change.
func (g *Row) clone() *Row {
	c := *g
	c.Nodes = append([]Node(nil), g.Nodes...)
	c.Stores = append([]Store(nil), g.Stores...)
	c.Loops = append([]int(nil), g.Loops...)
	c.Stages = make([]Stage, len(g.Stages))
	for i, st := range g.Stages {
		c.Stages[i] = Stage{Width: st.Width, Stores: append([]Store(nil), st.Stores...)}
	}
	if g.Q8 != nil {
		q := *g.Q8
		c.Q8 = &q
	}
	return &c
}
