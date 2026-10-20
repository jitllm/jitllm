package compose

import (
	"maps"
	"slices"
	"sort"

	"github.com/jitllm/jitllm/jit/gpu/ir"
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
	inLoop := make([]bool, len(g.Loops))
	for c2 := range g.Loops {
		if c2 == c {
			continue
		}
		for _, nd := range g.Nodes {
			if nd.Kind == NFold && nd.At == c2 && (reads(nd.B) || nd.A >= 0 && reads(nd.A)) {
				inLoop[c2] = true
			}
		}
		if !inLoop[c2] {
			continue
		}
		v2 := g.varies(c2)
		for i := range nested {
			nested[i] = nested[i] || v2[i]
		}
	}
	// unrolled is whether loop c runs unrolled in another's body: there a
	// step wider than one is an add from the trip before, a unit step the
	// access's offset.
	unrolled := false
	if g.Loops[c] > 0 && g.Loops[c] <= unrollTrips && (c > 0 || s.Split <= 1) {
		for c0 := range g.Loops {
			if c0 != c && g.foldReadsIter(c, c0) {
				unrolled = true
			}
		}
	}
	// offs is which nodes read a tile offset: an invariant sum holding one is
	// read through, so the offset reaches the uses and not the fold.
	offs := make([]bool, len(g.Nodes))
	for changed := true; changed; {
		changed = false
		for i := range g.Nodes {
			if offs[i] {
				continue
			}
			x := g.isTileOff(i)
			for _, o := range g.operands(i) {
				x = x || offs[o]
			}
			if x {
				offs[i], changed = true, true
			}
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
		case (n == iter || nested[n] && !vary[n] || !vary[n] && !nested[n]) && !(nd.Kind == NUBinary && offs[n]):
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
	// The indices: what a load reads at, and where a fold of a loop nested
	// in c starts.
	var cands []int
	init := map[int]bool{}
	// loads is how many loads read each index; an index written as a word
	// scaled to its elements (X*k + Z) counts for X,
	// the word, which is what the cursor counts, Z a place known per env.
	loads := map[int]int{}
	for _, nd := range g.Nodes {
		switch {
		case nd.Kind == NLoad && nd.Indexed:
			ix := nd.Ix
			if x := g.Nodes[ix]; x.Kind == NUBinary && x.Op == OpAdd && g.fixedPlace(x.B) {
				if m := g.Nodes[x.A]; m.Kind == NUBinary && m.Op == OpMul && g.Nodes[m.B].Kind == NUConst && vary[m.A] {
					ix = m.A
				}
			}
			cands = append(cands, ix)
			loads[ix]++
		case nd.Kind == NFold && nd.At != c && inLoop[nd.At] && g.typeOf(nd.B) == ir.U32:
			cands = append(cands, nd.B)
			init[nd.B] = true
		}
	}
	// A cursor serves where several loads step together; one load reads at
	// its iterator. A nested or unrolled loop's index, and a nested loop's
	// start, always ride one.
	// formed is whether the iterator's terms are formed anyway: read by a node
	// that is no affine index (a clamp, a mask, a page's place). Then an
	// index adds them as cheaply as a cursor steps, and no cursor serves.
	formed := g.iterFormed(iter)
	ridden := map[int]bool{}
	// sole is the one index the iterator serves, where it serves one: its
	// cursor replaces the loop's counter -- in a run-time loop, unless it
	// steps by one, when it is the counter and a base.
	sole := -1
	{
		var in []int
		for _, ix := range cands {
			if vary[ix] && !slices.Contains(in, ix) {
				in = append(in, ix)
			}
		}
		if len(in) == 1 {
			tree := map[int]bool{}
			var walk func(n int)
			walk = func(n int) {
				if tree[n] {
					return
				}
				tree[n] = true
				for _, o := range g.operands(n) {
					walk(o)
				}
			}
			walk(in[0])
			sole = in[0]
			for i := range g.Nodes {
				if !tree[i] && slices.Contains(g.operands(i), iter) {
					sole = -1
				}
			}
		}
	}
	for _, ix := range cands {
		if seen[ix] || !vary[ix] || ix == iter {
			continue
		}
		seen[ix] = true
		a := aff(ix)
		if a == nil || a.terms[iter] == 0 {
			continue
		}
		ridden[ix] = nested[ix] || unrolled && a.terms[iter] != 1 || init[ix] || loads[ix] > 1 && !formed || ix == sole && (a.terms[iter] != 1 || g.Loops[c] > 0)
		var base []int
		for k, v := range a.terms {
			if k != iter && v != 0 && !nested[k] && !g.isTileOff(k) {
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
		f.uses = append(f.uses, ix)
	}
	fams = slices.DeleteFunc(fams, func(f *fam) bool {
		return (len(f.uses) < 2 || formed) && !slices.ContainsFunc(f.uses, func(ix int) bool { return ridden[ix] })
	})
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
			for _, k := range slices.Sorted(maps.Keys(a.terms)) {
				v := a.terms[k]
				if (nested[k] || g.isTileOff(k)) && k != iter && v != 0 {
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
		// A cursor of a loop in c's body that starts here and steps the whole
		// stride over its trips: this one steps by where that one ends.
		for _, n := range f.uses {
			if r.Nodes[n].B != u(0) && r.Nodes[r.Nodes[n].B].At != 0 {
				continue
			}
			for i, nd := range r.Nodes {
				if nd.Kind != NFold || nd.At == c || nd.B != n || nd.A < 0 {
					continue
				}
				if st := r.Nodes[nd.A]; st.Kind == NUBinary && st.Op == OpAdd && st.A == i && r.Nodes[st.B].Kind == NUConst &&
					int64(r.Loops[nd.At]*r.Nodes[st.B].At) == f.stride {
					r.SetFold(fold, i)
				}
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
	c.Trip = append([]int(nil), g.Trip...)
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

// foldReadsIter is whether a fold of loop c starts or steps by loop c0's
// iterator: c runs in c0's body.
func (g *Row) foldReadsIter(c, c0 int) bool {
	memo := map[int]bool{}
	var reads func(n int) bool
	reads = func(n int) bool {
		if v, ok := memo[n]; ok {
			return v
		}
		memo[n] = false
		nd := g.Nodes[n]
		v := nd.Kind == NIter && nd.At == c0
		for _, o := range g.operands(n) {
			v = v || reads(o)
		}
		memo[n] = v
		return v
	}
	for _, nd := range g.Nodes {
		if nd.Kind == NFold && nd.At == c && (reads(nd.B) || nd.A >= 0 && reads(nd.A)) {
			return true
		}
	}
	return false
}

// tiled is g with its column and row split for schedule s's tile: each the
// tile's first plus the env's place in it (NColBase + NColOff, NRowBase +
// NRowOff), so what reads only the first is shared by the tile's envs.
func (g *Row) tiled(s Schedule) *Row {
	tr, tc := max(s.Tile, 1), max(s.TileCols, 1)
	if !s.Flat || s.MMA || tr*tc == 1 {
		return g
	}
	r := g.clone()
	// A strided tile's first place and step, as nodes: the tile t's first
	// t/k*n*k + t%k and its j-th that plus j*k.
	place := func(row bool) Node {
		base, tile, off, n, k := NColBase, NColTile, NColOff, tc, s.ColStride
		if row {
			base, tile, off, n, k = NRowBase, NRowTile, NRowOff, tr, s.RowStride
		}
		if k <= 1 {
			return Node{Kind: NUBinary, Op: OpAdd, A: r.Add(Node{Kind: base}), B: r.Add(Node{Kind: off})}
		}
		u, ub := r.UConst, r.UBin
		t := r.Add(Node{Kind: tile})
		first := ub(OpAdd, ub(OpMul, ub(OpDiv, t, u(k)), u(n*k)), ub(OpRem, t, u(k)))
		return Node{Kind: NUBinary, Op: OpAdd, A: first, B: ub(OpMul, r.Add(Node{Kind: off}), u(k))}
	}
	for i, nd := range g.Nodes {
		switch {
		case nd.Kind == NColumn && tc > 1:
			r.Nodes[i] = place(false)
		case nd.Kind == NRowIdx && tr > 1:
			r.Nodes[i] = place(true)
		}
	}
	r = r.divmods(s)
	r.liftOffsets(s)
	return r
}

// isTileOff is whether node n is an env's place in its tile: a part of an
// index each env adds to what the tile shares.
func (g *Row) isTileOff(n int) bool { k := g.Nodes[n].Kind; return k == NColOff || k == NRowOff }

// fixedPlace is whether index node n is the same for every thread: constants
// and an env's place in its tile, summed and scaled.
func (g *Row) fixedPlace(n int) bool {
	nd := g.Nodes[n]
	switch nd.Kind {
	case NUConst, NColOff, NRowOff:
		return true
	case NUBinary:
		return (nd.Op == OpAdd || nd.Op == OpMul) && g.fixedPlace(nd.A) && g.fixedPlace(nd.B)
	}
	return false
}

// pipelined is g with every indirection a loop's trips read one trip
// ahead: a table load whose value addresses other loads and whose index
// reads the loop's iterator (and nothing else of the loop) becomes a value
// the loop carries, the first trip's loaded before it and each next trip's
// at the end of the one before. g is copied before it is changed.
func (g *Row) pipelined() *Row {
	addr := make([]bool, len(g.Nodes))
	var mark func(n int)
	mark = func(n int) {
		if addr[n] {
			return
		}
		addr[n] = true
		for _, o := range g.operands(n) {
			mark(o)
		}
	}
	for _, nd := range g.Nodes {
		if nd.Kind == NLoad && nd.Indexed {
			mark(nd.Ix)
		}
	}
	var r *Row
	for c := range g.Loops {
		iter := -1
		for i, nd := range g.Nodes {
			if nd.Kind == NIter && nd.At == c {
				iter = i
			}
		}
		if iter < 0 {
			continue
		}
		vary := g.varies(c)
		for i, nd := range g.Nodes {
			if nd.Kind != NLoad || !nd.Indexed || !addr[i] || g.Bufs[nd.Buf].Kind != BufU32 || !vary[nd.Ix] ||
				!g.onlyIter(nd.Ix, iter, c) || !g.quotientOf(nd.Ix, iter) {
				continue
			}
			if r == nil {
				r = g.clone()
			}
			at := func(x int) int {
				memo := map[int]int{}
				var cp func(n int) int
				cp = func(n int) int {
					if n == iter {
						return x
					}
					if !vary[n] {
						return n
					}
					if v, ok := memo[n]; ok {
						return v
					}
					m := r.Nodes[n]
					switch m.Kind {
					case NUBinary, NBinary, NLess:
						m.A, m.B = cp(m.A), cp(m.B)
					case NSelect:
						m.A, m.B, m.C = cp(m.A), cp(m.B), cp(m.C)
					case NUnary:
						m.A = cp(m.A)
					case NLoad:
						m.Ix = cp(m.Ix)
					}
					v := r.Add(m)
					memo[n] = v
					return v
				}
				return r.LoadAt(nd.Buf, cp(nd.Ix))
			}
			first := at(r.UConst(0))
			step := at(r.UBin(OpAdd, iter, r.UConst(1)))
			r.Nodes[i] = Node{Kind: NFold, At: c, B: first, A: step}
		}
	}
	if r == nil {
		return g
	}
	return r
}

// onlyIter is whether index n reads loop c's iterator and no other value
// of the loop (no fold of it).
func (g *Row) onlyIter(n, iter, c int) bool {
	seen := map[int]bool{}
	var ok func(n int) bool
	ok = func(n int) bool {
		if seen[n] {
			return true
		}
		seen[n] = true
		if nd := g.Nodes[n]; nd.Kind == NFold && nd.At == c {
			return false
		}
		for _, o := range g.operands(n) {
			if !ok(o) {
				return false
			}
		}
		return true
	}
	return ok(n)
}

// quotientOf is whether index node n reads iterator iter through a
// quotient: a table entry several trips share (a page of positions), which
// is worth loading a trip ahead.
func (g *Row) quotientOf(n, iter int) bool {
	seen := map[int]bool{}
	var reads func(n int) bool
	reads = func(n int) bool {
		if n == iter {
			return true
		}
		if seen[n] {
			return false
		}
		seen[n] = true
		for _, o := range g.operands(n) {
			if reads(o) {
				return true
			}
		}
		return false
	}
	var q func(n int) bool
	q = func(n int) bool {
		nd := g.Nodes[n]
		if nd.Kind == NUBinary && (nd.Op == OpDiv || nd.Op == OpShr) && reads(nd.A) {
			return true
		}
		for _, o := range g.operands(n) {
			if q(o) {
				return true
			}
		}
		return false
	}
	return q(n)
}

// iterFormed is whether a node that is no affine index reads an affine
// expression of iterator node iter: its value is formed whatever cursors
// there are.
func (g *Row) iterFormed(iter int) bool {
	aff := make([]bool, len(g.Nodes))
	aff[iter] = true
	for changed := true; changed; {
		changed = false
		for i, nd := range g.Nodes {
			if aff[i] || nd.Kind != NUBinary || nd.Op != OpAdd && nd.Op != OpSub && nd.Op != OpMul {
				continue
			}
			if aff[nd.A] || aff[nd.B] {
				aff[i], changed = true, true
			}
		}
	}
	for i, nd := range g.Nodes {
		if aff[i] || nd.Kind == NLoad || nd.Kind == NFold {
			continue
		}
		for _, o := range g.operands(i) {
			if aff[o] {
				return true
			}
		}
	}
	return false
}
