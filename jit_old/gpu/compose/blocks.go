package compose

// Range is loop c's iteration over a coordinate: its value S = Lo + the
// iterator, over [Lo, Hi), every place it names inside [Origin,
// Origin+Ext), its blocks aligned from Origin.
type Range struct {
	S, Lo, Hi, Origin, Ext int
	ok                     bool
	Form                   BlockForm
}

// BlockForm is how the key-block pass cuts a range, its parameters: the
// zero form peels the block holding Lo and the one past the last whole
// block, masked, holds an affine read's block start inside the extent and
// selects every float read outside [Lo, Hi).
type BlockForm struct {
	// All masks every block, one loop of them from Lo.
	All bool
	// Base, where set, is 1 + the node a place's liveness counts from
	// (Lo otherwise).
	Base int
	// Paged selects only the reads a place's index reaches through more
	// than a sum (a page's remainder), leaving an affine read as it is.
	Paged bool
	// Align, where set, aligns a page's clamp to it (the block otherwise).
	Align int
}

// RangeLoop is a loop over the coordinate's [lo, hi), every place it names
// inside [origin, origin+ext): the loop and the coordinate's value in its
// body.
func (g *Row) RangeLoop(origin, lo, hi, ext int) (c, s int) {
	return g.RangeLoopForm(origin, lo, hi, ext, BlockForm{})
}

// RangeLoopForm is RangeLoop cut by the block pass in form f.
func (g *Row) RangeLoopForm(origin, lo, hi, ext int, f BlockForm) (c, s int) {
	c = g.LoopN(g.UBin(OpSub, hi, lo))
	s = g.UBin(OpAdd, lo, g.Iter(c))
	for len(g.Ranges) <= c {
		g.Ranges = append(g.Ranges, Range{})
	}
	g.Ranges[c] = Range{S: s, Lo: lo, Hi: hi, Origin: origin, Ext: ext, ok: true, Form: f}
	return c, s
}

// KeyAt is key t's page index clamped by the row's own facts: live where
// the row has a key, t held at or below last; under key blocks the block's
// first key, held at or below last aligned (BlockForm.Align), serves the
// block.
func (g *Row) KeyAt(t, live, last int) int {
	sel := g.Select(live, g.UBin(OpMin, t, last), g.UConst(0))
	return g.Add(Node{Kind: NHint, Hint: HintKey, A: sel, B: t, C: live, Ix: last, At: 1})
}

// KeyClamp is key t clamped into [lo, hi), hinted to be read only for its
// page: under key blocks the block's first key, clamped aligned, serves the
// block.
func (g *Row) KeyClamp(t, lo, hi int) int {
	return g.Add(Node{Kind: NHint, Hint: HintKey, A: g.clampInto(t, lo, hi, 1), B: t, C: lo, Ix: hi})
}

// blocks cuts each range loop of schedule s's Block into aligned blocks of
// the coordinate: the block holding Lo and the block past the last whole one
// masked -- what reads the coordinate selected to zero outside [Lo, Hi), an
// affine read's block start held inside [0, Ext-Block) -- and the blocks
// between unmasked, a block a trip; a key's page taken once a block.
func (g *Row) blocks(s Schedule) *Row {
	B := s.Block
	if B <= 1 {
		return g
	}
	var r *Row
	for c, rg := range g.Ranges {
		if !rg.ok || g.tripOf(c) < 0 {
			continue
		}
		if r == nil {
			r = g.clone()
		}
		r.blockLoop(c, rg, B)
	}
	if r == nil {
		return g
	}
	return r
}

func (r *Row) blockLoop(c int, rg Range, B int) {
	u, ub := r.UConst, r.UBin
	n0 := len(r.Nodes)
	vary, body := r.varies(c), r.bodyOf(c)
	iter := -1
	var folds []int
	for i, nd := range r.Nodes {
		switch {
		case nd.Kind == NIter && nd.At == c:
			iter = i
		case nd.Kind == NFold && nd.At == c:
			folds = append(folds, i)
		}
	}
	steps := map[int]int{}
	for _, f := range folds {
		steps[f] = r.Nodes[f].A
	}
	f := rg.Form
	aB := ub(OpAdd, rg.Origin, ub(OpMul, ub(OpDiv, ub(OpSub, rg.Lo, rg.Origin), u(B)), u(B)))
	nb := r.Select(r.Less(rg.Lo, rg.Hi), ub(OpDiv, ub(OpAdd, ub(OpSub, rg.Hi, aB), u(B-1)), u(B)), u(0))
	if f.All {
		aB = rg.Lo
		nb = ub(OpDiv, ub(OpAdd, ub(OpSub, rg.Hi, rg.Lo), u(B-1)), u(B))
	}
	mid := ub(OpSub, ub(OpMax, nb, u(2)), u(2))
	base := rg.Lo
	if f.Base > 0 {
		base = f.Base - 1
	}
	// block is the block from bs, its B places in order, the folds' values
	// from vals.
	block := func(bs int, masked bool, vals map[int]int) map[int]int {
		page := map[int]int{}
		var start func(n int) int
		start = func(n int) int {
			if n == rg.S {
				return bs
			}
			if !vary[n] {
				return n
			}
			return r.copyWith(n, start)
		}
		for kk := 0; kk < B; kk++ {
			sk := bs
			if kk > 0 {
				sk = ub(OpAdd, bs, u(kk))
			}
			live := r.Less(ub(OpSub, sk, base), ub(OpSub, rg.Hi, base))
			if f.All {
				// Every block from Lo: its place's offset from the base an
				// invariant, the block's step and the place in it.
				off := ub(OpAdd, ub(OpSub, rg.Lo, base), ub(OpMul, iter, u(B)))
				if kk > 0 {
					off = ub(OpAdd, off, u(kk))
				}
				live = r.Less(off, ub(OpSub, rg.Hi, base))
			}
			memo := map[int]int{}
			var at func(n int) int
			at = func(n int) int {
				if n == rg.S {
					return sk
				}
				if v, ok := vals[n]; ok {
					return v
				}
				if !vary[n] {
					return n
				}
				if m, ok := memo[n]; ok {
					return m
				}
				nd := r.Nodes[n]
				var m int
				switch {
				case nd.Kind == NHint && nd.Hint == HintKey:
					switch p, ok := page[n]; {
					case ok:
						m = p
					case nd.At == 1:
						al := B
						if f.Align > 0 {
							al = f.Align
						}
						last := ub(OpMul, ub(OpDiv, nd.Ix, u(al)), u(al))
						m = r.Select(nd.C, ub(OpMin, start(nd.B), last), u(0))
						page[n] = m
					default:
						m = r.clampInto(start(nd.B), nd.C, nd.Ix, B)
						page[n] = m
					}
				case nd.Kind == NLoad && nd.Indexed && masked && f.Paged && r.affineIn(nd.Ix, rg.S):
					// An affine read of a paged form is read as it is.
					m = r.copyWith(n, at)
				case nd.Kind == NLoad && nd.Indexed && masked:
					ix := r.Nodes[n].Ix
					var at2 func(x int) int
					if r.affineIn(ix, rg.S) && !f.Paged {
						held := ub(OpAdd, rg.Origin, ub(OpMin, ub(OpSub, bs, rg.Origin), u(rg.Ext-B)))
						if kk > 0 {
							held = ub(OpAdd, held, u(kk))
						}
						memo2 := map[int]int{}
						at2 = func(x int) int {
							if x == rg.S {
								return held
							}
							if !vary[x] {
								return x
							}
							if v, ok := memo2[x]; ok {
								return v
							}
							v := r.copyWith(x, at2)
							memo2[x] = v
							return v
						}
					} else {
						at2 = at
					}
					l := r.Nodes[n]
					l.Ix = at2(ix)
					m = r.Add(l)
					if r.typeOf(n) != r.typeOf(rg.S) {
						m = r.Select(live, m, r.Const(0))
					}
				default:
					m = r.copyWith(n, at)
				}
				memo[n] = m
				return m
			}
			next := map[int]int{}
			for _, f := range folds {
				next[f] = at(steps[f])
			}
			vals = next
		}
		return vals
	}
	if f.All {
		own := map[int]int{}
		for _, fo := range folds {
			own[fo] = fo
		}
		steps := block(ub(OpAdd, aB, ub(OpMul, iter, u(B))), true, own)
		for _, fo := range folds {
			r.Nodes[fo].A = steps[fo]
		}
		r.Trip[c] = nb
		r.Ranges[c].ok = false
		r.Nodes[rg.S] = Node{Kind: NUConst}
		return
	}
	first := map[int]int{}
	for _, f := range folds {
		first[f] = r.Nodes[f].B
	}
	first = block(aB, true, first)
	own := map[int]int{}
	for _, f := range folds {
		own[f] = f
	}
	bsMid := ub(OpAdd, ub(OpAdd, aB, u(B)), ub(OpMul, iter, u(B)))
	midSteps := block(bsMid, false, own)
	last := block(ub(OpAdd, aB, ub(OpMul, ub(OpAdd, mid, u(1)), u(B))), true, own)
	for k := 0; k < n0; k++ {
		if body[k] {
			continue
		}
		nd := &r.Nodes[k]
		if nd.Kind == NFold && nd.At == c {
			continue
		}
		for _, f := range folds {
			nd.replaceOperand(f, last[f])
		}
	}
	for j, st := range r.Stores {
		for _, f := range folds {
			if st.Val == f {
				r.Stores[j].Val = last[f]
			}
		}
	}
	for _, f := range folds {
		r.Nodes[f].B, r.Nodes[f].A = first[f], midSteps[f]
	}
	r.Trip[c] = mid
	r.Ranges[c].ok = false
	// What read the coordinate per place reads the blocks' places now; the
	// per-place body left behind reads nothing of the loop.
	r.Nodes[rg.S] = Node{Kind: NUConst}
}

// copyWith is node n with its operands mapped by at.
func (r *Row) copyWith(n int, at func(int) int) int {
	nd := r.Nodes[n]
	switch nd.Kind {
	case NUnary, NWindow, NShuffle, NHint, NReduce:
		nd.A = at(nd.A)
	case NBinary, NUBinary, NLess:
		nd.A, nd.B = at(nd.A), at(nd.B)
		if nd.ternary() {
			nd.C = at(nd.C)
		}
	case NSelect, NDot4:
		nd.A, nd.B, nd.C = at(nd.A), at(nd.B), at(nd.C)
	case NAt:
		nd.A, nd.Ix = at(nd.A), at(nd.Ix)
	case NLoad:
		if nd.Indexed {
			nd.Ix = at(nd.Ix)
		}
	default:
		return n
	}
	return r.Add(nd)
}

// affineIn reports whether index n reads s, and only as a term of a sum
// (not inside a quotient, remainder or any other operation).
func (g *Row) affineIn(n, s int) bool {
	has, bad := false, false
	var walk func(x int)
	walk = func(x int) {
		if x == s {
			has = true
			return
		}
		nd := g.Nodes[x]
		switch {
		case nd.Kind == NUBinary && (nd.Op == OpAdd || nd.Op == OpSub):
			walk(nd.A)
			walk(nd.B)
		case nd.Kind == NUBinary && nd.Op == OpMul && g.Nodes[nd.B].Kind == NUConst:
			walk(nd.A)
		case nd.Kind == NUBinary && nd.Op == OpMul && g.Nodes[nd.A].Kind == NUConst:
			walk(nd.B)
		default:
			bad = bad || g.readsNode(x, s)
		}
	}
	walk(n)
	return has && !bad
}

// readsNode reports whether node n reads node s.
func (g *Row) readsNode(n, s int) bool {
	seen := map[int]bool{}
	var walk func(x int) bool
	walk = func(x int) bool {
		if x == s {
			return true
		}
		if seen[x] {
			return false
		}
		seen[x] = true
		for _, o := range g.operands(x) {
			if walk(o) {
				return true
			}
		}
		return false
	}
	return walk(n)
}
