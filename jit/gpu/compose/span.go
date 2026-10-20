package compose

// SpanWord is word w of row row's descriptor.
func (g *Row) SpanWord(desc, words, row, w int) int {
	u, ub := g.UConst, g.UBin
	return g.LoadAt(desc, ub(OpAdd, ub(OpMul, row, u(words)), u(w)))
}

// TileFirst is n, hinted to serve every row of a tile taken at its first.
func (g *Row) TileFirst(n int) int { return g.Add(Node{Kind: NHint, Hint: HintTileFirst, A: n}) }

// TileLast is n, hinted to serve every row of a tile taken at its last.
func (g *Row) TileLast(n int) int { return g.Add(Node{Kind: NHint, Hint: HintTileLast, A: n}) }

// SpanClamp is index t clamped into row row's keys [start, end), aligned
// down to align where it is over one, position 0 for an empty span: the
// span's ends hinted to widen to a tile's, so the index is any key of the
// row's table outside its keys.
func (g *Row) SpanClamp(desc, words, row, start, end, t, align int) int {
	return g.clampInto(t, g.TileFirst(g.SpanWord(desc, words, row, start)), g.TileLast(g.SpanWord(desc, words, row, end)), align)
}

func (g *Row) clampInto(t, lo, hi, align int) int {
	u, ub := g.UConst, g.UBin
	live := g.Less(lo, hi)
	last := ub(OpSub, ub(OpMax, hi, u(1)), u(1))
	if align > 1 {
		lo, last = ub(OpMul, ub(OpDiv, lo, u(align)), u(align)), ub(OpMul, ub(OpDiv, last, u(align)), u(align))
	}
	return g.Select(live, ub(OpMin, ub(OpMax, t, lo), last), u(0))
}

// hulls takes every tile hint under a row tile of schedule s at the tile's
// first or last row, and drops every hint.
func (g *Row) hulls(s Schedule) *Row {
	tr := max(s.Tile, 1)
	widen := s.Flat && !s.MMA && tr > 1 && g.SpanGroup%tr == 0
	var r *Row
	for i, nd := range g.Nodes {
		if nd.Kind != NHint {
			continue
		}
		if r == nil {
			r = g.clone()
		}
		n := nd.A
		switch {
		case widen && nd.Hint == HintTileFirst:
			n = r.atRowOff(nd.A, 0)
		case widen && nd.Hint == HintTileLast:
			n = r.atRowOff(nd.A, tr-1)
		}
		r.Nodes[i] = r.Nodes[n]
	}
	if r == nil {
		return g
	}
	return r
}

// atRowOff is node n with the tile's row offset off.
func (g *Row) atRowOff(n, off int) int {
	memo := map[int]int{}
	var at func(n int) int
	at = func(n int) int {
		if m, ok := memo[n]; ok {
			return m
		}
		nd := g.Nodes[n]
		m := n
		switch nd.Kind {
		case NRowOff:
			m = g.UConst(off)
		case NUBinary, NLess, NSelect, NUnary, NBinary, NHint:
			c := nd
			c.A = at(nd.A)
			if nd.Kind != NUnary && nd.Kind != NHint {
				c.B = at(nd.B)
			}
			if nd.Kind == NSelect || nd.Kind == NBinary && nd.Op == OpFma {
				c.C = at(nd.C)
			}
			if c != nd {
				m = g.Add(c)
			}
		case NLoad:
			if nd.Indexed {
				c := nd
				c.Ix = at(nd.Ix)
				if c != nd {
					m = g.Add(c)
				}
			}
		}
		memo[n] = m
		return m
	}
	return at(n)
}
