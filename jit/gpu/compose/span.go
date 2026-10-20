package compose

// SpanWord is word w of row row's descriptor.
func (g *Row) SpanWord(desc, words, row, w int) int {
	u, ub := g.UConst, g.UBin
	return g.LoadAt(desc, ub(OpAdd, ub(OpMul, row, u(words)), u(w)))
}

// SpanUniform is n, hinted to have one value over its row's group.
func (g *Row) SpanUniform(n int) int {
	return g.Add(Node{Kind: NHint, Hint: HintUniform, A: n})
}

// SpanClamp is index t clamped into row row's keys [start, end), aligned
// down to align where it is over one, position 0 for an empty span; hinted
// to be any key of the row's table outside the keys.
func (g *Row) SpanClamp(desc, words, row, start, end, t, align int) int {
	n := g.clampInto(t, g.SpanWord(desc, words, row, start), g.SpanWord(desc, words, row, end), align)
	return g.Add(Node{Kind: NHint, Hint: HintSpan, A: n, B: t, C: row, Buf: desc, At: align, H: [3]int32{int32(words), int32(start), int32(end)}})
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

// hulls widens every span hint under a row tile of schedule s to the
// tile's -- a uniform value at the first row, a clamp into the first row's
// start and the last row's end -- and drops every hint.
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
		case widen && nd.Hint == HintUniform:
			n = r.atRowOff(nd.A, 0)
		case widen && nd.Hint == HintSpan:
			words, start, end := int(nd.H[0]), int(nd.H[1]), int(nd.H[2])
			first, last := r.atRowOff(nd.C, 0), r.atRowOff(nd.C, tr-1)
			n = r.clampInto(r.atRowOff(nd.B, 0), r.SpanWord(nd.Buf, words, first, start), r.SpanWord(nd.Buf, words, last, end), nd.At)
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
