package compose

// Span is an index into a row's span of a descriptor buffer, recorded so a
// pass may widen it: Clamp is index T clamped into row Row's keys [start,
// end) of descriptor Desc (Words a row), equal to T inside them and any key
// of the row's table outside; or, where Uniform, a value of the row that is
// the same over its group. Rows of an aligned group of SpanGroup share such
// values and their spans do not decrease, so under a tile of such rows the
// tile's first row's values and the tile's widest span serve every row.
type Span struct {
	Clamp, T, Row     int
	Uniform           bool
	Desc, Words       int
	Start, End, Align int
}

// SpanWord is word w of row row's descriptor.
func (g *Row) SpanWord(desc, words, row, w int) int {
	u, ub := g.UConst, g.UBin
	return g.LoadAt(desc, ub(OpAdd, ub(OpMul, row, u(words)), u(w)))
}

// SpanUniform states that node n has one value over its row's group.
func (g *Row) SpanUniform(n int) int {
	g.Spans = append(g.Spans, Span{Clamp: n, Uniform: true})
	return n
}

// SpanClamp is index t clamped into row row's keys [start, end), aligned
// down to align where it is over one; position 0 for an empty span.
func (g *Row) SpanClamp(desc, words, row, start, end, t, align int) int {
	n := g.clampInto(t, g.SpanWord(desc, words, row, start), g.SpanWord(desc, words, row, end), align)
	g.Spans = append(g.Spans, Span{Clamp: n, T: t, Row: row, Desc: desc, Words: words, Start: start, End: end, Align: align})
	return n
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

// hulls widens every span under a row tile of schedule s to the tile's:
// a uniform value at the first row, a clamp into the first row's start and
// the last row's end.
func (g *Row) hulls(s Schedule) *Row {
	tr := max(s.Tile, 1)
	if len(g.Spans) == 0 || !s.Flat || s.MMA || tr == 1 || g.SpanGroup%tr != 0 {
		return g
	}
	r := g.clone()
	last := tr - 1
	for _, sp := range g.Spans {
		var n int
		if sp.Uniform {
			n = r.atRowOff(sp.Clamp, 0)
		} else {
			first, end := r.atRowOff(sp.Row, 0), r.atRowOff(sp.Row, last)
			n = r.clampInto(r.atRowOff(sp.T, 0), r.SpanWord(sp.Desc, sp.Words, first, sp.Start), r.SpanWord(sp.Desc, sp.Words, end, sp.End), sp.Align)
		}
		r.Nodes[sp.Clamp] = r.Nodes[n]
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
		case NUBinary, NLess, NSelect, NUnary, NBinary:
			c := nd
			c.A = at(nd.A)
			if nd.Kind != NUnary {
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
