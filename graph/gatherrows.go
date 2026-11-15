package graph

import "fmt"

// The operands an OpGatherRows index expression reads (an ArgIn's Index).
const (
	GatherRow     = iota // the output row: past the op's bound, the last row written
	GatherOutRows        // Out[0]'s rows
	GatherInRows         // In[0]'s rows, 0 with no input
	GatherLive           // the step's live rows
	GatherPos            // the row's position
	GatherGridW          // the request's patch grid's width
	GatherGridH          // the request's patch grid's height
	GatherMark           // the request's run of table rows before: its first plus one, 0 for none
	GatherMarkN          // that run's length
	GatherTrail          // the request's run of table rows after: its first plus one, 0 for none
	GatherTrailN         // that run's length
	GatherSamples        // the request's clip's samples
	GatherCol            // the output element's column (OpGatherElems)
	GatherOperands
)

// Index builds an OpGatherRows index expression, unsigned integers
// throughout: each method appends a term and returns it as an operand of
// later ones, and the last term is the source row.
type Index struct {
	Terms []Term
	bound int
}

// In is operand k (GatherRow to GatherTrailN).
func (x *Index) In(k int) Arg { return Arg{Kind: ArgIn, Index: k} }

// K is the constant v.
func (x *Index) K(v int) Arg { return Arg{Kind: ArgConst, Const: float64(v)} }

func (x *Index) term(f MapFn, a ...Arg) Arg {
	t := Term{Fn: f}
	copy(t.Args[:], a)
	x.Terms = append(x.Terms, t)
	return Arg{Kind: ArgTerm, Index: len(x.Terms) - 1}
}

func (x *Index) Add(a, b Arg) Arg      { return x.term(MapAdd, a, b) }
func (x *Index) Sub(a, b Arg) Arg      { return x.term(MapSub, a, b) }
func (x *Index) Mul(a, b Arg) Arg      { return x.term(MapMul, a, b) }
func (x *Index) Div(a, b Arg) Arg      { return x.term(MapDiv, a, b) }
func (x *Index) Max(a, b Arg) Arg      { return x.term(MapMax, a, b) }
func (x *Index) Min(a, b Arg) Arg      { return x.term(MapMin, a, b) }
func (x *Index) Less(a, b Arg) Arg     { return x.term(MapLess, a, b) }
func (x *Index) Where(p, a, b Arg) Arg { return x.term(MapWhere, p, a, b) }

// Rem is a mod b, its quotient q's remainder.
func (x *Index) Rem(a, b, q Arg) Arg { return x.Sub(a, x.Mul(q, b)) }

// Bound makes a the last row the gather writes: a row past it is that row,
// recomputed and written again.
func (x *Index) Bound(a Arg) {
	if a.Kind != ArgTerm {
		a = x.Add(a, x.K(0))
	}
	x.bound = a.Index + 1
}

// Attrs is the gather's attributes with r its source row.
func (x *Index) Attrs(r Arg) Attrs {
	if r.Kind != ArgTerm || r.Index != len(x.Terms)-1 {
		x.Add(r, x.K(0))
	}
	return Attrs{Expr: x.Terms, Span: x.bound}
}

func (x *Index) inLast() Arg { return x.Sub(x.In(GatherInRows), x.K(1)) }

// UntileRows is the input's rows -- the cells of a grid in nested order,
// window x window windows of cells, each window's cells in raster order and
// consecutive, the windows in raster order and those at the grid's right and
// lower edges cut to it -- in the cells' raster order: the grid the request's
// patch grid over merge wide, its live rows over merge*merge over that tall.
func UntileRows(merge, window int) Attrs {
	var x Index
	m := max(merge, 1)
	gw := x.Max(x.Div(x.In(GatherGridW), x.K(m)), x.K(1))
	gh := x.Max(x.Div(x.Div(x.In(GatherLive), x.K(m*m)), gw), x.K(1))
	r := x.In(GatherRow)
	cy := x.Div(r, gw)
	cx := x.Rem(r, gw, cy)
	win := x.K(window)
	wy, wx := x.Div(cy, win), x.Div(cx, win)
	top, left := x.Mul(wy, win), x.Mul(wx, win)
	wh := x.Max(x.Min(win, x.Sub(x.Max(gh, top), top)), x.K(1))
	ww := x.Max(x.Min(win, x.Sub(x.Max(gw, left), left)), x.K(1))
	at := x.Add(x.Mul(top, gw), x.Mul(left, wh))
	at = x.Add(at, x.Add(x.Mul(x.Sub(cy, top), ww), x.Sub(cx, left)))
	return x.Attrs(x.Min(at, x.inLast()))
}

// LeadRows is the input with its first n rows the table's row.
func LeadRows(n int) Attrs {
	var x Index
	r := x.In(GatherRow)
	return x.Attrs(x.Where(x.Less(r, x.K(n)), x.In(GatherInRows), r))
}

// TailRow is the input with its last live row the table's row.
func TailRow() Attrs {
	var x Index
	r := x.In(GatherRow)
	return x.Attrs(x.Where(x.Less(x.Add(r, x.K(1)), x.In(GatherLive)), r, x.In(GatherInRows)))
}

// DropRows is the input from row n on.
func DropRows(n int) Attrs {
	var x Index
	return x.Attrs(x.Add(x.In(GatherRow), x.K(n)))
}

// UpsampleRows is the input, a grid of rows w wide, s times as large each
// way: output row (y, c) the input's (y/s, c/s), the nearest.
func UpsampleRows(s, w int) Attrs {
	var x Index
	r, line := x.In(GatherRow), x.K(w*s)
	y := x.Div(r, line)
	c := x.Rem(r, line, y)
	return x.Attrs(x.Add(x.Mul(x.Div(y, x.K(s)), x.K(w)), x.Div(c, x.K(s))))
}

// AppendRow is the input's rows, then the table's row k.
func AppendRow(k int) Attrs {
	var x Index
	r, n := x.In(GatherRow), x.In(GatherInRows)
	return x.Attrs(x.Where(x.Less(r, n), r, x.Add(n, x.K(k))))
}

// TableRows is the table's rows, each an output row.
func TableRows() Attrs {
	var x Index
	return x.Attrs(x.In(GatherRow))
}

// RowBreakRows is the input's rows -- a grid's raster, its width the request's
// patch grid's over s -- with the table's row after each of the grid's rows,
// or after each but the last where trim is 1; rows past the last are not
// written.
func RowBreakRows(s, trim int) Attrs {
	var x Index
	S := x.K(s)
	w := x.Div(x.Max(x.In(GatherGridW), S), S)
	h := x.Div(x.Max(x.In(GatherGridH), S), S)
	line := x.Add(w, x.K(1))
	x.Bound(x.Sub(x.Mul(h, line), x.K(1+trim)))
	r := x.In(GatherRow)
	q := x.Div(r, line)
	k := x.Rem(r, line, q)
	src := x.Min(x.Add(x.Mul(q, w), k), x.inLast())
	return x.Attrs(x.Where(x.Less(k, w), src, x.In(GatherInRows)))
}

// WrapRows is the first table's row, the input's rows and the second's: the
// input a grid of the request's patches over s, each of its rows extra rows
// longer, so the second table's row is output row H*(W+extra)+1.
func WrapRows(s, extra int) Attrs {
	var x Index
	S := x.K(s)
	gw, gh := x.Max(x.In(GatherGridW), S), x.Max(x.In(GatherGridH), S)
	w := x.Add(x.Div(gw, S), x.K(extra))
	end := x.Add(x.Mul(x.Div(gh, S), w), x.K(1))
	x.Bound(end)
	r, n := x.In(GatherRow), x.In(GatherInRows)
	src := x.Min(x.Sub(x.Max(r, x.K(1)), x.K(1)), x.inLast())
	return x.Attrs(x.Where(x.Less(r, x.K(1)), n, x.Where(x.Less(r, end), src, x.Add(n, x.K(1)))))
}

// MarkRows is the input's rows -- a square's merged grid, the request's
// patch grid over merge, or where merge is 0 every input row -- after the
// run of the table's rows the request names (GatherMark, at most lead long)
// and before the run it names after them (GatherTrail, at most trail long),
// the table rows rows long; rows past them are not written.
func MarkRows(merge, lead, trail, rows int) Attrs {
	var x Index
	n := x.In(GatherInRows)
	body := n
	if merge > 0 {
		S := x.K(merge)
		w := x.Div(x.Max(x.In(GatherGridW), S), S)
		h := x.Div(x.Max(x.In(GatherGridH), S), S)
		body = x.Mul(h, w)
	}
	run := func(at, cnt, most int) (first, length Arg) {
		m := x.In(at)
		first = x.Sub(x.Max(m, x.K(1)), x.K(1))
		length = x.Min(x.Mul(x.Min(m, x.K(1)), x.In(cnt)), x.K(most))
		return first, length
	}
	mark, before := run(GatherMark, GatherMarkN, lead)
	tr, after := x.K(0), x.K(0)
	if trail > 0 {
		tr, after = run(GatherTrail, GatherTrailN, trail)
	}
	end := x.Add(body, before)
	x.Bound(x.Sub(x.Add(end, after), x.K(1)))
	r := x.In(GatherRow)
	isBefore := x.Less(r, before)
	named := x.Where(isBefore, x.Add(mark, r), x.Add(tr, x.Sub(x.Max(r, end), end)))
	row := x.Add(n, x.Min(named, x.K(rows-1)))
	src := x.Min(x.Sub(x.Max(r, before), before), x.inLast())
	return x.Attrs(x.Where(isBefore, row, x.Where(x.Less(x.Sub(end, x.K(1)), r), row, src)))
}

// gridCell is the row's cell on the request's grid: its position less lead,
// and the grid's width.
func (x *Index) gridCell(lead int) (q, gw Arg) {
	gw = x.Max(x.In(GatherGridW), x.K(1))
	q = x.Sub(x.Max(x.In(GatherPos), x.K(lead)), x.K(lead))
	return q, gw
}

// PosBuckets is each row's row of an s x s position table at its cell of the
// request's grid (its position less lead) bucketed onto it: row
// floor(s*y/gh), column floor(s*x/gw).
func PosBuckets(s, lead int) Attrs {
	var x Index
	q, gw := x.gridCell(lead)
	gh := x.Max(x.In(GatherGridH), x.K(1))
	S := x.K(s)
	y := x.Div(q, gw)
	col := x.Div(x.Mul(x.Rem(q, gw, y), S), gw)
	row := x.Div(x.Mul(y, S), gh)
	return x.Attrs(x.Add(x.Mul(x.Min(row, x.K(s-1)), S), x.Min(col, x.K(s-1))))
}

// PosAxis is each row's row of a table of two axes of n rows each: the
// first axis's row at its cell's column (axis 0), or the second's at its
// row (axis 1), the cell the row's position less lead on the request's grid.
func PosAxis(n, lead, axis int) Attrs {
	var x Index
	q, gw := x.gridCell(lead)
	y := x.Div(q, gw)
	if axis == 0 {
		return x.Attrs(x.Min(x.Rem(q, gw, y), x.K(n-1)))
	}
	return x.Attrs(x.Add(x.Min(y, x.K(n-1)), x.K(n)))
}

// ClipRows is the input with the rows past a clip's valid length zero: the
// rows a grid w wide (0: one row a grid row), a grid row kept below L, L the
// request's samples over stride halved halvings times as a stride-two
// convolution of three taps padded by one halves it (L = (L-1)/2+1): row*2^h
// < L0 is (row*2^h + 1)*stride <= the samples, no division of the count.
func ClipRows(w, stride, halvings int) Attrs {
	var x Index
	r := x.In(GatherRow)
	row := r
	if w > 0 {
		row = x.Div(r, x.K(w))
	}
	end := x.Mul(x.Add(x.Mul(row, x.K(1<<halvings)), x.K(1)), x.K(stride))
	return x.Attrs(x.Where(x.Less(x.In(GatherSamples), end), x.K(GatherNone), r))
}

// liveRows is the step's live rows less drop, at least one.
func (x *Index) liveRows(drop int) Arg {
	live := x.In(GatherLive)
	return x.Max(x.Where(x.Less(live, x.K(drop+1)), x.K(0), x.Sub(live, x.K(drop))), x.K(1))
}

// LiveRows is the count of the step's live rows less drop, at least one
// (an OpMeanRows' rows).
func LiveRows(drop int) Attrs {
	var x Index
	return x.Attrs(x.liveRows(drop))
}

// KeepLiveRows is the input with its rows from the step's live rows less
// drop (at least one) on zero.
func KeepLiveRows(drop int) Attrs {
	var x Index
	n := x.liveRows(drop)
	r := x.In(GatherRow)
	return x.Attrs(x.Where(x.Less(r, n), r, x.K(GatherNone)))
}

// FirstRow is every row the input's first.
func FirstRow() Attrs {
	var x Index
	return x.Attrs(x.K(0))
}

// past is an element index no source holds.
const past = 1<<32 - 1

// FrameSamples is each element of a centred frame of width samples every
// stride of the request's clip, back samples before it: row t's element i
// the sample t*stride + i - width/2 - back, none where that is before the
// clip's first or t*stride + i - width/2 is not below its samples.
func FrameSamples(width, stride, back int) Attrs {
	var x Index
	k := x.Add(x.Mul(x.In(GatherRow), x.K(stride)), x.In(GatherCol))
	at := x.Sub(k, x.K(width/2))
	inside := x.Where(x.Less(at, x.In(GatherSamples)), x.Sub(at, x.K(back)), x.K(past))
	return x.Attrs(x.Where(x.Less(k, x.K(width/2+back)), x.K(past), inside))
}

// RelRows is row q's table row at key j for a table whose row k is the
// position top-1-k: top-1-q+j, the position q-j.
func RelRows(top int) Attrs {
	var x Index
	return x.Attrs(x.Sub(x.Add(x.K(top-1), x.In(GatherCol)), x.In(GatherRow)))
}

// GatherNone is the source index of a row or element no source holds: a
// gather writes zero there.
const GatherNone = past

// staticIndex is term upto-1 of terms with the operands known holds, false
// where it reads one it does not.
func staticIndex(terms []Term, upto int, ops *[GatherOperands]uint32, known *[GatherOperands]bool, vals []uint32, ok []bool) (uint32, bool) {
	for k := range upto {
		t := terms[k]
		var a [4]uint32
		ok[k] = true
		for j := range t.Fn.Arity() {
			switch x := t.Args[j]; x.Kind {
			case ArgIn:
				if x.Index < 0 || x.Index >= GatherOperands {
					return 0, false
				}
				a[j], ok[k] = ops[x.Index], ok[k] && known[x.Index]
			case ArgTerm:
				if x.Index < 0 || x.Index >= k {
					return 0, false
				}
				a[j], ok[k] = vals[x.Index], ok[k] && ok[x.Index]
			default:
				a[j] = uint32(x.Const)
			}
		}
		switch t.Fn {
		case MapAdd:
			vals[k] = a[0] + a[1]
		case MapSub:
			vals[k] = a[0] - a[1]
		case MapMul:
			vals[k] = a[0] * a[1]
		case MapDiv:
			if a[1] == 0 {
				ok[k] = false
			} else {
				vals[k] = a[0] / a[1]
			}
		case MapMax:
			vals[k] = max(a[0], a[1])
		case MapMin:
			vals[k] = min(a[0], a[1])
		case MapLess:
			vals[k] = 0
			if a[0] < a[1] {
				vals[k] = 1
			}
		case MapWhere:
			vals[k] = a[2]
			if a[0] > 0 {
				vals[k] = a[1]
			}
		}
	}
	return vals[upto-1], ok[upto-1]
}

// GatherReads checks a row gather's reads where its index reads only the
// row and the counts: of out rows from in rows and tables rows of tables,
// its bound writes every row and each row written reads a row of its sources
// or GatherNone. A row past the sources is otherwise a clamped read of the
// last, which no gather means.
func GatherReads(a Attrs, out, in, tables int) error {
	if out < 1 || len(a.Expr) == 0 || a.Span < 0 || a.Span > len(a.Expr) {
		return nil
	}
	var ops [GatherOperands]uint32
	var known [GatherOperands]bool
	ops[GatherOutRows], ops[GatherInRows] = uint32(out), uint32(in)
	known[GatherRow], known[GatherOutRows], known[GatherInRows] = true, true, true
	vals, ok := make([]uint32, len(a.Expr)), make([]bool, len(a.Expr))
	written := out
	if a.Span > 0 {
		b, static := staticIndex(a.Expr, a.Span, &ops, &known, vals, ok)
		if !static {
			return nil
		}
		if int64(b)+1 < int64(out) {
			return fmt.Errorf("writes %d of its %d rows", b+1, out)
		}
	}
	for r := range written {
		ops[GatherRow] = uint32(r)
		s, static := staticIndex(a.Expr, len(a.Expr), &ops, &known, vals, ok)
		if !static {
			return nil
		}
		if s != GatherNone && int64(s) >= int64(in+tables) {
			return fmt.Errorf("row %d reads row %d of %d", r, s, in+tables)
		}
	}
	return nil
}
