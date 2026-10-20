package compose

// A mixture's rows grouped by expert: each row's k (token, slot) pairs
// sorted by their expert, each expert's run padded to a whole unit of
// columns, every column named by its expert, and the block counts the
// two-level scatter places each pair by. Each is a flat group, an element a
// thread, its folds over at most GroupBlock pairs, the blocks or the
// experts.

// GroupBlock is the pairs one block of the scatter counts over.
const GroupBlock = 64

// GroupBlocks is the GroupBlock-pair blocks of rows*k pairs.
func GroupBlocks(rows, k int) int { return (rows*k + GroupBlock - 1) / GroupBlock }

// selAt is pair i's expert in a selection laid out k+1 a row.
func (g *Row) selAt(sel, i, k int) int {
	u, ub := g.UConst, g.UBin
	return g.LoadAt(sel, ub(OpAdd, ub(OpMul, ub(OpDiv, i, u(k)), u(k+1)), ub(OpRem, i, u(k))))
}

// eqU is 1 where indices x and y are equal, 0 elsewhere.
func (g *Row) eqU(x, y int) int {
	u, ub := g.UConst, g.UBin
	return ub(OpSub, u(1), ub(OpAdd, g.Flag(g.Less(x, y)), g.Flag(g.Less(y, x))))
}

// unless is 0 where cond holds and 1 where it does not.
func (g *Row) unless(cond int) int { return g.Select(cond, g.UConst(0), g.UConst(1)) }

func (g *Row) padTo(n, unit int) int {
	u, ub := g.UConst, g.UBin
	return ub(OpMul, ub(OpDiv, ub(OpAdd, n, u(unit-1)), u(unit)), u(unit))
}

// GroupBlockCount is block b's pairs routed to expert e, the element b*nExp+e.
func (g *Row) GroupBlockCount(sel, rows, k, nExp int) int {
	u, ub := g.UConst, g.UBin
	t := g.Column()
	blk, e := ub(OpDiv, t, u(nExp)), ub(OpRem, t, u(nExp))
	base, last := ub(OpMul, blk, u(GroupBlock)), u(rows*k-1)
	c := g.Loop(GroupBlock)
	acc := g.Fold(c, u(0))
	i := ub(OpAdd, base, g.Iter(c))
	g.SetFold(acc, ub(OpAdd, acc, ub(OpMul, g.unless(g.Less(last, i)), g.eqU(g.selAt(sel, ub(OpMin, i, last), k), e))))
	return acc
}

// GroupCount stores at off expert e's pairs in the blocks before b (the
// element b*nExp+e), and at cnt its total from the last block's thread (the
// others' into a row past the nExp).
func (g *Row) GroupCount(bc, off, cnt, rows, k, nExp int) {
	u, ub := g.UConst, g.UBin
	nb := GroupBlocks(rows, k)
	t := g.Column()
	blk, e := ub(OpDiv, t, u(nExp)), ub(OpRem, t, u(nExp))
	c := g.Loop(nb)
	f := g.Iter(c)
	acc := g.Fold(c, u(0))
	g.SetFold(acc, ub(OpAdd, acc, ub(OpMul, g.Flag(g.Less(f, blk)), g.LoadAt(bc, ub(OpAdd, ub(OpMul, f, u(nExp)), e)))))
	g.Store(off, acc)
	isLast := g.unless(g.Less(blk, u(nb-1)))
	g.StoreAt(cnt, ub(OpAdd, acc, g.LoadAt(bc, t)), ub(OpAdd, e, ub(OpMul, ub(OpSub, u(1), isLast), u(nExp))))
}

// GroupEnds is the end of expert e's padded run: the sum over f <= e of
// cnt[f] padded to unit.
func (g *Row) GroupEnds(cnt, nExp, unit int) int {
	u, ub := g.UConst, g.UBin
	e := g.Column()
	c := g.Loop(nExp)
	f := g.Iter(c)
	acc := g.Fold(c, u(0))
	g.SetFold(acc, ub(OpAdd, acc, ub(OpMul, g.unless(g.Less(e, f)), g.padTo(g.LoadAt(cnt, f), unit))))
	return acc
}

// expertOf is the expert whose run holds column j: the runs ending at or
// before it, expert 0 at or past the used columns.
func (g *Row) expertOf(ends, j, nExp int) int {
	u, ub := g.UConst, g.UBin
	c := g.Loop(nExp)
	f := g.Iter(c)
	acc := g.Fold(c, u(0))
	g.SetFold(acc, ub(OpAdd, acc, g.unless(g.Less(j, g.LoadAt(ends, f)))))
	return g.Select(g.Less(acc, u(nExp)), acc, u(0))
}

// GroupColumns stores each column's expert at colE, and the defaults the
// scatter overwrites: a zero row (rows) at perm, and 0 at pair (-1 none).
func (g *Row) GroupColumns(ends, colE, perm, pair, rows, nExp int) {
	j := g.Column()
	g.Store(colE, g.expertOf(ends, j, nExp))
	g.Store(perm, g.UConst(rows))
	if pair >= 0 {
		g.Store(pair, g.UConst(0))
	}
}

// GroupScatter places each pair (an element) at its expert's run start
// plus the expert's pairs in earlier blocks plus those before it in its
// own: its row at perm, its column at pos, and itself at pair (-1 none).
func (g *Row) GroupScatter(sel, cnt, ends, off, perm, pos, pair, rows, k, nExp, unit int) {
	u, ub := g.UConst, g.UBin
	i := g.Column()
	e := g.selAt(sel, i, k)
	blk := ub(OpDiv, i, u(GroupBlock))
	base := ub(OpMul, blk, u(GroupBlock))
	c := g.Loop(GroupBlock)
	acc := g.Fold(c, u(0))
	jj := ub(OpAdd, base, g.Iter(c))
	g.SetFold(acc, ub(OpAdd, acc, ub(OpMul, g.Flag(g.Less(jj, i)), g.eqU(g.selAt(sel, ub(OpMin, jj, i), k), e))))
	start := ub(OpSub, g.LoadAt(ends, e), g.padTo(g.LoadAt(cnt, e), unit))
	o := g.LoadAt(off, ub(OpAdd, ub(OpMul, blk, u(nExp)), e))
	col := ub(OpAdd, ub(OpAdd, start, o), acc)
	g.StoreAt(perm, ub(OpDiv, i, u(k)), col)
	g.Store(pos, col)
	if pair >= 0 {
		g.StoreAt(pair, i, col)
	}
}

// GroupTiles is the expert of the element's tile of tile columns.
func (g *Row) GroupTiles(ends, nExp, tile int) int {
	return g.expertOf(ends, g.UBin(OpMul, g.Column(), g.UConst(tile)), nExp)
}

// GatherRows is element i of sorted row j (rows of n, an element a thread):
// source row perm[j], or zero where perm[j] is past the nsrc source rows.
func (g *Row) GatherRows(src, perm, n, nsrc int) int {
	u, ub := g.UConst, g.UBin
	flat := g.Column()
	j, i := ub(OpDiv, flat, u(n)), ub(OpRem, flat, u(n))
	s, last := g.LoadAt(perm, j), u(nsrc-1)
	v := g.LoadAt(src, ub(OpAdd, ub(OpMul, ub(OpMin, s, last), u(n)), i))
	return g.Select(g.Less(last, s), g.Const(0), v)
}

// CombineGrouped is element i of row r of the rows' sorted expert outputs
// e (n a column) summed by their weights w in selection order, pair r*k+s
// at its column pos[r*k+s].
func (g *Row) CombineGrouped(e, w, pos, n, k int) int {
	u, ub := g.UConst, g.UBin
	flat := g.Column()
	r, i := ub(OpDiv, flat, u(n)), ub(OpRem, flat, u(n))
	j0 := ub(OpMul, r, u(k))
	c := g.Loop(k)
	j := ub(OpAdd, j0, g.Iter(c))
	acc := g.Fold(c, g.Const(0))
	col := g.LoadAt(pos, j)
	g.SetFold(acc, g.fma(g.LoadAt(w, j), g.LoadAt(e, ub(OpAdd, ub(OpMul, col, u(n)), i)), acc))
	return acc
}

// IndexedBias is an element of k rows of n plus the bias row its selection
// names: in[j*n+r] + bias[sel[j]*n+r].
func (g *Row) IndexedBias(in, sel, bias, n int) int {
	u, ub := g.UConst, g.UBin
	i := g.Column()
	j, r := ub(OpDiv, i, u(n)), ub(OpRem, i, u(n))
	e := g.LoadAt(sel, j)
	return g.Bin(OpAdd, g.LoadAt(in, i), g.LoadAt(bias, ub(OpAdd, ub(OpMul, e, u(n)), r)))
}
