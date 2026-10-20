package compose

// q8Stores is the int8 operand's three stores (Quantize8): the packed codes,
// the window's scale and the sums per 16, each a value and the index it
// lands at in its buffer.
type q8Stores struct {
	codes, codesAt, scale, scaleAt, sum, sumAt int
}

// quantize8 is the int8 operand by its definition, the element a word of
// four: word is the word's index among all the operand's words (nb their
// 32-element blocks), val(i) the float at the word's element i; the
// absolute maximum over each window of `window` elements (a whole number
// of 32), the block's scale d = amax/127 and its inverse 1/d (0 for an
// all-zero window) correctly rounded, each element rounded half away from
// zero to x/d, four codes packed into the word, d at [word/8] and the codes'
// sum over each 16 elements at [nb + word/4].
func (g *Row) quantize8(word, nb, window int, val func(i int) int) q8Stores {
	u, ub := g.UConst, g.UBin
	per := max(window/32, 1)
	const absMask, sign, half = 0x7FFFFFFF, 0x80000000, 0x3F000000 // 0.5f
	var v [4]int
	amax := g.Const(0)
	for i := range v {
		v[i] = val(i)
		abs := g.bits(ub(OpAnd, g.word(v[i]), u(absMask)))
		amax = g.Bin(OpMax, amax, abs)
	}
	wmax := g.Add(Node{Kind: NWindow, Red: RedMax, A: amax, At: per * 8})
	d := g.Add(Node{Kind: NBinary, Op: OpDivExact, A: wmax, B: g.Const(127)})
	inv := g.Select(g.Less(g.Const(0), d), g.Add(Node{Kind: NBinary, Op: OpDivExact, A: g.Const(1), B: d}), g.Const(0))
	sum := g.Add(Node{Kind: NIConst})
	packed := -1
	for i := range v {
		f := g.Bin(OpMul, v[i], inv)
		rnd := g.bits(ub(OpAdd, ub(OpAnd, g.word(f), u(sign)), u(half))) // copysign(0.5, f)
		q := g.Add(Node{Kind: NUnary, Op: OpInt, A: g.Bin(OpAdd, f, rnd)})
		sum = ub(OpAdd, sum, q)
		b := ub(OpAnd, g.Add(Node{Kind: NUnary, Op: OpAsWord, A: q}), u(0xFF))
		if i == 0 {
			packed = b
		} else {
			packed = ub(OpAdd, packed, ub(OpShl, b, u(8*i)))
		}
	}
	s16 := g.Add(Node{Kind: NWindow, Red: RedSum, A: g.Add(Node{Kind: NUnary, Op: OpFloat, A: sum}), At: 4})
	return q8Stores{codes: packed, codesAt: word, scale: d, scaleAt: ub(OpDiv, word, u(8)),
		sum: s16, sumAt: ub(OpAdd, ub(OpDiv, word, u(4)), u(nb))}
}

// Quantize8 is the int8 operand of a row of floats as the group's own
// elements, a word each: the row's element 4e+i at x (BufRow, read by
// index), the codes at a (BufOutU32), the scales and sums at ax (BufOut).
// The group's Width is the row's words; Rows 1.
func (g *Row) Quantize8(x, a, ax, window int) {
	u, ub := g.UConst, g.UBin
	e := g.Column()
	q := g.quantize8(e, g.Width/8, window, func(i int) int { return g.LoadAt(x, ub(OpAdd, ub(OpMul, e, u(4)), u(i))) })
	g.StoreAt(a, q.codes, q.codesAt)
	g.StoreAt(ax, q.scale, q.scaleAt)
	g.StoreAt(ax, q.sum, q.sumAt)
}

// Quantize8Stage is the int8 operand of node y over the group's rows, as a
// stage of the group (a word an element, Width/4 of them a row): y's value
// at element 4e+i of its row (NAt, recomputed there), the codes at a, the
// scales and sums at ax as Quantize8 writes them over every row.
func (g *Row) Quantize8Stage(y, a, ax, window int) {
	u, ub := g.UConst, g.UBin
	words := g.Width / 4
	e := g.Column()
	word := ub(OpAdd, ub(OpMul, g.RowIdx(), u(words)), e)
	q := g.quantize8(word, g.Rows*g.Width/32, window, func(i int) int {
		return g.Add(Node{Kind: NAt, A: y, Ix: ub(OpAdd, ub(OpMul, e, u(4)), u(i))})
	})
	g.Stages = append(g.Stages, Stage{Width: words, Stores: []Store{
		{Buf: a, Val: q.codes, Indexed: true, Ix: q.codesAt},
		{Buf: ax, Val: q.scale, Indexed: true, Ix: q.scaleAt},
		{Buf: ax, Val: q.sum, Indexed: true, Ix: q.sumAt}}})
}
