package compose

// Quantize8 is the int8 operand of a row of floats, by its definition, as
// the group's element a word of four: the row's element 4e+i at x (a BufRow
// buffer read by index), the absolute maximum over each window of `window`
// elements (a whole number of 32), the block's scale d = amax/127 and its
// inverse 1/d (0 for an all-zero window) correctly rounded, each element
// rounded half away from zero to x/d, four codes packed into the word at a
// (BufOutU32, the element's place), d at ax[e/8] (BufOut) and the codes' sum
// over each 16 elements at ax[nb + e/4], nb the row's 32-element blocks.
// The group's Width is the row's words; Rows 1.
func (g *Row) Quantize8(x, a, ax, window int) {
	u, ub := g.UConst, g.UBin
	e := g.Column()
	nb := g.Width / 8
	per := max(window/32, 1)
	const absMask, sign, half = 0x7FFFFFFF, 0x80000000, 0x3F000000 // 0.5f
	var v [4]int
	amax := g.Const(0)
	for i := range v {
		v[i] = g.LoadAt(x, ub(OpAdd, ub(OpMul, e, u(4)), u(i)))
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
	g.Store(a, packed)
	s16 := g.Add(Node{Kind: NWindow, Red: RedSum, A: g.Add(Node{Kind: NUnary, Op: OpFloat, A: sum}), At: 4})
	g.StoreAt(ax, d, ub(OpDiv, e, u(8)))
	g.StoreAt(ax, s16, ub(OpAdd, ub(OpDiv, e, u(4)), u(nb)))
}
