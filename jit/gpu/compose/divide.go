package compose

// Correctly rounded float division as nodes: a device's division is allowed
// an error (SPIR-V's OpFDiv 2.5 ulp), so a quotient that must equal the
// host's to the bit -- an int8 operand's scale -- is computed from the
// operands' bits in integer arithmetic.

func (g *Row) word(x int) int { return g.Add(Node{Kind: NUnary, Op: OpWord, A: x}) }
func (g *Row) bits(x int) int { return g.Add(Node{Kind: NUnary, Op: OpBits, A: x}) }

// Div127 is x/127 correctly rounded for x a normal float (the division
// otherwise): x's significand shifted up 8, divided by 127, rounded to 24
// bits by the bits below them, the exponent adjusted by the shift and a
// rounding carry.
func (g *Row) Div127(x int) int {
	u, ub := g.UConst, g.UBin
	nx := g.word(x)
	ex := ub(OpShr, nx, u(23))
	ma := ub(OpAdd, ub(OpAnd, nx, u(0x7FFFFF)), u(0x800000))
	q := ub(OpDiv, ub(OpShl, ma, u(8)), u(127)) // [2^24, 2^26)
	sh := ub(OpAdd, ub(OpShr, q, u(25)), u(1))  // bits below the 24 kept
	half := ub(OpShl, u(1), ub(OpSub, sh, u(1)))
	up := g.Select(g.Less(ub(OpAnd, q, ub(OpSub, ub(OpShl, half, u(1)), u(1))), half), u(0), u(1))
	m := ub(OpAdd, ub(OpShr, q, sh), up)
	carry := ub(OpShr, m, u(24)) // rounded up to 2^24
	// The biased exponent: ma*2^(ex-150)/127 = (q/2^sh) * 2^(ex-158+sh).
	e := ub(OpAdd, ub(OpSub, ub(OpAdd, ex, sh), u(8)), carry)
	bs := ub(OpAdd, ub(OpShl, e, u(23)), ub(OpAnd, ub(OpShr, m, carry), u(0x7FFFFF)))
	return g.Select(g.Less(ub(OpSub, ex, u(8)), u(247)), g.bits(bs), g.Bin(OpDiv, x, g.Const(127)))
}

// RoundQuotient is num/den correctly rounded from an estimate est within
// reach-1/2 floats of it: of the floats est-reach .. est+reach (positive
// floats' bit patterns order as the floats do), the lowest plus how many
// midpoints lie below the quotient, compared exactly in integers; est
// itself where an operand or the estimate is outside the normal range.
func (g *Row) RoundQuotient(num, den, est int, reach int) int {
	u, ub := g.UConst, g.UBin
	mant := func(bits int) int { return ub(OpAdd, ub(OpAnd, bits, u(0x7FFFFF)), u(0x800000)) }
	na, nd := g.word(num), g.word(den)
	ea, ed := ub(OpShr, na, u(23)), ub(OpShr, nd, u(23))
	ma, md := mant(na), mant(nd)
	ny := g.word(est)
	base := ub(OpSub, ny, u(reach))
	pick := base
	for j := 0; j < 2*reach; j++ {
		c := ub(OpAdd, base, u(j))
		ec := ub(OpShr, c, u(23))
		// s = (ea-150) - (ec-151) - (ed-150): ma*2^s against (2Mc+1)*md.
		s := ub(OpSub, ub(OpAdd, ea, u(151)), ub(OpAdd, ec, ed))
		mid := ub(OpAdd, ub(OpShl, mant(c), u(1)), u(1))
		d := ub(OpSub, ub(OpShl, ma, s), ub(OpMul, mid, md))
		// D > 0 as a signed value; the quotient of two floats is never
		// exactly halfway between two floats.
		above := g.Less(ub(OpSub, d, u(1)), u(0x7FFFFFFF))
		pick = ub(OpAdd, pick, g.Select(above, u(1), u(0)))
	}
	// Normal operands (biased exponent 1..254) and an estimate a binade
	// inside that range, so every candidate is normal and the shift defined.
	in := func(e int, lo, n int) int { return g.Less(ub(OpSub, e, u(lo)), u(n)) }
	ok := func(p, then int) int { return g.Select(p, then, est) }
	return ok(in(ea, 1, 254), ok(in(ed, 1, 254), ok(in(ub(OpShr, ny, u(23)), 2, 252), g.bits(pick))))
}

// Q8Scales is an int8 block's scale d = amax/127 and its inverse 127/amax,
// both correctly rounded (the host's), the inverse 0 for an all-zero block:
// the inverse's estimate refined by one Newton step, then rounded.
func (g *Row) Q8Scales(amax int) (d, inv int) {
	d = g.Div127(amax)
	y := g.Bin(OpDiv, g.Const(127), amax)
	y = g.Bin(OpAdd, y, g.Bin(OpMul, y, g.Bin(OpSub, g.Const(1), g.Bin(OpMul, d, y))))
	inv = g.Select(g.Less(g.Const(0), d), g.RoundQuotient(g.Const(1), d, y, 2), g.Const(0))
	return d, inv
}
