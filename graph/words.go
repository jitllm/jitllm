package graph

import (
	"math/bits"

	"github.com/jitllm/jitllm/format/jlm"
)

// WordTerms writes an expression into words (OpExpr into ElemU32) a term at
// a time: each Fn appends a term and is its operand.
type WordTerms struct{ T []Term }

// WordIn is the expression's operand i.
func WordIn(i int) Arg { return Arg{Kind: ArgIn, Index: i} }

// Word is the constant word v.
func Word(v uint32) Arg { return Arg{Kind: ArgConst, Const: float64(v)} }

// Fn is f over a, as a new term.
func (e *WordTerms) Fn(f MapFn, a ...Arg) Arg {
	var t Term
	t.Fn = f
	copy(t.Args[:], a)
	e.T = append(e.T, t)
	return Arg{Kind: ArgTerm, Index: len(e.T) - 1}
}

// Words is the expression e into a row of width words over in, each a row of
// words.
func (s *Builder) Words(name string, width int, e *WordTerms, in ...ValueID) ValueID {
	return s.Emit(OpExpr, in, []Result{{Name: name, Width: width, Elem: ElemU32}}, nil, Attrs{Expr: e.T}, nil, nil)[0]
}

// Wide is a 64-bit number a row, as two rows of words: its low word and its
// high one.
type Wide struct{ Lo, Hi ValueID }

const limb = 0xFFFF

// MulXor is (acc ^ x*m) modulo 2^64 for the row of words x, acc zero where
// it is nil. Every product is of 16-bit limbs, exact in a word, and its
// carries walked up a column at a time, so no word wraps.
func (s *Builder) MulXor(name string, acc *Wide, x ValueID, m uint64) Wide {
	half := func(e *WordTerms, hi bool) Arg {
		if hi {
			return e.Fn(MapShr, WordIn(0), Word(16))
		}
		return e.Fn(MapAnd, WordIn(0), Word(limb))
	}
	word := func(top bool) ValueID {
		var e WordTerms
		h := [2]Arg{half(&e, false), half(&e, true)}
		terms := map[[2]int]Arg{}
		term := func(i, k int) Arg {
			if t, ok := terms[[2]int{i, k}]; ok {
				return t
			}
			t := e.Fn(MapMul, h[i], Word(uint32(m>>(16*k)&limb)))
			terms[[2]int{i, k}] = t
			return t
		}
		cols := 2
		if top {
			cols = 4
		}
		var limbs [4]Arg
		carry := Word(0)
		for q := range cols {
			sum := carry
			for i := range 2 {
				if k := q - i; k >= 0 && m>>(16*k)&limb != 0 {
					sum = e.Fn(MapAdd, sum, e.Fn(MapAnd, term(i, k), Word(limb)))
				}
				if k := q - 1 - i; k >= 0 && m>>(16*k)&limb != 0 {
					sum = e.Fn(MapAdd, sum, e.Fn(MapShr, term(i, k), Word(16)))
				}
			}
			limbs[q], carry = e.Fn(MapAnd, sum, Word(limb)), e.Fn(MapShr, sum, Word(16))
		}
		lo := 0
		if top {
			lo = 2
		}
		v := e.Fn(MapOr, limbs[lo], e.Fn(MapShl, limbs[lo+1], Word(16)))
		in := []ValueID{x}
		if acc != nil {
			prev := acc.Lo
			if top {
				prev = acc.Hi
			}
			v = e.Fn(MapXor, v, WordIn(1))
			in = append(in, prev)
		}
		suffix := ".lo"
		if top {
			suffix = ".hi"
		}
		return s.Words(name+suffix, s.width(x), &e, in...)
	}
	return Wide{Lo: word(false), Hi: word(true)}
}

// ModAdd is (w modulo n) + off for the 64-bit rows w, n in [1, 2^31] and
// off+n at most 2^32: the remainder taken from the top a few bits at a time,
// a remainder below n shifted by those bits staying in a word.
func (s *Builder) ModAdd(name string, w Wide, n, off uint64) ValueID {
	var e WordTerms
	limbs := [4]Arg{
		e.Fn(MapAnd, WordIn(0), Word(limb)), e.Fn(MapShr, WordIn(0), Word(16)),
		e.Fn(MapAnd, WordIn(1), Word(limb)), e.Fn(MapShr, WordIn(1), Word(16)),
	}
	bitsOf := func(lo, ln int) Arg {
		q, sh := lo/16, lo%16
		v := limbs[q]
		if sh > 0 {
			v = e.Fn(MapShr, v, Word(uint32(sh)))
		}
		if sh+ln > 16 {
			v = e.Fn(MapAdd, v, e.Fn(MapShl, limbs[q+1], Word(uint32(16-sh))))
		}
		return e.Fn(MapAnd, v, Word(uint32(1)<<ln-1))
	}
	step := min(16, 32-bits.Len64(n-1))
	rem := Word(0)
	for top := 64; top > 0; {
		ln := min(step, top)
		top -= ln
		rem = e.Fn(MapRem, e.Fn(MapAdd, e.Fn(MapShl, rem, Word(uint32(ln))), bitsOf(top, ln)), Word(uint32(n)))
	}
	e.Fn(MapAdd, rem, Word(uint32(off)))
	return s.Words(name, s.width(w.Lo), &e, w.Lo, w.Hi)
}

// TableRows is the row of words g's gathers of a row table (jlm.RowTable)
// read their rows from -- the row each gather's index is sliced from, or
// the index itself -- and -1 where g gathers no row table's rows or its
// gathers read rows from two rows.
func (g *Graph) TableRows() ValueID {
	from := map[ValueID]ValueID{}
	for _, o := range g.Ops {
		if o.Kind == OpSlice && len(o.In) == 1 && len(o.Out) == 1 {
			from[o.Out[0]] = o.In[0]
		}
	}
	rv := ValueID(-1)
	for _, o := range g.Ops {
		if o.Kind != OpEmbed || len(o.In) != 1 || len(o.Weight) != 1 || !jlm.RowTableRole(o.Weight[0].Role) {
			continue
		}
		v, ok := from[o.In[0]]
		if !ok {
			v = o.In[0]
		}
		if rv >= 0 && v != rv {
			return -1
		}
		rv = v
	}
	return rv
}
