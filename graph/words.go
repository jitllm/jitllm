package graph

import (
	"fmt"
	"math/bits"
	"slices"

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
func (s *Builder) ModAdd(name string, w Wide, n, off uint64) (ValueID, error) {
	if n < 1 || n > 1<<31 || off > 1<<32-n {
		return -1, fmt.Errorf("graph: a remainder modulo %d plus %d is not a word", n, off)
	}
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
	return s.Words(name, s.width(w.Lo), &e, w.Lo, w.Hi), nil
}

// NGramRows is each row's table row a head of h over its window win (n
// words: the row's token, then its predecessors nearest first, each its id
// plus one, 0 for none), in words: the window's ids, a predecessor that is
// none or past an end of sequence read as the end of sequence; each g-gram's
// product, the (g-1)-gram's xored with its last id times that id's
// multiplier, modulo 2^64 (the 2-gram's the first two ids'); and each head's
// row, its gram's product modulo the head's vocabulary plus its offset.
func (s *Builder) NGramRows(win ValueID, n int, h *jlm.NGramHash) (ValueID, error) {
	if heads := (n - 1) * int(h.HeadsPer); n < 2 || len(h.Mul) < n || len(h.Voc) < heads || len(h.Off) < heads {
		return -1, fmt.Errorf("graph: %d-grams over %d heads, %d multipliers, %d ranges", n, heads, len(h.Mul), len(h.Voc))
	}
	at := func(i int) ValueID {
		return s.Emit(OpSlice, []ValueID{win}, []Result{{Name: fmt.Sprintf("ng.w%d", i), Width: 1, Elem: ElemU32}}, nil,
			Attrs{Offset: i, Width: 1}, nil, nil)[0]
	}
	eos := Word(h.EOS)
	ids := []ValueID{at(0)}
	for i := 1; i < n; i++ {
		var e WordTerms
		none := e.Fn(MapEqual, WordIn(0), Word(0))
		in := []ValueID{at(i)}
		if i > 1 {
			none = e.Fn(MapAdd, none, e.Fn(MapEqual, WordIn(1), eos))
			in = append(in, ids[i-1])
		}
		e.Fn(MapWhere, none, eos, e.Fn(MapSub, WordIn(0), Word(1)))
		ids = append(ids, s.Words(fmt.Sprintf("ng.id%d", i), 1, &e, in...))
	}
	gram := s.MulXor("ng.g1", nil, ids[0], h.Mul[0])
	var rows []ValueID
	for g := 2; g <= n; g++ {
		gram = s.MulXor(fmt.Sprintf("ng.g%d", g), &gram, ids[g-1], h.Mul[g-1])
		for k := range int(h.HeadsPer) {
			i := (g-2)*int(h.HeadsPer) + k
			r, err := s.ModAdd(fmt.Sprintf("ng.h%d", i), gram, h.Voc[i], h.Off[i])
			if err != nil {
				return -1, err
			}
			rows = append(rows, r)
		}
	}
	return s.Emit(OpConcat, rows, []Result{{Name: "ng.rows", Width: len(rows), Elem: ElemU32}}, nil, Attrs{}, nil, nil)[0], nil
}

// TableRow is a row table (jlm.RowTable) a graph gathers from: its role,
// and the row of words its gathers read their rows from -- the row each
// gather's index is sliced from, or the index itself.
type TableRow struct {
	Role jlm.Role
	Rows ValueID
}

// TableRows is each row table of tables g gathers from, in the order g
// first gathers it, any number of them; a table whose gathers read rows
// from two rows is left out, so no view is opened for it and a form binding
// it is declined.
func (g *Graph) TableRows(tables []jlm.Role) []TableRow {
	from := map[ValueID]ValueID{}
	for _, o := range g.Ops {
		if o.Kind == OpSlice && len(o.In) == 1 && len(o.Out) == 1 {
			from[o.Out[0]] = o.In[0]
		}
	}
	var ts []TableRow
	two := map[jlm.Role]bool{}
	for _, o := range g.Ops {
		if o.Kind != OpEmbed || len(o.In) != 1 || len(o.Weight) != 1 || !slices.Contains(tables, o.Weight[0].Role) {
			continue
		}
		v, ok := from[o.In[0]]
		if !ok {
			v = o.In[0]
		}
		r := o.Weight[0].Role
		i := slices.IndexFunc(ts, func(t TableRow) bool { return t.Role == r })
		switch {
		case i < 0:
			ts = append(ts, TableRow{Role: r, Rows: v})
		case ts[i].Rows != v:
			two[r] = true
		}
	}
	return slices.DeleteFunc(ts, func(t TableRow) bool { return two[t.Role] })
}
