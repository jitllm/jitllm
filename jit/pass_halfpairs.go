package jit

import (
	"math"
	"slices"

	"github.com/jitllm/jitllm/jit/ir"
)

// HalfPairs decodes, in a kernel of a kind in Kinds on a device that runs
// packed binary16 arithmetic (Device.HalfPairs), two words a trip stores to
// workgroup memory together when each is two binary16 elements, every
// element one scale times an integer code plus one bias, and the second
// word's codes are the first's bit fields sixteen bits higher in the same
// words. Each field is masked out of its word for an element of the first
// word and the one of the second at once, into the low and high half of
// one word; OR'd into 1024's exponent each half is the binary16 1024+q,
// exact while q is below 1024, and less 1024 -- and less the code's offset
// where the bias is the scale times it -- exactly q. One packed multiply
// and add by the scale and the bias rounds each element once, and the two
// pairs are dealt back into the words' order.
type HalfPairs struct {
	Kinds []PrimKind
}

func (HalfPairs) Name() string { return "halfpairs" }

func (p HalfPairs) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	if !d.HalfPairs {
		return ks, nil
	}
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if !slices.Contains(p.Kinds, k.Kind) {
			return nil, false, nil
		}
		m := k.clone()
		if !m.halfPairs() {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

// codeField is one bit field of a code: word >> (shift + delta) masked to
// mask, then shifted left by left. shift is -1 where the shift is delta
// alone.
type codeField struct {
	word, shift       int
	delta, mask, left uint32
}

// codeElem is an element scale*code + bias (less, where neg), its code the
// sum of its fields; bias is -1 where there is none.
type codeElem struct {
	fields      []codeField
	scale, bias int
	neg         bool
}

func (n *Kernel) halfPairs() bool {
	type word struct {
		store  int
		lo, hi codeElem
	}
	var words []word
	for i, s := range n.Stores {
		if s.Barrier || s.Tile || s.At < 0 || n.Bufs[s.Buf].Kind != BufShared {
			continue
		}
		lo, hi, ok := n.halfWord(s.Val)
		if !ok {
			continue
		}
		el, okl := n.codeElemOf(lo)
		eh, okh := n.codeElemOf(hi)
		if okl && okh && el.scale == eh.scale && el.bias == eh.bias && el.neg == eh.neg {
			words = append(words, word{i, el, eh})
		}
	}
	used := make([]bool, len(words))
	changed := false
	for x, X := range words {
		for y, Y := range words {
			sx, sy := n.Stores[X.store], n.Stores[Y.store]
			if x == y || used[x] || used[y] || sx.At != sy.At || sx.Buf != sy.Buf || X.lo.scale != Y.lo.scale ||
				X.lo.bias != Y.lo.bias || X.lo.neg != Y.lo.neg || !X.lo.above(Y.lo) || !X.hi.above(Y.hi) {
				continue
			}
			ac, bd := n.decodePair(X.lo), n.decodePair(X.hi)
			U := ir.U32
			n.Stores[X.store].Val = n.bin(OpOr, U, n.bin(OpAnd, U, ac, n.u32(0xFFFF)), n.bin(OpShl, U, bd, n.u32(16)))
			n.Stores[Y.store].Val = n.bin(OpOr, U, n.bin(OpShr, U, ac, n.u32(16)), n.bin(OpAnd, U, bd, n.u32(0xFFFF0000)))
			used[x], used[y] = true, true
			changed = true
		}
	}
	return changed
}

// above reports whether f's code is e's fields sixteen bits higher in the
// same words, field for field.
func (e codeElem) above(f codeElem) bool {
	if len(e.fields) != len(f.fields) {
		return false
	}
	for i, a := range e.fields {
		b := f.fields[i]
		if a.word != b.word || a.shift != b.shift || a.mask != b.mask || a.left != b.left || b.delta != a.delta+16 {
			return false
		}
	}
	return true
}

// decodePair is e and the element sixteen bits above it (codeElem.above)
// as a binary16 pair, e in the low half.
func (n *Kernel) decodePair(e codeElem) int {
	U := ir.U32
	var codes int
	for i, f := range e.fields {
		sh := n.u32(int(f.delta))
		if f.shift >= 0 {
			sh = n.bin(OpAdd, U, sh, f.shift)
		}
		v := n.bin(OpAnd, U, n.bin(OpShr, U, f.word, sh), n.u32(int(f.mask|f.mask<<16)))
		if f.left > 0 {
			v = n.bin(OpShl, U, v, n.u32(int(f.left)))
		}
		if i == 0 {
			codes = v
		} else {
			codes = n.bin(OpAdd, U, codes, v)
		}
	}
	both := func(x int) int { return n.bin(OpOr, U, x, n.bin(OpShl, U, x, n.u32(16))) }
	off, bias := uint32(0), -1
	if e.bias >= 0 {
		if k, ok := n.scaledBy(e.bias, e.scale); ok && e.neg && k < 1024 {
			off = k
		} else {
			b := e.bias
			if e.neg {
				b = n.bin(OpSub, ir.F32, n.f32(0), b)
			}
			bias = both(n.un(OpHalf, U, b))
		}
	}
	if bias < 0 {
		bias = n.u32(0)
	}
	q := n.bin(OpOr, U, codes, n.u32(0x64006400))
	q = n.bin(OpSubF16x2, U, q, n.u32(int((0x6400+off)*0x10001)))
	return n.add(Node{Op: OpFmaF16x2, Type: U, A: q, B: both(n.un(OpHalf, U, e.scale)), C: bias})
}

// scaledBy reports whether node b is node s times a whole constant, and it.
func (n *Kernel) scaledBy(b, s int) (uint32, bool) {
	nd := n.Nodes[b]
	if nd.Op != OpMul || nd.Type != ir.F32 {
		return 0, false
	}
	for _, x := range [][2]int{{nd.A, nd.B}, {nd.B, nd.A}} {
		c := n.Nodes[x[1]]
		if x[0] != s || c.Op != OpConst || c.Type != ir.F32 {
			continue
		}
		v := math.Float32frombits(uint32(c.Imm))
		if u := uint32(v); v >= 0 && v < 1024 && float32(u) == v {
			return u, true
		}
	}
	return 0, false
}

// halfWord is the two binary16 elements of a word Half(lo) | Half(hi)<<16.
func (n *Kernel) halfWord(v int) (lo, hi int, ok bool) {
	nd := n.Nodes[v]
	if nd.Op != OpOr || nd.Type != ir.U32 {
		return 0, 0, false
	}
	for _, x := range [][2]int{{nd.A, nd.B}, {nd.B, nd.A}} {
		l, s := n.Nodes[x[0]], n.Nodes[x[1]]
		if l.Op == OpHalf && s.Op == OpShl && n.isConst(s.B, 16) && n.Nodes[s.A].Op == OpHalf {
			return l.A, n.Nodes[s.A].A, true
		}
	}
	return 0, 0, false
}

// codeElemOf is float node f as scale*code + bias, its code a sum of bit
// fields whose most is below 1024.
func (n *Kernel) codeElemOf(f int) (codeElem, bool) {
	e := codeElem{bias: -1}
	nd := n.Nodes[f]
	if nd.Type != ir.F32 {
		return e, false
	}
	product := func(i int) bool {
		p := n.Nodes[i]
		if p.Op != OpMul || p.Type != ir.F32 {
			return false
		}
		for _, x := range [][2]int{{p.A, p.B}, {p.B, p.A}} {
			if fs, ok := n.codeOf(x[0]); ok {
				e.fields, e.scale = fs, x[1]
				return true
			}
		}
		return false
	}
	ok := false
	switch nd.Op {
	case OpMul:
		ok = product(f)
	case OpSub:
		ok, e.bias, e.neg = product(nd.A), nd.B, true
	case OpAdd:
		if ok, e.bias = product(nd.A), nd.B; !ok {
			ok, e.bias = product(nd.B), nd.A
		}
	}
	if !ok {
		return e, false
	}
	most := uint32(0)
	for _, fd := range e.fields {
		most += fd.mask << fd.left
	}
	return e, most < 1024
}

// codeOf is node c, a float conversion of a non-negative integer, as the
// sum of its bit fields.
func (n *Kernel) codeOf(c int) ([]codeField, bool) {
	nd := n.Nodes[c]
	if nd.Op != OpCvtF32 {
		return nil, false
	}
	if b := n.Nodes[nd.A]; b.Op == OpBitcast && b.Type == ir.I32 {
		return n.fieldsOf(b.A, nil)
	}
	return nil, false
}

func (n *Kernel) fieldsOf(c int, fs []codeField) ([]codeField, bool) {
	nd := n.Nodes[c]
	if nd.Type != ir.U32 {
		return nil, false
	}
	if nd.Op == OpAdd || nd.Op == OpOr {
		fs, ok := n.fieldsOf(nd.A, fs)
		if !ok {
			return nil, false
		}
		return n.fieldsOf(nd.B, fs)
	}
	left := uint32(0)
	if nd.Op == OpShl {
		s := n.Nodes[nd.B]
		if s.Op != OpConst || uint32(s.Imm) > 16 {
			return nil, false
		}
		left = uint32(s.Imm)
		nd = n.Nodes[nd.A]
	}
	mk := n.Nodes[nd.B]
	if nd.Op != OpAnd || mk.Op != OpConst || uint32(mk.Imm) >= 1<<16 {
		return nil, false
	}
	f := codeField{word: nd.A, shift: -1, mask: uint32(mk.Imm), left: left}
	if w := n.Nodes[nd.A]; w.Op == OpShr {
		f.word = w.A
		sh := n.Nodes[w.B]
		switch {
		case sh.Op == OpConst:
			f.delta = uint32(sh.Imm)
		case sh.Op == OpAdd && n.Nodes[sh.A].Op == OpConst:
			f.delta, f.shift = uint32(n.Nodes[sh.A].Imm), sh.B
		case sh.Op == OpAdd && n.Nodes[sh.B].Op == OpConst:
			f.delta, f.shift = uint32(n.Nodes[sh.B].Imm), sh.A
		default:
			f.shift = w.B
		}
	}
	return append(fs, f), true
}
