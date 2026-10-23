package jit

import (
	"fmt"
	"sort"
	"strings"

	"github.com/jitllm/jitllm/jit/ir"
)

// alg writes a kernel's integer nodes as sums of monomials times constants:
// a monomial a product of atoms, an atom a node or a bit field of one, the
// empty monomial the constant. Bit fields are kept so that a length split
// into trips of 32, 8 and 1 sums back to the length.
type alg struct {
	k     *Kernel
	atoms []atom
	ids   map[atom]int
}

// atom is node Node, or where Field is set bits [Lo, Lo+Bits) of atom Of
// shifted down by Shift... kept as the field and(shr(of, shift), mask).
type atom struct {
	node        int
	of          int
	shift, mask int64
	field       bool
}

func newAlg(k *Kernel) *alg { return &alg{k: k, ids: map[atom]int{}} }

func (g *alg) id(a atom) int {
	if i, ok := g.ids[a]; ok {
		return i
	}
	g.atoms = append(g.atoms, a)
	g.ids[a] = len(g.atoms) - 1
	return len(g.atoms) - 1
}

// lin maps a monomial, its atoms' ids sorted and joined, to its
// coefficient.
type lin map[string]int64

func mono(ids ...int) string {
	sort.Ints(ids)
	var b strings.Builder
	for j, i := range ids {
		if j > 0 {
			b.WriteByte('*')
		}
		fmt.Fprint(&b, i)
	}
	return b.String()
}

func monoIDs(m string) []int {
	if m == "" {
		return nil
	}
	var out []int
	for _, s := range strings.Split(m, "*") {
		var i int
		fmt.Sscan(s, &i)
		out = append(out, i)
	}
	return out
}

func (l lin) add(o lin, f int64) {
	for m, v := range o {
		l[m] += f * v
		if l[m] == 0 {
			delete(l, m)
		}
	}
}

func (l lin) constant() (int64, bool) {
	switch len(l) {
	case 0:
		return 0, true
	case 1:
		v, ok := l[""]
		return v, ok
	}
	return 0, false
}

// single is l's one atom where l is that atom alone.
func (l lin) single() (int, bool) {
	if len(l) != 1 {
		return 0, false
	}
	for m, v := range l {
		ids := monoIDs(m)
		return ids[0], v == 1 && len(ids) == 1
	}
	return 0, false
}

func (l lin) equal(o lin) bool {
	if len(l) != len(o) {
		return false
	}
	for m, v := range l {
		if o[m] != v {
			return false
		}
	}
	return true
}

func (g *alg) of(i int) lin {
	nd := g.k.Nodes[i]
	leaf := lin{mono(g.id(atom{node: i})): 1}
	if nd.Type == ir.F32 {
		return leaf
	}
	switch nd.Op {
	case OpConst:
		if nd.Imm == 0 {
			return lin{}
		}
		return lin{"": nd.Imm}
	case OpAdd, OpSub:
		out := lin{}
		out.add(g.of(nd.A), 1)
		f := int64(1)
		if nd.Op == OpSub {
			f = -1
		}
		out.add(g.of(nd.B), f)
		return g.merged(out)
	case OpMul:
		return g.merged(g.times(g.of(nd.A), g.of(nd.B)))
	case OpShl:
		if c, ok := g.of(nd.B).constant(); ok {
			return g.merged(g.times(g.of(nd.A), lin{"": 1 << c}))
		}
	case OpShr, OpAnd:
		c, ok := g.of(nd.B).constant()
		x, single := g.of(nd.A).single()
		if !ok || !single {
			return leaf
		}
		a := g.atoms[x]
		if nd.Op == OpShr {
			if a.field {
				return lin{mono(g.id(atom{of: a.of, shift: a.shift + c, mask: a.mask >> c, field: true})): 1}
			}
			return lin{mono(g.id(atom{of: x, shift: c, mask: -1, field: true})): 1}
		}
		if a.field {
			m := c
			if a.mask >= 0 {
				m &= a.mask
			}
			return lin{mono(g.id(atom{of: a.of, shift: a.shift, mask: m, field: true})): 1}
		}
		return lin{mono(g.id(atom{of: x, mask: c, field: true})): 1}
	}
	return leaf
}

func (g *alg) times(a, b lin) lin {
	out := lin{}
	for m1, v1 := range a {
		for m2, v2 := range b {
			out.add(lin{mono(append(monoIDs(m1), monoIDs(m2)...)...): 1}, v1*v2)
		}
	}
	return out
}

// merged writes each field times the power of two it was shifted by as a
// field of the unshifted value, a field of every bit from some on as the
// value less its low bits, and fields of one value and coefficient whose
// masks do not overlap as one.
func (g *alg) merged(l lin) lin {
	for changed := true; changed; {
		changed = false
		for m, v := range l {
			ids := monoIDs(m)
			if len(ids) != 1 {
				continue
			}
			a := g.atoms[ids[0]]
			if !a.field {
				continue
			}
			switch {
			case a.shift > 0 && v%(1<<a.shift) == 0:
				delete(l, m)
				mask := a.mask
				if mask >= 0 {
					mask <<= a.shift
				} else {
					mask = -1 << a.shift
				}
				l.add(lin{mono(g.id(atom{of: a.of, mask: mask, field: true})): 1}, v>>a.shift)
				changed = true
			case a.shift == 0 && a.mask < -1:
				delete(l, m)
				l.add(lin{mono(a.of): 1, mono(g.id(atom{of: a.of, mask: ^a.mask, field: true})): -1}, v)
				changed = true
			}
			if changed {
				break
			}
		}
		if changed {
			continue
		}
	}
	return g.refined(l)
}

// refined writes the fields of each value in l as sums of the disjoint bit
// ranges their masks cut it into, so that fields that sum to the same bits
// are the same lin.
func (g *alg) refined(l lin) lin {
	byOf := map[int][]string{}
	for m := range l {
		if ids := monoIDs(m); len(ids) == 1 && g.atoms[ids[0]].field && g.atoms[ids[0]].shift == 0 && g.atoms[ids[0]].mask >= 0 {
			of := g.atoms[ids[0]].of
			byOf[of] = append(byOf[of], m)
		}
	}
	for of, ms := range byOf {
		if len(ms) < 2 {
			continue
		}
		sort.Strings(ms)
		masks := make([]int64, len(ms))
		for i, m := range ms {
			masks[i] = g.atoms[monoIDs(m)[0]].mask
		}
		pieces := map[int64]int64{}
		for bit := 0; bit < 63; bit++ {
			var sig int64
			for i, mk := range masks {
				if mk>>bit&1 == 1 {
					sig |= 1 << i
				}
			}
			if sig != 0 {
				pieces[sig] |= 1 << bit
			}
		}
		coef := map[int64]int64{}
		for i, m := range ms {
			v := l[m]
			delete(l, m)
			for sig, piece := range pieces {
				if sig>>i&1 == 1 {
					coef[piece] += v
				}
			}
		}
		for piece, v := range coef {
			if v != 0 {
				l.add(lin{mono(g.id(atom{of: of, mask: piece, field: true})): 1}, v)
			}
		}
	}
	return l
}

// coef splits l by atom a: the lin multiplying a, and the rest.
func (g *alg) coef(l lin, a int) (lin, lin) {
	with, rest := lin{}, lin{}
	for m, v := range l {
		ids := monoIDs(m)
		at := -1
		for j, x := range ids {
			if x == a {
				at = j
			}
		}
		if at < 0 {
			rest.add(lin{m: 1}, v)
			continue
		}
		with.add(lin{mono(append(ids[:at:at], ids[at+1:]...)...): 1}, v)
	}
	return with, rest
}

// nodes is every node an atom names.
func (g *alg) nodes(a int) []int {
	at := g.atoms[a]
	if at.field {
		return g.nodes(at.of)
	}
	return []int{at.node}
}

// moves reports whether loop c moves any atom of l.
func (g *alg) moves(l lin, c int) bool {
	for m := range l {
		for _, a := range monoIDs(m) {
			for _, n := range g.nodes(a) {
				if g.k.Varies(n, c) {
					return true
				}
			}
		}
	}
	return false
}

// build is l as nodes of the kernel.
func (g *alg) build(l lin) int {
	k := g.k
	var ms []string
	for m := range l {
		ms = append(ms, m)
	}
	sort.Strings(ms)
	acc := -1
	var c int64
	for _, m := range ms {
		if m == "" {
			c = l[m]
			continue
		}
		v := -1
		for _, a := range monoIDs(m) {
			x := g.atomNode(a)
			if v < 0 {
				v = x
			} else {
				v = k.bin(OpMul, ir.U32, v, x)
			}
		}
		if f := l[m]; f != 1 {
			v = k.bin(OpMul, ir.U32, v, k.u32(int(f)))
		}
		if acc < 0 {
			acc = v
		} else {
			acc = k.bin(OpAdd, ir.U32, acc, v)
		}
	}
	switch {
	case acc < 0:
		return k.u32(int(c))
	case c != 0:
		return k.bin(OpAdd, ir.U32, acc, k.u32(int(c)))
	}
	return acc
}

func (g *alg) atomNode(a int) int {
	at := g.atoms[a]
	if !at.field {
		return at.node
	}
	k := g.k
	x := g.atomNode(at.of)
	if at.shift > 0 {
		x = k.bin(OpShr, ir.U32, x, k.u32(int(at.shift)))
	}
	if at.mask >= 0 {
		x = k.bin(OpAnd, ir.U32, x, k.u32(int(at.mask)))
	}
	return x
}

// still reports whether no loop of the kernel moves l.
func (g *alg) still(l lin) bool {
	for c := range g.k.Loops {
		if g.moves(l, c) {
			return false
		}
	}
	return true
}
