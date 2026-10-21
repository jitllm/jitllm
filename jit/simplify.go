package jit

import "github.com/jitllm/jitllm/jit/ir"

// Simplify folds a nest's unsigned index arithmetic: constants, the
// identities of one and zero, and a quotient or remainder by c of a sum
// whose part below c is bounded below c.
type Simplify struct{}

func (Simplify) Name() string { return "simplify" }

func (s Simplify) Apply(prog []*Nest, t Target) ([]*Nest, error) {
	return eachNest(s.Name(), prog, func(n *Nest) ([]*Nest, bool, error) {
		m := n.clone()
		if !m.simplify() {
			return nil, false, nil
		}
		m.Prim = n.Prim
		return []*Nest{m}, true, nil
	})
}

// bound is the largest a u32 node can be, false where unknown.
func (n *Nest) bound(i int, depth int) (uint64, bool) {
	if depth == 0 {
		return 0, false
	}
	nd := n.Nodes[i]
	if nd.Type != ir.U32 {
		return 0, false
	}
	c := func(j int) (uint64, bool) {
		x := n.Nodes[j]
		return uint64(uint32(x.Imm)), x.Op == OpConst
	}
	switch nd.Op {
	case OpConst:
		return uint64(uint32(nd.Imm)), true
	case OpCoord:
		if nd.A < len(n.Space) {
			return uint64(n.Space[nd.A] - 1), true
		}
	case OpThread:
		return uint64(n.Group - 1), true
	case OpIter:
		l := n.Loops[nd.At]
		if l.Trip > 0 && !l.Spread {
			return uint64(l.Trip - 1), true
		}
		if l.Trip > 0 {
			return uint64((l.Trip+n.Group-1)/n.Group*n.Group - 1), true
		}
	case OpMin:
		x, okx := n.bound(nd.A, depth-1)
		y, oky := n.bound(nd.B, depth-1)
		switch {
		case okx && oky:
			return min(x, y), true
		case okx:
			return x, true
		case oky:
			return y, true
		}
	case OpRem:
		if d, ok := c(nd.B); ok && d > 0 {
			hi := d - 1
			if x, ok := n.bound(nd.A, depth-1); ok {
				hi = min(hi, x)
			}
			return hi, true
		}
	case OpDiv, OpShr:
		x, ok := n.bound(nd.A, depth-1)
		d, okd := c(nd.B)
		if ok && okd {
			if nd.Op == OpShr {
				return x >> (d & 31), true
			}
			if d > 0 {
				return x / d, true
			}
		}
	case OpAnd:
		if m, ok := c(nd.B); ok {
			return m, true
		}
	case OpAdd, OpMul:
		x, okx := n.bound(nd.A, depth-1)
		y, oky := n.bound(nd.B, depth-1)
		if okx && oky {
			r := x + y
			if nd.Op == OpMul {
				r = x * y
			}
			if r <= 0xffffffff {
				return r, true
			}
		}
	}
	return 0, false
}

// split is node i as its multiple of c and the rest: i = hi*c + lo, where
// it reads so through sums and products.
func (n *Nest) split(i int, c uint64) (hi, lo int, ok bool) {
	nd := n.Nodes[i]
	if nd.Type != ir.U32 {
		return 0, 0, false
	}
	isConst := func(j int) (uint64, bool) {
		x := n.Nodes[j]
		return uint64(uint32(x.Imm)), x.Op == OpConst
	}
	switch nd.Op {
	case OpMul:
		for _, ab := range [2][2]int{{nd.A, nd.B}, {nd.B, nd.A}} {
			if k, ok := isConst(ab[1]); ok && k%c == 0 {
				return n.bin(OpMul, ir.U32, ab[0], n.u32(int(k/c))), n.u32(0), true
			}
		}
	case OpAdd:
		for _, ab := range [2][2]int{{nd.A, nd.B}, {nd.B, nd.A}} {
			if h, l, ok := n.split(ab[0], c); ok && n.Nodes[l].Op == OpConst && uint32(n.Nodes[l].Imm) == 0 {
				return h, ab[1], true
			}
		}
	}
	return 0, 0, false
}

func (n *Nest) simplify() bool {
	changed := false
	for i := range n.Nodes {
		nd := n.Nodes[i]
		if nd.Type != ir.U32 || nd.Op != OpDiv && nd.Op != OpRem {
			continue
		}
		cn := n.Nodes[nd.B]
		if cn.Op != OpConst || cn.Imm <= 0 {
			continue
		}
		c := uint64(uint32(cn.Imm))
		hi, lo, ok := n.split(nd.A, c)
		if !ok {
			continue
		}
		b, okb := n.bound(lo, 12)
		if !okb || b >= c {
			continue
		}
		if nd.Op == OpDiv {
			n.Nodes[i] = n.Nodes[hi]
		} else {
			n.Nodes[i] = n.Nodes[lo]
		}
		changed = true
	}
	return changed
}
