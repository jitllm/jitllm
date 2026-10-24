package jit

import "github.com/jitllm/jitllm/jit/ir"

// Affine simplifies the index arithmetic of a kernel's loads, exact in the
// u32 ring: equal nodes merged, the arithmetic of constants done, constants
// carried outward -- (a+c1)*c2 as a*c2 + c1*c2, (a+c1)+b as (a+b)+c1 -- and
// folded, so the unrolled steps of a walk share one address and differ by a
// constant, and a load of a+c reads a at c past it, the constant the
// access's immediate offset: a workgroup's partials read as one register
// and a displacement each, not an address each.
type Affine struct{}

func (Affine) Name() string { return "affine" }

func (p Affine) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		m := k.clone()
		changed := false
		for {
			step := m.cse()
			step = m.foldConstants() || step
			step = m.constantsOutward() || step
			if !step {
				break
			}
			changed = true
			m.compact()
		}
		if m.immediates() {
			changed = true
			m.compact()
		}
		if !changed {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

// foldConstants does the u32 arithmetic of two constants.
func (k *Kernel) foldConstants() bool {
	changed := false
	for i, nd := range k.Nodes {
		if nd.Type != ir.U32 {
			continue
		}
		switch nd.Op {
		case OpAdd, OpSub, OpMul, OpDiv, OpRem, OpShr, OpShl, OpAnd, OpXor, OpMin, OpMax:
		default:
			continue
		}
		a, b := k.Nodes[nd.A], k.Nodes[nd.B]
		if a.Op != OpConst || b.Op != OpConst || a.Type != ir.U32 || b.Type != ir.U32 {
			continue
		}
		x, y := uint32(a.Imm), uint32(b.Imm)
		var v uint32
		switch nd.Op {
		case OpAdd:
			v = x + y
		case OpSub:
			v = x - y
		case OpMul:
			v = x * y
		case OpDiv, OpRem:
			if y == 0 {
				continue
			}
			v = x / y
			if nd.Op == OpRem {
				v = x % y
			}
		case OpShr:
			v = x >> (y & 31)
		case OpShl:
			v = x << (y & 31)
		case OpAnd:
			v = x & y
		case OpXor:
			v = x ^ y
		case OpMin:
			v = min(x, y)
		case OpMax:
			v = max(x, y)
		}
		k.Nodes[i] = Node{Op: OpConst, Type: ir.U32, Imm: int64(v)}
		changed = true
	}
	return changed
}

// addConst is node i as x + c for a constant c, when it is one.
func (k *Kernel) addConst(i int) (x int, c uint32, ok bool) {
	nd := k.Nodes[i]
	if nd.Op != OpAdd || nd.Type != ir.U32 || k.Nodes[nd.B].Op != OpConst {
		return 0, 0, false
	}
	return nd.A, uint32(k.Nodes[nd.B].Imm), true
}

// globalIndex is every u32 add and multiply a load's flat index is made of.
func (k *Kernel) globalIndex() map[int]bool {
	out := map[int]bool{}
	var walk func(i int)
	walk = func(i int) {
		nd := k.Nodes[i]
		if out[i] || nd.Type != ir.U32 || nd.Op != OpAdd && nd.Op != OpMul && nd.Op != OpDiv && nd.Op != OpRem {
			return
		}
		out[i] = true
		walk(nd.A)
		walk(nd.B)
	}
	for _, nd := range k.Nodes {
		if nd.Op == OpLoad && nd.Ax == nil {
			walk(nd.A)
		}
	}
	return out
}

// constantsOutward carries u32 constants outward in a load's index and
// folds them.
func (k *Kernel) constantsOutward() bool {
	changed := false
	index := k.globalIndex()
	U := ir.U32
	cst := func(i int) (uint32, bool) {
		if nd := k.Nodes[i]; nd.Op == OpConst && nd.Type == U {
			return uint32(nd.Imm), true
		}
		return 0, false
	}
	set := func(i int, nd Node) {
		k.Nodes[i] = nd
		changed = true
	}
	for i := 0; i < len(k.Nodes); i++ {
		nd := k.Nodes[i]
		if !index[i] {
			continue
		}
		a, b := nd.A, nd.B
		ca, aok := cst(a)
		cb, bok := cst(b)
		if (nd.Op == OpDiv || nd.Op == OpRem) && bok && cb > 0 {
			// (y+c) % m is c % m, (y+c)/m is y/m + c/m, y a multiple of m.
			y, c, ok := k.addConst(a)
			if !ok {
				y, c = a, 0
			}
			if !k.divisibleBy(y, cb) {
				continue
			}
			if nd.Op == OpRem {
				set(i, Node{Op: OpConst, Type: U, Imm: int64(c % cb)})
				continue
			}
			if q, ok := k.dividedBy(y, cb); ok {
				if c/cb == 0 {
					k.redirectAll(i, q)
					changed = true
				} else {
					set(i, Node{Op: OpAdd, Type: U, A: q, B: k.u32(int(c / cb))})
				}
			}
			continue
		}
		if nd.Op != OpAdd && nd.Op != OpMul {
			continue
		}
		switch {
		case aok && bok && nd.Op == OpAdd:
			set(i, Node{Op: OpConst, Type: U, Imm: int64(ca + cb)})
		case aok && bok:
			set(i, Node{Op: OpConst, Type: U, Imm: int64(ca * cb)})
		case aok:
			set(i, Node{Op: nd.Op, Type: U, A: b, B: a})
		case nd.Op == OpAdd && bok && cb == 0:
			k.redirectAll(i, a)
			changed = true
		case nd.Op == OpMul && bok && cb == 1:
			k.redirectAll(i, a)
			changed = true
		case nd.Op == OpAdd:
			if x, c1, ok := k.addConst(a); ok && bok {
				set(i, Node{Op: OpAdd, Type: U, A: x, B: k.u32(int(c1 + cb))})
			} else if ok {
				set(i, Node{Op: OpAdd, Type: U, A: k.bin(OpAdd, U, x, b), B: k.u32(int(c1))})
			} else if x, c1, ok := k.addConst(b); ok {
				set(i, Node{Op: OpAdd, Type: U, A: k.bin(OpAdd, U, a, x), B: k.u32(int(c1))})
			}
		case nd.Op == OpMul && bok:
			if x, c1, ok := k.addConst(a); ok {
				set(i, Node{Op: OpAdd, Type: U, A: k.bin(OpMul, U, x, b), B: k.u32(int(c1 * cb))})
			} else if an := k.Nodes[a]; an.Op == OpMul && an.Type == U {
				if c1, ok := cst(an.B); ok {
					set(i, Node{Op: OpMul, Type: U, A: an.A, B: k.u32(int(c1 * cb))})
				}
			}
		}
	}
	return changed
}

// divisibleBy reports whether node i is a multiple of m by its structure.
func (k *Kernel) divisibleBy(i int, m uint32) bool {
	nd := k.Nodes[i]
	switch {
	case nd.Type != ir.U32:
		return false
	case nd.Op == OpConst:
		return uint32(nd.Imm)%m == 0
	case nd.Op == OpMul:
		return k.divisibleBy(nd.A, m) || k.divisibleBy(nd.B, m)
	case nd.Op == OpAdd:
		return k.divisibleBy(nd.A, m) && k.divisibleBy(nd.B, m)
	}
	return false
}

// dividedBy is node i, a multiple of m, divided by m where its structure
// states it: a product with a constant multiple of m, or a sum of such.
func (k *Kernel) dividedBy(i int, m uint32) (int, bool) {
	nd := k.Nodes[i]
	switch {
	case nd.Op == OpConst:
		return k.u32(int(uint32(nd.Imm) / m)), true
	case nd.Op == OpMul && k.Nodes[nd.B].Op == OpConst && uint32(k.Nodes[nd.B].Imm)%m == 0:
		return k.bin(OpMul, ir.U32, nd.A, k.u32(int(uint32(k.Nodes[nd.B].Imm)/m))), true
	case nd.Op == OpAdd:
		x, ok := k.dividedBy(nd.A, m)
		if !ok {
			return 0, false
		}
		y, ok := k.dividedBy(nd.B, m)
		if !ok {
			return 0, false
		}
		return k.bin(OpAdd, ir.U32, x, y), true
	}
	return 0, false
}

// immediates moves a load's constant term into its offset: a flat index
// a+c, c below 2^31, read as a at c elements past it.
func (k *Kernel) immediates() bool {
	changed := false
	for i, nd := range k.Nodes {
		if nd.Op != OpLoad || nd.Ax != nil {
			continue
		}
		x, c, ok := k.addConst(nd.A)
		if !ok || c >= 1<<31 {
			continue
		}
		k.Nodes[i].A, k.Nodes[i].Imm = x, nd.Imm+int64(c)
		changed = true
	}
	return changed
}
