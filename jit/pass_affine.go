package jit

import "github.com/jitllm/jitllm/jit/ir"

// Affine simplifies the index arithmetic of a kernel's global loads, exact
// in the u32 ring: equal nodes merged, constants carried outward --
// (a+c1)*c2 as a*c2 + c1*c2, (a+c1)+b as (a+b)+c1 -- and folded, so the
// unrolled steps of a walk share one address and differ by a constant, and
// a load of a+c reads a at c past it, the constant the access's immediate
// offset. A workgroup-memory address main computes whole, and so it stays.
type Affine struct{}

func (Affine) Name() string { return "affine" }

func (p Affine) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		m := k.clone()
		changed := false
		for {
			step := m.cse()
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

// addConst is node i as x + c for a constant c, when it is one.
func (k *Kernel) addConst(i int) (x int, c uint32, ok bool) {
	nd := k.Nodes[i]
	if nd.Op != OpAdd || nd.Type != ir.U32 || k.Nodes[nd.B].Op != OpConst {
		return 0, 0, false
	}
	return nd.A, uint32(k.Nodes[nd.B].Imm), true
}

// globalIndex is every u32 add and multiply a global load's flat index is
// made of.
func (k *Kernel) globalIndex() map[int]bool {
	out := map[int]bool{}
	var walk func(i int)
	walk = func(i int) {
		nd := k.Nodes[i]
		if out[i] || nd.Type != ir.U32 || nd.Op != OpAdd && nd.Op != OpMul {
			return
		}
		out[i] = true
		walk(nd.A)
		walk(nd.B)
	}
	for _, nd := range k.Nodes {
		if nd.Op == OpLoad && nd.Ax == nil && k.Bufs[nd.Buf].Kind != BufShared {
			walk(nd.A)
		}
	}
	return out
}

// constantsOutward carries u32 constants outward in a global load's index
// and folds them.
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

// immediates moves a load's constant term into its offset: a flat index
// a+c, c below 2^31, read as a at c elements past it.
func (k *Kernel) immediates() bool {
	changed := false
	for i, nd := range k.Nodes {
		if nd.Op != OpLoad || nd.Ax != nil || k.Bufs[nd.Buf].Kind == BufShared {
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
