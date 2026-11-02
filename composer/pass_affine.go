package composer

import (
	"slices"

	"github.com/jitllm/jitllm/composer/ir"
)

// Affine simplifies the index arithmetic of a kernel's loads, exact in the
// u32 ring: equal nodes merged, the arithmetic of constants done, constants
// carried outward -- (a+c1)*c2 as a*c2 + c1*c2, (a+c1)+b as (a+b)+c1 -- and
// folded, so the unrolled steps of a walk share one address and differ by a
// constant, and a load of a+c reads a at c past it, the constant the
// access's immediate offset: a workgroup's partials read as one register
// and a displacement each, not an address each. In a kernel of a kind in
// Weights, loads of one weight word merge too, so the elements a word packs
// share its load and the arithmetic of its scales, and a term a shift or a
// mask cannot see by its bound is dropped (bounded).
type Affine struct {
	Weights []PrimKind
}

func (Affine) Name() string { return "affine" }

func (p Affine) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		m := k.clone()
		changed := false
		weights := slices.Contains(p.Weights, k.Kind)
		for {
			step := m.cse(weights)
			step = m.foldConstants() || step
			step = m.identities() || step
			step = m.powersOfTwo() || step
			step = m.headroom() || step
			if weights {
				step = m.bounded() || step
			}
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
// powersOfTwo takes an unsigned division by 2^n as a shift by n and a
// remainder as a mask, which they are exactly: a device compiler that cannot
// see the divisor is a constant -- one made outside the loop that divides --
// otherwise emits a whole division in the loop.
func (k *Kernel) powersOfTwo() bool {
	changed := false
	for i, nd := range k.Nodes {
		if nd.Type != ir.U32 || nd.Op != OpDiv && nd.Op != OpRem {
			continue
		}
		n, ok := k.powerOfTwo(nd.B)
		if !ok || k.Nodes[nd.B].Type != ir.U32 {
			continue
		}
		if nd.Op == OpDiv {
			k.Nodes[i] = Node{Op: OpShr, Type: ir.U32, A: nd.A, B: k.u32(n)}
		} else {
			k.Nodes[i] = Node{Op: OpAnd, Type: ir.U32, A: nd.A, B: k.u32(1<<n - 1)}
		}
		changed = true
	}
	return k.aligned() || changed
}

// aligned takes the low and high bits of a*C + c apart where C is a multiple
// of 2^n: the mask 2^n-1 keeps c's bits alone, exactly in the u32 ring, and
// the shift by n is a*(C>>n) for c below 2^n where a is a loop's iteration
// whose trips keep a*C + c below 2^32 -- so two places of one word, reached
// by different constants, become one address.
func (k *Kernel) aligned() bool {
	cst := func(i int) (uint32, bool) {
		nd := k.Nodes[i]
		return uint32(nd.Imm), nd.Op == OpConst && nd.Type == ir.U32
	}
	// split is node i as a*C + c with C and c constants.
	split := func(i int) (a int, C, c uint32, ok bool) {
		nd := k.Nodes[i]
		if nd.Op == OpAdd {
			if v, ok := cst(nd.B); ok {
				c, i, nd = v, nd.A, k.Nodes[nd.A]
			}
		}
		if nd.Op != OpMul {
			return 0, 0, 0, false
		}
		v, ok := cst(nd.B)
		return nd.A, v, c, ok
	}
	changed := false
	for i, nd := range k.Nodes {
		if nd.Type != ir.U32 || nd.Op != OpAnd && nd.Op != OpShr {
			continue
		}
		b, ok := cst(nd.B)
		if !ok {
			continue
		}
		a, C, c, ok := split(nd.A)
		if !ok {
			continue
		}
		switch nd.Op {
		case OpAnd:
			if b&(b+1) != 0 || C&b != 0 {
				continue
			}
			k.Nodes[i] = Node{Op: OpConst, Type: ir.U32, Imm: int64(c & b)}
		case OpShr:
			n := b
			it := k.Nodes[a]
			if n == 0 || n > 31 || C&(1<<n-1) != 0 || c >= 1<<n || it.Op != OpIter || k.Loops[it.At].Count > 0 ||
				uint64(k.Loops[it.At].Trip)*uint64(C)+uint64(c) >= 1<<32 {
				continue
			}
			k.Nodes[i] = Node{Op: OpMul, Type: ir.U32, A: a, B: k.u32(int(C >> n))}
		}
		changed = true
	}
	return changed
}

// headroom drops the c of x+c that a shift or a mask cannot see, exactly in
// the u32 ring: where x is a multiple of 2^m by its structure and c is below
// 2^m, x+c is x|c, so (x+c)>>n is x>>n for n at least m, and (x+c)&(2^n-1)
// is (x&(2^n-1))+c; x&(2^n-1) is 0 for x a multiple of 2^n. Where x is a
// multiple of 2^n itself, (x+c)&(2^n-1) is c&(2^n-1) whatever c, and
// (x+c)>>n is (x>>n)+(c>>n) where x+c stays below 2^32 by x's bound. The
// elements of one word reached at x plus each place then read the word at
// one address, and the words past it at constants from it.
func (k *Kernel) headroom() bool {
	changed := false
	bound := map[int]uint64{}
	for i := range k.Nodes {
		nd := k.Nodes[i]
		if nd.Type != ir.U32 || nd.Op != OpShr && nd.Op != OpAnd || k.Nodes[nd.B].Op != OpConst {
			continue
		}
		b := uint32(k.Nodes[nd.B].Imm)
		if nd.Op == OpAnd && b&(b+1) == 0 && b != 0 && b != ^uint32(0) && k.divisibleBy(nd.A, b+1) {
			k.Nodes[i] = Node{Op: OpConst, Type: ir.U32}
			changed = true
			continue
		}
		x, c, ok := k.addConst(nd.A)
		if !ok || c == 0 {
			continue
		}
		if nd.Op == OpAnd && b&(b+1) == 0 && b != ^uint32(0) && k.divisibleBy(x, b+1) {
			k.Nodes[i] = Node{Op: OpConst, Type: ir.U32, Imm: int64(c & b)}
			changed = true
			continue
		}
		if nd.Op == OpShr && b > 0 && b < 32 && c>>b > 0 && k.divisibleBy(x, 1<<b) && k.below(x, bound)+uint64(c) < 1<<32 {
			k.Nodes[i] = Node{Op: OpAdd, Type: ir.U32, A: k.bin(OpShr, ir.U32, x, nd.B), B: k.u32(int(c >> b))}
			changed = true
			continue
		}
		n := uint32(32)
		if nd.Op == OpShr {
			n = b
		} else if b&(b+1) == 0 {
			n = uint32(bitsOf(int(b) + 1))
		}
		m := uint32(0)
		for e := n; e > 0; e-- {
			if c < 1<<e && k.divisibleBy(x, 1<<e) {
				m = e
				break
			}
		}
		if m == 0 || n > 31 {
			continue
		}
		if nd.Op == OpShr {
			k.Nodes[i].A = x
		} else {
			k.Nodes[i] = Node{Op: OpAdd, Type: ir.U32, A: k.bin(OpAnd, ir.U32, x, nd.B), B: k.u32(int(c))}
		}
		changed = true
	}
	return changed
}

// bounded drops what a shift or a mask by 2^n cannot see, exactly in the
// u32 ring, where a term is a multiple of 2^n by its structure and the
// other stays below 2^n by its bound rather than by being a constant: x+y
// is then x|y, so (x+y)>>n is x>>n, and (x+y)&(2^n-1) is y&(2^n-1) whatever
// y. A mask 2^n-1 of a value its bound keeps at or under it is the value,
// and (a*C)>>n is a*(C>>n) for C a multiple of 2^n where a's bound keeps
// a*C below 2^32. An index of a trip and a thread's place in it then reads
// as the trip's and the place's apart.
func (k *Kernel) bounded() bool {
	changed := false
	bound := map[int]uint64{}
	for i := range k.Nodes {
		nd := k.Nodes[i]
		c := k.Nodes[nd.B]
		if nd.Type != ir.U32 || nd.Op != OpShr && nd.Op != OpAnd || c.Op != OpConst || c.Type != ir.U32 {
			continue
		}
		b := uint32(c.Imm)
		n := b
		if nd.Op == OpAnd {
			if b&(b+1) != 0 || b == ^uint32(0) {
				continue
			}
			n = uint32(bitsOf(int(b) + 1))
		} else if b == 0 || b > 31 {
			continue
		}
		low := uint64(1)<<n - 1
		if nd.Op == OpAnd && k.below(nd.A, bound) <= low {
			k.Nodes[i] = Node{Op: OpAdd, Type: ir.U32, A: nd.A, B: k.u32(0)}
			changed = true
			continue
		}
		a := k.Nodes[nd.A]
		switch {
		case a.Op == OpAdd && a.Type == ir.U32:
			for _, xy := range [][2]int{{a.A, a.B}, {a.B, a.A}} {
				x, y := xy[0], xy[1]
				if k.Nodes[y].Op == OpConst || !k.divisibleBy(x, 1<<n) {
					continue
				}
				if nd.Op == OpAnd {
					k.Nodes[i].A = y
				} else if k.below(y, bound) <= low && k.below(x, bound)+k.below(y, bound) < 1<<32 {
					k.Nodes[i].A = x
				} else {
					continue
				}
				changed = true
				break
			}
		case a.Op == OpMul && a.Type == ir.U32 && nd.Op == OpShr:
			for _, xc := range [][2]int{{a.A, a.B}, {a.B, a.A}} {
				x, C := xc[0], k.Nodes[xc[1]]
				if C.Op != OpConst || C.Type != ir.U32 || uint32(C.Imm)&uint32(low) != 0 || k.below(x, bound)*uint64(uint32(C.Imm)) >= 1<<32 {
					continue
				}
				k.Nodes[i] = Node{Op: OpMul, Type: ir.U32, A: x, B: k.u32(int(uint32(C.Imm) >> n))}
				changed = true
				break
			}
		}
	}
	return changed
}

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

// identities takes x+0, x-0, x|0 shifted by 0, x*1 and x/1 as x.
func (k *Kernel) identities() bool {
	same := map[int]int{}
	for i, nd := range k.Nodes {
		if nd.Type != ir.U32 {
			continue
		}
		c := func(x int, v int64) bool { return k.Nodes[x].Op == OpConst && k.Nodes[x].Imm == v }
		switch nd.Op {
		case OpAdd:
			if c(nd.B, 0) {
				same[i] = nd.A
			} else if c(nd.A, 0) {
				same[i] = nd.B
			}
		case OpSub, OpShr, OpShl, OpXor:
			if c(nd.B, 0) {
				same[i] = nd.A
			}
		case OpMul:
			if c(nd.B, 1) {
				same[i] = nd.A
			} else if c(nd.A, 1) {
				same[i] = nd.B
			}
		case OpDiv:
			if c(nd.B, 1) {
				same[i] = nd.A
			}
		}
	}
	for i, j := range same {
		for {
			t, ok := same[j]
			if !ok {
				break
			}
			j = t
		}
		same[i] = j
	}
	k.redirectMany(same)
	return len(same) > 0
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

// below is a bound node i's u32 value stays at or under by its structure:
// its constants, its loops' trips, the space and the workgroup; 2^32-1
// where the structure states none. memo holds the bounds found.
func (k *Kernel) below(i int, memo map[int]uint64) uint64 {
	const top = 1<<32 - 1
	if v, ok := memo[i]; ok {
		return v
	}
	memo[i] = top
	nd := k.Nodes[i]
	v := uint64(top)
	cst := func(j int) (uint64, bool) {
		c := k.Nodes[j]
		return uint64(uint32(c.Imm)), c.Op == OpConst && c.Type == ir.U32
	}
	if nd.Type == ir.U32 {
		switch nd.Op {
		case OpConst:
			v = uint64(uint32(nd.Imm))
		case OpIter:
			if l := k.Loops[nd.At]; l.Count == 0 && l.Trip > 0 {
				v = uint64(l.Trip - 1)
			}
		case OpCoord:
			if nd.A < len(k.Space) && k.Space[nd.A] > 0 {
				v = uint64(k.Space[nd.A] - 1)
			}
		case OpThread:
			if k.Group > 0 {
				v = uint64(k.Group - 1)
			}
		case OpAdd:
			v = min(k.below(nd.A, memo)+k.below(nd.B, memo), top)
		case OpMul:
			if a, b := k.below(nd.A, memo), k.below(nd.B, memo); a == 0 || b == 0 || a <= top/b {
				v = a * b
			}
		case OpAnd:
			v = min(k.below(nd.A, memo), k.below(nd.B, memo))
		case OpMin:
			v = min(k.below(nd.A, memo), k.below(nd.B, memo))
		case OpRem:
			if c, ok := cst(nd.B); ok && c > 0 {
				v = min(k.below(nd.A, memo), c-1)
			}
		case OpDiv:
			if c, ok := cst(nd.B); ok && c > 0 {
				v = k.below(nd.A, memo) / c
			}
		case OpShr:
			if c, ok := cst(nd.B); ok && c < 32 {
				v = k.below(nd.A, memo) >> c
			}
		case OpShl:
			if c, ok := cst(nd.B); ok && c < 32 && k.below(nd.A, memo)<<c <= top {
				v = k.below(nd.A, memo) << c
			}
		}
	}
	memo[i] = v
	return v
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
	case m&(m-1) != 0:
		// The cases below hold for a power of two alone.
	case nd.Op == OpAnd:
		return k.divisibleBy(nd.A, m) || k.divisibleBy(nd.B, m)
	case nd.Op == OpShl && k.Nodes[nd.B].Op == OpConst && k.Nodes[nd.B].Imm < 32:
		return k.divisibleBy(nd.A, m) || uint64(1)<<uint64(k.Nodes[nd.B].Imm)%uint64(m) == 0
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
