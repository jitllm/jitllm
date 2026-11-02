package composer

import "github.com/jitllm/jitllm/composer/ir"

// vecLoad is a load the lowering reads with three others in one access
// (ir.OpLoadV): the group's loads by lane, and this one's lane.
type vecLoad struct {
	group *[4]int
	lane  int
}

// vectorGroups finds, on a device that reads four consecutive words of a
// buffer in one access (Device.VectorLoads), the 32-bit loads of a scratch
// buffer at one index and the four offsets of an aligned run of four: one
// sixteen-byte access reads them. A scratch buffer is an allocation of its
// own, never bound at an offset (Binding), so an index a multiple of four
// is aligned to sixteen bytes. Indices equal by their structure are one
// index: a load carried a trip ahead (Prefetch) is a copy of its index.
func (e *lowerer) vectorGroups() {
	n := e.n
	canon := map[int]int{}
	ids := map[[8]int64]int{}
	var id func(i int) int
	id = func(i int) int {
		if v, ok := canon[i]; ok {
			return v
		}
		nd := n.Nodes[i]
		key := [8]int64{int64(nd.Op), int64(nd.Type), nd.Imm, int64(nd.At), int64(nd.Buf), -1, -1, -1}
		if nd.Op == OpFold || nd.Ax != nil || nd.Tile != nil {
			key[5] = int64(i)
		} else {
			for j, a := range n.Operands(&nd) {
				key[5+j] = int64(id(*a))
			}
		}
		v, ok := ids[key]
		if !ok {
			v = len(ids)
			ids[key] = v
		}
		canon[i] = v
		return v
	}
	type at struct {
		buf, ix int
		base    int64
		typ     ir.Type
	}
	groups := map[at]*[4]int{}
	have := map[at]int{}
	for i, nd := range n.Nodes {
		if nd.Op != OpLoad || nd.Ax != nil || nd.Imm < 0 || nd.Type != ir.U32 && nd.Type != ir.F32 && nd.Type != ir.I32 {
			continue
		}
		bf := n.Bufs[nd.Buf]
		if bf.Kind != BufShared && bf.Operand.Kind != OperandScratch || !n.alignedTo(nd.A, 4) {
			continue
		}
		k := at{nd.Buf, id(nd.A), nd.Imm &^ 3, nd.Type}
		g := groups[k]
		if g == nil {
			g = &[4]int{-1, -1, -1, -1}
			groups[k] = g
		}
		if g[nd.Imm-k.base] < 0 {
			g[nd.Imm-k.base] = i
			have[k] |= 1 << (nd.Imm - k.base)
		}
	}
	e.storeGroups(id)
	for k, g := range groups {
		if have[k] != 15 {
			continue
		}
		if e.vec == nil {
			e.vec = map[int]vecLoad{}
		}
		for lane, i := range g {
			e.vec[i] = vecLoad{group: g, lane: lane}
		}
	}
}

// storeGroups finds runs of four consecutive stores to a workgroup array at
// an aligned index and the three words after it: one 16-byte access.
func (e *lowerer) storeGroups(id func(int) int) {
	n := e.n
	split := func(i int) (int, int64) {
		var off int64
		for {
			nd := n.Nodes[i]
			if nd.Op != OpAdd || nd.Type != ir.U32 {
				return i, off
			}
			if c := n.Nodes[nd.B]; c.Op == OpConst {
				off, i = off+c.Imm, nd.A
				continue
			}
			if c := n.Nodes[nd.A]; c.Op == OpConst {
				off, i = off+c.Imm, nd.B
				continue
			}
			return i, off
		}
	}
	for si := 0; si+3 < len(n.Stores); si++ {
		st := n.Stores[si]
		if st.Barrier || st.Run || st.Tile || st.Ax != nil || n.Bufs[st.Buf].Kind != BufShared {
			continue
		}
		base, off := split(st.Ix)
		if off != 0 || !n.alignedTo(base, 4) {
			continue
		}
		run := [4]int{si}
		for j := 1; j < 4; j++ {
			o := n.Stores[si+j]
			b, f := split(o.Ix)
			if o.Barrier || o.Run || o.Tile || o.Ax != nil || o.Buf != st.Buf || o.At != st.At || f != int64(j) || id(b) != id(base) {
				run[0] = -1
				break
			}
			run[j] = si + j
		}
		if run[0] < 0 {
			continue
		}
		if e.vecStore == nil {
			e.vecStore = map[int][4]int{}
		}
		e.vecStore[si] = run
		si += 3
	}
}

// loadVector reads load i's group in one access and holds each lane's
// value; it is i's.
func (e *lowerer) loadVector(i int, v vecLoad) ir.Value {
	nd := e.n.Nodes[i]
	lanes := e.b.LoadV(nd.Type, e.params[nd.Buf], e.get(nd.A), nd.Imm-int64(v.lane), 4)
	for lane, j := range v.group {
		e.val[j] = lanes[lane]
	}
	return lanes[v.lane]
}

// alignedTo reports whether node i is a multiple of m, a power of two, by
// its structure: divisibleBy, and a shift right by s of a multiple of m<<s.
func (n *Kernel) alignedTo(i int, m uint32) bool {
	nd := n.Nodes[i]
	if nd.Type != ir.U32 {
		return false
	}
	switch nd.Op {
	case OpAdd:
		return n.alignedTo(nd.A, m) && n.alignedTo(nd.B, m)
	case OpMul, OpAnd:
		return n.alignedTo(nd.A, m) || n.alignedTo(nd.B, m)
	case OpShr:
		if s := n.Nodes[nd.B]; s.Op == OpConst && s.Imm >= 0 && s.Imm < 32 && uint64(m)<<uint64(s.Imm) < 1<<32 {
			return n.alignedTo(nd.A, m<<uint32(s.Imm))
		}
		return false
	}
	return n.divisibleBy(i, m)
}
