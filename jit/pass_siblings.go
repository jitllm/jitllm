package jit

import "github.com/jitllm/jitllm/jit/ir"

// Siblings runs consecutive kernels that read one activation and touch
// nothing of each other's -- the projections of one row -- as one launch.
// Each keeps its own workgroups, its body under a loop that runs once in
// them and never in the others' (the IR's conditional: there is no branch
// outside a loop, and a count that is one value for a whole workgroup is
// one), its coordinates its own workgroup's within its share. A small
// matrix rides a large one's grid instead of a launch of its own.
//
// A kernel joins when its element is a workgroup of the first's size, it
// is of the first's kind, it carries no positions, heads or live rows, it
// reads no index of the whole launch, and every store it makes is its
// workgroup's own, outside its loops. A guard is a scope to the lowering:
// what a share reads is made inside its guard and dies there, so the launch
// holds one share's values at a time.
type Siblings struct{}

func (Siblings) Name() string { return "siblings" }

func (s Siblings) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	var out []*Kernel
	for i := 0; i < len(ks); {
		j := i + 1
		for j < len(ks) && sibling(ks[i:j], ks[j]) {
			j++
		}
		if j-i < 2 {
			out = append(out, ks[i])
			i++
			continue
		}
		k := s.merge(ks[i:j])
		k.compact()
		out = append(out, k)
		i = j
	}
	return out, nil
}

// sibling says k joins run: a launch like run's first, sharing an
// activation with one of them, and independent of all.
func sibling(run []*Kernel, k *Kernel) bool {
	f := run[0]
	if !joinable(k) || !joinable(f) || k.Group != f.Group || k.Kind != f.Kind || k.Part != f.Part || k.Of != f.Of || k.Least != f.Least {
		return false
	}
	shares := false
	for _, m := range run {
		if writesInto(m, k) || writesInto(k, m) {
			return false
		}
		shares = shares || sharesActivation(m, k)
	}
	return shares
}

// joinable says k can be one share of a launch: a workgroup an element,
// nothing a fact about its space, every store at the element.
func joinable(k *Kernel) bool {
	if !k.Work || k.Group <= 0 || k.Positions != nil || k.Heads != nil || k.Quarter != nil || k.Rows != 0 || k.Clamp != 0 ||
		k.Seq != nil || len(k.Args) > 0 || len(k.Ptrs) > 0 {
		return false
	}
	for _, st := range k.Stores {
		if st.At != -1 || st.Run {
			return false
		}
	}
	for _, nd := range k.Nodes {
		if nd.Op == OpFlat || nd.Op == OpLocal {
			return false
		}
	}
	return true
}

// writesInto says a stores to something b reads or writes.
func writesInto(a, b *Kernel) bool {
	for _, st := range a.Stores {
		if st.Barrier {
			continue
		}
		o := a.Bufs[st.Buf].Operand
		for _, x := range b.Bufs {
			if x.Kind != BufShared && x.Operand == o {
				return true
			}
		}
	}
	return false
}

// sharesActivation says a and b read one operand that is not a weight.
func sharesActivation(a, b *Kernel) bool {
	for _, x := range a.Bufs {
		if x.Kind != BufIn || x.Operand.Kind == OperandWeight {
			continue
		}
		for _, y := range b.Bufs {
			if y.Kind == BufIn && y.Operand == x.Operand {
				return true
			}
		}
	}
	return false
}

// merge is ms as one launch of their workgroups in order.
func (s Siblings) merge(ms []*Kernel) *Kernel {
	f := ms[0]
	k := &Kernel{Name: f.Name, Work: true, Group: f.Group, Kind: f.Kind, Part: f.Part, Of: f.Of, Least: f.Least,
		Passes: append(append([]string(nil), f.Passes...), s.Name())}
	total := 0
	for _, m := range ms {
		total += m.Elements()
	}
	k.Space = []int{total}
	e := k.add(Node{Op: OpCoord, A: 0, Type: ir.U32})
	zero, one := k.u32(0), k.u32(1)
	off := 0
	for _, m := range ms {
		n := m.Elements()
		lo, hi := k.u32(off), k.u32(off+n)
		below := k.add(Node{Op: OpLess, Type: ir.Pred, A: e, B: lo})
		under := k.add(Node{Op: OpLess, Type: ir.Pred, A: e, B: hi})
		count := k.add(Node{Op: OpSelect, Type: ir.U32, A: under, B: k.add(Node{Op: OpSelect, Type: ir.U32, A: below, B: zero, C: one}), C: zero})
		g := k.loop(1, -1)
		k.Loops[g].Count, k.Loops[g].Kept, k.Loops[g].Uniform = count+1, true, true
		local := k.bin(OpSub, ir.U32, e, lo)
		k.Stores = append(k.Stores, Store{Run: true, At: g})
		s.share(k, m, g, local)
		off += n
	}
	return k
}

// share is kernel m in loop g of k, its element local: its coordinates
// unflattened from local, its loops inside g, its stores at g's trip.
func (Siblings) share(k, m *Kernel, g, local int) {
	coord := make([]int, len(m.Space))
	rest := local
	for a := len(m.Space) - 1; a >= 0; a-- {
		switch ext := m.Space[a]; {
		case ext == 1:
			coord[a] = k.u32(0)
		case a == 0:
			coord[a] = rest
		default:
			coord[a] = k.bin(OpRem, ir.U32, rest, k.u32(ext))
			rest = k.bin(OpDiv, ir.U32, rest, k.u32(ext))
		}
	}
	// One buffer an operand across the shares, but workgroup memory of each
	// its own: only one share's body runs in a workgroup.
	bufOf := make([]int, len(m.Bufs))
	for i, b := range m.Bufs {
		if b.Kind == BufShared {
			bufOf[i] = k.buf(b)
		} else {
			bufOf[i] = k.bufLike(b)
		}
	}
	loopOf := make([]int, len(m.Loops))
	for i, l := range m.Loops {
		if l.Parent >= 0 {
			l.Parent = loopOf[l.Parent]
		} else {
			l.Parent = g
		}
		if l.Of > 0 {
			l.Of = loopOf[l.Of-1] + 1
		}
		cs := make([]Cursor, len(l.Cursors))
		for j, c := range l.Cursors {
			c.Buf = bufOf[c.Buf]
			cs[j] = c
		}
		l.Cursors = cs
		loopOf[i] = len(k.Loops)
		k.Loops = append(k.Loops, l)
	}
	nodeOf := make([]int, len(m.Nodes))
	for i := range nodeOf {
		nodeOf[i] = -1
	}
	var walk func(i int) int
	walk = func(i int) int {
		if nodeOf[i] >= 0 {
			return nodeOf[i]
		}
		nd := m.Nodes[i]
		if nd.Op == OpCoord {
			nodeOf[i] = coord[nd.A]
			return nodeOf[i]
		}
		if nd.Op == OpIter || nd.Op == OpFold {
			nd.At = loopOf[nd.At]
		}
		if nd.Op == OpLoad {
			nd.Buf = bufOf[nd.Buf]
		}
		nodeOf[i] = k.add(Node{})
		for _, a := range k.Operands(&nd) {
			*a = walk(*a)
		}
		k.Nodes[nodeOf[i]] = nd
		return nodeOf[i]
	}
	for i := range m.Nodes {
		walk(i)
	}
	ref := func(v int) int {
		if v > 0 {
			return walk(v-1) + 1
		}
		return v
	}
	for i, l := range m.Loops {
		at := &k.Loops[loopOf[i]]
		at.Count, at.Span, at.End, at.Elem = ref(l.Count), ref(l.Span), ref(l.End), ref(l.Elem)
	}
	for _, st := range m.Stores {
		ns := Store{Barrier: st.Barrier, At: g, Narrow: st.Narrow}
		if !st.Barrier {
			ns.Buf, ns.Val = bufOf[st.Buf], walk(st.Val)
			// The flat index is read even beside coordinates: it is the
			// kernel's node, not a stranger's.
			ns.Ix = walk(st.Ix)
			for _, a := range st.Ax {
				ns.Ax = append(ns.Ax, walk(a))
			}
		}
		k.Stores = append(k.Stores, ns)
	}
}
