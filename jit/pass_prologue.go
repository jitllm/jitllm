package jit

import "github.com/jitllm/jitllm/jit/ir"

// Prologue runs a producer whose outputs only the kernels right after it read
// -- a row's norm and int8 form, an activation's int8 form -- inside each of
// their workgroups, its outputs in workgroup memory: its launch and its trip
// through memory go, every workgroup making the vector it reads.
//
// A producer of one workgroup runs as it is: its spread loops are dealt over
// the consumer's threads, a group's reductions are the consumer's group's. A
// producer of threads runs as one spread loop over its elements, a surplus
// thread clamped to the last element (a pure function of what it reads,
// RULE 13). Where the elements come in whole warps, a surplus thread stores
// nothing, and a combine by shuffles within a warp stays among the warp's
// own elements. An exchange through workgroup slots holds only where its
// loop takes one trip in the consumer, a unit's slot its thread's. Words
// caps the workgroup memory of each kernel it forms, its own arrays and the
// producer's together (the 48 KiB every backend gives a workgroup statically).
type Prologue struct {
	Words int
}

func (Prologue) Name() string { return "prologue" }

func (p Prologue) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	out := append([]*Kernel(nil), ks...)
	for i := 0; i+1 < len(out); i++ {
		n := p.readers(out, i, d)
		if n == 0 {
			continue
		}
		for j := i + 1; j <= i+n; j++ {
			k := p.merge(out[i], out[j], d)
			k.compact()
			out[j] = k
		}
		out = append(out[:i], out[i+1:]...)
	}
	return out, nil
}

// readers is how many kernels right after producer i read what it writes,
// each one it runs in -- every reader of its outputs among them, nothing
// between it and them writing what it reads -- or zero where they do not.
func (p Prologue) readers(ks []*Kernel, i int, d Device) int {
	pr := ks[i]
	n := 0
	for j := i + 1; j < len(ks); j++ {
		reads := false
		for _, b := range pr.Bufs {
			reads = reads || b.Kind == BufOut && readsOperand(ks[j], b.Operand)
		}
		if !reads {
			break
		}
		if !p.fits(pr, ks[j], d) || writesInto(ks[j], pr) {
			return 0
		}
		n++
	}
	if n == 0 {
		return 0
	}
	// No kernel past them reads or writes what the producer writes.
	for _, b := range pr.Bufs {
		if b.Kind != BufOut {
			continue
		}
		for _, k := range ks[i+1+n:] {
			for _, x := range k.Bufs {
				if x.Operand == b.Operand {
					return 0
				}
			}
		}
	}
	return n
}

// fits says producer pr runs in kernel c's workgroups: a producer with no
// facts about its space, the kernel the two make within Words.
func (p Prologue) fits(pr, c *Kernel, d Device) bool {
	if !c.Work || c.Group <= 0 || pr.Positions != nil || pr.Heads != nil || pr.Quarter != nil || pr.Rows != 0 || pr.Clamp != 0 ||
		pr.Seq != nil || len(pr.Args) > 0 || len(pr.Ptrs) > 0 || pr.Part != c.Part || pr.Of != c.Of {
		return false
	}
	shuffles, exchanges := false, false
	for _, nd := range pr.Nodes {
		switch nd.Op {
		// A reduction over the whole workgroup -- a norm's sum of squares --
		// is paid for in every consumer workgroup before its own work starts:
		// a row's norm inside q/k/v and gate/up measured slower than apart
		// (docs/engineering-history/gpu-kernels.md).
		case OpFlat, OpThread, OpGroupSum, OpGroupMax:
			return false
		case OpShuffle:
			shuffles = true
		case OpExchange:
			exchanges = true
		}
	}
	// A combine by lanes needs the consumer's warps whole.
	if shuffles && (d.warp() == 0 || c.Group%d.warp() != 0) {
		return false
	}
	if pr.Work {
		if pr.Elements() != 1 {
			return false
		}
		for li, l := range pr.Loops {
			switch {
			case l.Parent < 0 && l.Spread:
				if exchanges && l.Trip > c.Group {
					return false
				}
			case l.Parent >= 0 && pr.Loops[l.Parent].Parent < 0 && pr.Loops[l.Parent].Spread && l.Kept && l.Count > 0 && len(pr.foldsOf(li)) == 0:
			default:
				return false
			}
		}
		for _, st := range pr.Stores {
			if st.Run {
				return false
			}
		}
	} else {
		n := pr.Elements()
		if len(pr.Loops) > 0 || exchanges && n > c.Group || shuffles && n%d.warp() != 0 {
			return false
		}
	}
	words, read := 0, false
	for _, b := range c.Bufs {
		if b.Kind == BufShared {
			words += (b.Size*ElemBytes(b) + 3) / 4
		}
	}
	for _, b := range pr.Bufs {
		switch {
		case b.Kind == BufShared:
			words += (b.Size*ElemBytes(b) + 3) / 4
		case b.Kind != BufOut:
		case readsOperand(pr, b.Operand):
			// An output the producer reads back stays where it reads it.
			return false
		case readsOperand(c, b.Operand):
			n := b.bytes()
			if n == 0 {
				return false
			}
			words += int((n + 3) / 4)
			read = true
		}
	}
	return read && words <= p.Words
}

// readsOperand says k reads operand o.
func readsOperand(k *Kernel, o Operand) bool {
	for _, b := range k.Bufs {
		if b.Kind == BufIn && b.Operand == o {
			return true
		}
	}
	return false
}

// merge is c with pr run first in each of its workgroups.
func (p Prologue) merge(pr, c *Kernel, d Device) *Kernel {
	k := c.clone()
	k.Passes = append(append([]string(nil), c.Passes...), p.Name())
	// What c reads of pr is workgroup memory: its buffer, and pr's stores
	// to it, one slot of k.
	shared := map[Operand]int{}
	for bi, b := range k.Bufs {
		if b.Kind != BufIn {
			continue
		}
		for _, x := range pr.Bufs {
			if x.Kind == BufOut && x.Operand == b.Operand {
				b.Kind, b.Size = BufShared, int(x.bytes()/int64(ElemBytes(x)))
				b.Shape, b.Extent = x.Shape, x.Extent
				k.Bufs[bi] = b
				shared[b.Operand] = bi
			}
		}
	}
	bufOf := make([]int, len(pr.Bufs))
	for i, b := range pr.Bufs {
		switch {
		case b.Kind == BufOut:
			if at, ok := shared[b.Operand]; ok {
				bufOf[i] = at
			} else {
				bufOf[i] = -1
			}
		case b.Kind == BufShared:
			bufOf[i] = k.buf(b)
		default:
			bufOf[i] = k.bufLike(b)
		}
	}
	// A producer of threads is a spread loop over its elements, its stores
	// in a guard of the real ones where they come in whole warps. There the
	// element is the guard's iteration, so a warp past the last element
	// skips the work -- unless it exchanges through workgroup slots, which
	// every thread meets, a surplus one at the last element.
	wrap, guard, raw, coord := -1, -1, -1, make([]int, len(pr.Space))
	if pr.Work {
		for a := range coord {
			coord[a] = k.u32(0)
		}
	} else {
		n := pr.Elements()
		wrap = k.loop(n, -1)
		k.Loops[wrap].Spread = true
		raw = k.add(Node{Op: OpIter, At: wrap, Type: ir.U32})
		rest := -1
		if w := d.warp(); w > 0 && n%w == 0 && k.Group%w == 0 {
			guard = k.loop(1, wrap)
			in := k.add(Node{Op: OpLess, Type: ir.Pred, A: raw, B: k.u32(n)})
			count := k.add(Node{Op: OpSelect, Type: ir.U32, A: in, B: k.u32(1), C: k.u32(0)})
			k.Loops[guard].Count, k.Loops[guard].Kept = count+1, true
			exchanges := false
			for _, nd := range pr.Nodes {
				exchanges = exchanges || nd.Op == OpExchange
			}
			if !exchanges {
				rest = k.bin(OpAdd, ir.U32, raw, k.add(Node{Op: OpIter, At: guard, Type: ir.U32}))
			}
		}
		if rest < 0 {
			rest = k.bin(OpMin, ir.U32, raw, k.u32(n-1))
		}
		for a := len(pr.Space) - 1; a >= 0; a-- {
			switch ext := pr.Space[a]; {
			case ext == 1:
				coord[a] = k.u32(0)
			case a == 0:
				coord[a] = rest
			default:
				coord[a] = k.bin(OpRem, ir.U32, rest, k.u32(ext))
				rest = k.bin(OpDiv, ir.U32, rest, k.u32(ext))
			}
		}
	}
	loopOf := make([]int, len(pr.Loops))
	for i, l := range pr.Loops {
		if l.Parent >= 0 {
			l.Parent = loopOf[l.Parent]
		}
		if l.Of > 0 {
			l.Of = loopOf[l.Of-1] + 1
		}
		cs := make([]Cursor, len(l.Cursors))
		for j, cu := range l.Cursors {
			cu.Buf = bufOf[cu.Buf]
			cs[j] = cu
		}
		l.Cursors = cs
		loopOf[i] = len(k.Loops)
		k.Loops = append(k.Loops, l)
	}
	nodeOf := make([]int, len(pr.Nodes))
	for i := range nodeOf {
		nodeOf[i] = -1
	}
	var walk func(i int) int
	walk = func(i int) int {
		if nodeOf[i] >= 0 {
			return nodeOf[i]
		}
		nd := pr.Nodes[i]
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
	ref := func(v int) int {
		if v > 0 {
			return walk(v-1) + 1
		}
		return v
	}
	for i, l := range pr.Loops {
		at := &k.Loops[loopOf[i]]
		at.Count, at.Span, at.End, at.Elem = ref(l.Count), ref(l.Span), ref(l.End), ref(l.Elem)
	}
	var first []Store
	for si, st := range pr.Stores {
		if !st.Barrier && bufOf[st.Buf] < 0 {
			continue
		}
		// The producer's own last barrier is the one the pass puts after it.
		if st.Barrier && st.At == -1 && si == len(pr.Stores)-1 {
			continue
		}
		ns := Store{Barrier: st.Barrier, At: st.At, Narrow: st.Narrow}
		switch {
		case st.At >= 0:
			ns.At = loopOf[st.At]
		case guard >= 0:
			ns.At = guard
		case wrap >= 0:
			ns.At = wrap
		}
		if !st.Barrier {
			ns.Buf, ns.Val = bufOf[st.Buf], walk(st.Val)
			// The flat index is read even beside coordinates: it is the
			// kernel's node, not a stranger's.
			ns.Ix = walk(st.Ix)
			for _, a := range st.Ax {
				ns.Ax = append(ns.Ax, walk(a))
			}
		}
		first = append(first, ns)
	}
	// What the producer wrote is seen by every thread before the kernel's
	// own body reads it.
	first = append(first, Store{Barrier: true, At: -1})
	k.Stores = append(first, k.Stores...)
	return k
}
