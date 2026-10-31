package jit

import (
	"slices"

	"github.com/jitllm/jitllm/jit/ir"
)

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
	// Lead is how many of each thread's weight words a kernel it forms asks
	// the cache for before the producer runs, its first trips' words.
	Lead int
	// Trips caps the trips a thread of the kernel takes of a producer's
	// loop where the producer reduces over its workgroup: its latency is
	// paid before any weight word is summed.
	Trips int
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
	if n == 0 || !pays(pr, ks[i+1:i+1+n], d) {
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

// pays says running pr in each workgroup of cs moves no more words than a
// launch of pr holds d for: every workgroup reads pr's inputs whole before
// its own first load -- what it then reads of pr's outputs streamed beside
// its weights apart, so saves it no wait -- and pr's own reads and writes go.
func pays(pr *Kernel, cs []*Kernel, d Device) bool {
	if d.Launch <= 0 {
		return true
	}
	words := func(b Buf) int64 { return (b.bytes() + 3) / 4 }
	var in, out int64
	for _, b := range pr.Bufs {
		switch b.Kind {
		case BufIn:
			in += words(b)
		case BufOut:
			out += words(b)
		}
	}
	extra := -in - out
	for _, c := range cs {
		extra += int64(c.Elements()) * in
	}
	return extra <= int64(d.Launch)
}

// fits says producer pr runs in kernel c's workgroups: a producer with no
// facts about its space, the kernel the two make within Words.
func (p Prologue) fits(pr, c *Kernel, d Device) bool {
	if !c.Work || c.Group <= 0 || c.Kind == PrimMatrix || pr.Positions != nil || pr.Heads != nil || pr.Quarter != nil || pr.Rows != 0 || pr.Clamp != 0 ||
		pr.Seq != nil || len(pr.Args) > 0 || len(pr.Ptrs) > 0 || pr.Part != c.Part || pr.Of != c.Of {
		return false
	}
	shuffles, exchanges, whole := false, false, false
	for _, nd := range pr.Nodes {
		switch nd.Op {
		case OpFlat, OpThread:
			return false
		case OpGroupSum, OpGroupMax:
			whole = true
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
	// A value stored at a computed index -- a scatter -- has no shape for
	// the consumer's reads at coordinates to lay its workgroup array out by.
	for _, st := range pr.Stores {
		if st.Ax == nil && readsShaped(c, pr.Bufs[st.Buf].Operand) {
			return false
		}
	}
	words, read := 0, false
	for _, b := range c.Bufs {
		if b.Kind == BufShared {
			words += (b.Size*ElemBytes(b) + 3) / 4
		}
	}
	// The lowering gives each group sum, maximum and exchange an array of
	// the workgroup's width (lowergpu): they share the workgroup's memory.
	for _, k := range []*Kernel{c, pr} {
		for _, nd := range k.Nodes {
			if nd.Op == OpGroupSum || nd.Op == OpGroupMax || nd.Op == OpExchange {
				words += c.Group
			}
		}
	}
	for _, b := range pr.Bufs {
		switch {
		case b.Kind == BufShared:
			words += (b.Size*ElemBytes(b) + 3) / 4
		case b.Kind == BufOut && b.Operand.Kind == OperandTable:
			// A table it stores -- a history's pool -- is state the
			// consumer reads beyond what the producer writes: other
			// positions' rows, which a workgroup copy would not hold.
			return false
		case b.Kind != BufOut:
		case readsOperand(pr, b.Operand):
			// An output the producer reads back stays where it reads it.
			return false
		case readsOperand(c, b.Operand):
			n := b.bytes()
			if n == 0 || !sameShapeRead(c, b) {
				return false
			}
			words += int((n + 3) / 4)
			read = true
		}
	}
	return read && words <= p.Words && !readsASlice(pr, c) && (!whole || oneWave(c, 4*words, d) && p.short(pr, c))
}

// readsASlice says a workgroup of c reads only part of what pr writes: a
// load of it whose index differs between c's first and last workgroups,
// each thread at its first trip, so every workgroup making the whole of it
// repeats the producer's work for parts it never reads.
func readsASlice(pr, c *Kernel) bool {
	last := map[int]uint32{}
	for a, x := range c.Space {
		last[a] = uint32(x - 1)
	}
	for _, nd := range c.Nodes {
		if nd.Op != OpLoad || nd.Ax != nil {
			continue
		}
		for _, x := range pr.Bufs {
			if x.Kind != BufOut || x.Operand != c.Bufs[nd.Buf].Operand {
				continue
			}
			lo, ok0 := c.intAt(nd.A, nil)
			hi, ok1 := c.intAt(nd.A, last)
			if ok0 && ok1 && lo != hi {
				return true
			}
		}
	}
	return false
}

// intAt is node i's u32 value at the coordinates coord gives (zero where
// absent), every thread and trip at zero; false where it reads anything
// else.
func (k *Kernel) intAt(i int, coord map[int]uint32) (uint32, bool) {
	nd := k.Nodes[i]
	switch nd.Op {
	case OpConst:
		return uint32(nd.Imm), true
	case OpCoord:
		return coord[nd.A], true
	case OpThread, OpIter:
		return 0, true
	}
	a, ok := k.intAt(nd.A, coord)
	if !ok {
		return 0, false
	}
	b, ok := k.intAt(nd.B, coord)
	if !ok {
		return 0, false
	}
	switch nd.Op {
	case OpAdd:
		return a + b, true
	case OpSub:
		return a - b, true
	case OpMul:
		return a * b, true
	case OpDiv:
		if b == 0 {
			return 0, false
		}
		return a / b, true
	case OpRem:
		if b == 0 {
			return 0, false
		}
		return a % b, true
	case OpMin:
		return min(a, b), true
	case OpMax:
		return max(a, b), true
	case OpShl:
		return a << (b & 31), true
	case OpShr:
		return a >> (b & 31), true
	case OpAnd:
		return a & b, true
	case OpOr:
		return a | b, true
	case OpXor:
		return a ^ b, true
	}
	return 0, false
}

// short says each of pr's spread loops takes at most Trips trips a thread
// of c.
func (p Prologue) short(pr, c *Kernel) bool {
	for _, l := range pr.Loops {
		if l.Parent < 0 && l.Spread && (l.Trip+c.Group-1)/c.Group > p.Trips {
			return false
		}
	}
	return true
}

// oneWave says the device holds every workgroup of c at once, each taking
// shared bytes, with registers past those of a full unit's threads for the
// producer's values: a reduction over the workgroup run in each of them is
// paid once, not once a wave.
func oneWave(c *Kernel, shared int, d Device) bool {
	g := c.groups()
	if d.Units <= 0 || d.Slots <= 0 || c.Group <= 0 {
		return false
	}
	per := (g + d.Units - 1) / d.Units
	return per*c.Group < d.Slots/d.Units && (d.Shared <= 0 || per*shared <= d.Shared)
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

// sameShapeRead says every load of k at coordinates of producer buffer b's
// operand is at b's shape: the workgroup array is laid out by b's.
func sameShapeRead(k *Kernel, b Buf) bool {
	for _, nd := range k.Nodes {
		if x := k.Bufs[nd.Buf]; nd.Op == OpLoad && nd.Ax != nil && x.Operand == b.Operand && !slices.Equal(x.Shape, b.Shape) {
			return false
		}
	}
	return true
}

// readsShaped says k loads operand o at coordinates of its shape.
func readsShaped(k *Kernel, o Operand) bool {
	for _, nd := range k.Nodes {
		if nd.Op == OpLoad && nd.Ax != nil && k.Bufs[nd.Buf].Operand == o {
			return true
		}
	}
	return false
}

// merge is c with pr run first in each of its workgroups.
func (p Prologue) merge(pr, c *Kernel, d Device) *Kernel {
	k := c.clone()
	lead := k.lead(p.Lead)
	k.Passes = append(append([]string(nil), c.Passes...), p.Name())
	// What c reads of pr is workgroup memory: its buffer, and pr's stores
	// to it, one slot of k.
	shared := map[Operand]int{}
	for bi, b := range k.Bufs {
		if _, ok := shared[b.Operand]; ok || b.Kind != BufIn {
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
	// One array an output: c's other buffers of it -- a gather's flat
	// reads beside reads at coordinates -- read the array the producer fills.
	for i, nd := range k.Nodes {
		if nd.Op != OpLoad || k.Bufs[nd.Buf].Kind != BufIn {
			continue
		}
		if at, ok := shared[k.Bufs[nd.Buf].Operand]; ok {
			k.Nodes[i].Buf = at
		}
	}
	// A value the graph hands back is stored whole as well: every workgroup
	// makes the same words, and the caller reads them past every kernel.
	bufOf, heldOf := make([]int, len(pr.Bufs)), make([]int, len(pr.Bufs))
	for i, b := range pr.Bufs {
		heldOf[i] = -1
		if b.Kind == BufOut && b.Operand.Out {
			heldOf[i] = k.bufLike(b)
		}
		switch {
		case b.Kind == BufOut:
			if at, ok := shared[b.Operand]; ok {
				bufOf[i] = at
			} else {
				bufOf[i] = heldOf[i]
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
		held := !st.Barrier && heldOf[st.Buf] >= 0 && heldOf[st.Buf] != bufOf[st.Buf]
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
		if held {
			whole := ns
			whole.Buf, whole.Ax = heldOf[st.Buf], slices.Clone(ns.Ax)
			first = append(first, whole)
		}
	}
	// What the producer wrote is seen by every thread before the kernel's
	// own body reads it.
	first = append(first, Store{Barrier: true, At: -1})
	k.Stores = append(append(lead, first...), k.Stores...)
	return k
}

// lead is up to n touches of the weight words each of k's threads reads
// first: each weight load whose index is arithmetic of the thread, the
// workgroup and the trip of one loop of fixed trips, at that loop's trips
// in order. A loop in a guard -- a sibling's, run by its own workgroups --
// takes its touches in a guard of the same count.
func (k *Kernel) lead(n int) []Store {
	if n <= 0 {
		return nil
	}
	guardOf := func(l Loop) (int, bool) {
		if l.Parent < 0 {
			return -1, true
		}
		g := k.Loops[l.Parent]
		return l.Parent, g.Parent < 0 && g.Kept && g.Trip == 1 && g.Count > 0 && !g.Spread && !g.Range
	}
	type site struct{ load, loop, guard int }
	var sites []site
	for i, nd := range k.Nodes {
		if nd.Op != OpLoad || nd.Ax != nil || k.Bufs[nd.Buf].Operand.Kind != OperandWeight {
			continue
		}
		loop, ok := -1, true
		var walk func(v int)
		walk = func(v int) {
			x := k.Nodes[v]
			switch x.Op {
			case OpConst, OpCoord, OpThread:
				return
			case OpIter:
				l := k.Loops[x.At]
				_, guarded := guardOf(l)
				if loop >= 0 && loop != x.At || !guarded || l.Count != 0 || l.Spread || l.Range || l.Lanes != 0 || l.Copies != 0 || l.Wide != 0 || len(l.Cursors) > 0 {
					ok = false
				}
				loop = x.At
				return
			case OpAdd, OpSub, OpMul, OpDiv, OpRem, OpMin, OpMax, OpShl, OpShr, OpAnd, OpOr, OpXor:
			default:
				ok = false
				return
			}
			for _, a := range k.Operands(&x) {
				walk(*a)
			}
		}
		walk(nd.A)
		if !ok {
			continue
		}
		g := -1
		if loop >= 0 {
			g, _ = guardOf(k.Loops[loop])
		}
		sites = append(sites, site{i, loop, g})
	}
	var out []Store
	type word struct {
		buf, ix int
		imm     int64
		trip    int
	}
	guards := map[int]int{}
	taken := map[int]int{}
	seen := map[word]bool{}
	for t := 0; ; t++ {
		more := false
		for _, s := range sites {
			if taken[s.guard] == n || s.loop < 0 && t > 0 || s.loop >= 0 && t >= k.Loops[s.loop].Trip {
				continue
			}
			more = true
			nd := k.Nodes[s.load]
			if w := (word{nd.Buf, nd.A, nd.Imm, t}); seen[w] {
				continue
			} else {
				seen[w] = true
			}
			memo := map[int]int{}
			var at func(v int) int
			at = func(v int) int {
				if r, ok := memo[v]; ok {
					return r
				}
				x := k.Nodes[v]
				r := v
				switch {
				case x.Op == OpIter:
					r = k.u32(t)
				case x.Op != OpConst && x.Op != OpCoord && x.Op != OpThread && s.loop >= 0 && k.Varies(v, s.loop):
					for _, a := range k.Operands(&x) {
						*a = at(*a)
					}
					r = k.add(x)
				}
				memo[v] = r
				return r
			}
			ix := at(nd.A)
			if nd.Imm != 0 {
				ix = k.bin(OpAdd, ir.U32, ix, k.u32(int(nd.Imm)))
			}
			st := Store{Buf: nd.Buf, Ix: ix, Val: ix, At: -1, Touch: true}
			if s.guard >= 0 {
				g, ok := guards[s.guard]
				if !ok {
					g = k.loop(1, -1)
					k.Loops[g].Kept, k.Loops[g].Count = true, k.Loops[s.guard].Count
					guards[s.guard] = g
				}
				st.At = g
			}
			out = append(out, st)
			taken[s.guard]++
		}
		if !more {
			break
		}
	}
	return out
}
