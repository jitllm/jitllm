package jit

import (
	"sort"

	"github.com/jitllm/jitllm/jit/ir"
)

// Order writes a kernel as the sequence its code runs: the argument words
// first, in the order of their fields but a vector loop's counts last, a
// value no loop moves before the first loop that needs it, and in each loop
// its values in the order they depend on each other, each inner loop where
// a value first needs what it carries, and its stores; the sums a loop
// carries start just before it, and a choice another sum carries a value
// with steps once the trip's other reads of it have run. A vector's lanes
// rearranged are made just after it is loaded. A float zero is made where it is read. A loop advances the
// address it stores through itself first, then its pointers in the order
// its code first reaches them. A pair's heads run statement by statement
// side by side, each head's statements from where it reads a finished sum
// on run whole, head after head.
type Order struct{}

func (Order) Name() string { return "order" }

func (p Order) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if k.Args == nil || k.Seq != nil {
			return nil, false, nil
		}
		m := k.clone()
		o := orderer{k: m, done: map[int]bool{}, loopDone: map[int]bool{}, home: map[int]int{}, depths: map[int]int{}, words: m.vectorWords()}
		o.run()
		m.Seq = m.interleave(o.seq)
		m.orderCursors()
		return []*Kernel{m}, true, nil
	})
}

type orderer struct {
	k        *Kernel
	seq      []Entry
	done     map[int]bool
	loopDone map[int]bool
	home     map[int]int
	depths   map[int]int
	words    map[int]bool
	// after is, for a fold of the loop being written that other values of
	// its trip read, those values: its step runs once they have.
	after   map[int][]int
	scopeOf int
}

func (o *orderer) run() {
	k := o.k
	live := k.live()
	vecCount := map[int]bool{}
	for _, l := range k.Loops {
		if l.Count > 0 && (l.Lanes > 0 || o.isTail(l)) {
			vecCount[l.Count-1] = true
		}
	}
	var params, late []int
	for i, nd := range k.Nodes {
		if nd.Op != OpParam || !live[i] && !o.isBase(i) {
			continue
		}
		if vecCount[i] {
			late = append(late, i)
		} else {
			params = append(params, i)
		}
	}
	sort.SliceStable(params, func(a, b int) bool { return k.Nodes[params[a]].Imm < k.Nodes[params[b]].Imm })
	sort.SliceStable(late, func(a, b int) bool { return o.countLoop(late[a]) < o.countLoop(late[b]) })
	for _, i := range append(params, late...) {
		o.node(i)
	}
	hs := k.homes()
	for i := range k.Nodes {
		if live[i] {
			o.home[i] = hs[i]
		}
	}
	o.scope(-1)
}

func (o *orderer) isTail(l Loop) bool {
	for _, v := range o.k.Loops {
		if v.Lanes > 0 && v.Parent == l.Parent && v.Range == l.Range && v.Count > 0 && l.Count > 0 {
			return true
		}
	}
	return false
}

func (o *orderer) isBase(i int) bool {
	for _, b := range o.k.Bufs {
		if b.Base == i+1 {
			return true
		}
	}
	return false
}

func (o *orderer) countLoop(i int) int {
	for c, l := range o.k.Loops {
		if l.Count == i+1 {
			return c
		}
	}
	return len(o.k.Loops)
}

func (o *orderer) node(i int) {
	if o.done[i] {
		return
	}
	o.done[i] = true
	o.seq = append(o.seq, Entry{Kind: EntryNode, I: i})
	if o.k.Nodes[i].Op == OpLoad {
		for j, nd := range o.k.Nodes {
			if (nd.Op == OpDupEven || nd.Op == OpDupOdd || nd.Op == OpSwapPairs) && nd.A == i && o.home[j] == o.home[i] {
				o.node(j)
			}
		}
	}
	for f, rs := range o.after {
		ready := true
		for _, r := range rs {
			ready = ready && o.done[r]
		}
		if ready {
			delete(o.after, f)
			o.need(o.k.Nodes[f].A, o.scopeOf)
		}
	}
}

// need emits node i and what it reads, within scope c: a value of an inner
// loop emits that loop.
func (o *orderer) need(i, c int) {
	if o.done[i] {
		return
	}
	k := o.k
	nd := k.Nodes[i]
	if nd.Op == OpConst && !o.words[i] && !(nd.Type == ir.F32 && nd.Imm == 0) || nd.Op == OpIter {
		return
	}
	if nd.Op == OpFold {
		if nd.At != c {
			o.loop(o.childOf(nd.At, c))
			return
		}
		o.done[i] = true
		return
	}
	if h := o.home[i]; h != c && k.within(h, c) && h >= 0 {
		o.loop(o.childOf(h, c))
		return
	}
	for _, a := range k.evalArgs(&nd, o.depths) {
		o.need(*a, c)
	}
	o.node(i)
}

// childOf is the loop directly inside c that holds loop h.
func (o *orderer) childOf(h, c int) int {
	for h >= 0 && o.k.Loops[h].Parent != c {
		h = o.k.Loops[h].Parent
	}
	return h
}

// scope emits the values homed in loop c that its inner loops need, the
// inner loops in order, then c's stores.
func (o *orderer) scope(c int) {
	k := o.k
	var inner []int
	for d, l := range k.Loops {
		if l.Parent == c {
			inner = append(inner, d)
		}
	}
	for _, d := range inner {
		for _, f := range k.foldsOf(d) {
			if b := k.Nodes[f].B; k.Nodes[b].Op != OpFold && !k.startsOnly(b) && !k.isZero(b) {
				o.need(b, c)
			}
		}
	}
	for _, d := range inner {
		for i := range k.Nodes {
			if h, ok := o.home[i]; ok && h == c && k.Nodes[i].Op != OpFold && !k.isZero(i) && o.reads(d, i) && !o.counts(d)[i] {
				o.need(i, c)
			}
		}
		o.countsFor(c, d)
		o.loop(d)
	}
	outer, scope := o.after, o.scopeOf
	o.after, o.scopeOf = map[int][]int{}, c
	carried := map[int]bool{}
	for _, f := range k.foldsOf(c) {
		if v := k.carriedBy(f); v >= 0 {
			carried[v] = true
		}
	}
	for _, f := range k.foldsOf(c) {
		if !carried[f] {
			continue
		}
		step := k.Nodes[f].A
		for i, nd := range k.Nodes {
			if h, ok := o.home[i]; !ok || h != c || i == step || o.done[i] {
				continue
			}
			for _, a := range k.args(&nd) {
				if *a == f {
					o.after[f] = append(o.after[f], i)
					break
				}
			}
		}
	}
	for _, f := range k.foldsOf(c) {
		if _, wait := o.after[f]; !wait {
			o.need(k.Nodes[f].A, c)
		}
	}
	for f := range o.after {
		o.need(k.Nodes[f].A, c)
	}
	o.after, o.scopeOf = outer, scope
	for i, s := range k.Stores {
		if s.At != c || s.Barrier || s.Run {
			continue
		}
		for _, a := range k.storeArgs(&s) {
			o.need(*a, c)
		}
		o.seq = append(o.seq, Entry{Kind: EntryStore, I: i})
	}
}

// reads reports whether loop d's code reads node i: a node homed in d or a
// loop inside it, or one of d's stores, reaching it.
func (o *orderer) reads(d, i int) bool {
	k := o.k
	for j, nd := range k.Nodes {
		if h, ok := o.home[j]; ok && h >= 0 && k.within(h, d) {
			for _, a := range k.args(&nd) {
				if *a == i {
					return true
				}
			}
		}
	}
	for _, s := range k.Stores {
		if s.At >= 0 && k.within(s.At, d) && !s.Barrier && !s.Run {
			for _, a := range k.storeArgs(&s) {
				if *a == i {
					return true
				}
			}
		}
	}
	return false
}

// counts is the nodes the counts of loop d and the loops in it are made
// from.
func (o *orderer) counts(d int) map[int]bool {
	k := o.k
	out := map[int]bool{}
	var walk func(i int)
	walk = func(i int) {
		if out[i] {
			return
		}
		out[i] = true
		nd := k.Nodes[i]
		for _, a := range k.args(&nd) {
			walk(*a)
		}
	}
	for e, l := range k.Loops {
		if l.Count > 0 && k.within(e, d) {
			walk(l.Count - 1)
		}
	}
	return out
}

// countsFor emits, in scope c, the counts of loop d and the loops in it,
// in the order of their loops, a count made by one operation from a value a
// later count still reads after that count.
func (o *orderer) countsFor(c, d int) {
	k := o.k
	var cs []int
	for _, e := range k.preorder() {
		if l := k.Loops[e]; l.Count > 0 && k.within(e, d) {
			if n := l.Count - 1; o.home[n] == c && !o.done[n] {
				cs = append(cs, n)
			}
		}
	}
	var front, last []int
	for j, x := range cs {
		n := k.Nodes[x]
		deferred := false
		if arity(n.Op) == 2 && k.Nodes[n.B].Op == OpConst {
			for _, later := range cs[j+1:] {
				if later != x && o.readsNode(later, n.A) {
					deferred = true
				}
			}
		}
		if deferred {
			last = append(last, x)
		} else {
			front = append(front, x)
		}
	}
	for _, n := range append(front, last...) {
		o.need(n, c)
	}
}

// readsNode reports whether node i is computed from node x.
func (o *orderer) readsNode(i, x int) bool {
	if i == x {
		return true
	}
	nd := o.k.Nodes[i]
	if nd.Op == OpFold {
		return false
	}
	for _, a := range o.k.args(&nd) {
		if o.readsNode(*a, x) {
			return true
		}
	}
	return false
}

func (o *orderer) loop(d int) {
	if d < 0 || o.loopDone[d] {
		return
	}
	o.loopDone[d] = true
	for _, f := range o.k.foldsOf(d) {
		o.done[f] = true
		o.seq = append(o.seq, Entry{Kind: EntryNode, I: f})
	}
	o.seq = append(o.seq, Entry{Kind: EntryLoop, I: d})
	o.scope(d)
	o.seq = append(o.seq, Entry{Kind: EntryEnd, I: d})
}

// orderCursors orders each loop's cursors: a buffer the loop stores to
// through its own address first, then in the order the sequence first
// reaches each pointer.
func (k *Kernel) orderCursors() {
	first := map[[2]int]int{}
	for pos, e := range k.Seq {
		var b, ptr int
		switch e.Kind {
		case EntryNode:
			nd := k.Nodes[e.I]
			if nd.Op != OpLoad {
				continue
			}
			b, ptr = nd.Buf, nd.Ptr
		case EntryStore:
			s := k.Stores[e.I]
			b, ptr = s.Buf, s.Ptr
		default:
			continue
		}
		key := [2]int{k.Bufs[b].Base, ptr}
		if _, ok := first[key]; !ok {
			first[key] = pos
		}
	}
	for c := range k.Loops {
		cs := k.Loops[c].Cursors
		rank := func(cu Cursor) (int, int) {
			stored := 1
			for _, s := range k.Stores {
				if s.At == c && !s.Barrier && !s.Run && s.Ptr == 0 && cu.Ptr == 0 && k.Bufs[s.Buf].Base == k.Bufs[cu.Buf].Base {
					stored = 0
				}
			}
			at, ok := first[[2]int{k.Bufs[cu.Buf].Base, cu.Ptr}]
			if !ok {
				at = len(k.Seq)
			}
			return stored, at
		}
		sort.SliceStable(cs, func(i, j int) bool {
			si, ai := rank(cs[i])
			sj, aj := rank(cs[j])
			if si != sj {
				return si < sj
			}
			return ai < aj
		})
	}
}

// interleave puts each statement of a pair's later head beside its first
// head's twin, but after the run of finishing statements -- those reading a
// sum after its loop -- a twin belongs to.
func (k *Kernel) interleave(seq []Entry) []Entry {
	cp := k.copies()
	later := false
	for _, c := range cp {
		later = later || c > 0
	}
	if !later {
		return seq
	}
	tw := k.twins(cp)
	of := map[int][]int{}
	for i, t := range tw {
		if t >= 0 {
			of[t] = append(of[t], i)
		}
	}
	for t := range of {
		sort.Slice(of[t], func(a, b int) bool { return cp[of[t][a]] < cp[of[t][b]] })
	}
	storeOf := map[int][]int{}
	for i, s := range k.Stores {
		if s.Barrier || s.Run || k.Bufs[s.Buf].Operand.Copy == 0 {
			continue
		}
		for j, o := range k.Stores {
			if k.Bufs[o.Buf].Operand.Copy == 0 && !o.Barrier && !o.Run && tw[s.Val] == o.Val && o.At == s.At {
				storeOf[j] = append(storeOf[j], i)
				break
			}
		}
	}
	hs := k.homes()
	finishing := map[int]bool{}
	var fin func(i int) bool
	fin = func(i int) bool {
		if v, ok := finishing[i]; ok {
			return v
		}
		finishing[i] = false
		nd := k.Nodes[i]
		if nd.Op == OpFold {
			return false
		}
		v := false
		for _, a := range k.args(&nd) {
			x := k.Nodes[*a]
			if x.Op == OpFold && x.At != hs[i] && !(hs[i] >= 0 && k.within(hs[i], x.At)) || fin(*a) {
				v = true
			}
		}
		finishing[i] = v
		return v
	}
	isLater := func(e Entry) bool {
		switch e.Kind {
		case EntryNode:
			return cp[e.I] > 0
		case EntryStore:
			s := k.Stores[e.I]
			return !s.Barrier && !s.Run && k.Bufs[s.Buf].Operand.Copy > 0
		}
		return false
	}
	isFinish := func(e Entry) bool {
		switch e.Kind {
		case EntryNode:
			return fin(e.I)
		case EntryStore:
			s := k.Stores[e.I]
			return !s.Barrier && !s.Run && fin(s.Val)
		}
		return false
	}
	var out, pending []Entry
	flush := func() {
		out = append(out, pending...)
		pending = nil
	}
	for _, e := range seq {
		if isLater(e) {
			continue
		}
		if len(pending) > 0 && !isFinish(e) {
			flush()
		}
		out = append(out, e)
		var copies []Entry
		switch e.Kind {
		case EntryNode:
			for _, c := range of[e.I] {
				copies = append(copies, Entry{Kind: EntryNode, I: c})
			}
		case EntryStore:
			for _, c := range storeOf[e.I] {
				copies = append(copies, Entry{Kind: EntryStore, I: c})
			}
		}
		if isFinish(e) {
			pending = append(pending, copies...)
		} else {
			out = append(out, copies...)
		}
	}
	flush()
	return out
}

// startsOnly reports whether node i is a load, at the buffer's own address
// or of constants, only sums start from: the sum's start loads it.
func (k *Kernel) startsOnly(i int) bool {
	if nd := k.Nodes[i]; nd.Op != OpLoad || nd.Ptr > 0 && k.Bufs[nd.Buf].Operand.Kind != OperandConsts {
		return false
	}
	for j, x := range k.Nodes {
		for _, a := range k.args(&x) {
			if *a == i && !(x.Op == OpFold && x.B == i && j != i) {
				return false
			}
		}
	}
	return true
}

// isZero reports whether node i is a float zero, made where it is read.
func (k *Kernel) isZero(i int) bool {
	nd := k.Nodes[i]
	return nd.Op == OpConst && nd.Type == ir.F32 && nd.Imm == 0
}
