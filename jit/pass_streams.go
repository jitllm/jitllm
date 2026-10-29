package jit

import (
	"slices"

	"github.com/jitllm/jitllm/jit/ir"
)

// Streams turns a loop over a call's rows around the one loop it holds over
// a contraction's sub-blocks, where each trip of that loop sums words of a
// weight as an integer sum it then scales into a float sum, and every load
// the rows move takes consecutive elements of them: the sub-blocks outside,
// the rows inside, each sum held in its output between sub-blocks (zeroed
// before the first, loaded, added and stored again a sub-block). A trip then
// reads each word's plane along the rows, a stream the hardware follows,
// rather than every plane a line at a time row group after row group. The
// sums add in the same order, so the result is the same. With Apart, each
// position of a whole period is a rows loop of its own: a few planes a loop.
type Streams struct {
	// Dots is the fewest byte dots a trip's integer sum takes: below it a
	// trip is too little work to pay its output's load and store.
	Dots int
	// Most is the longest period of the sub-blocks' iteration a trip runs
	// whole (Periods).
	Most int
	// Apart, where set, is the positions of the period a loop over the
	// rows takes, where a plane's run of a call's rows is at least the
	// core's line fill buffers (Fill), each Line bytes: a loop then keeps
	// them busy on the few planes it reads, where every plane at once
	// would want more lines in flight than there are buffers.
	Apart, Fill, Line int
}

func (Streams) Name() string { return "streams" }

func (p Streams) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if k.Unstreamed {
			return nil, false, nil
		}
		for r, l := range k.Loops {
			in := k.innerLoops(r)
			if l.Lanes > 0 || l.Wide > 0 || l.Range || l.Spread || l.Walk || len(in) != 1 || len(k.innerLoops(in[0])) > 0 {
				continue
			}
			apart := p.Apart
			if l.Trip*4 < p.Fill*p.Line {
				apart = 0
			}
			if m, ok := k.streams(r, in[0], p.Dots, p.Most, apart); ok {
				return []*Kernel{m}, true, nil
			}
		}
		return nil, false, nil
	})
}

// streams is k with row loop r turned around sub-block loop s, where it
// qualifies.
func (k *Kernel) streams(r, s, dots, most, apart int) (*Kernel, bool) {
	ls := k.Loops[s]
	if ls.Lanes > 0 || ls.Wide > 0 || ls.Range || ls.Spread || ls.Walk || ls.Count > 0 || len(k.foldsOf(r)) > 0 {
		return nil, false
	}
	it, sit := k.iterOf(r), k.iterOf(s)
	folds := k.foldsOf(s)
	// One sum: a call of several of a step's rows (RowJam) reads each weight
	// word once for all of them already.
	if it < 0 || sit < 0 || len(folds) != 1 {
		return nil, false
	}
	stored := map[int]int{}
	for i, st := range k.Stores {
		if st.Barrier || st.Run || st.At < 0 {
			continue
		}
		if st.At != r || k.Nodes[st.Val].Op != OpFold || k.Nodes[st.Val].At != s || k.Bufs[st.Buf].Half || k.Bufs[st.Buf].Kind != BufOut {
			return nil, false
		}
		if _, dup := stored[st.Val]; dup {
			return nil, false
		}
		stored[st.Val] = i
	}
	hs := k.Homes()
	for _, f := range folds {
		nd := k.Nodes[f]
		if nd.Type != ir.F32 || !k.sumFold(f) || !k.isZero(nd.B) || k.byteDotRun(nd.A) < dots {
			return nil, false
		}
		if _, ok := stored[f]; !ok {
			return nil, false
		}
		// The fold is read by its step and its store alone.
		for i := range k.Nodes {
			if i == nd.A || i == f {
				continue
			}
			var buf [3]int
			for _, a := range k.args(i, &buf) {
				if a == f {
					return nil, false
				}
			}
		}
	}
	if len(stored) != len(folds) {
		return nil, false
	}
	// Every load the rows move takes consecutive elements of them, and one
	// at least moves with the sub-blocks too: a weight's stream.
	g := newAlg(k)
	a := g.id(atom{node: it})
	unit := func(x int) bool {
		w, rest := g.coef(g.of(x), a)
		if c, ok := w.constant(); !ok || c != 1 {
			return false
		}
		for m := range rest {
			for _, id := range monoIDs(m) {
				for _, nd := range g.nodes(id) {
					if k.readsNode(nd, it) {
						return false
					}
				}
			}
		}
		return true
	}
	last := func(nd Node) (int, bool) {
		if nd.Ax == nil {
			return nd.A, true
		}
		for _, x := range nd.Ax[:len(nd.Ax)-1] {
			if k.readsNode(x, it) {
				return 0, false
			}
		}
		return nd.Ax[len(nd.Ax)-1], true
	}
	stream := false
	for x, nd := range k.Nodes {
		if nd.Op != OpLoad || x >= len(hs) || hs[x] < 0 || !k.within(hs[x], r) || !k.readsNode(x, it) {
			continue
		}
		ix, ok := last(nd)
		if !ok || !unit(ix) {
			return nil, false
		}
		stream = stream || k.readsNode(x, sit)
	}
	if !stream || k.readsAsValue(s, it) {
		return nil, false
	}
	for _, st := range k.Stores {
		if st.At != r || st.Barrier || st.Run {
			continue
		}
		ix := st.Ix
		if st.Ax != nil {
			ix, _ = last(Node{Op: OpLoad, A: st.Ix, Ax: st.Ax})
		}
		if !unit(ix) {
			return nil, false
		}
	}

	m := k.clone()
	// A trip a whole period of the sub-blocks' iteration (Periods): what
	// the sub-block's low bits picked is then a fixed word and offset, and
	// every plane an address the rows advance.
	u := m.period(s, most)
	whole := u >= 2 && ls.Trip%u == 0
	apart = min(apart, u)
	if apart > 0 && whole && u%apart != 0 {
		apart = 0
	}
	var chain []int
	if whole {
		chain = m.unrollBy(s, u)[0]
		if apart == 0 {
			m.compact()
		}
		it, folds, hs = m.iterOf(r), m.foldsOf(s), m.Homes()
		clear(stored)
		for i, st := range m.Stores {
			if !st.Barrier && !st.Run && st.At >= 0 {
				stored[st.Val] = i
			}
		}
	}
	zero := m.u32(0)
	// canon is index x as an offset the rows do not move plus the row's
	// iteration, the form a vector loop reads consecutive elements in.
	canon := func(x int) int {
		if x == it {
			return x
		}
		return m.bin(OpAdd, ir.U32, m.copyOf(x, map[int]int{it: zero}), it)
	}
	for x := range m.Nodes {
		nd := m.Nodes[x]
		if nd.Op != OpLoad || x >= len(hs) || hs[x] < 0 || !m.within(hs[x], r) || !m.readsNode(x, it) {
			continue
		}
		if nd.Ax == nil {
			v := canon(nd.A)
			m.Nodes[x].A = v
		} else {
			ax := append([]int(nil), nd.Ax...)
			ax[len(ax)-1] = canon(ax[len(ax)-1])
			m.Nodes[x].Ax = ax
		}
	}
	var inits []Store
	acc := -1
	for _, f := range folds {
		i := stored[f]
		st := m.Stores[i]
		var ax []int
		ix := canon(st.Ix)
		if st.Ax != nil {
			ax = append([]int(nil), st.Ax...)
			ax[len(ax)-1] = canon(ax[len(ax)-1])
			ix = st.Ix
		}
		acc = m.add(Node{Op: OpLoad, Type: ir.F32, Buf: st.Buf, A: ix, Ax: ax})
		step := m.Nodes[f].A
		for x := range m.Nodes {
			if x == f || m.Nodes[x].Op == OpFold {
				continue
			}
			for _, p := range m.Operands(&m.Nodes[x]) {
				if *p == f {
					*p = acc
				}
			}
		}
		m.Stores[i].Ix, m.Stores[i].Ax, m.Stores[i].Val = ix, ax, step
		inits = append(inits, Store{Buf: st.Buf, Ix: ix, Ax: ax, Val: m.Nodes[f].B})
	}
	// The rows' loop inside the sub-blocks', and a loop over the same rows
	// zeroing the sums before it.
	parent := m.Loops[r].Parent
	m.Loops[s].Parent, m.Loops[r].Parent = parent, s
	z := m.insertLoop(0, Loop{Trip: m.Loops[r].Trip, Count: m.Loops[r].Count, Parent: parent})
	r++
	zit := m.add(Node{Op: OpIter, At: z, Type: ir.U32})
	for j := range inits {
		in := &inits[j]
		in.At = z
		if in.Ax != nil {
			ax := make([]int, len(in.Ax))
			for x, v := range in.Ax {
				ax[x] = m.copyOf(v, map[int]int{it: zit})
			}
			in.Ax = ax
		}
		in.Ix = m.copyOf(in.Ix, map[int]int{it: zit})
	}
	m.Stores = append(inits, m.Stores...)
	for _, f := range folds {
		m.Nodes[f] = Node{Op: OpConst, Type: ir.F32}
	}
	if whole && apart > 0 {
		m.apart(s+1, r, apart, chain, acc)
	}
	return m, true
}

// apart cuts rows loop r into a loop for each g positions of chain, each
// summing on from acc where the one before stored.
func (k *Kernel) apart(s, r, g int, chain []int, acc int) {
	U := ir.U32
	rits := k.itersOf(r)
	var body Store
	var stores []Store
	for _, st := range k.Stores {
		if st.At == r {
			body = st
		} else {
			stores = append(stores, st)
		}
	}
	for n := range len(chain) / g {
		end := chain[(n+1)*g-1]
		st := body
		if n == 0 {
			st.Val = end
			stores = append(stores, st)
			continue
		}
		c := k.insertLoop(r+n, Loop{Trip: k.Loops[r].Trip, Count: k.Loops[r].Count, Parent: s})
		if s >= c {
			s++
		}
		it := k.add(Node{Op: OpIter, At: c, Type: U})
		sub := map[int]int{}
		for _, o := range rits {
			sub[o] = it
		}
		memo := map[int]int{}
		st.At = c
		st.Ix = k.copyMemo(st.Ix, sub, memo)
		if st.Ax != nil {
			ax := make([]int, len(st.Ax))
			for x, v := range st.Ax {
				ax[x] = k.copyMemo(v, sub, memo)
			}
			st.Ax = ax
		}
		sub[chain[n*g-1]] = k.copyMemo(acc, sub, memo)
		st.Val = k.copyOf(end, sub)
		stores = append(stores, st)
	}
	k.Stores = stores
	k.compact()
}

// byteDotRun is the most byte dots in one chain of integer sums node x
// reads: a trip's integer sub-reduction.
func (k *Kernel) byteDotRun(x int) int {
	best := 0
	seen := map[int]bool{}
	var walk func(i int)
	walk = func(i int) {
		if seen[i] {
			return
		}
		seen[i] = true
		nd := k.Nodes[i]
		if nd.Op == OpFold || nd.Op == OpLoad {
			return
		}
		if nd.Op.byteDot() {
			n := 0
			for c := i; k.Nodes[c].Op.byteDot(); c = k.Nodes[c].C {
				n++
			}
			best = max(best, n)
		}
		var buf [3]int
		for _, a := range k.args(i, &buf) {
			walk(a)
		}
	}
	walk(x)
	return best
}

// StreamWords loads the words a loop of rows reads that the sub-blocks'
// loop around it loads, in the rows' loop, each time a trip reads them:
// where the trips of a sub-block's rows (Streams) read many such words, the
// vector registers cannot hold them all through the rows, and a load from
// the cache is cheaper than a register's spill. A loop around rows that
// carries sums keeps them (Jam's form holds them a trip).
type StreamWords struct {
	// Least is the fewest such words a loop reads before they move.
	Least int
}

func (StreamWords) Name() string { return "streamwords" }

func (p StreamWords) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		m := k.clone()
		moved := false
		for c, l := range k.Loops {
			o := l.Parent
			if o < 0 || k.hasInner(c) || len(k.foldsOf(o)) > 0 || !k.storesAt(c) {
				continue
			}
			moved = m.streamWords(c, o, p.Least) || moved
		}
		if !moved {
			return nil, false, nil
		}
		m.compact()
		return []*Kernel{m}, true, nil
	})
}

// streamWords makes in loop c the values loop o makes from the words it
// loads that c reads as data, where they are at least least: each such
// value's loads and float arithmetic copied into c, the loads read there,
// c's readers reading the copies. Addresses and integer values stay where
// they are.
func (k *Kernel) streamWords(c, o, least int) bool {
	hs := k.Homes()
	written := map[int]bool{}
	for _, s := range k.Stores {
		if !s.Barrier && !s.Run {
			written[s.Buf] = true
		}
	}
	in := func(i int) bool { return i < len(hs) && hs[i] >= 0 && k.within(hs[i], c) }
	// c sums into its output: it loads what it stores.
	sums := false
	for x, nd := range k.Nodes {
		sums = sums || nd.Op == OpLoad && written[nd.Buf] && in(x)
	}
	if !sums {
		return false
	}
	ok := map[int]bool{}
	var movable func(x int) bool
	movable = func(x int) bool {
		if v, seen := ok[x]; seen {
			return v
		}
		ok[x] = false
		nd := k.Nodes[x]
		v := false
		switch {
		case x >= len(hs) || hs[x] != o:
			v = true
		case nd.Op == OpLoad:
			v = nd.At == 0 && !written[nd.Buf] && !k.Bufs[nd.Buf].Shared
		case nd.Op == OpFold || nd.Op == OpIter || nd.Type != ir.F32 && nd.Op != OpBroadcast:
			v = true
		default:
			v = true
			for _, a := range k.Operands(&nd) {
				v = v && movable(*a)
			}
		}
		ok[x] = v
		return v
	}
	// moves reports whether x is a value c's copy makes: one o makes from
	// loads.
	moves := func(x int) bool {
		nd := k.Nodes[x]
		return x < len(hs) && hs[x] == o && (nd.Op == OpLoad || nd.Op == OpBroadcast || nd.Type == ir.F32 && nd.Op != OpFold && nd.Op != OpIter) && movable(x)
	}
	roots := map[int]bool{}
	for i, nd := range k.Nodes {
		if !in(i) || nd.Op == OpLoad {
			continue
		}
		for _, a := range k.Operands(&nd) {
			if moves(*a) {
				roots[*a] = true
			}
		}
	}
	if len(roots) < least {
		return false
	}
	memo := map[int]int{}
	var copyIn func(x int) int
	copyIn = func(x int) int {
		if !moves(x) {
			return x
		}
		if v, done := memo[x]; done {
			return v
		}
		nd := k.Nodes[x]
		if nd.Op == OpLoad {
			nd.At = c + 1
		} else {
			for _, a := range k.Operands(&nd) {
				*a = copyIn(*a)
			}
		}
		v := k.add(nd)
		memo[x] = v
		return v
	}
	for i := range hs {
		if !in(i) || k.Nodes[i].Op == OpLoad {
			continue
		}
		ops := k.Operands(&k.Nodes[i])
		for j := range ops {
			if roots[*ops[j]] {
				v := copyIn(*ops[j])
				ops = k.Operands(&k.Nodes[i])
				*ops[j] = v
			}
		}
	}
	return true
}

// streamed reports whether a pass turned k's rows around its sub-blocks
// (Streams).
func (k *Kernel) streamed() bool { return slices.Contains(k.Passes, Streams{}.Name()) }
