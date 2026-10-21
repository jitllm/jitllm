package jit

import (
	"maps"
	"math"
	"math/bits"

	"github.com/jitllm/jitllm/jit/ir"
)

// Vectorize runs an innermost loop that reads and writes consecutive
// elements of its iteration Lanes iterations a trip, one in each lane, and
// the trips that do not fill a vector after it in a loop of one a trip. A
// sum the loop carries becomes a sum in each lane, the tail adding into its
// first lane, its term kept to that lane where it is not zero in the others
// (a store of the term stores the kept term, the same in that lane),
// and the lanes are summed where the sum is read. A loop that
// stores a pair of values at 2i and 2i+1 stores the two vectors' lanes
// zipped, the low pairs then the high.
type Vectorize struct {
	Lanes int
	// Pairs, where set, are the registers of a complex product's values:
	// the second operand, its pairs' first lanes, their second lanes, the
	// row and its pairs swapped.
	Pairs [5]int
	// Carry, where set, are the registers of a carried extreme's tail
	// value, the tail's carried value and the copy of the lanes' extremes
	// the comparison after the loops reads.
	Carry []int
}

func (Vectorize) Name() string { return "vectorize" }

func (p Vectorize) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		m := k.clone()
		changed := false
		for c := 0; c < len(m.Loops); c++ {
			if cp, ok := m.complexPairOf(c, p.Lanes); ok {
				m.vectorizeComplex(c, p.Lanes, cp, p.Pairs)
				changed = true
				continue
			}
			if m.vectorizable(c) {
				m.vectorize(c, p.Lanes, p.Carry)
				changed = true
				c++
			}
		}
		if !changed {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

func (k *Kernel) vectorizable(c int) bool {
	l := k.Loops[c]
	if l.Lanes > 0 || l.Wide > 0 || len(k.innerLoops(c)) > 0 || k.iterOf(c) < 0 {
		return false
	}
	folds := k.foldsOf(c)
	for _, f := range folds {
		if k.Nodes[f].Type == ir.F32 && (k.sumFold(f) || k.extremeTerm(f) >= 0) || k.carriedBy(f) >= 0 {
			continue
		}
		return false
	}
	it := k.iterOf(c)
	stores := 0
	_, paired := k.pairStores(c)
	for _, s := range k.Stores {
		if s.At != c || s.Barrier || s.Run {
			continue
		}
		stores++
		if !paired && !k.consecutive(s.Buf, s.Ix, s.Ax, it, c) {
			return false
		}
	}
	if stores == 0 && len(folds) == 0 {
		return false
	}
	for i, nd := range k.Nodes {
		if nd.Op == OpLoad && k.varies(i, c) && !k.consecutive(nd.Buf, nd.A, nd.Ax, it, c) {
			return false
		}
	}
	return true
}

// consecutive reports whether an access of buffer b at flat index ix, or at
// coordinates ax of its shape, takes consecutive elements over loop c's
// iterations it: the last coordinate the iteration, the others fixed in c.
func (k *Kernel) consecutive(b, ix int, ax []int, it, c int) bool {
	if ax != nil {
		for _, a := range ax[:len(ax)-1] {
			if k.varies(a, c) {
				return false
			}
		}
		last := ax[len(ax)-1]
		if last == it {
			return true
		}
		nd := k.Nodes[last]
		return nd.Op == OpAdd && (nd.B == it && !k.varies(nd.A, c) || nd.A == it && !k.varies(nd.B, c))
	}
	if ix == it {
		return true
	}
	nd := k.Nodes[ix]
	return nd.Op == OpAdd && (nd.B == it && !k.varies(nd.A, c) || nd.A == it && !k.varies(nd.B, c))
}

func (k *Kernel) vectorize(c, L int, carry []int) {
	l := k.Loops[c]
	U := ir.U32
	shift := bits.TrailingZeros(uint(L))
	var tailCount, done int
	tail := Loop{Parent: l.Parent, Range: l.Range, Of: c + 1}
	if l.Of > 0 {
		tail.Of = l.Of
	}
	if l.Count > 0 {
		n := l.Count - 1
		vec := k.bin(OpShr, U, n, k.u32(shift))
		k.Loops[c].Count = vec + 1
		tailCount = k.bin(OpAnd, U, n, k.u32(L-1))
		tail.Count = tailCount + 1
		done = k.bin(OpShl, U, vec, k.u32(shift))
	} else {
		tail.Trip = l.Trip % L
		k.Loops[c].Trip = l.Trip / L
		done = k.u32(l.Trip / L * L)
	}
	k.Loops[c].Lanes = L
	k.broadcastInto(c)
	t := k.insertLoop(c+1, tail)
	ti := k.add(Node{Op: OpIter, At: t, Type: U})
	sub := map[int]int{k.iterOf(c): k.bin(OpAdd, U, done, ti)}
	carries, kept := map[int]int{}, map[int]int{}
	for _, f := range k.foldsOf(c) {
		if v := k.carriedBy(f); v >= 0 {
			carries[v] = f
		}
	}
	for v, cf := range carries {
		k.vectorizeCarry(c, t, v, cf, sub, carry)
	}
	for _, f := range k.foldsOf(c) {
		if _, ok := carries[f]; ok || k.carriedBy(f) >= 0 {
			continue
		}
		if k.extremeTerm(f) >= 0 {
			k.vectorizeExtreme(c, t, f, sub)
			continue
		}
		ft := k.add(Node{Op: OpFold, At: t, Type: k.Nodes[f].Type, B: f})
		step := k.Nodes[f].A
		sub[f] = ft
		k.Nodes[ft].A = k.copyKeep(step, sub)
		if st := &k.Nodes[k.Nodes[ft].A]; st.Op == OpAdd && st.A == ft && !k.zeroKept(st.B, t) {
			term := st.B
			st.B = k.add(Node{Op: OpFirstLane, Type: ir.F32, A: term, B: k.add(Node{Op: OpConst, Type: ir.F32})})
			kept[term] = st.B
		}
		h := k.add(Node{Op: OpHSum, Type: k.Nodes[f].Type, A: ft})
		hs := k.homes()
		for i := range k.Nodes {
			if i == step || i == ft || i == h || i < len(hs) && hs[i] >= 0 && (k.within(hs[i], c) || k.within(hs[i], t)) {
				continue
			}
			for _, a := range k.args(&k.Nodes[i]) {
				if *a == f {
					*a = h
				}
			}
		}
		for i := range k.Stores {
			if k.Stores[i].Val == f {
				k.Stores[i].Val = h
			}
		}
	}
	var stores []Store
	last := -1
	for i, s := range k.Stores {
		if s.At != c || s.Barrier || s.Run {
			continue
		}
		last = i
		n := s
		n.At = t
		n.Val = k.copyKeep(s.Val, sub)
		if v, ok := kept[n.Val]; ok {
			n.Val = v
		}
		if s.Ax != nil {
			n.Ax = make([]int, len(s.Ax))
			for j, a := range s.Ax {
				n.Ax[j] = k.copyKeep(a, sub)
			}
		} else {
			n.Ix = k.copyKeep(s.Ix, sub)
		}
		stores = append(stores, n)
	}
	k.Stores = append(k.Stores[:last+1:last+1], append(stores, k.Stores[last+1:]...)...)
	if pair, ok := k.pairStores(c); ok {
		a, b := k.Stores[pair[0]], k.Stores[pair[1]]
		lo := Store{Buf: a.Buf, At: c, Ix: a.Ix, Val: k.add(Node{Op: OpZip, Type: ir.F32, A: a.Val, B: b.Val})}
		hi := Store{Buf: a.Buf, At: c, Ix: k.bin(OpAdd, U, a.Ix, k.u32(L)), Val: k.add(Node{Op: OpZip, Type: ir.F32, A: a.Val, B: b.Val, Imm: 1})}
		k.Stores[pair[0]], k.Stores[pair[1]] = lo, hi
	}
}

// zeroKept reports whether node i is zero in the lanes past the first
// where the loads of tail t, which fill the first lane alone, are.
func (k *Kernel) zeroKept(i, t int) bool {
	nd := k.Nodes[i]
	switch nd.Op {
	case OpLoad:
		return k.varies(i, t)
	case OpConst:
		return nd.Imm == 0
	case OpMul:
		return k.zeroKept(nd.A, t) || k.zeroKept(nd.B, t)
	case OpAdd, OpSub:
		return k.zeroKept(nd.A, t) && k.zeroKept(nd.B, t)
	case OpFma:
		return (k.zeroKept(nd.A, t) || k.zeroKept(nd.B, t)) && k.zeroKept(nd.C, t)
	}
	return false
}

// vectorizeExtreme continues fold f, the largest or least of loop c's
// terms, in tail t with each term in every lane, and takes its lanes'
// largest or least where it is read after.
func (k *Kernel) vectorizeExtreme(c, t, f int, sub map[int]int) {
	st := k.Nodes[k.Nodes[f].A]
	ft := k.add(Node{Op: OpFold, At: t, Type: ir.F32, B: f})
	sub[f] = ft
	term := k.add(Node{Op: OpBroadcast, Type: ir.F32, A: k.copyOf(k.extremeTerm(f), sub)})
	a, b := term, ft
	if st.A == f {
		a, b = ft, term
	}
	k.Nodes[ft].A = k.bin(st.Op, ir.F32, a, b)
	lanes := OpHMax
	if st.Op == OpMin {
		lanes = OpHMin
	}
	h := k.add(Node{Op: lanes, Type: ir.F32, A: ft})
	k.readAfter(c, t, map[int]int{f: h}, h)
}

// vectorizeCarry runs fold v, the largest or least of loop c's terms, and
// fold cf, the value v carries, in each lane, and in tail t on folds of
// their own from v's start, cf from each lane's first term's: after the loops the lanes' extreme, the least
// carried value of the lanes that hold it, and the tail's where its own is
// beyond it. The lanes' extreme is taken in place, the comparison reading
// a copy of the lanes' values made by their own combine; the extreme and
// the value after the loops share the lanes' register, the carried values
// after them the carried lanes'.
func (k *Kernel) vectorizeCarry(c, t, v, cf int, sub map[int]int, carry []int) {
	ct := k.Nodes[cf].Type
	op := k.Nodes[k.Nodes[v].A].Op
	vt := k.add(Node{Op: OpFold, At: t, Type: ir.F32, B: k.Nodes[v].B})
	cft := k.add(Node{Op: OpFold, At: t, Type: ct, B: k.add(Node{Op: OpConst, Type: ct})})
	if len(carry) == 3 {
		k.Nodes[vt].Hint, k.Nodes[cft].Hint = carry[0]+1, carry[1]+1
	}
	sub[v], sub[cf] = vt, cft
	from := maps.Clone(sub)
	k.Nodes[vt].A = k.copyKeep(k.Nodes[v].A, sub)
	k.Nodes[cft].A = k.copyKeep(k.Nodes[cf].A, sub)
	if len(carry) == 3 {
		label := 1
		for _, nd := range k.Nodes {
			label = max(label, nd.Share+1)
		}
		hs := k.homes()
		for x, y := range sub {
			if _, ok := from[x]; !ok && k.Nodes[x].Op == OpLoad && x < len(hs) && hs[x] == c {
				k.Nodes[x].Share, k.Nodes[y].Share = label, label
				label++
			}
		}
	}
	lanes0 := k.add(Node{Op: OpLanes, Type: ir.U32, Imm: int64(k.Loops[c].Lanes)})
	k.Nodes[cf].B = k.copyOf(k.Nodes[k.Nodes[cf].A].B, map[int]int{k.iterOf(c): lanes0})
	lanes, least := OpHMax, int64(math.MaxInt32)
	if op == OpMin {
		lanes = OpHMin
	}
	first := len(k.Nodes)
	kept := v
	if len(carry) == 3 {
		kept = k.add(Node{Op: op, Type: ir.F32, A: v, B: v, Hint: carry[2] + 1})
	}
	ext := k.add(Node{Op: lanes, Type: ir.F32, A: v})
	m := k.add(Node{Op: OpBroadcast, Type: ir.F32, A: ext})
	held := k.add(Node{Op: OpSelect, Type: ct, A: k.add(Node{Op: OpEq, Type: ir.Pred, A: kept, B: m}), B: cf,
		C: k.add(Node{Op: OpConst, Type: ct, Imm: least})})
	ci := k.add(Node{Op: OpHMin, Type: ct, A: held})
	val := k.bin(op, ir.F32, vt, m)
	car := k.add(Node{Op: OpSelect, Type: ct, A: k.add(Node{Op: OpLess, Type: ir.Pred, A: m, B: vt}), B: cft, C: ci})
	if len(carry) == 3 {
		label := 1
		for _, nd := range k.Nodes {
			label = max(label, nd.Share+1)
		}
		for _, i := range []int{v, ext, m, val} {
			k.Nodes[i].Share = label
		}
		for _, i := range []int{cf, held, ci, car} {
			k.Nodes[i].Share = label + 1
		}
	}
	k.readAfter(c, t, map[int]int{v: val, cf: car}, first)
}

// readAfter points each reader of a fold in sub outside loops c and t, the
// nodes from first on aside, at its replacement.
func (k *Kernel) readAfter(c, t int, sub map[int]int, first int) {
	hs := k.homes()
	for i := range k.Nodes {
		if i >= first || i < len(hs) && hs[i] >= 0 && (k.within(hs[i], c) || k.within(hs[i], t)) {
			continue
		}
		if nd := k.Nodes[i]; nd.Op == OpFold && (nd.At == c || nd.At == t) {
			continue
		}
		for _, a := range k.args(&k.Nodes[i]) {
			if r, ok := sub[*a]; ok {
				*a = r
			}
		}
	}
	for i := range k.Stores {
		if r, ok := sub[k.Stores[i].Val]; ok {
			k.Stores[i].Val = r
		}
	}
}

// complexPair is a loop's product of two interleaved pairs as complex
// numbers: stores lo and hi of a*c - b*s and a*s + b*c, a and b the loads
// of one buffer at 2i and 2i+1, c and s of another.
type complexPair struct{ a, c, lo, hi int }

// complexPairOf is innermost loop cl's complex product, where its pairs
// fill whole vectors of lanes L.
func (k *Kernel) complexPairOf(cl, L int) (complexPair, bool) {
	l := k.Loops[cl]
	it := k.iterOf(cl)
	if l.Lanes > 0 || l.Count > 0 || it < 0 || len(k.innerLoops(cl)) > 0 || len(k.foldsOf(cl)) > 0 || 2*l.Trip%L != 0 {
		return complexPair{}, false
	}
	var st []int
	for i, s := range k.Stores {
		if s.At == cl && !s.Barrier && !s.Run {
			st = append(st, i)
		}
	}
	if len(st) != 2 {
		return complexPair{}, false
	}
	g := newAlg(k)
	last := func(ix int, ax []int) int {
		if ax != nil {
			return ax[len(ax)-1]
		}
		return ix
	}
	same := func(x, y []int) bool {
		if len(x) != len(y) {
			return false
		}
		for i := range x[:max(0, len(x)-1)] {
			if x[i] != y[i] {
				return false
			}
		}
		return true
	}
	// pair reports whether index lo moves two a trip and hi is one past it.
	pair := func(lo, hi int) bool {
		w, _ := g.coef(g.of(lo), g.id(atom{node: it}))
		if two, ok := w.constant(); !ok || two != 2 {
			return false
		}
		d := g.of(hi)
		d.add(g.of(lo), -1)
		one, ok := g.merged(d).constant()
		return ok && one == 1
	}
	s0, s1 := k.Stores[st[0]], k.Stores[st[1]]
	if k.Bufs[s0.Buf].Operand != k.Bufs[s1.Buf].Operand || !same(s0.Ax, s1.Ax) {
		return complexPair{}, false
	}
	if !pair(last(s0.Ix, s0.Ax), last(s1.Ix, s1.Ax)) {
		s0, s1, st[0], st[1] = s1, s0, st[1], st[0]
		if !pair(last(s0.Ix, s0.Ax), last(s1.Ix, s1.Ax)) {
			return complexPair{}, false
		}
	}
	mul := func(i int) (int, int, bool) {
		nd := k.Nodes[i]
		return nd.A, nd.B, nd.Op == OpMul && nd.Type == ir.F32
	}
	y0, y1 := k.Nodes[s0.Val], k.Nodes[s1.Val]
	if y0.Op != OpSub || y1.Op != OpAdd {
		return complexPair{}, false
	}
	a, c, ok1 := mul(y0.A)
	b, s, ok2 := mul(y0.B)
	if !ok1 || !ok2 {
		return complexPair{}, false
	}
	load := func(i int) (Node, bool) {
		nd := k.Nodes[i]
		return nd, nd.Op == OpLoad && k.varies(i, cl)
	}
	la, oka := load(a)
	lb, okb := load(b)
	lc, okc := load(c)
	ls, oks := load(s)
	if !oka || !okb || !okc || !oks {
		return complexPair{}, false
	}
	of := func(nd Node) Operand { return k.Bufs[nd.Buf].Operand }
	if of(la) != of(lb) {
		a, c, la, lc = c, a, lc, la
		b, s, lb, ls = s, b, ls, lb
	}
	if of(la) != of(lb) || of(lc) != of(ls) || of(la) == of(lc) || !same(la.Ax, lb.Ax) || !same(lc.Ax, ls.Ax) ||
		!pair(last(la.A, la.Ax), last(lb.A, lb.Ax)) || !pair(last(lc.A, lc.Ax), last(ls.A, ls.Ax)) {
		return complexPair{}, false
	}
	is := func(i, x, y int) bool {
		p, q, ok := mul(i)
		return ok && (p == x && q == y || p == y && q == x)
	}
	if !(is(y1.A, a, s) && is(y1.B, b, c) || is(y1.A, b, c) && is(y1.B, a, s)) {
		return complexPair{}, false
	}
	return complexPair{a: a, c: c, lo: st[0], hi: st[1]}, true
}

// vectorizeComplex runs loop c's complex product over elements, L a trip:
// each lane its element times its pair's first of the other operand, plus
// or minus the pair's other element times the pair's second, as
// addsub(x * even(t), swap(x) * odd(t)).
func (k *Kernel) vectorizeComplex(c, L int, cp complexPair, regs [5]int) {
	g := newAlg(k)
	it := k.iterOf(c)
	halves := map[int]int{}
	half := func(ix int) int {
		if h, ok := halves[ix]; ok {
			return h
		}
		_, rest := g.coef(g.of(ix), g.id(atom{node: it}))
		halves[ix] = k.bin(OpAdd, ir.U32, g.build(rest), it)
		return halves[ix]
	}
	at := func(nd Node) Node {
		if nd.Ax != nil {
			nd.Ax = append([]int(nil), nd.Ax...)
			nd.Ax[len(nd.Ax)-1] = half(nd.Ax[len(nd.Ax)-1])
		} else {
			nd.A = half(nd.A)
		}
		return nd
	}
	hint := func(n, r int) int {
		if regs != [5]int{} {
			k.Nodes[n].Hint = r + 1
		}
		return n
	}
	x := hint(k.add(at(k.Nodes[cp.a])), regs[3])
	t := hint(k.add(at(k.Nodes[cp.c])), regs[0])
	p := k.bin(OpMul, ir.F32, x, hint(k.add(Node{Op: OpDupEven, Type: ir.F32, A: t}), regs[1]))
	q := k.bin(OpMul, ir.F32, hint(k.add(Node{Op: OpSwapPairs, Type: ir.F32, A: x}), regs[4]), hint(k.add(Node{Op: OpDupOdd, Type: ir.F32, A: t}), regs[2]))
	s := k.Stores[cp.lo]
	if s.Ax != nil {
		s.Ax = append([]int(nil), s.Ax...)
		s.Ax[len(s.Ax)-1] = half(s.Ax[len(s.Ax)-1])
	} else {
		s.Ix = half(s.Ix)
	}
	s.Val = k.bin(OpAddSub, ir.F32, p, q)
	k.Stores[cp.lo] = s
	k.Stores = append(k.Stores[:cp.hi:cp.hi], k.Stores[cp.hi+1:]...)
	k.Loops[c].Trip = 2 * k.Loops[c].Trip / L
	k.Loops[c].Lanes = L
}

// pairStores is loop c's two stores, where they are a pair: one buffer at
// 2i and 2i+1 for iteration i, flat.
func (k *Kernel) pairStores(c int) ([2]int, bool) {
	it := k.iterOf(c)
	var at []int
	for i, s := range k.Stores {
		if s.At == c && !s.Barrier && !s.Run {
			at = append(at, i)
		}
	}
	if it < 0 || len(at) != 2 || k.Stores[at[0]].Ax != nil || k.Stores[at[1]].Ax != nil ||
		k.Bufs[k.Stores[at[0]].Buf].Operand != k.Bufs[k.Stores[at[1]].Buf].Operand {
		return [2]int{}, false
	}
	g := newAlg(k)
	x, y := g.of(k.Stores[at[0]].Ix), g.of(k.Stores[at[1]].Ix)
	w, rx := g.coef(x, g.id(atom{node: it}))
	if two, ok := w.constant(); !ok || two != 2 || g.moves(rx, c) {
		return [2]int{}, false
	}
	y.add(x, -1)
	if one, ok := g.merged(y).constant(); !ok || one != 1 {
		return [2]int{}, false
	}
	return [2]int{at[0], at[1]}, true
}

// broadcastInto puts each value made from a lane sum, maximum or minimum
// that vector loop c reads, made outside it, in every lane once before the loop.
func (k *Kernel) broadcastInto(c int) {
	hs := k.homes()
	made := map[int]int{}
	for i := range k.Nodes {
		if hs[i] >= 0 && k.within(hs[i], c) {
			for _, a := range k.args(&k.Nodes[i]) {
				x := *a
				if x >= len(hs) || hs[x] >= 0 && k.within(hs[x], c) || k.Nodes[x].Type != ir.F32 || k.Nodes[x].Op == OpBroadcast || !k.computedFrom(x, OpHSum) && !k.computedFrom(x, OpHMax) && !k.computedFrom(x, OpHMin) {
					continue
				}
				b, ok := made[x]
				if !ok {
					b = k.add(Node{Op: OpBroadcast, Type: ir.F32, A: x})
					made[x] = b
				}
				*a = b
			}
		}
	}
}
