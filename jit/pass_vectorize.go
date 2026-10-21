package jit

import (
	"math"
	"math/bits"

	"github.com/jitllm/jitllm/jit/ir"
)

// Vectorize runs an innermost loop that reads and writes consecutive
// elements of its iteration Lanes iterations a trip, one in each lane, and
// the trips that do not fill a vector after it in a loop of one a trip. A
// sum the loop carries becomes a sum in each lane, the tail adding into its
// first lane, and the lanes are summed where the sum is read. A loop that
// stores a pair of values at 2i and 2i+1 stores the two vectors' lanes
// zipped, the low pairs then the high.
type Vectorize struct {
	Lanes int
}

func (Vectorize) Name() string { return "vectorize" }

func (p Vectorize) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		m := k.clone()
		changed := false
		for c := 0; c < len(m.Loops); c++ {
			if m.vectorizable(c) {
				m.vectorize(c, p.Lanes)
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
	if l.Lanes > 0 || len(k.innerLoops(c)) > 0 || k.iterOf(c) < 0 {
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

func (k *Kernel) vectorize(c, L int) {
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
	carries := map[int]int{}
	for _, f := range k.foldsOf(c) {
		if v := k.carriedBy(f); v >= 0 {
			carries[v] = f
		}
	}
	for v, cf := range carries {
		k.vectorizeCarry(c, t, v, cf, sub)
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
		k.Nodes[ft].A = k.copyOf(step, sub)
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
		n.Val = k.copyOf(s.Val, sub)
		if s.Ax != nil {
			n.Ax = make([]int, len(s.Ax))
			for j, a := range s.Ax {
				n.Ax[j] = k.copyOf(a, sub)
			}
		} else {
			n.Ix = k.copyOf(s.Ix, sub)
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
// beyond it.
func (k *Kernel) vectorizeCarry(c, t, v, cf int, sub map[int]int) {
	ct := k.Nodes[cf].Type
	op := k.Nodes[k.Nodes[v].A].Op
	vt := k.add(Node{Op: OpFold, At: t, Type: ir.F32, B: k.Nodes[v].B})
	cft := k.add(Node{Op: OpFold, At: t, Type: ct, B: k.add(Node{Op: OpConst, Type: ct})})
	sub[v], sub[cf] = vt, cft
	k.Nodes[vt].A = k.copyOf(k.Nodes[v].A, sub)
	k.Nodes[cft].A = k.copyOf(k.Nodes[cf].A, sub)
	lanes0 := k.add(Node{Op: OpLanes, Type: ir.U32, Imm: int64(k.Loops[c].Lanes)})
	k.Nodes[cf].B = k.copyOf(k.Nodes[k.Nodes[cf].A].B, map[int]int{k.iterOf(c): lanes0})
	lanes, least := OpHMax, int64(math.MaxInt32)
	if op == OpMin {
		lanes = OpHMin
	}
	first := len(k.Nodes)
	m := k.add(Node{Op: OpBroadcast, Type: ir.F32, A: k.add(Node{Op: lanes, Type: ir.F32, A: v})})
	held := k.add(Node{Op: OpSelect, Type: ct, A: k.add(Node{Op: OpEq, Type: ir.Pred, A: v, B: m}), B: cf,
		C: k.add(Node{Op: OpConst, Type: ct, Imm: least})})
	ci := k.add(Node{Op: OpHMin, Type: ct, A: held})
	val := k.bin(op, ir.F32, vt, m)
	car := k.add(Node{Op: OpSelect, Type: ct, A: k.add(Node{Op: OpLess, Type: ir.Pred, A: m, B: vt}), B: cft, C: ci})
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
		if hs[i] >= 0 && k.within(hs[i], c) || k.Nodes[i].Op == OpBroadcast {
			for _, a := range k.args(&k.Nodes[i]) {
				x := *a
				if x >= len(hs) || hs[x] >= 0 && k.within(hs[x], c) || k.Nodes[x].Type != ir.F32 || !k.computedFrom(x, OpHSum) && !k.computedFrom(x, OpHMax) && !k.computedFrom(x, OpHMin) {
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
