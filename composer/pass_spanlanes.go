package composer

import (
	"fmt"

	"github.com/jitllm/jitllm/composer/ir"
)

// SpanLanes forms an iteration's sums over runs (OpSpanSum) on a device
// whose code runs a call at a time, after Serial has made its elements a
// loop around its trips' loop: the elements' loop takes a run's elements a
// trip, together. Where each element's loads and stores move with it and
// the row fills a run and a vector of Lanes, its trips are vectors of
// Lanes, a run within a vector summed by a butterfly of exchanges across
// its lanes (OpShuffle) and a longer run's vectors, each so summed, added;
// a row that does not fill its last vector runs a vector again at its end,
// computing what the trips before computed for the elements they share.
// Otherwise a trip is a run, its elements copies of the body, the run's
// sum their values added.
type SpanLanes struct{ Lanes int }

func (SpanLanes) Name() string { return "spanlanes" }

func (p SpanLanes) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		S, err := k.spanOf()
		if S == 0 || err != nil {
			return nil, false, err
		}
		i := -1
		for c, l := range k.Loops {
			if l.Serial {
				i = c
			}
		}
		o := -1
		if i >= 0 {
			o = k.Loops[i].Parent
		}
		if o < 0 || len(k.innerLoops(o)) != 1 || len(k.innerLoops(i)) != 0 || k.iterOf(o) < 0 {
			return nil, false, fmt.Errorf("jit: %s sums runs of %d outside an iteration's trips", k.Name, S)
		}
		if l := k.Loops[o]; l.Count > 0 || l.Lanes > 0 || l.Wide > 0 || l.Trip%S != 0 {
			return nil, false, fmt.Errorf("jit: %s sums runs of %d over a loop of %d trips", k.Name, S, l.Trip)
		}
		m := k.clone()
		m.Spans = S
		m.flattenStrided(o)
		W := m.Loops[o].Trip
		L := p.Lanes
		n := max(1, S/L)
		_, ok := m.spanSteps(o, i)
		if L < 2 || W < n*L || k.Unvector || k.Unwide || !ok {
			L, n = 1, S
		}
		if W%(n*L) != 0 {
			o2, i2 := m.spanTail(o, i)
			if err := m.spanJam(o2, i2, L, n, W-n*L, 1, S); err != nil {
				return nil, false, err
			}
		}
		if err := m.spanJam(o, i, L, n, 0, W/(n*L), S); err != nil {
			return nil, false, err
		}
		return []*Kernel{m}, true, nil
	})
}

// flattenStrided makes each store of loop o whose last coordinate moves
// other than one element a trip a store at a flat index.
func (k *Kernel) flattenStrided(o int) {
	it := k.iterOf(o)
	g := newAlg(k)
	a := g.id(atom{node: it})
	for x := range k.Stores {
		s := &k.Stores[x]
		if s.At != o || s.Barrier || s.Run || s.Ax == nil || len(s.Ax) != len(k.Bufs[s.Buf].Shape) {
			continue
		}
		w, _ := g.coef(g.of(s.Ax[len(s.Ax)-1]), a)
		if c, ok := w.constant(); ok && c == 1 {
			continue
		}
		sh := k.Bufs[s.Buf].Shape
		ix := s.Ax[0]
		for d := 1; d < len(sh); d++ {
			ix = k.bin(OpAdd, ir.U32, k.bin(OpMul, ir.U32, ix, k.u32(sh[d])), s.Ax[d])
		}
		s.Ix, s.Ax = ix, nil
	}
}

// spanSteps is the elements each store of loop o moves by a trip of it,
// and whether o's elements can be lanes of a vector: each of the loads
// within o reads at o's iteration, moving one element a trip and reading
// no value inner loop i carries, no value reads the iteration but through
// an address, and o stores i's values alone, each at its iteration plus a
// place it fixes.
func (k *Kernel) spanSteps(o, i int) ([]int, bool) {
	it := k.iterOf(o)
	g := newAlg(k)
	a := g.id(atom{node: it})
	step := func(x int) (int, bool) {
		w, rest := g.coef(g.of(x), a)
		c, ok := w.constant()
		if !ok || c < 1 {
			return 0, false
		}
		for mo := range rest {
			for _, id := range monoIDs(mo) {
				for _, nd := range g.nodes(id) {
					if k.readsNode(nd, it) {
						return 0, false
					}
				}
			}
		}
		return int(c), true
	}
	folds := k.foldsOf(i)
	hs := k.Homes()
	for x, nd := range k.Nodes {
		if nd.Op != OpLoad || x >= len(hs) || hs[x] < 0 || !k.within(hs[x], o) {
			continue
		}
		if !k.readsNode(x, it) {
			return nil, false
		}
		for _, f := range folds {
			if k.readsNode(x, f) {
				return nil, false
			}
		}
		at := nd.A
		if nd.Ax != nil {
			for _, ax := range nd.Ax[:len(nd.Ax)-1] {
				if k.readsNode(ax, it) {
					return nil, false
				}
			}
			at = nd.Ax[len(nd.Ax)-1]
		}
		if c, ok := step(at); !ok || c != 1 {
			return nil, false
		}
	}
	values := []int{}
	for _, f := range folds {
		values = append(values, k.Nodes[f].A, k.Nodes[f].B)
	}
	steps := make([]int, len(k.Stores))
	for x, s := range k.Stores {
		if s.Barrier || s.Run || s.At != o {
			continue
		}
		if k.Nodes[s.Val].Op != OpFold || k.Nodes[s.Val].At != i {
			return nil, false
		}
		at := s.Ix
		if s.Ax != nil {
			for _, ax := range s.Ax[:len(s.Ax)-1] {
				if k.readsNode(ax, it) {
					return nil, false
				}
			}
			at = s.Ax[len(s.Ax)-1]
		}
		c, ok := step(at)
		if !ok || s.Ax != nil && c != 1 {
			return nil, false
		}
		steps[x] = c
	}
	seen := map[int]bool{}
	var asValue func(x int) bool
	asValue = func(x int) bool {
		if x == it {
			return true
		}
		if seen[x] {
			return false
		}
		seen[x] = true
		nd := k.Nodes[x]
		if nd.Op == OpLoad || nd.Op == OpFold {
			return false
		}
		for _, p := range k.Operands(&nd) {
			if asValue(*p) {
				return true
			}
		}
		return false
	}
	for _, v := range values {
		if asValue(v) {
			return nil, false
		}
	}
	return steps, true
}

// spanTail is a copy of loop o and its inner loop i after them, run at
// iterations a jam of the copy places: the copy's folds and stores o's and
// i's under the copy's iterations.
func (k *Kernel) spanTail(o, i int) (int, int) {
	lo, li := k.Loops[o], k.Loops[i]
	to := len(k.Loops)
	k.Loops = append(k.Loops, Loop{Trip: 1, Parent: lo.Parent, Range: lo.Range, Of: o + 1})
	inner := li
	inner.Parent, inner.Cursors, inner.Reg = to, nil, 0
	ti := len(k.Loops)
	k.Loops = append(k.Loops, inner)
	sub := map[int]int{k.iterOf(o): k.add(Node{Op: OpIter, At: to, Type: ir.U32})}
	if x := k.iterOf(i); x >= 0 {
		sub[x] = k.add(Node{Op: OpIter, At: ti, Type: ir.U32})
	}
	folds := k.foldsOf(i)
	for _, f := range folds {
		sub[f] = k.add(Node{Op: OpFold, At: ti, Type: k.Nodes[f].Type})
	}
	for _, f := range folds {
		b := k.copyKeep(k.Nodes[f].B, sub)
		a := k.copyKeep(k.Nodes[f].A, sub)
		k.Nodes[sub[f]].A, k.Nodes[sub[f]].B = a, b
	}
	var copied []Store
	for _, s := range k.Stores {
		if s.Barrier || s.Run || s.At != o {
			continue
		}
		copied = append(copied, k.storeCopy(s, sub, to))
	}
	k.Stores = append(k.Stores, copied...)
	return to, ti
}

// spanJam runs loop o over trips trips of n copies of inner loop i's body,
// copy j at element base + n*L*trip + j*L, a vector of L where L is more
// than one: each run's elements in the trip's copies, a sum over runs
// each copy's lanes summed (laneSum) and the copies' sums added, a store
// whose lanes are apart storing each alone.
func (k *Kernel) spanJam(o, i, L, n, base, trips, S int) error {
	U := ir.U32
	var steps []int
	if L > 1 {
		var ok bool
		if steps, ok = k.spanSteps(o, i); !ok {
			return fmt.Errorf("jit: %s's elements are no vector's lanes", k.Name)
		}
	}
	it := k.iterOf(o)
	folds := k.foldsOf(i)
	subs := make([]map[int]int, n)
	memo := make([]map[int]int, n)
	for j := range n {
		first := k.bin(OpAdd, U, k.bin(OpMul, U, it, k.u32(n*L)), k.u32(base+j*L))
		subs[j], memo[j] = map[int]int{it: first}, map[int]int{}
		for _, f := range folds {
			subs[j][f] = k.add(Node{Op: OpFold, At: i, Type: k.Nodes[f].Type})
		}
	}
	sums := map[int]int{}
	var walk func(j, x int) int
	walk = func(j, x int) int {
		if v, ok := subs[j][x]; ok {
			return v
		}
		if v, ok := memo[j][x]; ok {
			return v
		}
		nd := k.Nodes[x]
		if nd.Op == OpFold {
			return x
		}
		if nd.Op == OpSpanSum {
			v, ok := sums[x]
			if !ok {
				parts := make([]int, n)
				for jj := range n {
					parts[jj] = k.laneSum(walk(jj, nd.A), min(S, L))
				}
				for len(parts) > 1 {
					var next []int
					for q := 0; q+1 < len(parts); q += 2 {
						next = append(next, k.bin(OpAdd, ir.F32, parts[q], parts[q+1]))
					}
					parts = append(next, parts[len(parts)&^1:]...)
				}
				v = parts[0]
				sums[x] = v
			}
			memo[j][x] = v
			return v
		}
		changed := false
		for _, p := range k.Operands(&nd) {
			if w := walk(j, *p); w != *p {
				*p, changed = w, true
			}
		}
		v := x
		if changed {
			v = k.add(nd)
		}
		memo[j][x] = v
		return v
	}
	for j := range n {
		for _, f := range folds {
			g := subs[j][f]
			a, b := walk(j, k.Nodes[f].A), walk(j, k.Nodes[f].B)
			k.Nodes[g].A, k.Nodes[g].B = a, b
		}
	}
	var stores []Store
	for x, s := range k.Stores {
		if s.Barrier || s.Run || s.At != o {
			stores = append(stores, s)
			continue
		}
		if k.Nodes[s.Val].Op != OpFold || k.Nodes[s.Val].At != i {
			return fmt.Errorf("jit: %s stores a value its trips do not carry beside a sum over runs", k.Name)
		}
		for j := range n {
			ns := s
			ns.Val = subs[j][s.Val]
			if s.Ax != nil {
				ns.Ax = make([]int, len(s.Ax))
				for d, a := range s.Ax {
					ns.Ax[d] = walk(j, a)
				}
			} else {
				ns.Ix = walk(j, s.Ix)
			}
			if L > 1 && x < len(steps) && steps[x] > 1 {
				ns.Step = steps[x]
			}
			stores = append(stores, ns)
		}
	}
	k.Stores = stores
	k.Loops[o].Trip = trips
	if L > 1 {
		// A vector of one trip stays a loop: the loads made in it are its
		// lanes'.
		k.Loops[o].Wide, k.Loops[i].Wide, k.Loops[o].Kept = L, L, trips == 1
	}
	return nil
}
