package composer

import (
	"fmt"
	"maps"
	"slices"
	"strings"
)

// Cursors replaces a loop's indexing of a buffer by an address the loop
// advances. Each access becomes a constant displacement from a pointer: the
// buffer's address, or a copy of it. A loop advances the pointer the stride
// a trip less what its inner loops already advanced it; a loop after it
// picks the pointer up where it was left when its accesses start there; and
// where a later loop, or the next trip of the loop around, needs the
// address as it was, the loops advance a copy taken as the first of them
// starts. Where a scope makes several passes over a row of a value, each
// pass reads and writes the row through a copy of its own.
type Cursors struct{}

func (Cursors) Name() string { return "cursors" }

func (p Cursors) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if k.Args == nil || k.Uncursored {
			return nil, false, nil
		}
		m := k.clone()
		changed := false
		for _, a := range m.Args {
			if a.Buf < 0 {
				continue
			}
			if newCursorer(m, a.Buf).run() {
				changed = true
				continue
			}
			// A buffer read at several strides walks a copy of its address
			// for each.
			if groups := strideGroups(m, a.Buf); len(groups) > 1 {
				t, ok := m.clone(), true
				for _, g := range groups {
					cu := newCursorer(t, a.Buf)
					cu.only, cu.copied = g, true
					ok = ok && cu.run()
				}
				if ok {
					m, changed = t, true
				}
			}
		}
		if !changed {
			return nil, false, nil
		}
		field := map[int]int{}
		for _, a := range m.Args {
			if a.Buf >= 0 {
				field[a.Buf] = a.Field
			}
		}
		for c := range m.Loops {
			if !m.Loops[c].Walk {
				continue
			}
			slices.SortStableFunc(m.Loops[c].Cursors, func(a, b Cursor) int { return field[a.Buf] - field[b.Buf] })
		}
		return []*Kernel{m}, true, nil
	})
}

// cAccess is one load or store of the buffer: where it sits and its index.
type cAccess struct {
	access
	home int
	ix   lin
}

type cursorer struct {
	k      *Kernel
	g      *alg
	base   int
	buf    int
	as     []cAccess
	disp   map[int]int64
	ptr    map[int]int
	cursor map[int]Cursor
	// inits is the term a loop's take adds to its source (Cursor.Init), by
	// loop; hs the kernel's homes as the walk began.
	inits map[int]lin
	hs    []int
	// only, where set, is the accesses the cursorer lays (accessID), and
	// copied walks them on a copy of the address rather than the address.
	only   map[int]bool
	copied bool
}

// accessID names a load by its node and a store by its index.
func accessID(node, store int) int {
	if node >= 0 {
		return node
	}
	return -1 - store
}

// strideGroups is the accesses of buffer b's address grouped by how the
// iteration of each loop moves their index.
func strideGroups(k *Kernel, b int) []map[int]bool {
	g := newAlg(k)
	live := k.Live()
	base := k.Bufs[b].Base
	var keys []string
	groups := map[string]map[int]bool{}
	add := func(id, ix int) {
		var key strings.Builder
		for c := range k.Loops {
			if it := k.iterOf(c); it >= 0 {
				w, _ := g.coef(g.of(ix), g.id(atom{node: it}))
				ms := slices.Sorted(maps.Keys(w))
				fmt.Fprintf(&key, "%d:", c)
				for _, m := range ms {
					fmt.Fprintf(&key, "%s=%d,", m, w[m])
				}
				key.WriteByte(';')
			}
		}
		s := key.String()
		if groups[s] == nil {
			groups[s] = map[int]bool{}
			keys = append(keys, s)
		}
		groups[s][id] = true
	}
	for i, nd := range k.Nodes {
		if nd.Op == OpLoad && live[i] && k.Bufs[nd.Buf].Base == base {
			add(accessID(i, -1), nd.A)
		}
	}
	for i, s := range k.Stores {
		if !s.Barrier && !s.Run && k.Bufs[s.Buf].Base == base {
			add(accessID(-1, i), s.Ix)
		}
	}
	out := make([]map[int]bool, len(keys))
	for i, s := range keys {
		out[i] = groups[s]
	}
	return out
}

func newCursorer(k *Kernel, b int) *cursorer {
	return &cursorer{k: k, g: newAlg(k), base: k.Bufs[b].Base, buf: b, disp: map[int]int64{}, ptr: map[int]int{}, cursor: map[int]Cursor{}, inits: map[int]lin{}}
}

func (cu *cursorer) run() bool {
	k := cu.k
	live := k.Live()
	hs := k.Homes()
	cu.hs = hs
	for i, nd := range k.Nodes {
		if nd.Op == OpLoad && live[i] && k.Bufs[nd.Buf].Base == cu.base && (cu.only == nil || cu.only[accessID(i, -1)]) {
			cu.as = append(cu.as, cAccess{access: access{node: i, store: -1}, home: hs[i], ix: cu.g.of(nd.A)})
		}
	}
	for i, s := range k.Stores {
		if !s.Barrier && !s.Run && k.Bufs[s.Buf].Base == cu.base && (cu.only == nil || cu.only[accessID(-1, i)]) {
			cu.as = append(cu.as, cAccess{access: access{node: -1, store: i}, home: s.At, ix: cu.g.of(s.Ix)})
		}
	}
	if len(cu.as) == 0 {
		return false
	}
	all := make([]int, len(cu.as))
	ix := make([]lin, len(cu.as))
	for i := range cu.as {
		all[i] = i
		ix[i] = cu.as[i].ix
	}
	ptrs := len(k.Ptrs)
	start := 0
	if cu.copied {
		start = cu.newPtr()
	}
	if _, ok := cu.walk(-1, start, 0, cu.copied, lin{}, all, ix); !ok {
		k.Ptrs = k.Ptrs[:ptrs]
		return false
	}
	for c, t := range cu.inits {
		if l, ok := cu.cursor[c]; ok && l.Take {
			l.Init = cu.g.build(t) + 1
			cu.cursor[c] = l
		}
	}
	for c, l := range cu.cursor {
		k.Loops[c].Cursors = append(k.Loops[c].Cursors, l)
	}
	for i, a := range cu.as {
		d := k.u32(int(cu.disp[i]))
		if a.node >= 0 {
			k.Nodes[a.node].A, k.Nodes[a.node].At, k.Nodes[a.node].Ptr = d, a.home+1, cu.ptr[i]
		} else {
			k.Stores[a.store].Ix, k.Stores[a.store].Ptr = d, cu.ptr[i]
		}
	}
	return true
}

// in reports whether access a sits in loop c or a loop inside it.
func (cu *cursorer) in(a, c int) bool {
	h := cu.as[a].home
	return h >= 0 && cu.k.within(h, c)
}

// walk lays the accesses as, their indices ix with the iterations of the
// loops around scope p taken out, on pointer ptr standing at pos as the
// scope starts; with fresh set, ptr is a copy of pointer from the scope's
// first loop takes. It returns how far the scope moves the pointer.
func (cu *cursorer) walk(p, ptr, from int, fresh bool, pos lin, as []int, ix []lin) (lin, bool) {
	k, g := cu.k, cu.g
	at := copyLin(pos)
	direct := false
	for j, a := range as {
		if cu.as[a].home != p {
			continue
		}
		direct = true
		// A fresh copy is taken as the scope's first loop starts: an
		// access the scope makes itself would read it before then.
		if fresh {
			return nil, false
		}
		d := copyLin(ix[j])
		d.add(at, -1)
		c, ok := g.merged(d).constant()
		if !ok {
			return nil, false
		}
		cu.disp[a], cu.ptr[a] = c, ptr
	}
	var loops []int
	for c, l := range k.Loops {
		if l.Parent == p {
			loops = append(loops, c)
		}
	}
	family := func(c int) int {
		if of := k.Loops[c].Of; of > 0 {
			return of - 1
		}
		return c
	}
	passes := cu.passes(loops, family) > 1 && cu.row()
	pass, start, scope, scopeFresh := -1, copyLin(at), ptr, fresh
	src := ptr
	if fresh {
		src = from
	}
	// A loop split from one that walked a copy aside (a vector loop's tail)
	// walks on where that copy stands.
	famPtr, famAt := map[int]int{}, map[int]lin{}
	for li, c := range loops {
		var mine []int
		var mix []lin
		for j, a := range as {
			if cu.in(a, c) {
				mine = append(mine, a)
				mix = append(mix, ix[j])
			}
		}
		if len(mine) == 0 {
			continue
		}
		it := k.iterOf(c)
		var stride lin
		rest := make([]lin, len(mine))
		for j := range mine {
			var w lin
			if it >= 0 {
				w, rest[j] = g.coef(mix[j], g.id(atom{node: it}))
			} else {
				w, rest[j] = lin{}, copyLin(mix[j])
			}
			if j == 0 {
				stride = w
			} else if !stride.equal(w) {
				return nil, false
			}
		}
		step := g.times(stride, lin{"": int64(max(1, k.Loops[c].Lanes) * max(1, k.Loops[c].Copies))})
		if k.Loops[c].once() {
			step = lin{}
		}
		trips := lin{"": int64(k.Loops[c].Trip)}
		if k.Loops[c].Count > 0 {
			trips = g.of(k.Loops[c].Count - 1)
		}
		after := g.merged(addLin(at, g.times(trips, step)))
		q, take, qFrom := ptr, fresh, from
		fq, cont := famPtr[family(c)]
		cont = cont && !passes
		if cont {
			q, take = fq, false
		}
		// The scope's own accesses stand where it started, wherever they
		// run among its loops: a loop that moves the pointer moves a copy.
		// A fresh copy no loop has taken yet stands where its source does.
		aside := !cont && (!fresh && (k.Loops[c].Walk || direct && len(g.merged(g.times(trips, step))) > 0) || cu.laterNeeds(loops[li+1:], as, ix, after))
		if passes {
			aside = false
			if family(c) != pass {
				pass, at = family(c), copyLin(start)
				after = g.merged(addLin(at, g.times(trips, step)))
				q, take = scope, false
				if scopeFresh || cu.familyMoves(loops, family, family(c), as, ix) {
					q, take, qFrom = cu.newPtr(), true, src
				}
			}
		}
		if aside {
			q, take, qFrom = cu.newPtr(), true, ptr
			if fresh {
				qFrom = from
			}
		}
		if cu.k.Copied && q == 0 && !take {
			q, take, qFrom = cu.newPtr(), true, 0
		}
		walkAt := at
		if cont {
			walkAt = famAt[family(c)]
		}
		// Accesses that all add a base the loop around picks walk a copy
		// taken at that base.
		if !cont && !passes {
			if t, ok := cu.invariant(c, rest, walkAt); ok {
				if !aside {
					q, take, qFrom, aside = cu.newPtr(), true, ptr, true
					if fresh {
						qFrom = from
					}
				}
				cu.inits[c] = t
				walkAt = g.merged(addLin(walkAt, t))
			}
		}
		net, ok := cu.walk(c, q, 0, false, walkAt, mine, rest)
		if !ok {
			return nil, false
		}
		bump := g.merged(addLin(step, scaleLin(net, -1)))
		if _, ok := bump.constant(); !ok {
			if _, ok = cu.walk(c, cu.newPtr(), q, true, walkAt, mine, rest); !ok {
				return nil, false
			}
			bump = step
			if _, ok := bump.constant(); !ok {
				return nil, false
			}
		}
		if b, _ := bump.constant(); b != 0 || take {
			cu.cursor[c] = Cursor{Buf: cu.buf, Elems: int(b), Ptr: q, Take: take, From: qFrom}
		}
		if aside || cont {
			famPtr[family(c)], famAt[family(c)] = q, g.merged(addLin(walkAt, g.times(trips, step)))
			continue
		}
		at, ptr = after, q
		fresh = false
	}
	if passes {
		return lin{}, true
	}
	moved := copyLin(at)
	moved.add(pos, -1)
	return g.merged(moved), true
}

// invariant is the term past at that every access of a weight in loop c
// adds (rest, c's iteration taken out) where no loop in c moves it and it is
// no constant: a base the loop around picks, an expert's sheet read from the
// selection, every node of it made before c in a scope around c, each of its
// terms read from memory and added. False where the accesses add
// different terms, or the element is wider than an address's scale.
func (cu *cursorer) invariant(c int, rest []lin, at lin) (lin, bool) {
	k, g := cu.k, cu.g
	if k.Uninit || k.Bufs[cu.buf].Operand.Kind != OperandWeight {
		return nil, false
	}
	switch ElemBytes(k.Bufs[cu.buf]) {
	case 1, 2, 4, 8:
	default:
		return nil, false
	}
	var t lin
	for j := range rest {
		r := copyLin(rest[j])
		for d := range k.Loops {
			if d != c && k.within(d, c) {
				if it := k.iterOf(d); it >= 0 {
					_, r = g.coef(r, g.id(atom{node: it}))
				}
			}
		}
		r.add(at, -1)
		r = g.merged(r)
		delete(r, "")
		if j == 0 {
			t = r
		} else if !t.equal(r) {
			return nil, false
		}
	}
	if len(t) == 0 || g.moves(t, c) {
		return nil, false
	}
	// The term is a base read from memory -- an expert's sheet, a page --
	// added, never subtracted: a take's index is the term's unsigned value
	// widened to an address.
	for m, f := range t {
		loaded := false
		for _, a := range monoIDs(m) {
			for _, n := range g.nodes(a) {
				loaded = loaded || cu.loaded(n)
			}
		}
		if f < 0 || !loaded {
			return nil, false
		}
	}
	for m := range t {
		for _, a := range monoIDs(m) {
			for _, n := range g.nodes(a) {
				if h := cu.hs[n]; h >= 0 && (h == c || !k.within(c, h)) {
					return nil, false
				}
			}
		}
	}
	return t, true
}

// loaded reports whether node n is, or is computed from, a value loaded from
// memory.
func (cu *cursorer) loaded(n int) bool {
	nd := cu.k.Nodes[n]
	if nd.Op == OpLoad {
		return true
	}
	for _, a := range cu.k.Operands(&nd) {
		if cu.loaded(*a) {
			return true
		}
	}
	return false
}

// familyMoves reports whether a loop of family f among loops moves the
// pointer its accesses as read.
func (cu *cursorer) familyMoves(loops []int, family func(int) int, f int, as []int, ix []lin) bool {
	k, g := cu.k, cu.g
	for _, d := range loops {
		if family(d) != f {
			continue
		}
		if k.Loops[d].Walk {
			return true
		}
		it := k.iterOf(d)
		if it < 0 || k.Loops[d].once() {
			continue
		}
		for j, a := range as {
			if cu.in(a, d) {
				if w, _ := g.coef(ix[j], g.id(atom{node: it})); len(g.merged(w)) > 0 {
					return true
				}
			}
		}
	}
	return false
}

// passes is how many passes the loops make over rows of values: the
// families of loops split from one that read or write one.
func (cu *cursorer) passes(loops []int, family func(int) int) int {
	k := cu.k
	hs := k.Homes()
	fams := map[int]bool{}
	isRow := func(b int) bool {
		o := k.Bufs[b].Operand
		return o.Kind == OperandValue || o.Kind == OperandScratch
	}
	for _, c := range loops {
		for i, nd := range k.Nodes {
			if nd.Op == OpLoad && isRow(nd.Buf) && hs[i] >= 0 && k.within(hs[i], c) {
				fams[family(c)] = true
			}
		}
		for _, s := range k.Stores {
			if !s.Barrier && !s.Run && isRow(s.Buf) && s.At >= 0 && k.within(s.At, c) {
				fams[family(c)] = true
			}
		}
	}
	return len(fams)
}

// row reports whether the buffer is a row of a value rather than a
// weight, a table or constants.
func (cu *cursorer) row() bool {
	o := cu.k.Bufs[cu.buf].Operand
	return o.Kind == OperandValue || o.Kind == OperandScratch
}

// laterNeeds reports whether a loop of loops reaches an access whose index,
// its iterations aside, is not where the pointer will stand when the loop
// starts: will after the loop before, each loop then moving it its trips
// times its stride.
func (cu *cursorer) laterNeeds(loops, as []int, ix []lin, will lin) bool {
	g, k := cu.g, cu.k
	for _, c := range loops {
		var stride lin
		seen := false
		for j, a := range as {
			if !cu.in(a, c) {
				continue
			}
			r := copyLin(ix[j])
			if it := k.iterOf(c); it >= 0 {
				var w lin
				w, r = g.coef(r, g.id(atom{node: it}))
				if !seen {
					stride, seen = w, true
				}
			}
			for d := range k.Loops {
				if d != c && k.within(d, c) {
					if it := k.iterOf(d); it >= 0 {
						_, r = g.coef(r, g.id(atom{node: it}))
					}
				}
			}
			r.add(will, -1)
			if _, ok := g.merged(r).constant(); !ok {
				return true
			}
		}
		if seen {
			trips := lin{"": int64(k.Loops[c].Trip)}
			if k.Loops[c].Count > 0 {
				trips = g.of(k.Loops[c].Count - 1)
			}
			step := g.times(stride, lin{"": int64(max(1, k.Loops[c].Lanes) * max(1, k.Loops[c].Copies))})
			will = g.merged(addLin(will, g.times(trips, step)))
		}
	}
	return false
}

func (cu *cursorer) newPtr() int {
	cu.k.Ptrs = append(cu.k.Ptrs, Ptr{Buf: cu.buf})
	return len(cu.k.Ptrs)
}

func copyLin(l lin) lin {
	out := lin{}
	out.add(l, 1)
	return out
}

func addLin(a, b lin) lin {
	out := copyLin(a)
	out.add(b, 1)
	return out
}

func scaleLin(a lin, f int64) lin {
	out := lin{}
	out.add(a, f)
	return out
}

// cursored reports whether a loop of k walks an address (Cursors).
func (k *Kernel) cursored() bool {
	for _, l := range k.Loops {
		if len(l.Cursors) > 0 {
			return true
		}
	}
	return false
}
