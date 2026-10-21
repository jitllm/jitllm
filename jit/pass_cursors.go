package jit

import "slices"

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
		if k.Args == nil {
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
}

func newCursorer(k *Kernel, b int) *cursorer {
	return &cursorer{k: k, g: newAlg(k), base: k.Bufs[b].Base, buf: b, disp: map[int]int64{}, ptr: map[int]int{}, cursor: map[int]Cursor{}}
}

func (cu *cursorer) run() bool {
	k := cu.k
	live := k.live()
	hs := k.homes()
	for i, nd := range k.Nodes {
		if nd.Op == OpLoad && live[i] && k.Bufs[nd.Buf].Base == cu.base {
			cu.as = append(cu.as, cAccess{access: access{node: i, store: -1}, home: hs[i], ix: cu.g.of(nd.A)})
		}
	}
	for i, s := range k.Stores {
		if !s.Barrier && !s.Run && k.Bufs[s.Buf].Base == cu.base {
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
	if _, ok := cu.walk(-1, 0, false, lin{}, all, ix); !ok {
		k.Ptrs = k.Ptrs[:ptrs]
		return false
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
// scope starts; with fresh set, ptr is a copy the scope's first loop takes.
// It returns how far the scope moves the pointer.
func (cu *cursorer) walk(p, ptr int, fresh bool, pos lin, as []int, ix []lin) (lin, bool) {
	k, g := cu.k, cu.g
	at := copyLin(pos)
	for j, a := range as {
		if cu.as[a].home != p {
			continue
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
	pass, start := -1, copyLin(at)
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
		q, take := ptr, fresh
		aside := !fresh && (k.Loops[c].Walk || cu.laterNeeds(loops[li+1:], as, ix, after))
		if passes {
			aside = false
			if family(c) != pass {
				pass, at = family(c), copyLin(start)
				after = g.merged(addLin(at, g.times(trips, step)))
				q, take = cu.newPtr(), true
			}
		}
		if aside {
			q, take = cu.newPtr(), true
		}
		net, ok := cu.walk(c, q, false, at, mine, rest)
		if !ok {
			return nil, false
		}
		bump := g.merged(addLin(step, scaleLin(net, -1)))
		if _, ok := bump.constant(); !ok {
			if _, ok = cu.walk(c, cu.newPtr(), true, at, mine, rest); !ok {
				return nil, false
			}
			bump = step
			if _, ok := bump.constant(); !ok {
				return nil, false
			}
		}
		if b, _ := bump.constant(); b != 0 || take {
			cu.cursor[c] = Cursor{Buf: cu.buf, Elems: int(b), Ptr: q, Take: take}
		}
		if !aside {
			at, ptr = after, q
		}
		fresh = false
	}
	if passes {
		return lin{}, true
	}
	moved := copyLin(at)
	moved.add(pos, -1)
	return g.merged(moved), true
}

// passes is how many passes the loops make over rows of values: the
// families of loops split from one that read or write one.
func (cu *cursorer) passes(loops []int, family func(int) int) int {
	k := cu.k
	hs := k.homes()
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
