package jit

// Pair is a producer's kind a consumer's may be fused into. Additive asks
// that the consumer add a term to the producer's value; AtStores fuses a
// consumer of a value the producer writes at several places an element, the
// consumer computed at each of the producer's stores; Unrolled fuses a
// reduction into a consumer reading it at leading coordinates of its own,
// the reduction's terms unrolled at each read. Then fuses, on a device
// whose code runs a call at a time, a consumer reading what the producer
// stores once its loops are done, the same at every trip of the
// consumer's: the consumer's code after the producer's in one kernel.
// Alongside fuses a reduction over what the producer stores along one of
// its loops, element for element, into that loop, its terms the values
// stored as they are made; Combine, where set, is the one combine it fuses.
type Pair struct {
	From, To  PrimKind
	Additive  bool
	AtStores  bool
	Unrolled  bool
	Then      bool
	Alongside bool
	Combine   Op
}

// Fuse inlines a consumer into its producer where their pair is allowed, the
// consumer reads the producer's value at its own element and is that value's
// one reader, and nothing between them touches what either does.
type Fuse struct {
	Pairs []Pair
}

func (Fuse) Name() string { return "fuse" }

func (f Fuse) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	out := append([]*Kernel(nil), ks...)
	for ci := 0; ci < len(out); ci++ {
		c := out[ci]
		for pi := ci - 1; pi >= 0; pi-- {
			p := out[pi]
			var m *Kernel
			if v, st, pl, ok := f.joinsAlongside(p, c); ok {
				m = mergeAlongside(p, c, v, st, pl)
			} else if v, st, ok := f.joinsThen(p, c, out); ok {
				m = merge(p, c, v, st)
				m.Kind = c.Kind
			} else if v, st, ok := f.joins(p, c, out); ok {
				m = merge(p, c, v, st)
			} else if v, ok := f.joinsAtStores(p, c, out); ok {
				m = mergeAtStores(p, c, v)
			} else if v, ok := f.joinsUnrolled(p, c, out); ok {
				u := unrolled(p, c, v)
				u.Passes = append(append([]string(nil), c.Passes...), f.Name())
				out[ci] = u
				out = append(out[:pi], out[pi+1:]...)
				ci--
				break
			}
			if m != nil {
				m.Passes = append(append([]string(nil), p.Passes...), f.Name())
				out[pi] = m
				out = append(out[:ci], out[ci+1:]...)
				ci--
				break
			}
			if conflicts(p, c) {
				break
			}
		}
	}
	return out, nil
}

// joinsThen finds what p stores after its loops, for every call alike,
// that c alone reads, nothing in c's loops moving where it reads it.
func (f Fuse) joinsThen(p, c *Kernel, ks []*Kernel) (Operand, int, bool) {
	allowed := false
	for _, pr := range f.Pairs {
		allowed = allowed || pr.Then && pr.From == p.Kind && pr.To == c.Kind
	}
	if !allowed || p.Work || c.Work || p.Group > 0 || c.Group > 0 || !sameSpace(p, c) || len(p.Loops) == 0 {
		return Operand{}, 0, false
	}
	for i, s := range p.Stores {
		if s.Barrier || s.Run || s.At != -1 {
			continue
		}
		o := p.Bufs[s.Buf].Operand
		if !soleReader(ks, c, o) {
			continue
		}
		fixed := true
		any := false
		hs := c.Homes()
		for j, nd := range c.Nodes {
			if nd.Op == OpLoad && c.Bufs[nd.Buf].Operand == o {
				any = true
				if hs[j] >= 0 {
					fixed = false
				}
			}
		}
		if any && fixed {
			return o, i, true
		}
	}
	return Operand{}, 0, false
}

func (f Fuse) allows(p, c *Kernel, atStores bool) bool {
	for _, pr := range f.Pairs {
		if pr.Then || pr.Alongside {
			continue
		}
		if pr.From == p.Kind && pr.To == c.Kind && pr.AtStores == atStores && !pr.Unrolled && (!pr.Additive || c.adds()) {
			return true
		}
	}
	return false
}

// joinsUnrolled finds a reduction's output c alone reads, each read at
// leading coordinates of c's own -- or, where c is a reduction too, at its
// coordinates and its own loop's iteration, a short reduction (Trip at most
// unrolledTrips) summed once a trip of c's.
func (f Fuse) joinsUnrolled(p, c *Kernel, ks []*Kernel) (Operand, bool) {
	allowed := false
	for _, pr := range f.Pairs {
		allowed = allowed || pr.Unrolled && pr.From == p.Kind && pr.To == c.Kind
	}
	fd, ok := p.fold()
	if !allowed || !ok || p.Work || c.Work || len(c.Loops) > 0 && c.Kind != PrimReduce || len(p.Stores) != 1 ||
		!p.storesOwn(p.Stores[0]) || p.Stores[0].Val != fd || p.Loops[0].Count > 0 ||
		c.Kind == PrimReduce && (len(p.Loops) != 1 || p.Loops[0].Trip > unrolledTrips) {
		return Operand{}, false
	}
	o := p.Bufs[p.Stores[0].Buf].Operand
	if !soleReader(ks, c, o) {
		return Operand{}, false
	}
	any := false
	for _, nd := range c.Nodes {
		if nd.Op != OpLoad || c.Bufs[nd.Buf].Operand != o {
			continue
		}
		if len(nd.Ax) != len(p.Space) {
			return Operand{}, false
		}
		for i, a := range nd.Ax {
			x := c.Nodes[a]
			if !(x.Op == OpCoord && x.A == i || c.Kind == PrimReduce && x.Op == OpIter) {
				return Operand{}, false
			}
		}
		any = true
	}
	return o, any
}

// unrolledTrips is the longest reduction joinsUnrolled sums inside another's
// loop, every trip of it written out.
const unrolledTrips = 32

// unrolled is c with each read of p's output v p's sum at the read's
// coordinates, its terms unrolled from its start.
func unrolled(p, c *Kernel, v Operand) *Kernel {
	m := c.clone()
	m.Name = p.Name + "+" + c.Name
	f, _ := p.fold()
	it := p.iterOf(0)
	for i, nd := range c.Nodes {
		if nd.Op != OpLoad || c.Bufs[nd.Buf].Operand != v {
			continue
		}
		ax := nd.Ax
		coord := func(j int, x Node) (int, bool) {
			if x.Op == OpCoord {
				return ax[x.A], true
			}
			return 0, false
		}
		acc := copyInto(m, p, p.Nodes[f].B, coord)
		for j := 0; j < p.Loops[0].Trip; j++ {
			at, a := m.u32(j), acc
			acc = copyInto(m, p, p.Nodes[f].A, func(k int, x Node) (int, bool) {
				switch {
				case k == f:
					return a, true
				case k == it:
					return at, true
				}
				return coord(k, x)
			})
		}
		m.redirect(i, acc, acc)
	}
	return m
}

func (f Fuse) joins(p, c *Kernel, ks []*Kernel) (Operand, int, bool) {
	if !f.allows(p, c, false) || !sameSpace(p, c) || len(p.Loops) > 0 && c.Kind != PrimMap {
		return Operand{}, 0, false
	}
	for i, s := range p.Stores {
		o := p.Bufs[s.Buf].Operand
		if s.At != -1 || o.Kind != OperandValue || !p.storesOwn(s) || !c.readsOwn(o) || !soleReader(ks, c, o) {
			continue
		}
		return o, i, true
	}
	return Operand{}, 0, false
}

// adds reports whether a map stores, at its own place and nothing else, the
// sum of two reads at its own place.
func (k *Kernel) adds() bool {
	if len(k.Stores) != 1 || len(k.Loops) != 0 || !k.storesOwn(k.Stores[0]) {
		return false
	}
	v := k.Nodes[k.Stores[0].Val]
	if v.Op != OpAdd {
		return false
	}
	for _, a := range []int{v.A, v.B} {
		if l := k.Nodes[a]; l.Op != OpLoad || !k.own(l.Buf, l.Ax) {
			return false
		}
	}
	return true
}

// conflicts reports whether c may not move before p: p writes what c reads,
// or reads or writes what c writes.
func conflicts(p, c *Kernel) bool {
	for _, b := range c.Bufs {
		if p.writes(b.Operand) {
			return true
		}
	}
	for _, s := range c.Stores {
		if s.Barrier || s.Run {
			continue
		}
		o := c.Bufs[s.Buf].Operand
		if p.reads(o) || p.writes(o) {
			return true
		}
	}
	return false
}

// graft copies c's nodes into m: its reads of v as node val, its
// coordinates through coord where set, its buffers and loops m's.
func graft(m, c *Kernel, v Operand, val int, coord func(i int) (int, bool)) (nodeOf []int, bufOf []int, loopOf []int) {
	return graftOnto(m, c, v, val, coord, nil)
}

// graftOnto is graft with each of c's loops in onto run as the loop of m
// it names.
func graftOnto(m, c *Kernel, v Operand, val int, coord func(i int) (int, bool), onto map[int]int) (nodeOf []int, bufOf []int, loopOf []int) {
	bufOf = make([]int, len(c.Bufs))
	for i, b := range c.Bufs {
		bufOf[i] = -1
		if b.Operand == v && b.Kind == BufIn {
			continue
		}
		for j, mb := range m.Bufs {
			if mb.Operand == b.Operand && mb.Kind == b.Kind && sameShape(mb.Shape, b.Shape) {
				bufOf[i] = j
			}
		}
		if bufOf[i] < 0 {
			bufOf[i] = m.buf(b)
		}
	}
	loopOf = make([]int, len(c.Loops))
	for i, l := range c.Loops {
		if x, ok := onto[i]; ok {
			loopOf[i] = x
			continue
		}
		if l.Parent >= 0 {
			l.Parent = loopOf[l.Parent]
		}
		m.Loops = append(m.Loops, l)
		loopOf[i] = len(m.Loops) - 1
	}
	nodeOf = make([]int, len(c.Nodes))
	for i := range nodeOf {
		nodeOf[i] = -1
	}
	var walk func(i int) int
	walk = func(i int) int {
		if nodeOf[i] >= 0 {
			return nodeOf[i]
		}
		nd := c.Nodes[i]
		switch {
		case nd.Op == OpLoad && c.Bufs[nd.Buf].Operand == v:
			nodeOf[i] = val
			return val
		case nd.Op == OpCoord && coord != nil:
			if x, ok := coord(nd.A); ok {
				nodeOf[i] = x
				return x
			}
		case nd.Op == OpIter:
			if x, ok := onto[nd.At]; ok && m.iterOf(x) >= 0 {
				nodeOf[i] = m.iterOf(x)
				return nodeOf[i]
			}
		}
		if nd.Op == OpIter || nd.Op == OpFold {
			nd.At = loopOf[nd.At]
		}
		if nd.Op == OpLoad {
			nd.Buf = bufOf[nd.Buf]
		}
		nodeOf[i] = m.add(Node{})
		args := m.Operands(&nd)
		for _, a := range args {
			*a = walk(*a)
		}
		m.Nodes[nodeOf[i]] = nd
		return nodeOf[i]
	}
	for i := range c.Nodes {
		walk(i)
	}
	for i, l := range c.Loops {
		if _, ok := onto[i]; !ok && l.Count > 0 {
			m.Loops[loopOf[i]].Count = walk(l.Count-1) + 1
		}
	}
	return nodeOf, bufOf, loopOf
}

// joinsAlongside finds what p stores along one of its innermost loops at
// its own element and c, a reduction of one loop over the same terms,
// reads at its own element and nothing else p writes: the stored operand,
// the store and the loop.
func (f Fuse) joinsAlongside(p, c *Kernel) (Operand, int, int, bool) {
	fd, ok := c.fold()
	allowed := false
	for _, pr := range f.Pairs {
		allowed = allowed || ok && pr.Alongside && pr.From == p.Kind && pr.To == c.Kind &&
			(pr.Combine == 0 || c.Nodes[c.Nodes[fd].A].Op == pr.Combine)
	}
	if !allowed || p.Work || c.Work || p.Group > 0 || c.Group > 0 || !sameSpace(p, c) {
		return Operand{}, 0, 0, false
	}
	own := func(k *Kernel, ax []int, it int) bool {
		if it < 0 || len(ax) != len(k.Space)+1 || ax[len(ax)-1] != it {
			return false
		}
		for i, a := range ax[:len(ax)-1] {
			if nd := k.Nodes[a]; nd.Op != OpCoord || nd.A != i {
				return false
			}
		}
		return true
	}
	writes := map[Operand]bool{}
	for _, s := range p.Stores {
		if !s.Barrier && !s.Run {
			writes[p.Bufs[s.Buf].Operand] = true
		}
	}
	for si, s := range p.Stores {
		if s.Barrier || s.Run || s.At < 0 || len(p.innerLoops(s.At)) > 0 || !own(p, s.Ax, p.iterOf(s.At)) {
			continue
		}
		o := p.Bufs[s.Buf].Operand
		reads, ok := false, true
		for _, nd := range c.Nodes {
			if nd.Op != OpLoad {
				continue
			}
			switch b := c.Bufs[nd.Buf].Operand; {
			case b == o:
				reads = true
				ok = ok && own(c, nd.Ax, c.iterOf(0))
			case writes[b]:
				ok = false
			}
		}
		pl, cl := p.Loops[s.At], c.Loops[0]
		if !reads || !ok || pl.Trip != cl.Trip || (pl.Count > 0) != (cl.Count > 0) ||
			pl.Count > 0 && !sameExpr(p, pl.Count-1, c, cl.Count-1) {
			continue
		}
		return o, si, s.At, true
	}
	return Operand{}, 0, 0, false
}

// mergeAlongside is p with c's reduction carried by p's loop pl, its reads
// of v the value p's store st stores there.
func mergeAlongside(p, c *Kernel, v Operand, st, pl int) *Kernel {
	m := p.clone()
	m.Name, m.Kind = p.Name+"+"+c.Name, PrimReduce
	coords := map[int]int{}
	for i, nd := range m.Nodes {
		if nd.Op == OpCoord {
			coords[nd.A] = i
		}
	}
	nodeOf, bufOf, loopOf := graftOnto(m, c, v, p.Stores[st].Val, func(a int) (int, bool) { x, ok := coords[a]; return x, ok }, map[int]int{0: pl})
	graftStores(m, c, nodeOf, bufOf, loopOf)
	return m
}

// sameExpr reports whether node i of a and node j of b compute one value:
// the same operations on the same operands, loads of the same operands.
func sameExpr(a *Kernel, i int, b *Kernel, j int) bool {
	x, y := a.Nodes[i], b.Nodes[j]
	if x.Op != y.Op || x.Type != y.Type || x.Imm != y.Imm || x.Op == OpIter || x.Op == OpFold {
		return false
	}
	if x.Op == OpLoad && a.Bufs[x.Buf].Operand != b.Bufs[y.Buf].Operand {
		return false
	}
	if x.Op == OpCoord {
		return x.A == y.A
	}
	ax, bx := a.Operands(&x), b.Operands(&y)
	if len(ax) != len(bx) {
		return false
	}
	for k := range ax {
		if !sameExpr(a, *ax[k], b, *bx[k]) {
			return false
		}
	}
	return true
}

func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// graftStores appends c's stores to m through the maps graft made.
func graftStores(m, c *Kernel, nodeOf, bufOf, loopOf []int) {
	for _, s := range c.Stores {
		at := s.At
		if at >= 0 {
			at = loopOf[at]
		}
		ns := Store{At: at, Barrier: s.Barrier, Run: s.Run}
		if !s.Barrier && !s.Run {
			ns.Buf, ns.Ix, ns.Val = bufOf[s.Buf], nodeOf[s.Ix], nodeOf[s.Val]
			for _, a := range s.Ax {
				ns.Ax = append(ns.Ax, nodeOf[a])
			}
		}
		m.Stores = append(m.Stores, ns)
	}
}

// merge is p with c's nodes beside its own, c's reads of v p's stored node,
// p's store st of v dropped.
func merge(p, c *Kernel, v Operand, st int) *Kernel {
	m := p.clone()
	m.Name = p.Name + "+" + c.Name
	val := p.Stores[st].Val
	m.Stores = append(m.Stores[:st:st], m.Stores[st+1:]...)
	coords := map[int]int{}
	for i, nd := range m.Nodes {
		if nd.Op == OpCoord {
			coords[nd.A] = i
		}
	}
	nodeOf, bufOf, loopOf := graft(m, c, v, val, func(a int) (int, bool) { x, ok := coords[a]; return x, ok })
	graftStores(m, c, nodeOf, bufOf, loopOf)
	return m
}

// joinsAtStores finds a value p writes by several stores an element, all of
// it, that c alone reads, at its own place.
func (f Fuse) joinsAtStores(p, c *Kernel, ks []*Kernel) (Operand, bool) {
	if !f.allows(p, c, true) || p.Work || c.Work || len(p.Loops) > 0 || len(c.Loops) > 0 {
		return Operand{}, false
	}
	count := map[Operand]int{}
	for _, s := range p.Stores {
		if !s.Barrier && !s.Run && s.At == -1 && s.Ax != nil {
			count[p.Bufs[s.Buf].Operand]++
		}
	}
	for o, n := range count {
		if o.Kind != OperandValue || n < 2 || n*p.Elements() != c.Elements() || !c.readsOwn(o) || !soleReader(ks, c, o) {
			continue
		}
		return o, true
	}
	return Operand{}, false
}

// mergeAtStores is p with c computed at each of p's stores of v: c's
// coordinates the store's, its read of v the stored node.
func mergeAtStores(p, c *Kernel, v Operand) *Kernel {
	m := p.clone()
	m.Name = p.Name + "+" + c.Name
	var kept, at []Store
	for _, s := range m.Stores {
		if !s.Barrier && !s.Run && s.At == -1 && m.Bufs[s.Buf].Operand == v {
			at = append(at, s)
			continue
		}
		kept = append(kept, s)
	}
	m.Stores = kept
	for _, s := range at {
		nodeOf, bufOf, loopOf := graft(m, c, v, s.Val, func(a int) (int, bool) { return s.Ax[a], true })
		graftStores(m, c, nodeOf, bufOf, loopOf)
	}
	return m
}
