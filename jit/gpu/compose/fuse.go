package compose

// Fuse is the fusion pass over a program's lowered groups: each group is
// merged into the group before it where it reads a value that group stores
// and the merged group computes the same, the read replaced by the node that
// computed the value.
//
//   - Over the same rows and width, a read of the value's own element is the
//     producer's node.
//   - A consumer over a flat row of words (Rows 1) after a producer that
//     reduces over its rows becomes a stage of the producer's rows: its
//     words a row's, each read the producer's node at the element in its row.
//   - The same consumer after an elementwise producer takes the producer into
//     its own space: each read the producer's node at the element read, the
//     producer's store made there.
//
// What the merge cannot express, what the parameters do not allow, and a
// merged group that would read a buffer it writes (RULE 13) leave the
// groups apart. Every store a group made is kept.
func Fuse(pr *Program, gs []Group, f FuseParams) []Group {
	if len(gs) == 0 || f.Off {
		return gs
	}
	out := []Group{gs[0]}
	for _, c := range gs[1:] {
		if c.Extern {
			out = append(out, c)
			continue
		}
		// The nearest earlier group c may join: nothing between writes what
		// c reads, or reads or writes what c writes.
		merged := false
		for j := len(out) - 1; j >= 0 && !merged; j-- {
			p := &out[j]
			if !p.Extern && f.allows(pr, *p, c) {
				if m, ok := fuse(*p, c); ok && !pr.readsWhatItWrites(m) {
					*p, merged = m, true
					break
				}
			}
			if pr.conflicts(*p, c) {
				break
			}
		}
		if !merged {
			out = append(out, c)
		}
	}
	return out
}

// conflicts reports whether c may not move before group p: p writes what c
// reads, or reads or writes what c writes, by value or by buffer.
func (pr *Program) conflicts(p, c Group) bool {
	type key struct{ v, b int }
	set := func(g Group, out bool) []key {
		var ks []key
		for _, i := range g.Prims {
			vs := pr.Prims[i].In
			if out {
				vs = pr.Prims[i].Out
			}
			for _, v := range vs {
				ks = append(ks, key{v, pr.Vals[v].Buf})
			}
		}
		return ks
	}
	meet := func(a, b []key) bool {
		for _, x := range a {
			for _, y := range b {
				if x.v == y.v || x.b >= 0 && x.b == y.b {
					return true
				}
			}
		}
		return false
	}
	pin, pout, cin, cout := set(p, false), set(p, true), set(c, false), set(c, true)
	return meet(pout, cin) || meet(pin, cout) || meet(pout, cout)
}

// FuseParams is the fusion pass's switch and parameters: Off runs no
// fusion; Pairs[k] has bit c where a value prim kind k computes may be
// fused into the prim of kind c reading it.
type FuseParams struct {
	Off   bool
	Pairs [numPrimKinds]uint32
	// Only[k][c], where set, is 1 + the buffer a pair's value must live in.
	Only [numPrimKinds][numPrimKinds]int
}

// Allow lets a value of a prim of kind from fuse into a prim of kind to.
func (f *FuseParams) Allow(from, to PrimKind) { f.Pairs[from] |= 1 << to }

// AllowIn is Allow for values in buffer buf alone.
func (f *FuseParams) AllowIn(from, to PrimKind, buf int) {
	f.Allow(from, to)
	f.Only[from][to] = buf + 1
}

// allows reports whether consumer c's prim may fuse every value of p's it
// reads.
func (f FuseParams) allows(pr *Program, p, c Group) bool {
	maker := map[int]PrimKind{}
	for _, i := range p.Prims {
		for _, v := range pr.Prims[i].Out {
			maker[v] = pr.Prims[i].Kind
		}
	}
	any := false
	for _, i := range c.Prims {
		o := pr.Prims[i]
		for _, v := range o.In {
			if k, ok := maker[v]; ok {
				if f.Pairs[k]&(1<<o.Kind) == 0 {
					return false
				}
				if b := f.Only[k][o.Kind]; b != 0 && pr.Vals[v].Buf != b-1 {
					return false
				}
				any = true
			}
		}
	}
	return any
}

// readsWhatItWrites reports whether group g reads a value whose buffer it
// writes.
func (pr *Program) readsWhatItWrites(g Group) bool {
	read, wrote := map[int]bool{}, map[int]bool{}
	for _, n := range g.Row.Nodes {
		if n.Kind == NLoad {
			if b := g.Binds[n.Buf]; b.Kind == BindVal || b.Kind == BindQ8 || b.Kind == BindQ8X {
				if c := pr.Vals[b.Index].Buf; c >= 0 {
					read[c] = true
				}
			}
		}
	}
	mark := func(sts []Store) {
		for _, st := range sts {
			if b := g.Binds[st.Buf]; b.Kind == BindVal || b.Kind == BindQ8 || b.Kind == BindQ8X {
				if c := pr.Vals[b.Index].Buf; c >= 0 {
					wrote[c] = true
				}
			}
		}
	}
	mark(g.Row.Stores)
	for _, s := range g.Row.Stages {
		mark(s.Stores)
	}
	for c := range wrote {
		if read[c] {
			return true
		}
	}
	return false
}

// fuse is producer p and consumer c as one group, false where no form of
// the merge applies.
func fuse(p, c Group) (Group, bool) {
	stored := map[int]int{} // value -> the node p stores it as
	for _, st := range p.Row.Stores {
		if b := p.Binds[st.Buf]; b.Kind == BindVal && !st.Indexed {
			stored[b.Index] = st.Val
		}
	}
	reads := false
	for _, n := range c.Row.Nodes {
		if n.Kind == NLoad && c.Binds[n.Buf].Kind == BindVal {
			if _, ok := stored[c.Binds[n.Buf].Index]; ok {
				reads = true
			}
		}
	}
	if !reads || len(p.Row.Stages) > 0 || len(c.Row.Stages) > 0 {
		return Group{}, false
	}
	switch {
	case c.Row.Width == p.Row.Width && c.Row.Rows == p.Row.Rows:
		return fuseSame(p, c, stored)
	case c.Row.Rows == 1 && p.Row.reduces():
		return fuseStage(p, c, stored)
	case c.Row.Rows == 1 && !p.Row.reduces():
		return fuseInto(p, c, stored)
	}
	return Group{}, false
}

// merger copies a group's nodes into another's.
type merger struct {
	dst   *Group
	src   *Group
	m     map[int]int
	sub   func(n int) (int, bool)
	bad   bool
	binds map[[2]int]int
}

func newMerger(dst, src *Group) *merger {
	mg := &merger{dst: dst, src: src, m: map[int]int{}, binds: map[[2]int]int{}}
	for i, b := range dst.Binds {
		mg.binds[bindKey(dst.Row.Bufs[i].Kind, b)] = i
	}
	return mg
}

func bindKey(k BufKind, b Bind) [2]int { return [2]int{int(k)<<8 | int(b.Kind), b.Index} }

// buf is src's buffer i in dst: the one binding the same, or a new one.
func (mg *merger) buf(i int) int {
	k := bindKey(mg.src.Row.Bufs[i].Kind, mg.src.Binds[i])
	if j, ok := mg.binds[k]; ok {
		return j
	}
	mg.dst.Row.Bufs = append(mg.dst.Row.Bufs, mg.src.Row.Bufs[i])
	mg.dst.Binds = append(mg.dst.Binds, mg.src.Binds[i])
	j := len(mg.dst.Row.Bufs) - 1
	mg.binds[k] = j
	return j
}

// node is src's node n in dst.
func (mg *merger) node(n int) int {
	if v, ok := mg.m[n]; ok {
		return v
	}
	if mg.sub != nil {
		if v, ok := mg.sub(n); ok {
			mg.m[n] = v
			return v
		}
	}
	nd := mg.src.Row.Nodes[n]
	f := mg.node
	switch nd.Kind {
	case NConst, NUConst, NIConst, NColumn, NRowIdx:
	case NLoad:
		nd.Buf = mg.buf(nd.Buf)
		if nd.Indexed {
			nd.Ix = f(nd.Ix)
		}
	case NUnary, NWindow, NShuffle, NReduce:
		nd.A = f(nd.A)
	case NBinary, NUBinary:
		nd.A, nd.B = f(nd.A), f(nd.B)
		if nd.ternary() {
			nd.C = f(nd.C)
		}
	case NLess:
		nd.A, nd.B = f(nd.A), f(nd.B)
	case NSelect, NDot4:
		nd.A, nd.B, nd.C = f(nd.A), f(nd.B), f(nd.C)
	case NAt:
		nd.A, nd.Ix = f(nd.A), f(nd.Ix)
	default:
		mg.bad = true
		return 0
	}
	v := mg.dst.Row.Add(nd)
	mg.m[n] = v
	return v
}

func (mg *merger) store(st Store) Store {
	st.Buf, st.Val = mg.buf(st.Buf), mg.node(st.Val)
	if st.Indexed {
		st.Ix = mg.node(st.Ix)
	}
	return st
}

// readOf is the value consumer c's load n reads where p stores it, else -1.
func readOf(c *Group, n int, stored map[int]int) int {
	nd := c.Row.Nodes[n]
	if nd.Kind != NLoad || c.Binds[nd.Buf].Kind != BindVal {
		return -1
	}
	if _, ok := stored[c.Binds[nd.Buf].Index]; !ok {
		return -1
	}
	return c.Binds[nd.Buf].Index
}

func clone(g Group) Group {
	g.Row.Nodes = append([]Node(nil), g.Row.Nodes...)
	g.Row.Bufs = append([]Buf(nil), g.Row.Bufs...)
	g.Row.Stores = append([]Store(nil), g.Row.Stores...)
	g.Binds = append([]Bind(nil), g.Binds...)
	g.Prims = append([]int(nil), g.Prims...)
	return g
}

// fuseSame is c after p over the same rows: a plain read of a value p
// stores is p's node.
func fuseSame(p, c Group, stored map[int]int) (Group, bool) {
	d := clone(p)
	mg := newMerger(&d, &c)
	mg.sub = func(n int) (int, bool) {
		if v := readOf(&c, n, stored); v >= 0 && !c.Row.Nodes[n].Indexed {
			return stored[v], true
		}
		return 0, false
	}
	for i := range c.Row.Nodes {
		if v := readOf(&c, i, stored); v >= 0 && c.Row.Nodes[i].Indexed {
			return Group{}, false // a read elsewhere in the row
		}
	}
	for _, st := range c.Row.Stores {
		d.Row.Stores = append(d.Row.Stores, mg.store(st))
	}
	if mg.bad {
		return Group{}, false
	}
	d.Prims = append(d.Prims, c.Prims...)
	return d.compact(), true
}

// fuseStage is consumer c over a flat row of words made a stage of
// producer p's rows: a word of c is a row's when c's words are p's rows
// times a row's words, c's flat word the row's offset plus the stage's
// column, and a read of a value p stores, at an index affine in the word
// with a row of words a row of elements, p's node at the element in its
// row.
func fuseStage(p, c Group, stored map[int]int) (Group, bool) {
	R, w := p.Row.Rows, p.Row.Width
	if c.Row.Width%R != 0 {
		return Group{}, false
	}
	words := c.Row.Width / R
	d := clone(p)
	mg := newMerger(&d, &c)
	r := &d.Row
	col := r.Column()
	flat := r.UBin(OpAdd, r.UBin(OpMul, r.RowIdx(), r.UConst(words)), col)
	var local func(n int) (int, bool)
	mg.sub = func(n int) (int, bool) {
		nd := c.Row.Nodes[n]
		if nd.Kind == NColumn {
			return flat, true
		}
		if v := readOf(&c, n, stored); v >= 0 {
			if !nd.Indexed {
				return 0, false
			}
			a, ok := c.Row.colCoef(nd.Ix)
			if !ok || a*words != w {
				mg.bad = true
				return 0, true
			}
			ix, ok := local(nd.Ix)
			if !ok {
				mg.bad = true
				return 0, true
			}
			return r.Add(Node{Kind: NAt, A: stored[v], Ix: ix}), true
		}
		return 0, false
	}
	// local is index node n with the flat word the stage's column: the
	// element in the row.
	lm := map[int]int{}
	local = func(n int) (int, bool) {
		if v, ok := lm[n]; ok {
			return v, true
		}
		nd := c.Row.Nodes[n]
		var v int
		switch {
		case nd.Kind == NColumn:
			v = col
		case nd.Kind == NUConst:
			v = r.UConst(nd.At)
		case nd.Kind == NUBinary && (nd.Op == OpAdd || nd.Op == OpMul):
			a, ok1 := local(nd.A)
			b, ok2 := local(nd.B)
			if !ok1 || !ok2 {
				return 0, false
			}
			v = r.UBin(nd.Op, a, b)
		default:
			return 0, false
		}
		lm[n] = v
		return v, true
	}
	st := Stage{Width: words}
	for _, s := range c.Row.Stores {
		if !s.Indexed {
			return Group{}, false
		}
		st.Stores = append(st.Stores, mg.store(s))
	}
	if mg.bad {
		return Group{}, false
	}
	r.Stages = append(r.Stages, st)
	d.Prims = append(d.Prims, c.Prims...)
	return d.compact(), true
}

// colCoef is index node n's coefficient on the column, where n is the
// column scaled and offset by constants.
func (g *Row) colCoef(n int) (int, bool) {
	nd := g.Nodes[n]
	switch {
	case nd.Kind == NColumn:
		return 1, true
	case nd.Kind == NUConst:
		return 0, true
	case nd.Kind == NUBinary && nd.Op == OpAdd:
		a, ok1 := g.colCoef(nd.A)
		b, ok2 := g.colCoef(nd.B)
		return a + b, ok1 && ok2
	case nd.Kind == NUBinary && nd.Op == OpMul:
		if g.Nodes[nd.B].Kind == NUConst {
			a, ok := g.colCoef(nd.A)
			return a * g.Nodes[nd.B].At, ok
		}
		if g.Nodes[nd.A].Kind == NUConst {
			a, ok := g.colCoef(nd.B)
			return a * g.Nodes[nd.A].At, ok
		}
	}
	return 0, false
}

// fuseInto is elementwise producer p taken into consumer c's flat space:
// each read of a value p stores is p's node at the element read (NAt), and
// p's store of it made at that element. Every store of p must be of a
// value c reads, and p reads nothing but its rows' elements and scalars.
func fuseInto(p, c Group, stored map[int]int) (Group, bool) {
	for _, n := range p.Row.Nodes {
		switch {
		case n.Kind == NColumn || n.Kind == NRowIdx || n.Kind == NLoad && n.Indexed:
			return Group{}, false
		case n.Kind == NLoad && p.Row.Bufs[n.Buf].Kind != BufRow && p.Row.Bufs[n.Buf].Kind != BufScalar:
			return Group{}, false
		}
	}
	read := map[int]bool{}
	for i := range c.Row.Nodes {
		if v := readOf(&c, i, stored); v >= 0 {
			if !c.Row.Nodes[i].Indexed {
				return Group{}, false
			}
			read[v] = true
		}
	}
	for v := range stored {
		if !read[v] {
			return Group{}, false
		}
	}
	d := Group{Row: Row{Width: c.Row.Width, Rows: c.Row.Rows}, Prims: append(append([]int(nil), p.Prims...), c.Prims...)}
	pm := newMerger(&d, &p)
	cm := &merger{dst: &d, src: &c, m: map[int]int{}, binds: pm.binds}
	outBuf := map[int]int{}
	for _, st := range p.Row.Stores {
		outBuf[p.Binds[st.Buf].Index] = st.Buf
	}
	var pst []Store
	cm.sub = func(n int) (int, bool) {
		nd := c.Row.Nodes[n]
		v := readOf(&c, n, stored)
		if v < 0 {
			return 0, false
		}
		y := pm.node(stored[v])
		ix := cm.node(nd.Ix)
		at := d.Row.Add(Node{Kind: NAt, A: y, Ix: ix})
		pst = append(pst, Store{Buf: pm.buf(outBuf[v]), Val: at, Indexed: true, Ix: ix})
		return at, true
	}
	var cst []Store
	for _, st := range c.Row.Stores {
		cst = append(cst, cm.store(st))
	}
	if pm.bad || cm.bad {
		return Group{}, false
	}
	d.Row.Stores = append(pst, cst...)
	return d.compact(), true
}

// compact drops the buffers nothing reads or writes, renumbering the rest.
func (g Group) compact() Group {
	used := make([]bool, len(g.Row.Bufs))
	for _, n := range g.Row.Nodes {
		if n.Kind == NLoad {
			used[n.Buf] = true
		}
	}
	mark := func(sts []Store) {
		for _, s := range sts {
			used[s.Buf] = true
		}
	}
	mark(g.Row.Stores)
	for _, s := range g.Row.Stages {
		mark(s.Stores)
	}
	at := make([]int, len(g.Row.Bufs))
	var bufs []Buf
	var binds []Bind
	for i, u := range used {
		at[i] = len(bufs)
		if u {
			bufs = append(bufs, g.Row.Bufs[i])
			binds = append(binds, g.Binds[i])
		}
	}
	if len(bufs) == len(g.Row.Bufs) {
		return g
	}
	for i := range g.Row.Nodes {
		if g.Row.Nodes[i].Kind == NLoad {
			g.Row.Nodes[i].Buf = at[g.Row.Nodes[i].Buf]
		}
	}
	remap := func(sts []Store) {
		for i := range sts {
			sts[i].Buf = at[sts[i].Buf]
		}
	}
	remap(g.Row.Stores)
	for i := range g.Row.Stages {
		remap(g.Row.Stages[i].Stores)
	}
	g.Row.Bufs, g.Binds = bufs, binds
	return g
}

// PruneParams is the dead-store pass's switch and parameters: Drop[k] lets
// it drop the store of a value a prim of kind k computes.
type PruneParams struct {
	Off  bool
	Drop [numPrimKinds]bool
}

// Prune is the dead-store pass: a group's store of a value it computes,
// which no prim outside the group reads and the program does not hand out
// (Out), is dropped where the parameters let the kind's be.
func Prune(pr *Program, gs []Group, f PruneParams) []Group {
	if f.Off {
		return gs
	}
	maker := map[int]int{}
	for i, o := range pr.Prims {
		for _, v := range o.Out {
			maker[v] = i
		}
	}
	for gi := range gs {
		g := &gs[gi]
		if g.Extern {
			continue
		}
		in := map[int]bool{}
		for _, i := range g.Prims {
			in[i] = true
		}
		dead := func(v int) bool {
			m, ok := maker[v]
			if !ok || !in[m] || !f.Drop[pr.Prims[m].Kind] || v < len(pr.Out) && pr.Out[v] {
				return false
			}
			for i, o := range pr.Prims {
				if in[i] {
					continue
				}
				for _, x := range o.In {
					if x == v {
						return false
					}
				}
			}
			return true
		}
		var keep []Store
		for _, st := range g.Row.Stores {
			if b := g.Binds[st.Buf]; b.Kind == BindVal && !st.Indexed && dead(b.Index) {
				continue
			}
			keep = append(keep, st)
		}
		if len(keep) != len(g.Row.Stores) {
			g.Row.Stores = keep
			*g = g.compact()
		}
	}
	return gs
}
