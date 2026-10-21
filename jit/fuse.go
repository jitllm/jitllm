package jit

// Pair is a producer's primitive kind a consumer's may be fused into;
// Additive asks that the consumer add a term to the producer's value. AtStores
// fuses a consumer of a value the producer writes at several places an
// element: the consumer computed at each of the producer's stores.
type Pair struct {
	From, To PrimKind
	Additive bool
	AtStores bool
}

// Fuse inlines a consumer into its producer where their pair is allowed, the
// consumer reads the producer's value at its own element and is that
// value's one reader, and nothing between them touches what either does.
type Fuse struct {
	Pairs []Pair
}

func (Fuse) Name() string { return "fuse" }

func (f Fuse) Apply(prog []*Nest, t Target) ([]*Nest, error) {
	out := append([]*Nest(nil), prog...)
	for ci := 0; ci < len(out); ci++ {
		c := out[ci]
		for pi := ci - 1; pi >= 0; pi-- {
			p := out[pi]
			v, st, ok := f.joins(p, c, out)
			if ok {
				out[pi] = merge(p, c, v, st)
			} else if v, ok = f.joinsAtStores(p, c, out); ok {
				out[pi] = mergeAtStores(p, c, v)
			}
			if ok {
				out[pi].Passes = append(append([]string(nil), p.Passes...), f.Name())
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

func (f Fuse) joins(p, c *Nest, prog []*Nest) (v Operand, st int, ok bool) {
	allowed := false
	for _, pr := range f.Pairs {
		allowed = allowed || pr.From == p.Kind && pr.To == c.Kind && (!pr.Additive || c.adds())
	}
	if !allowed || !sameSpace(p, c) {
		return Operand{}, 0, false
	}
	for i, s := range p.Stores {
		o := p.Bufs[s.Buf].Operand
		if s.At != -1 || o.Kind != OperandValue || !p.atElement(s.Ix) || !c.readsAtElement(o) {
			continue
		}
		readers := 0
		for _, n := range prog {
			if n != p && n.reads(o) {
				readers++
			}
		}
		if readers == 1 {
			return o, i, true
		}
	}
	return Operand{}, 0, false
}

func sameSpace(a, b *Nest) bool {
	if len(a.Space) != len(b.Space) {
		return false
	}
	for i := range a.Space {
		if a.Space[i] != b.Space[i] {
			return false
		}
	}
	return true
}

// atElement reports whether index node i is the element's own place in a
// one-dimensional space.
func (n *Nest) atElement(i int) bool {
	nd := n.Nodes[i]
	return len(n.Space) == 1 && nd.Op == OpCoord && nd.A == 0
}

func (n *Nest) reads(o Operand) bool {
	for _, nd := range n.Nodes {
		if nd.Op == OpLoad && n.Bufs[nd.Buf].Operand == o {
			return true
		}
	}
	return false
}

func (n *Nest) writes(o Operand) bool {
	for _, s := range n.Stores {
		if !s.Barrier && n.Bufs[s.Buf].Operand == o {
			return true
		}
	}
	return false
}

// readsAtElement reports whether n reads o and only at its element.
func (n *Nest) readsAtElement(o Operand) bool {
	any := false
	for _, nd := range n.Nodes {
		if nd.Op == OpLoad && n.Bufs[nd.Buf].Operand == o {
			if !n.atElement(nd.A) {
				return false
			}
			any = true
		}
	}
	return any
}

// conflicts reports whether c may not move before p: p writes what c reads,
// or reads or writes what c writes.
func conflicts(p, c *Nest) bool {
	for _, b := range c.Bufs {
		if p.writes(b.Operand) {
			return true
		}
	}
	for _, s := range c.Stores {
		if s.Barrier {
			continue
		}
		o := c.Bufs[s.Buf].Operand
		if p.reads(o) || p.writes(o) {
			return true
		}
	}
	return false
}

// merge is p with c's nodes after its own, c's reads of v p's stored node,
// p's store st of v dropped.
func merge(p, c *Nest, v Operand, st int) *Nest {
	m := p.clone()
	m.Name = p.Name + "+" + c.Name
	val := p.Stores[st].Val
	m.Stores = append(m.Stores[:st:st], m.Stores[st+1:]...)
	bufOf := map[int]int{}
	for i, b := range c.Bufs {
		j := -1
		for k, mb := range m.Bufs {
			if mb.Operand == b.Operand && mb.Kind == b.Kind {
				j = k
			}
		}
		if j < 0 {
			j = m.buf(b)
		}
		bufOf[i] = j
	}
	loopOf := make([]int, len(c.Loops))
	for i, l := range c.Loops {
		if l.Parent >= 0 {
			l.Parent = loopOf[l.Parent]
		}
		loopOf[i] = m.loop(l.Trip, l.Parent)
	}
	nodeOf := make([]int, len(c.Nodes))
	for i := range nodeOf {
		nodeOf[i] = -1
	}
	for i, nd := range c.Nodes {
		if nd.Op == OpLoad && c.Bufs[nd.Buf].Operand == v {
			nodeOf[i] = val
			continue
		}
		if nd.Op == OpIter || nd.Op == OpFold {
			nd.At = loopOf[nd.At]
		}
		if nd.Op == OpLoad {
			nd.Buf = bufOf[nd.Buf]
		}
		nodeOf[i] = m.add(nd)
	}
	for i := range c.Nodes {
		if c.Nodes[i].Op == OpLoad && c.Bufs[c.Nodes[i].Buf].Operand == v {
			continue
		}
		nd := &m.Nodes[nodeOf[i]]
		for _, a := range m.args(nd) {
			*a = nodeOf[*a]
		}
	}
	for _, s := range c.Stores {
		at := s.At
		if at >= 0 {
			at = loopOf[at]
		}
		if s.Barrier {
			m.Stores = append(m.Stores, Store{Barrier: true, At: at})
			continue
		}
		m.Stores = append(m.Stores, Store{Buf: bufOf[s.Buf], Ix: nodeOf[s.Ix], Val: nodeOf[s.Val], At: at})
	}
	return m
}

// adds reports whether a map stores, at its element and nothing else, the
// sum of two reads at its element.
func (n *Nest) adds() bool {
	if len(n.Stores) != 1 || len(n.Loops) != 0 {
		return false
	}
	s := n.Stores[0]
	v := n.Nodes[s.Val]
	if s.Barrier || s.At != -1 || !n.atElement(s.Ix) || v.Op != OpAdd {
		return false
	}
	for _, a := range []int{v.A, v.B} {
		if l := n.Nodes[a]; l.Op != OpLoad || !n.atElement(l.A) {
			return false
		}
	}
	return true
}

// joinsAtStores finds a value p writes by several stores an element, all of
// it, that c alone reads, at its own element.
func (f Fuse) joinsAtStores(p, c *Nest, prog []*Nest) (Operand, bool) {
	allowed := false
	for _, pr := range f.Pairs {
		allowed = allowed || pr.AtStores && pr.From == p.Kind && pr.To == c.Kind
	}
	if !allowed || p.Work || c.Work || len(p.Space) != 1 || len(c.Space) != 1 {
		return Operand{}, false
	}
	count := map[Operand]int{}
	for _, s := range p.Stores {
		if !s.Barrier && s.At == -1 {
			count[p.Bufs[s.Buf].Operand]++
		}
	}
	for o, k := range count {
		if o.Kind != OperandValue || k < 2 || k*p.elements() != c.elements() || !c.readsAtElement(o) || !soleReader(prog, c, o) {
			continue
		}
		return o, true
	}
	return Operand{}, false
}

// mergeAtStores is p with c computed at each of p's stores of v: c's
// element the store's place, its read of v the stored node.
func mergeAtStores(p, c *Nest, v Operand) *Nest {
	m := p.clone()
	m.Name = p.Name + "+" + c.Name
	var kept []Store
	var at [][2]int
	for _, s := range m.Stores {
		if !s.Barrier && s.At == -1 && m.Bufs[s.Buf].Operand == v {
			at = append(at, [2]int{s.Ix, s.Val})
			continue
		}
		kept = append(kept, s)
	}
	m.Stores = kept
	bufOf := map[int]int{}
	for i, b := range c.Bufs {
		if b.Operand == v {
			continue
		}
		j := -1
		for k, mb := range m.Bufs {
			if mb.Operand == b.Operand && mb.Kind == b.Kind {
				j = k
			}
		}
		if j < 0 {
			j = m.buf(b)
		}
		bufOf[i] = j
	}
	for _, s := range at {
		nodeOf := make([]int, len(c.Nodes))
		for i, nd := range c.Nodes {
			switch {
			case nd.Op == OpCoord:
				nodeOf[i] = s[0]
				continue
			case nd.Op == OpLoad && c.Bufs[nd.Buf].Operand == v:
				nodeOf[i] = s[1]
				continue
			case nd.Op == OpLoad:
				nd.Buf = bufOf[nd.Buf]
			}
			for _, a := range c.args(&nd) {
				*a = nodeOf[*a]
			}
			nodeOf[i] = m.add(nd)
		}
		for _, st := range c.Stores {
			m.Stores = append(m.Stores, Store{Buf: bufOf[st.Buf], Ix: nodeOf[st.Ix], Val: nodeOf[st.Val], At: -1})
		}
	}
	return m
}
