package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/ir"
)

// Coarsen runs a chain of maps and reductions, each after the first reading
// what an earlier one wrote, as one thread a Unit of the chain's finest
// elements. A member is computed where its value is read, from its own
// kernel's nodes at the reading coordinates: a finer one at each element of
// the thread's unit, a coarser reduction as each thread's partial over its
// own terms exchanged through a workgroup slot and combined over its group's
// slots. What is not a temporary is stored. Where the chain
// reads the last phase of a workgroup row, it joins that row as a further
// phase, the row's value recomputed from the phase's own expression.
type Coarsen struct {
	Unit int
}

func (Coarsen) Name() string { return "coarsen" }

func (c Coarsen) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	out := append([]*Kernel(nil), ks...)
	for i := 0; i < len(out); i++ {
		n := c.chain(out, i)
		if n == 0 {
			continue
		}
		var host *Kernel
		if i > 0 && out[i-1].Work && out[i-1].Part == out[i].Part {
			host = out[i-1]
		}
		m, ok := c.form(out[i:i+n], host)
		if !ok {
			continue
		}
		if host != nil {
			m.Passes = append(append([]string(nil), host.Passes...), c.Name())
			out[i-1] = m
			out = append(out[:i], out[i+n:]...)
			i--
			continue
		}
		m.Passes = append(append([]string(nil), out[i].Passes...), c.Name())
		out = append(append(out[:i:i], m), out[i+n:]...)
	}
	return out, nil
}

// terms is a member's finest elements: its elements times its loop's trips.
func terms(k *Kernel) int {
	t := k.elements()
	if len(k.Loops) == 1 {
		t *= k.Loops[0].Trip + firstTerm(k)
	}
	return t
}

// firstTerm is 1 where a reduction begins from its first term.
func firstTerm(k *Kernel) int {
	if f, ok := k.fold(); ok && k.Nodes[k.Nodes[f].B].Op != OpConst {
		return 1
	}
	return 0
}

// fold is a reduction's one fold.
func (k *Kernel) fold() (int, bool) {
	if len(k.Loops) != 1 {
		return 0, false
	}
	fs := k.foldsOf(0)
	if len(fs) != 1 {
		return 0, false
	}
	return fs[0], true
}

// chain is how many kernels from i form a chain: naive maps and reductions
// of one part, each after the first reading what an earlier one wrote, one
// of them at the unit's granularity. Zero for none.
func (c Coarsen) chain(ks []*Kernel, i int) int {
	written := map[Operand]bool{}
	j := i
	for ; j < len(ks); j++ {
		k := ks[j]
		if k.Work || k.Part != ks[i].Part || k.Kind != PrimMap && k.Kind != PrimReduce || len(k.Loops) > 1 {
			break
		}
		if k.Kind == PrimReduce {
			if _, ok := k.fold(); !ok {
				break
			}
		}
		if j > i {
			linked := false
			for _, b := range k.Bufs {
				linked = linked || b.Kind == BufIn && written[b.Operand]
			}
			if !linked {
				break
			}
		}
		for _, s := range k.Stores {
			if !s.Barrier && !s.Run {
				written[k.Bufs[s.Buf].Operand] = true
			}
		}
	}
	if j-i < 2 {
		return 0
	}
	E := 0
	for _, k := range ks[i:j] {
		E = max(E, terms(k))
	}
	for _, k := range ks[i:j] {
		if E/k.elements() == c.Unit {
			return j - i
		}
	}
	return 0
}

type coarse struct {
	c       Coarsen
	m       *Kernel
	members []*Kernel
	E, T    int
	u       int
	local   int
	memo    map[string]int
	host    *Kernel
	hostVal map[Operand]Store
	err     error
}

func (c Coarsen) form(members []*Kernel, host *Kernel) (*Kernel, bool) {
	E := 0
	for _, k := range members {
		E = max(E, terms(k))
	}
	U := c.Unit
	if E%U != 0 {
		return nil, false
	}
	for _, k := range members {
		g := E / k.elements()
		if E%k.elements() != 0 || g%U != 0 && U%g != 0 {
			return nil, false
		}
	}
	T := E / U
	k := &coarse{c: c, members: members, E: E, T: T, memo: map[string]int{}}
	if host != nil {
		if host.Space[0] != 1 || T > host.Group {
			return nil, false
		}
		k.m = host.clone()
		k.m.Name = host.Name + "+quantized"
		k.host, k.hostVal = host, map[Operand]Store{}
		for _, s := range host.Stores {
			if s.At >= 0 && !s.Barrier && !s.Run && host.Loops[s.At].Elem > 0 && s.Ax != nil {
				k.hostVal[host.Bufs[s.Buf].Operand] = s
			}
		}
		th := k.m.add(Node{Op: OpThread, Type: ir.U32})
		k.u = k.m.bin(OpMin, ir.U32, k.m.bin(OpAdd, ir.U32, th, k.m.u32(0)), k.m.u32(T-1))
	} else {
		k.m = &Kernel{Name: "chain", Kind: members[0].Kind, Space: []int{T}, Group: members[0].Group, Part: members[0].Part}
		k.u = k.m.add(Node{Op: OpCoord, A: 0, Type: ir.U32})
	}
	k.local = k.m.add(Node{Op: OpLocal, Type: ir.U32, A: k.u})
	for _, n := range members {
		for _, s := range n.Stores {
			if s.Barrier || s.Run || n.Bufs[s.Buf].Operand.Temp {
				continue
			}
			for _, idx := range k.indices(n) {
				coords := k.m.decode(idx, n.Space)
				v := k.produce(n, s, coords)
				ns := Store{Buf: k.m.bufLike(n.Bufs[s.Buf]), At: -1}
				ns.Val = v
				leaf := k.coordLeaf(n, coords)
				if s.Ax != nil {
					for _, a := range s.Ax {
						ns.Ax = append(ns.Ax, copyInto(k.m, n, a, leaf))
					}
				} else {
					ns.Ix = copyInto(k.m, n, s.Ix, leaf)
				}
				k.m.Stores = append(k.m.Stores, ns)
			}
		}
	}
	if k.err != nil {
		return nil, false
	}
	if host != nil {
		k.m.Stores = append(k.m.Stores, Store{Barrier: true, At: -1})
	}
	return k.m, true
}

// indices is the flat elements of member n the thread computes.
func (k *coarse) indices(n *Kernel) []int {
	m, U := k.m, k.c.Unit
	g := k.E / n.elements()
	switch {
	case g < U:
		var out []int
		for j := 0; j < U/g; j++ {
			out = append(out, m.bin(OpAdd, ir.U32, m.bin(OpMul, ir.U32, k.u, m.u32(U/g)), m.u32(j)))
		}
		return out
	case g == U:
		return []int{k.u}
	}
	return []int{m.bin(OpDiv, ir.U32, k.u, m.u32(g/U))}
}

// decode is flat index e's coordinates in a space of extents ext.
func (k *Kernel) decode(e int, ext []int) []int {
	c := make([]int, len(ext))
	for d := len(ext) - 1; d > 0; d-- {
		c[d] = k.bin(OpRem, ir.U32, e, k.u32(ext[d]))
		e = k.bin(OpDiv, ir.U32, e, k.u32(ext[d]))
	}
	if len(ext) > 0 {
		c[0] = e
	}
	return c
}

// coordLeaf maps member n's coordinates to coords and its chain reads to
// their producers.
func (k *coarse) coordLeaf(n *Kernel, coords []int) func(i int, nd Node) (int, bool) {
	return func(i int, nd Node) (int, bool) {
		switch nd.Op {
		case OpCoord:
			return coords[nd.A], true
		case OpLoad:
			return k.read(n, i, nd, coords)
		}
		return 0, false
	}
}

// read is member n's load i (operands copied, in nd) where the chain holds
// what it reads: the producing member's value at the place, or the host's.
func (k *coarse) read(n *Kernel, i int, nd Node, coords []int) (int, bool) {
	o := n.Bufs[nd.Buf].Operand
	if s, ok := k.hostVal[o]; ok && nd.Ax != nil && len(nd.Ax) == len(s.Ax) {
		jc := k.host.Loops[s.At].Elem - 1
		row := s.Ax[0]
		sub := map[int]int{jc: nd.Ax[len(nd.Ax)-1]}
		if h := k.host.Nodes[row]; h.Op == OpCoord && k.m.Nodes[nd.Ax[0]].Op == OpConst {
			sub[row] = nd.Ax[0]
		}
		return k.m.copyOf(s.Val, sub), true
	}
	for _, p := range k.members {
		for _, s := range p.Stores {
			if s.Barrier || s.Run || p.Bufs[s.Buf].Operand != o {
				continue
			}
			if p.storesOwn(s) && nd.Ax != nil && len(nd.Ax) == len(p.Space) {
				return k.produce(p, s, nd.Ax), true
			}
			if ix, ok := sameIndex(n, n.Nodes[i], p, s); ok && ix {
				return k.produce(p, s, coords), true
			}
		}
	}
	return 0, false
}

// sameIndex reports whether n's load reads the place p's store writes, the
// two index expressions alike with each coordinate the same axis.
func sameIndex(n *Kernel, load Node, p *Kernel, s Store) (bool, bool) {
	var a, b []int
	switch {
	case load.Ax != nil && s.Ax != nil && len(load.Ax) == len(s.Ax):
		a, b = load.Ax, s.Ax
	case load.Ax == nil && s.Ax == nil:
		a, b = []int{load.A}, []int{s.Ix}
	default:
		return false, false
	}
	if !sameSpace(n, p) {
		return false, false
	}
	for j := range a {
		if !alike(n, a[j], p, b[j]) {
			return false, true
		}
	}
	return true, true
}

// alike reports whether node x of a and node y of b compute the same from
// the same coordinates.
func alike(a *Kernel, x int, b *Kernel, y int) bool {
	p, q := a.Nodes[x], b.Nodes[y]
	if p.Op != q.Op || p.Type != q.Type || p.Imm != q.Imm || p.Op == OpLoad || p.Op == OpFold || p.Op == OpIter {
		return false
	}
	if p.Op == OpCoord {
		return p.A == q.A
	}
	pa, qa := a.args(&p), b.args(&q)
	if len(pa) != len(qa) {
		return false
	}
	for j := range pa {
		if !alike(a, *pa[j], b, *qa[j]) {
			return false
		}
	}
	return true
}

// produce is member n's value stored by s at its coordinates coords.
func (k *coarse) produce(n *Kernel, s Store, coords []int) int {
	key := fmt.Sprint(n.Name, &n.Nodes[0], s.Val, coords)
	g := k.E / n.elements()
	if n.Kind == PrimReduce && g > k.c.Unit {
		key = fmt.Sprint(n.Name, &n.Nodes[0], s.Val)
	}
	if v, ok := k.memo[key]; ok {
		return v
	}
	var v int
	if n.Kind == PrimReduce {
		v = k.reduce(n, s, coords, g)
	} else {
		v = copyInto(k.m, n, s.Val, k.coordLeaf(n, coords))
	}
	k.memo[key] = v
	return v
}

// term is reduction n's term j at coordinates coords.
func (k *coarse) term(n *Kernel, coords []int, j int) int {
	return k.termAt(n, coords, k.m.u32(j))
}

// reduce is reduction n's output: within the thread where its terms are,
// else the thread's partial exchanged over its group of threads.
func (k *coarse) reduce(n *Kernel, s Store, coords []int, g int) int {
	m, U := k.m, k.c.Unit
	f, _ := n.fold()
	t := n.Nodes[f].Type
	comb := n.Nodes[n.Nodes[f].A].Op
	K := n.Loops[0].Trip + firstTerm(n)
	start := -1
	if firstTerm(n) == 0 {
		start = m.add(n.Nodes[n.Nodes[f].B])
	}
	if g <= U {
		acc := start
		for j := 0; j < K; j++ {
			v := k.term(n, coords, j)
			if acc < 0 {
				acc = v
				continue
			}
			acc = m.bin(comb, t, acc, v)
		}
		return acc
	}
	tg := g / K
	if tg > U || U%tg != 0 {
		k.err = fmt.Errorf("a reduction of %d terms over %d", K, g)
		return m.u32(0)
	}
	count, slots := U/tg, g/U
	o := m.bin(OpDiv, ir.U32, k.u, m.u32(slots))
	oc := m.decode(o, n.Space)
	j0 := m.bin(OpMul, ir.U32, m.bin(OpRem, ir.U32, k.u, m.u32(slots)), m.u32(count))
	acc := start
	for c := 0; c < count; c++ {
		jj := j0
		if c > 0 {
			jj = m.bin(OpAdd, ir.U32, j0, m.u32(c))
		}
		v := k.termAt(n, oc, jj)
		if acc < 0 {
			acc = v
			continue
		}
		acc = m.bin(comb, t, acc, v)
	}
	at := m.bin(OpMul, ir.U32, m.bin(OpDiv, ir.U32, k.local, m.u32(slots)), m.u32(slots))
	init := -1
	if start >= 0 {
		init = m.add(n.Nodes[n.Nodes[f].B])
	}
	return m.add(Node{Op: OpExchange, Type: ir.F32, A: acc, B: at, C: init, Imm: int64(slots), At: int(comb)})
}

// termAt is reduction n's term at the index node jj: its step's term with
// the term's coordinate jj. A reduction from its first term counts its
// trips from the second, its term's coordinate the trip plus one.
func (k *coarse) termAt(n *Kernel, coords []int, jj int) int {
	f, _ := n.fold()
	it := n.iterOf(0)
	at := it
	if firstTerm(n) == 1 {
		at = -1
		for i, nd := range n.Nodes {
			if nd.Op == OpAdd && nd.A == it && n.Nodes[nd.B].Op == OpConst && n.Nodes[nd.B].Imm == 1 {
				at = i
			}
		}
		if at < 0 {
			k.err = fmt.Errorf("a reduction from its first term without its term's coordinate")
			return k.m.u32(0)
		}
	}
	leaf := k.coordLeaf(n, coords)
	step := n.Nodes[n.Nodes[f].A]
	t := step.B
	if t == f {
		t = step.A
	}
	return copyInto(k.m, n, t, func(i int, nd Node) (int, bool) {
		if i == at {
			return jj, true
		}
		return leaf(i, nd)
	})
}
