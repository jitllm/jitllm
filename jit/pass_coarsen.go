package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/ir"
)

// Coarsen runs a chain of maps and reductions, each after the first reading
// what an earlier one wrote, as one thread a unit of the chain's finest
// elements: Unit, or more where a reduction's group would span more than a
// warp's lanes -- the unit that keeps every combine within one warp. A member is computed where its value is read, from its own
// kernel's nodes at the reading coordinates: a finer one at each element of
// the thread's unit, a coarser reduction as each thread's partial over its
// own terms exchanged through a workgroup slot and combined over its group's
// slots. What is not a temporary is stored. Where the chain
// reads the last phase of a workgroup row -- one workgroup, so the phase
// sees every element the row wrote -- it joins that row as a further phase,
// the row's value recomputed from the phase's own expression.
type Coarsen struct {
	Unit int
}

func (Coarsen) Name() string { return "coarsen" }

func (c Coarsen) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	out := append([]*Kernel(nil), ks...)
	for i := 0; i < len(out); i++ {
		n, unit := c.chain(out, i, d.Lanes)
		if n == 0 {
			continue
		}
		var host *Kernel
		if i > 0 && out[i-1].Work && out[i-1].Part == out[i].Part {
			host = out[i-1]
		}
		m, ok := Coarsen{Unit: unit}.form(out[i:i+n], host, d.Lanes)
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
	t := k.Elements()
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

// readsUnnamed reports whether k reads one of written at a place no
// member's store names as read resolves it -- a gather's computed index, or
// coordinates of another rank than the producer's -- which the formed
// kernel would read from memory it writes itself.
func readsUnnamed(k *Kernel, members []*Kernel, written map[Operand]bool) bool {
	for _, nd := range k.Nodes {
		o := k.Bufs[nd.Buf].Operand
		if nd.Op != OpLoad || !written[o] {
			continue
		}
		named := false
		for _, p := range members {
			for _, s := range p.Stores {
				if s.Barrier || s.Run || p.Bufs[s.Buf].Operand != o {
					continue
				}
				if p.storesOwn(s) && nd.Ax != nil && len(nd.Ax) == len(p.Space) {
					named = true
				}
				if same, ok := sameIndex(k, nd, p, s); ok && same {
					named = true
				}
			}
		}
		if !named {
			return true
		}
	}
	return false
}

// readsMatrix is whether k reads a weight matrix: a projection, whose rows
// are the work to spread, not a row's chain.
func readsMatrix(k *Kernel) bool {
	for _, b := range k.Bufs {
		if b.Kind == BufIn && b.Operand.Kind == OperandWeight {
			return true
		}
	}
	return false
}

// chain is how many kernels from i form a chain -- naive maps and reductions
// of one part, none reading a weight matrix, each after the first reading
// what an earlier one wrote, one of them at the unit's granularity or a
// divisor of it -- and the unit it forms at. Zero for none.
func (c Coarsen) chain(ks []*Kernel, i, lanes int) (int, int) {
	written := map[Operand]bool{}
	j := i
	for ; j < len(ks); j++ {
		k := ks[j]
		if k.Work || k.Part != ks[i].Part || k.Kind != PrimMap && k.Kind != PrimReduce || len(k.Loops) > 1 || readsMatrix(k) {
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
			if !linked || readsUnnamed(k, ks[i:j], written) {
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
		return 0, 0
	}
	E := 0
	for _, k := range ks[i:j] {
		E = max(E, terms(k))
	}
	unit := c.Unit
	if lanes > 1 {
		for _, k := range ks[i:j] {
			if g := E / k.Elements(); k.Kind == PrimReduce && g > unit*lanes && g%lanes == 0 {
				unit = g / lanes
			}
		}
	}
	for _, k := range ks[i:j] {
		if g := E / k.Elements(); g <= unit && unit%g == 0 && g%c.Unit == 0 {
			return j - i, unit
		}
	}
	return 0, 0
}

type coarse struct {
	c     Coarsen
	lanes int
	// guard, where set, is the loop in which a host's phase stores: the
	// units dealt over the workgroup, the threads past the last one clamped
	// to it and left out of the guard. -1 where they store the last unit too.
	guard   int
	m       *Kernel
	members []*Kernel
	E, T    int
	u       int
	local   int
	memo    map[string]int
	values  map[int][]uint32
	host    *Kernel
	hostVal map[Operand]Store
	err     error
}

func (c Coarsen) form(members []*Kernel, host *Kernel, lanes int) (*Kernel, bool) {
	E := 0
	for _, k := range members {
		E = max(E, terms(k))
	}
	U := c.Unit
	if E%U != 0 {
		return nil, false
	}
	for _, k := range members {
		g := E / k.Elements()
		if E%k.Elements() != 0 || g%U != 0 && U%g != 0 {
			return nil, false
		}
	}
	T := E / U
	k := &coarse{c: c, members: members, E: E, T: T, lanes: lanes, guard: -1, memo: map[string]int{}, values: map[int][]uint32{}}
	if host != nil {
		// The phase reads what the host's whole row wrote: one workgroup.
		if host.Elements() != 1 || T > host.Group {
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
		if lanes > 1 && T%lanes == 0 {
			// The units dealt over the workgroup in whole warps: every
			// thread computes its unit, a surplus one the last, and only a
			// real one stores -- so the warps past the last unit hold only
			// copies, and a window combines within a warp by its lanes.
			sp := k.m.loop(T, -1)
			k.m.Loops[sp].Spread = true
			it := k.m.add(Node{Op: OpIter, At: sp, Type: ir.U32})
			k.u = k.m.bin(OpMin, ir.U32, it, k.m.u32(T-1))
			k.guard = k.m.loop(1, sp)
			in := k.m.add(Node{Op: OpLess, Type: ir.Pred, A: it, B: k.m.u32(T)})
			count := k.m.add(Node{Op: OpSelect, Type: ir.U32, A: in, B: k.m.u32(1), C: k.m.u32(0)})
			k.m.Loops[k.guard].Count, k.m.Loops[k.guard].Kept = count+1, true
		} else {
			th := k.m.add(Node{Op: OpThread, Type: ir.U32})
			k.u = k.m.bin(OpMin, ir.U32, k.m.bin(OpAdd, ir.U32, th, k.m.u32(0)), k.m.u32(T-1))
		}
	} else {
		// The widest workgroup, in whole warps, the units fill: no thread is
		// clamped, so a window combines within a warp by its lanes.
		group := members[0].Group
		if lanes > 1 && group > 0 && T%group != 0 {
			for g := group - lanes; g >= lanes; g -= lanes {
				if T%g == 0 && g%lanes == 0 {
					group = g
					break
				}
			}
		}
		k.m = &Kernel{Name: "chain", Kind: members[0].Kind, Space: []int{T}, Group: group, Part: members[0].Part, Of: members[0].Of}
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
				ns := Store{Buf: k.m.bufLike(n.Bufs[s.Buf]), At: k.guard}
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
	g := k.E / n.Elements()
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
	pa, qa := a.Operands(&p), b.Operands(&q)
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

// keys names each coordinate by its value: a coordinate that is a function
// of the thread's unit alone, by its value at every unit, so two
// expressions of one value -- (4u+j)/32 for each j of a unit's four
// elements -- are one; any other by its node.
func (k *coarse) keys(coords []int) []string {
	out := make([]string, len(coords))
	for i, c := range coords {
		if v, ok := k.value(c); ok {
			out[i] = fmt.Sprint(v)
			continue
		}
		out[i] = fmt.Sprint("n", c)
	}
	return out
}

// value is node i of the formed kernel at each of the T units, or false
// where it is not a function of the unit alone.
func (k *coarse) value(i int) ([]uint32, bool) {
	if v, ok := k.values[i]; ok {
		return v, v != nil
	}
	k.values[i] = nil
	nd := k.m.Nodes[i]
	out := make([]uint32, k.T)
	switch {
	case i == k.u:
		for t := range out {
			out[t] = uint32(t)
		}
	case nd.Op == OpConst && nd.Type == ir.U32:
		for t := range out {
			out[t] = uint32(nd.Imm)
		}
	case nd.Type == ir.U32 && (nd.Op == OpAdd || nd.Op == OpSub || nd.Op == OpMul || nd.Op == OpDiv || nd.Op == OpRem || nd.Op == OpMin ||
		nd.Op == OpMax || nd.Op == OpShl || nd.Op == OpShr || nd.Op == OpAnd):
		a, ok := k.value(nd.A)
		if !ok {
			return nil, false
		}
		b, ok := k.value(nd.B)
		if !ok {
			return nil, false
		}
		for t := range out {
			x, y := a[t], b[t]
			switch nd.Op {
			case OpAdd:
				out[t] = x + y
			case OpSub:
				out[t] = x - y
			case OpMul:
				out[t] = x * y
			case OpDiv, OpRem:
				if y == 0 {
					return nil, false
				}
				if nd.Op == OpDiv {
					out[t] = x / y
				} else {
					out[t] = x % y
				}
			case OpMin:
				out[t] = min(x, y)
			case OpMax:
				out[t] = max(x, y)
			case OpShl:
				out[t] = x << (y & 31)
			case OpShr:
				out[t] = x >> (y & 31)
			case OpAnd:
				out[t] = x & y
			}
		}
	default:
		return nil, false
	}
	k.values[i] = out
	return out, true
}

// produce is member n's value stored by s at its coordinates coords.
func (k *coarse) produce(n *Kernel, s Store, coords []int) int {
	key := fmt.Sprintf("%p %d %v", n, s.Val, k.keys(coords))
	g := k.E / n.Elements()
	if n.Kind == PrimReduce && g > k.c.Unit {
		key = fmt.Sprintf("%p %d", n, s.Val)
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
	if t != ir.F32 {
		// The exchange between threads is f32; an integer sum over threads
		// (a top-k's ranks) leaves the chain unformed.
		k.err = fmt.Errorf("an exchange of %v", t)
		return m.u32(0)
	}
	if tg > U || U%tg != 0 {
		k.err = fmt.Errorf("a reduction of %d terms over %d", K, g)
		return m.u32(0)
	}
	count, slots := U/tg, g/U
	// The exchange is the workgroup's: the threads sharing an output must
	// be one workgroup's, which a host's row is (form) and a chain of
	// threads in workgroups of Group is only where its slots fit them.
	if wg := m.Group; k.host == nil && wg > 0 && (slots > wg || wg%slots != 0) {
		k.err = fmt.Errorf("a reduction over %d threads in workgroups of %d", slots, wg)
		return m.u32(0)
	}
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
	init := -1
	if start >= 0 {
		init = m.add(n.Nodes[n.Nodes[f].B])
	}
	// A group of slots inside one warp -- consecutive lanes, aligned to the
	// group -- is combined by an xor butterfly that leaves the total in
	// every lane: no workgroup memory and no barrier. Only where no thread is
	// clamped to the last element: the exchange reads a clamped thread's
	// group by its clamped place, where it finds the real group's slots; a
	// butterfly pairs it by lane, with copies of one slot, and its total
	// would land on the last output beside the real one.
	whole := k.T%m.Group == 0
	if k.host != nil {
		whole = k.T == m.Group || k.guard >= 0
	}
	if whole && k.lanes > 1 && slots&(slots-1) == 0 && k.lanes%slots == 0 {
		v := acc
		for s := 1; s < slots; s *= 2 {
			v = m.bin(comb, t, v, m.add(Node{Op: OpShuffle, Type: t, A: v, Imm: int64(s)}))
		}
		if init >= 0 {
			v = m.bin(comb, t, init, v)
		}
		return v
	}
	at := m.bin(OpMul, ir.U32, m.bin(OpDiv, ir.U32, k.local, m.u32(slots)), m.u32(slots))
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
