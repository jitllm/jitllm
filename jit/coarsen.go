package jit

import "github.com/jitllm/jitllm/jit/ir"

// Coarsen runs a chain of maps and reductions over a row, linked through
// what they write, as one thread a Unit of the row's elements. A finer
// primitive is computed at each element of the thread's unit and read where
// it is computed; a coarser reduction is each thread's partial over its own
// terms, exchanged through a workgroup slot and combined over its group's
// slots; a coarser map is computed by every thread of its group, and so is
// its store. Where the chain reads the last phase of a workgroup row, it
// joins that row as a further phase, the value recomputed from the phase's
// own expression.
type Coarsen struct {
	Unit int
}

func (Coarsen) Name() string { return "coarsen" }

func (c Coarsen) Apply(prog []*Nest, t Target) ([]*Nest, error) {
	out := append([]*Nest(nil), prog...)
	for i := 0; i < len(out); i++ {
		members := c.chain(out, i)
		if members == 0 {
			continue
		}
		var host *Nest
		hi := -1
		if i > 0 && out[i-1].Work {
			host, hi = out[i-1], i-1
		}
		m, ok := c.form(out[i:i+members], host)
		if !ok {
			continue
		}
		m.Passes = append(append([]string(nil), m.Passes...), c.Name())
		if host != nil {
			out[hi] = m
			out = append(out[:i], out[i+members:]...)
			i--
			continue
		}
		out = append(append(out[:i:i], m), out[i+members:]...)
	}
	return out, nil
}

// chain is how many nests from i form a chain: naive maps and reductions,
// each after the first reading what an earlier one wrote, that hold a
// primitive at the unit's granularity and keep their temporaries to
// themselves. Zero for none.
func (c Coarsen) chain(prog []*Nest, i int) int {
	written := map[Operand]bool{}
	j := i
	for ; j < len(prog); j++ {
		n := prog[j]
		if n.Prim == nil || n.Kind != PrimMap && n.Kind != PrimReduce {
			break
		}
		if j > i {
			linked := false
			for _, o := range n.Prim.In {
				linked = linked || written[o]
			}
			if !linked {
				break
			}
		}
		written[n.Prim.Out] = true
	}
	if j-i < 2 {
		return 0
	}
	E := chainElements(prog[i:j])
	unit := false
	for _, n := range prog[i:j] {
		unit = unit || E/n.Prim.N == c.Unit
	}
	for o := range written {
		if !o.Temp {
			continue
		}
		for k, n := range prog {
			if (k < i || k >= j) && n.reads(o) {
				return 0
			}
		}
	}
	if !unit {
		return 0
	}
	return j - i
}

// chainElements is the row's elements a chain covers: its finest
// primitive's.
func chainElements(ns []*Nest) int {
	E := 0
	for _, n := range ns {
		e := n.Prim.N
		if n.Kind == PrimReduce {
			e *= n.Prim.K
		}
		E = max(E, e)
	}
	return E
}

type coarse struct {
	c       Coarsen
	m       *Nest
	members []*Nest
	E       int
	u       int
	local   int
	memo    map[valKey]int
	bufs    map[Operand]int
	host    *Nest
	hostVal map[Operand][2]int
}

type valKey struct {
	o   Operand
	off int
	ix  int
}

func (c Coarsen) form(members []*Nest, host *Nest) (*Nest, bool) {
	E := chainElements(members)
	U := c.Unit
	if E%U != 0 {
		return nil, false
	}
	for _, n := range members {
		g := E / n.Prim.N
		if E%n.Prim.N != 0 || g%U != 0 && U%g != 0 {
			return nil, false
		}
	}
	T := E / U
	k := &coarse{c: c, members: members, E: E, memo: map[valKey]int{}, bufs: map[Operand]int{}}
	if host != nil {
		if host.Space[0] != 1 || T > host.Group {
			return nil, false
		}
		k.m = host.clone()
		k.m.Name = host.Name + "+quantized"
		k.host, k.hostVal = host, map[Operand][2]int{}
		for _, s := range host.Stores {
			if s.At >= 0 && !s.Barrier && host.Loops[s.At].Elem > 0 {
				k.hostVal[host.Bufs[s.Buf].Operand] = [2]int{s.Val, host.Loops[s.At].Elem - 1}
			}
		}
		for i, b := range k.m.Bufs {
			k.bufs[b.Operand] = i
		}
		th := k.m.add(Node{Op: OpThread, Type: ir.U32})
		k.u = k.m.bin(OpMin, ir.U32, k.m.bin(OpAdd, ir.U32, th, k.m.u32(0)), k.m.u32(T-1))
	} else {
		k.m = &Nest{Name: "chain", Space: []int{T}, Group: members[0].Group}
		k.u = k.m.add(Node{Op: OpCoord, A: 0, Type: ir.U32})
	}
	k.local = k.m.add(Node{Op: OpLocal, Type: ir.U32, A: k.u})
	for _, n := range members {
		p := n.Prim
		if p.Out.Temp {
			continue
		}
		g := E / p.N
		var idxs []int
		switch {
		case g < U:
			for j := 0; j < U/g; j++ {
				idxs = append(idxs, k.m.bin(OpAdd, ir.U32, k.m.bin(OpMul, ir.U32, k.u, k.m.u32(U/g)), k.m.u32(j)))
			}
		case g == U:
			idxs = []int{k.u}
		default:
			idxs = []int{k.m.bin(OpDiv, ir.U32, k.u, k.m.u32(g/U))}
		}
		for _, ix := range idxs {
			v, ok := k.produce(n, ix)
			if !ok {
				return nil, false
			}
			b := k.buffer(p.Out, BufOut, elemOf(p.Out.Elem))
			k.m.Stores = append(k.m.Stores, Store{Buf: b, Ix: k.m.access(ix, p.OutAt), Val: v, At: -1})
		}
	}
	if host != nil {
		k.m.Stores = append(k.m.Stores, Store{Barrier: true, At: -1})
	}
	return k.m, true
}

func (k *coarse) buffer(o Operand, kind BufKind, elem ir.Type) int {
	if b, ok := k.bufs[o]; ok && k.m.Bufs[b].Kind == kind {
		return b
	}
	b := k.m.buf(Buf{Kind: kind, Elem: elem, Operand: o})
	k.bufs[o] = b
	return b
}

// producer is the member whose region of o holds offset off.
func (k *coarse) producer(o Operand, off int) *Nest {
	for _, n := range k.members {
		if r := n.Prim.OutAt.Off; n.Prim.Out == o && off >= r && off < r+n.Prim.N {
			return n
		}
	}
	return nil
}

// value is operand o at index ix of its storage, where the chain reads it.
func (k *coarse) value(o Operand, a Access, ix int) (int, bool) {
	if p := k.producer(o, a.Off); p != nil {
		a.Off -= p.Prim.OutAt.Off
		return k.produce(p, k.m.access(ix, a))
	}
	if hv, ok := k.hostVal[o]; ok && a == atElement {
		return k.m.copyOf(hv[0], map[int]int{hv[1]: ix}), true
	}
	b := k.buffer(o, BufIn, storageElem(o))
	return k.m.load(b, k.m.access(ix, a)), true
}

// produce is member n's value at its element ix.
func (k *coarse) produce(n *Nest, ix int) (int, bool) {
	p := n.Prim
	key := valKey{o: p.Out, off: p.OutAt.Off, ix: ix}
	g := k.E / p.N
	if p.Kind == PrimReduce && g > k.c.Unit {
		key.ix = -1
	}
	if v, ok := k.memo[key]; ok {
		return v, true
	}
	var v int
	var ok bool
	switch p.Kind {
	case PrimMap:
		v, ok = k.inline(p, ix)
	case PrimReduce:
		v, ok = k.reduce(p, ix, g)
	}
	if ok {
		k.memo[key] = v
	}
	return v, ok
}

// inline is p's function at element ix, each operand read where the chain
// has it.
func (k *coarse) inline(p *Prim, ix int) (int, bool) {
	m := k.m
	vals := make([]int, len(p.Expr))
	for i, x := range p.Expr {
		switch x.Op {
		case ExprIn:
			v, ok := k.value(p.In[x.In], p.At[x.In], ix)
			if !ok {
				return 0, false
			}
			vals[i] = v
		case OpConst:
			vals[i] = m.add(Node{Op: OpConst, Type: x.Type, Imm: x.Imm})
		default:
			vals[i] = m.add(Node{Op: x.Op, Type: x.Type, A: vals[x.A], B: vals[x.B], C: vals[x.C]})
		}
	}
	return vals[len(vals)-1], true
}

// reduce is reduction p's output: within the thread where its terms are,
// else the thread's partial exchanged over its group of threads.
func (k *coarse) reduce(p *Prim, ix, g int) (int, bool) {
	m, U := k.m, k.c.Unit
	t := elemOf(p.Type)
	tg := g / p.K
	var first, count int
	var base int
	if g <= U {
		base, count = m.bin(OpMul, ir.U32, ix, m.u32(p.K)), p.K
	} else {
		if tg > U || U%tg != 0 {
			return 0, false
		}
		count = U / tg
		base = m.bin(OpMul, ir.U32, k.u, m.u32(count))
	}
	start := -1
	if p.Start == StartConst {
		start = m.add(Node{Op: OpConst, Type: t, Imm: p.Init})
	}
	acc := start
	for j := first; j < count; j++ {
		ti := base
		if j > 0 {
			ti = m.bin(OpAdd, ir.U32, base, m.u32(j))
		}
		v, ok := k.inline(p, ti)
		if !ok {
			return 0, false
		}
		if acc < 0 {
			acc = v
			continue
		}
		acc = m.bin(p.Combine, t, acc, v)
	}
	if g <= U {
		return acc, true
	}
	slots := g / U
	at := m.bin(OpMul, ir.U32, m.bin(OpDiv, ir.U32, k.local, m.u32(slots)), m.u32(slots))
	init := -1
	if p.Start == StartConst {
		init = m.add(Node{Op: OpConst, Type: t, Imm: p.Init})
	}
	return m.add(Node{Op: OpExchange, Type: ir.F32, A: acc, B: at, C: init, Imm: int64(slots), At: int(p.Combine)}), true
}
