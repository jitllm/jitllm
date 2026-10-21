package jit

import (
	"sort"

	"github.com/jitllm/jitllm/jit/ir"
)

// Args binds a kernel to the words its call passes: every count a loop
// takes that no loop of the kernel moves is a word the caller computes, and
// every buffer the code reads or writes an address word, the part of its
// indices no loop moves added by the caller, the kernel indexing from the
// address it is given. Fields is the device's convention for which word
// each takes.
type Args struct {
	Fields ArgFields
}

// ArgFields are the byte offsets of the argument words: the buffers written
// in order, a weight's or the history's address, a block of constants, the
// other buffers read in order; a vector loop's count and its tail's, and the
// other counts in order.
type ArgFields struct {
	Out           []int
	Weight        int
	Consts        int
	In            []int
	Vectors, Tail int
	Counts        []int
}

func (Args) Name() string { return "args" }

func (p Args) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if k.Work || k.Group > 0 || k.Args != nil {
			return nil, false, nil
		}
		m := k.clone()
		m.bindCounts(p.Fields)
		m.bindBufs(p.Fields)
		return []*Kernel{m}, true, nil
	})
}

// fixed reports whether no loop of k moves node i.
func (k *Kernel) fixed(i int) bool {
	for c := range k.Loops {
		if k.varies(i, c) {
			return false
		}
	}
	return true
}

// bindCounts makes each loop count the call fixes an argument word.
func (k *Kernel) bindCounts(f ArgFields) {
	next := 0
	field := func(c int) int {
		l := k.Loops[c]
		if l.Lanes > 0 {
			return f.Vectors
		}
		for t, o := range k.Loops {
			if o.Lanes > 0 && o.Parent == l.Parent && t < c && o.Range == l.Range && k.tailOf(t) == c {
				return f.Tail
			}
		}
		v := f.Counts[min(next, len(f.Counts)-1)]
		next++
		return v
	}
	for c := range k.Loops {
		n := k.Loops[c].Count - 1
		if n < 0 || k.Nodes[n].Op == OpConst || k.Nodes[n].Op == OpParam || !k.fixed(n) {
			continue
		}
		fl := field(c)
		pv := k.add(Node{Op: OpParam, Type: ir.U32, Imm: int64(fl)})
		k.redirectAll(n, pv)
		k.Args = append(k.Args, Arg{Field: fl, Buf: -1, Node: n + 1})
	}
}

// tailOf is the loop taking vector loop c's remaining trips, or -1.
func (k *Kernel) tailOf(c int) int {
	for t := c + 1; t < len(k.Loops); t++ {
		if k.Loops[t].Lanes == 0 && k.Loops[t].Parent == k.Loops[c].Parent && len(k.innerLoops(t)) == 0 {
			return t
		}
	}
	return -1
}

// access is one load or store of a buffer: node Node's or store Store's.
type access struct {
	node, store int
}

// bindBufs gives each buffer the code reads or writes an address word.
func (k *Kernel) bindBufs(f ArgFields) {
	live := k.live()
	byOp := map[Operand][]access{}
	var order []Operand
	note := func(b int, a access) {
		o := k.Bufs[b].Operand
		if _, ok := byOp[o]; !ok {
			order = append(order, o)
		}
		byOp[o] = append(byOp[o], a)
	}
	written := map[Operand]bool{}
	for i, s := range k.Stores {
		if !s.Barrier && !s.Run && k.Bufs[s.Buf].Kind != BufShared {
			note(s.Buf, access{node: -1, store: i})
			written[k.Bufs[s.Buf].Operand] = true
		}
	}
	for i, nd := range k.Nodes {
		if nd.Op == OpLoad && live[i] && k.Bufs[nd.Buf].Kind != BufShared {
			note(nd.Buf, access{node: i, store: -1})
		}
	}
	sort.SliceStable(order, func(i, j int) bool { return k.firstBuf(order[i]) < k.firstBuf(order[j]) })
	outs, ins := 0, 0
	for _, o := range order {
		var fl int
		switch {
		case written[o]:
			fl = f.Out[min(outs, len(f.Out)-1)]
			outs++
		case o.Kind == OperandConsts:
			fl = f.Consts
		case o.Kind == OperandWeight || o.Kind == OperandTable:
			fl = f.Weight
		default:
			fl = f.In[min(ins, len(f.In)-1)]
			ins++
		}
		base := k.add(Node{Op: OpParam, Type: ir.U32, Imm: int64(fl)})
		off := k.split(byOp[o])
		bufs := map[int]bool{}
		for _, a := range byOp[o] {
			if a.node >= 0 {
				bufs[k.Nodes[a.node].Buf] = true
			} else {
				bufs[k.Stores[a.store].Buf] = true
			}
		}
		first := -1
		for b := range k.Bufs {
			if bufs[b] {
				k.Bufs[b].Base = base + 1
				k.Bufs[b].Offset = off
				if first < 0 {
					first = b
				}
			}
		}
		k.Args = append(k.Args, Arg{Field: fl, Buf: first})
	}
}

func (k *Kernel) firstBuf(o Operand) int {
	for b, x := range k.Bufs {
		if x.Operand == o {
			return b
		}
	}
	return len(k.Bufs)
}

// split moves the part of the accesses' indices no loop moves into an
// offset the caller adds, 1 + its node (0 for none); each access is left a
// flat index from the buffer's address.
func (k *Kernel) split(as []access) int {
	type lin struct {
		terms map[int]int64
		c     int64
	}
	ls := make([]lin, len(as))
	for i, a := range as {
		var t map[int]int64
		var c int64
		if a.node >= 0 {
			nd := k.Nodes[a.node]
			t, c = k.flatTerms(nd.Buf, nd.A, nd.Ax)
		} else {
			s := k.Stores[a.store]
			t, c = k.flatTerms(s.Buf, s.Ix, s.Ax)
		}
		ls[i] = lin{t, c}
	}
	fixed := map[int]int64{}
	for n, c := range ls[0].terms {
		if k.fixed(n) {
			fixed[n] = c
		}
	}
	for _, l := range ls[1:] {
		for n, c := range fixed {
			if l.terms[n] != c {
				delete(fixed, n)
			}
		}
	}
	for i, a := range as {
		for n := range fixed {
			delete(ls[i].terms, n)
		}
		ix := k.sumOf(ls[i].terms, ls[i].c)
		if a.node >= 0 {
			k.Nodes[a.node].A, k.Nodes[a.node].Ax = ix, nil
		} else {
			k.Stores[a.store].Ix, k.Stores[a.store].Ax = ix, nil
		}
	}
	if len(fixed) == 0 {
		return 0
	}
	return k.sumOf(fixed, 0) + 1
}

// flatTerms is an access's flat index as a sum of nodes times constants and
// a constant: index ix, or coordinates ax of buffer b's shape.
func (k *Kernel) flatTerms(b, ix int, ax []int) (map[int]int64, int64) {
	if ax == nil {
		return k.terms(ix)
	}
	sh := k.Bufs[b].Shape
	out, c := map[int]int64{}, int64(0)
	stride := int64(1)
	for j := len(ax) - 1; j >= 0; j-- {
		t, tc := k.terms(ax[j])
		for n, v := range t {
			out[n] += v * stride
		}
		c += tc * stride
		stride *= int64(sh[j])
	}
	return out, c
}

// terms is integer node i as a sum of nodes times constants and a
// constant.
func (k *Kernel) terms(i int) (map[int]int64, int64) {
	nd := k.Nodes[i]
	if nd.Type == ir.F32 {
		return map[int]int64{i: 1}, 0
	}
	switch nd.Op {
	case OpConst:
		return map[int]int64{}, nd.Imm
	case OpAdd, OpSub:
		a, ac := k.terms(nd.A)
		b, bc := k.terms(nd.B)
		sign := int64(1)
		if nd.Op == OpSub {
			sign = -1
		}
		for n, v := range b {
			a[n] += sign * v
			if a[n] == 0 {
				delete(a, n)
			}
		}
		return a, ac + sign*bc
	case OpMul, OpShl:
		x, y := nd.A, nd.B
		if k.Nodes[x].Op == OpConst && nd.Op == OpMul {
			x, y = y, x
		}
		if c := k.Nodes[y]; c.Op == OpConst {
			f := c.Imm
			if nd.Op == OpShl {
				f = 1 << c.Imm
			}
			t, tc := k.terms(x)
			for n := range t {
				t[n] *= f
			}
			return t, tc * f
		}
	}
	return map[int]int64{i: 1}, 0
}

// sumOf is the node of the sum of terms and c, in the order of the terms'
// nodes.
func (k *Kernel) sumOf(terms map[int]int64, c int64) int {
	var ns []int
	for n := range terms {
		ns = append(ns, n)
	}
	sort.Ints(ns)
	acc := -1
	for _, n := range ns {
		v := n
		if f := terms[n]; f != 1 {
			v = k.bin(OpMul, ir.U32, n, k.u32(int(f)))
		}
		if acc < 0 {
			acc = v
		} else {
			acc = k.bin(OpAdd, ir.U32, acc, v)
		}
	}
	switch {
	case acc < 0:
		return k.u32(int(c))
	case c != 0:
		return k.bin(OpAdd, ir.U32, acc, k.u32(int(c)))
	}
	return acc
}
