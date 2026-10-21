package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/ir"
)

// Simplify folds what the IR's structure decides: equal nodes merged, the
// integer arithmetic of constants done, identities removed, a loop of one
// trip and no carried value dropped, and what a loop's iteration is known
// to satisfy, being below its count and a range's position inside it and
// below its end, taken as decided. With ConstLast, a commutative float
// operation reads a constant as its second operand.
type Simplify struct {
	ConstLast bool
}

func (Simplify) Name() string { return "simplify" }

func (p Simplify) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		m := k.clone()
		changed := false
		for {
			step := m.cse()
			step = m.foldDecided() || step
			step = m.dropUnitLoops() || step
			step = m.dropEmptyLoops() || step
			step = p.ConstLast && m.constLast() || step
			if !step {
				break
			}
			changed = true
			m.compact()
		}
		if !changed {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

// constLast swaps a commutative float operation's constant first operand
// for its second.
func (k *Kernel) constLast() bool {
	changed := false
	for i, nd := range k.Nodes {
		switch nd.Op {
		case OpAdd, OpMul, OpMin, OpMax:
		default:
			continue
		}
		if nd.Type == ir.F32 && k.Nodes[nd.A].Op == OpConst && k.Nodes[nd.B].Op != OpConst {
			k.Nodes[i].A, k.Nodes[i].B = nd.B, nd.A
			changed = true
		}
	}
	return changed
}

// cse merges each node into the first node equal to it: the same op, type,
// operands and immediate, a load only of a buffer the kernel does not
// store.
func (k *Kernel) cse() bool {
	stored := map[Operand]bool{}
	for _, s := range k.Stores {
		if !s.Barrier && !s.Run {
			stored[k.Bufs[s.Buf].Operand] = true
		}
	}
	seen := map[string]int{}
	changed := false
	for i := range k.Nodes {
		nd := k.Nodes[i]
		switch nd.Op {
		case OpFold, OpExchange, OpShuffle, OpGroupSum, OpGroupMax:
			continue
		case OpLoad:
			if stored[k.Bufs[nd.Buf].Operand] {
				continue
			}
		}
		var ops []int
		for _, a := range k.args(&nd) {
			ops = append(ops, *a)
		}
		key := fmt.Sprint(nd.Op, nd.Type, ops, nd.At, nd.Buf, nd.Imm)
		if nd.Op == OpConst || nd.Op == OpCoord || nd.Op == OpParam || nd.Op == OpLength || nd.Op == OpSpan {
			key = fmt.Sprint(nd.Op, nd.Type, nd.A, nd.Imm)
		}
		if nd.Op == OpIter {
			key = fmt.Sprint(nd.Op, nd.At)
		}
		if j, ok := seen[key]; ok {
			k.redirectAll(i, j)
			changed = true
			continue
		}
		seen[key] = i
	}
	return changed
}

// redirectAll points every reader of node from at node to: nodes, stores,
// loops, buffers and facts.
func (k *Kernel) redirectAll(from, to int) {
	k.redirect(from, to, -1)
	re := func(x *int) {
		if *x == from+1 {
			*x = to + 1
		}
	}
	for i := range k.Loops {
		l := &k.Loops[i]
		re(&l.Count)
		re(&l.Span)
		re(&l.Elem)
		re(&l.End)
	}
	re(&k.Rows)
	re(&k.Clamp)
	for i := range k.Bufs {
		re(&k.Bufs[i].Base)
		re(&k.Bufs[i].Offset)
		for j := range k.Bufs[i].Dims {
			re(&k.Bufs[i].Dims[j])
		}
	}
	for i := range k.Args {
		re(&k.Args[i].Node)
	}
	if p := k.Positions; p != nil {
		q := *p
		if q.First == from {
			q.First = to
		}
		if q.End == from {
			q.End = to
		}
		q.Live = append([]int(nil), p.Live...)
		q.Pos = append([]int(nil), p.Pos...)
		for _, xs := range [][]int{q.Live, q.Pos} {
			for j := range xs {
				if xs[j] == from {
					xs[j] = to
				}
			}
		}
		re(&q.Span)
		re(&q.Part)
		k.Positions = &q
	}
}

func (k *Kernel) isConst(i int, v int64) bool {
	nd := k.Nodes[i]
	return nd.Op == OpConst && nd.Imm == v && nd.Type != ir.F32
}

// countOf is the loop whose count node i is, or -1.
func (k *Kernel) countOf(i int) int {
	for c, l := range k.Loops {
		if l.Count == i+1 {
			return c
		}
	}
	return -1
}

// foldDecided rewrites each node its operands decide.
func (k *Kernel) foldDecided() bool {
	changed := false
	to := func(i, j int) {
		k.redirectAll(i, j)
		changed = true
	}
	for i := range k.Nodes {
		nd := k.Nodes[i]
		if nd.Type == ir.F32 && nd.Op != OpSelect && nd.Op != OpBitcast {
			continue
		}
		a, b := nd.A, nd.B
		switch nd.Op {
		case OpAdd:
			if k.isConst(b, 0) {
				to(i, a)
			} else if k.isConst(a, 0) {
				to(i, b)
			} else if ca, cb := k.Nodes[a], k.Nodes[b]; ca.Op == OpConst && cb.Op == OpConst {
				k.Nodes[i] = Node{Op: OpConst, Type: nd.Type, Imm: int64(uint32(ca.Imm + cb.Imm))}
				changed = true
			}
		case OpSub:
			if k.isConst(b, 0) {
				to(i, a)
			} else if x := k.Nodes[a]; x.Op == OpAdd && x.A == b {
				to(i, x.B)
			} else if x.Op == OpAdd && x.B == b {
				to(i, x.A)
			}
		case OpMul:
			if k.isConst(b, 1) {
				to(i, a)
			} else if k.isConst(a, 1) {
				to(i, b)
			} else if k.isConst(a, 0) || k.isConst(b, 0) {
				k.Nodes[i] = Node{Op: OpConst, Type: nd.Type}
				changed = true
			} else if ca, cb := k.Nodes[a], k.Nodes[b]; ca.Op == OpConst && cb.Op == OpConst {
				k.Nodes[i] = Node{Op: OpConst, Type: nd.Type, Imm: int64(uint32(ca.Imm * cb.Imm))}
				changed = true
			}
		case OpDiv:
			if k.isConst(b, 1) {
				to(i, a)
			} else if k.isConst(a, 0) {
				to(i, a)
			} else if x, ok := k.multipleOf(a, b); ok {
				to(i, x)
			}
		case OpBitcast:
			if x := k.Nodes[a]; nd.Type == ir.F32 && x.Op == OpLoad && x.Type == ir.U32 && k.uses(a) == 1 && k.Bufs[x.Buf].Operand.Kind == OperandTable {
				x.Type = ir.F32
				if nd.Hint > 0 {
					x.Hint = nd.Hint
				}
				k.Nodes[i] = x
				changed = true
			} else if x.Type == nd.Type {
				to(i, a)
			} else if nd.Type != ir.F32 && x.Type != ir.F32 && (x.Op == OpConst || x.Op == OpLanes) {
				x.Type = nd.Type
				k.Nodes[i] = x
				changed = true
			}
		case OpShr:
			if x := k.Nodes[a]; x.Op == OpShr && k.Nodes[x.B].Op == OpConst && k.Nodes[b].Op == OpConst {
				k.Nodes[i] = Node{Op: OpShr, Type: nd.Type, A: x.A, B: k.u32(int(k.Nodes[x.B].Imm + k.Nodes[b].Imm))}
				changed = true
			}
		case OpRem:
			if k.isConst(b, 1) || k.isConst(a, 0) {
				k.Nodes[i] = Node{Op: OpConst, Type: nd.Type}
				changed = true
			}
		case OpLess:
			if it := k.Nodes[a]; it.Op == OpIter && k.countOf(b) == it.At || k.belowEnd(a, b) {
				k.Nodes[i] = Node{Op: OpConst, Type: ir.Pred, Imm: 1}
				changed = true
			}
		case OpSelect:
			if c := k.Nodes[a]; c.Op == OpConst && c.Type == ir.Pred {
				if c.Imm != 0 {
					to(i, b)
				} else {
					to(i, nd.C)
				}
			}
		case OpMax:
			if x := k.Nodes[a]; x.Op == OpAdd && x.A == b && k.Nodes[x.B].Op == OpIter {
				to(i, a)
			}
		case OpMin:
			if k.insideRange(a, b) {
				to(i, a)
			}
		}
	}
	return changed
}

// insideRange reports whether node pos, a range's first plus an iteration
// of a loop counting end less first, is at most node last, max(end, 1) - 1.
func (k *Kernel) insideRange(pos, last int) bool {
	p := k.Nodes[pos]
	if p.Op != OpAdd || k.Nodes[p.B].Op != OpIter {
		return false
	}
	l := k.Loops[k.Nodes[p.B].At]
	if l.Count == 0 {
		return false
	}
	cnt := k.Nodes[l.Count-1]
	if cnt.Op != OpSub || cnt.B != p.A {
		return false
	}
	ls := k.Nodes[last]
	if ls.Op != OpSub || !k.isConst(ls.B, 1) {
		return false
	}
	mx := k.Nodes[ls.A]
	return mx.Op == OpMax && mx.A == cnt.A && k.isConst(mx.B, 1)
}

// dropUnitLoops removes each loop of one static trip that carries no value:
// its iteration is 0, what ran in it runs in its parent.
func (k *Kernel) dropUnitLoops() bool {
	for c, l := range k.Loops {
		if l.Trip != 1 || l.Count != 0 || l.Range || len(k.foldsOf(c)) > 0 || l.Lanes > 0 {
			continue
		}
		zero := k.u32(0)
		for i := range k.Nodes {
			if k.Nodes[i].Op == OpIter && k.Nodes[i].At == c {
				k.redirectAll(i, zero)
			}
		}
		k.removeLoop(c)
		return true
	}
	return false
}

// removeLoop deletes loop c: its stores and inner loops move to its parent,
// later loops' indices close up.
func (k *Kernel) removeLoop(c int) {
	parent := k.Loops[c].Parent
	re := func(x int) int {
		switch {
		case x == c:
			return parent
		case x > c:
			return x - 1
		}
		return x
	}
	for i := range k.Loops {
		k.Loops[i].Parent = re(k.Loops[i].Parent)
		if of := k.Loops[i].Of; of > 0 {
			if of-1 == c {
				k.Loops[i].Of = 0
			} else {
				k.Loops[i].Of = re(of-1) + 1
			}
		}
	}
	k.Loops = append(k.Loops[:c:c], k.Loops[c+1:]...)
	for i := range k.Nodes {
		if op := k.Nodes[i].Op; op == OpIter || op == OpFold {
			k.Nodes[i].At = re(k.Nodes[i].At)
		}
	}
	for i := range k.Stores {
		k.Stores[i].At = re(k.Stores[i].At)
	}
	if p := k.Positions; p != nil && p.Loop >= 0 {
		q := *p
		q.Loop = re(p.Loop)
		k.Positions = &q
	}
	var seq []Entry
	for _, e := range k.Seq {
		if (e.Kind == EntryLoop || e.Kind == EntryEnd) && e.I == c {
			continue
		}
		if e.Kind == EntryLoop || e.Kind == EntryEnd {
			e.I = re(e.I)
		}
		seq = append(seq, e)
	}
	if k.Seq != nil {
		k.Seq = seq
	}
}

// dropEmptyLoops removes each loop of no static trips and nothing inside
// it: what it carries is where it started, what it stores is never stored.
func (k *Kernel) dropEmptyLoops() bool {
	for c, l := range k.Loops {
		if l.Trip != 0 || l.Count != 0 || len(k.innerLoops(c)) > 0 {
			continue
		}
		for _, f := range k.foldsOf(c) {
			k.redirectAll(f, k.Nodes[f].B)
		}
		var keep []Store
		for _, s := range k.Stores {
			if s.At != c {
				keep = append(keep, s)
			}
		}
		k.Stores = keep
		for i := range k.Nodes {
			if k.Nodes[i].Op == OpIter && k.Nodes[i].At == c {
				k.Nodes[i] = Node{Op: OpConst, Type: ir.U32}
			}
			if k.Nodes[i].Op == OpFold && k.Nodes[i].At == c {
				k.Nodes[i] = Node{Op: OpConst, Type: k.Nodes[i].Type}
			}
		}
		k.removeLoop(c)
		return true
	}
	return false
}

// multipleOf is x where node a is x*m + j, m node b's constant and j a
// constant below it: what a divided by m is.
func (k *Kernel) multipleOf(a, b int) (int, bool) {
	mb := k.Nodes[b]
	if mb.Op != OpConst || mb.Imm <= 0 {
		return 0, false
	}
	nd := k.Nodes[a]
	j := int64(0)
	if nd.Op == OpAdd {
		if c := k.Nodes[nd.B]; c.Op == OpConst && c.Imm >= 0 && c.Imm < mb.Imm {
			j, nd = c.Imm, k.Nodes[nd.A]
		}
	}
	if nd.Op == OpMul && k.isConst(nd.B, mb.Imm) && j < mb.Imm {
		return nd.A, true
	}
	return 0, false
}

// belowEnd reports whether node pos, a range's first plus an iteration of
// a loop counting end less first, is below node end.
func (k *Kernel) belowEnd(pos, end int) bool {
	p := k.Nodes[pos]
	if p.Op != OpAdd || k.Nodes[p.B].Op != OpIter {
		return false
	}
	l := k.Loops[k.Nodes[p.B].At]
	if l.Count == 0 {
		return false
	}
	cnt := k.Nodes[l.Count-1]
	return cnt.Op == OpSub && cnt.A == end && cnt.B == p.A
}
