package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/ir"
)

// gpuLowering lowers a kernel to the GPU IR.
type gpuLowering struct{}

func (gpuLowering) Lower(n *Kernel, d Device) (Lowered, error) {
	if n.Group <= 0 {
		return Lowered{}, fmt.Errorf("no workgroup: the profile binds none")
	}
	e := &lowerer{n: n, b: ir.New(n.Name, [3]int{n.Group, 1, 1})}
	k, err := e.run()
	return Lowered{GPU: k}, err
}

type lowerer struct {
	n      *Kernel
	b      *ir.Builder
	params []ir.Value
	val    []ir.Value
	home   []int
	coords []ir.Value
	flat   ir.Value
	tid    ir.Value
	shared map[int]ir.Value
	open   []int
	iter   map[int]ir.Value
	phi    map[int]ir.Value
	done   []bool
	err    error
}

func (e *lowerer) run() (*ir.Kernel, error) {
	n, b := e.n, e.b
	e.val = make([]ir.Value, len(n.Nodes))
	e.done = make([]bool, len(n.Loops))
	e.iter, e.phi, e.shared = map[int]ir.Value{}, map[int]ir.Value{}, map[int]ir.Value{}
	e.homes()
	for i, bf := range n.Bufs {
		if bf.Kind == BufShared {
			e.params = append(e.params, b.Shared(fmt.Sprintf("s%d", i), bf.Elem, bf.Size))
			continue
		}
		e.params = append(e.params, b.Param(fmt.Sprintf("p%d", i), bf.Elem))
	}
	for i, nd := range n.Nodes {
		if nd.Op == OpGroupSum || nd.Op == OpGroupMax || nd.Op == OpExchange {
			e.shared[i] = b.Shared(fmt.Sprintf("r%d", i), ir.F32, n.Group)
		}
	}
	var flat ir.Value
	if n.Work && len(n.Space) > 1 {
		e.coords = e.workCoords()
	} else if n.Work {
		flat = b.Min(ir.U32, b.CTAID(), b.Const(ir.U32, int64(n.elements()-1)))
	} else {
		flat = b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
		e.flat = flat
		flat = b.Min(ir.U32, flat, b.Const(ir.U32, int64(n.elements()-1)))
	}
	if e.coords == nil {
		e.coords = make([]ir.Value, len(n.Space))
		rest := flat
		for d := len(n.Space) - 1; d > 0; d-- {
			ext := b.Const(ir.U32, int64(n.Space[d]))
			e.coords[d] = b.Rem(ir.U32, rest, ext)
			rest = b.Div(ir.U32, rest, ext)
		}
		if len(n.Space) > 0 {
			e.coords[0] = rest
		}
	}
	for _, st := range n.Stores {
		switch {
		case st.At >= 0:
			if !e.done[st.At] {
				e.lowerLoop(st.At)
			}
		case st.Barrier:
			b.Barrier()
		default:
			ix, v := e.storeIndex(st), e.get(st.Val)
			b.Store(e.params[st.Buf], ix, v, 0)
		}
	}
	if e.err != nil {
		return nil, e.err
	}
	return b.Done()
}

// workCoords is a workgroup's coordinates from its index: an axis of one
// extent zero, the outermost other axis held to its extent, the rest its
// remainders.
func (e *lowerer) workCoords() []ir.Value {
	n, b := e.n, e.b
	outer := 0
	for outer < len(n.Space)-1 && n.Space[outer] == 1 {
		outer++
	}
	c := make([]ir.Value, len(n.Space))
	rest := b.CTAID()
	for d := len(n.Space) - 1; d >= 0; d-- {
		ext := int64(n.Space[d])
		switch {
		case ext == 1:
			c[d] = b.Const(ir.U32, 0)
		case d == outer:
			c[d] = b.Min(ir.U32, rest, b.Const(ir.U32, ext-1))
		default:
			c[d] = b.Rem(ir.U32, rest, b.Const(ir.U32, ext))
			rest = b.Div(ir.U32, rest, b.Const(ir.U32, ext))
		}
	}
	return c
}

func (e *lowerer) fail(format string, args ...any) {
	if e.err == nil {
		e.err = fmt.Errorf(format, args...)
	}
}

// homes is each node's loop: the deepest loop whose trips both reach it
// and change it. Everything else runs outside, after a loop where it reads
// the loop's folds.
func (e *lowerer) homes() {
	n := e.n
	e.home = make([]int, len(n.Nodes))
	depth := make([]int, len(n.Loops))
	for c := range n.Loops {
		for h := c; h >= 0; h = n.Loops[h].Parent {
			depth[c]++
		}
	}
	for i := range e.home {
		e.home[i] = -1
	}
	for i, nd := range n.Nodes {
		if nd.Op == OpIter || nd.Op == OpFold {
			e.home[i] = nd.At
		}
	}
	for c := range n.Loops {
		for _, i := range e.body(c) {
			if n.Nodes[i].Op == OpFold && n.Nodes[i].At == c || !n.varies(i, c) {
				continue
			}
			if h := e.home[i]; h < 0 || depth[c] > depth[h] {
				e.home[i] = c
			}
		}
	}
}

func (e *lowerer) isOpen(c int) bool {
	for _, o := range e.open {
		if o == c {
			return true
		}
	}
	return false
}

// get is node i's value where the lowerer stands, its loop emitted first
// where it is a fold read after the loop.
func (e *lowerer) get(i int) ir.Value {
	nd := e.n.Nodes[i]
	if h := e.home[i]; h >= 0 && !e.isOpen(h) {
		if nd.Op != OpFold || e.n.Loops[h].Parent != e.innermost() {
			e.fail("node %d read outside its loop %d", i, h)
			return 0
		}
		if !e.done[h] {
			e.lowerLoop(h)
		}
		return e.phi[i]
	}
	if e.val[i] != 0 {
		return e.val[i]
	}
	b := e.b
	var v ir.Value
	switch nd.Op {
	case OpConst:
		v = b.Const(nd.Type, nd.Imm)
	case OpCoord:
		v = e.coords[nd.A]
	case OpThread:
		v = e.thread()
	case OpFlat:
		v = e.flat
	case OpIter:
		v = e.iter[nd.At]
	case OpFold:
		v = e.phi[i]
	case OpLoad:
		ix := nd.A
		if nd.Ax != nil {
			v = b.Load(nd.Type, e.params[nd.Buf], e.flatten(nd.Buf, nd.Ax), nd.Imm)
			break
		}
		v = b.Load(nd.Type, e.params[nd.Buf], e.get(ix), nd.Imm)
	case OpBitcast:
		v = b.Bitcast(nd.Type, e.get(nd.A))
	case OpSqrt:
		v = b.Sqrt(e.get(nd.A))
	case OpExp:
		v = b.Exp(e.get(nd.A))
	case OpCvtF32:
		v = b.CvtF32(e.get(nd.A))
	case OpCvtI32:
		v = b.CvtI32(e.get(nd.A))
	case OpFma:
		x, y, z := e.get(nd.A), e.get(nd.B), e.get(nd.C)
		v = b.Fma(x, y, z)
	case OpLess:
		x, y := e.get(nd.A), e.get(nd.B)
		v = b.Lt(e.n.Nodes[nd.A].Type, x, y)
	case OpSelect:
		x, y, z := e.get(nd.A), e.get(nd.B), e.get(nd.C)
		v = b.Select(nd.Type, x, y, z)
	case OpGroupSum, OpGroupMax:
		v = e.groupReduce(i, e.get(nd.A), nd.Op == OpGroupMax)
	case OpExchange:
		v = e.exchange(i)
	case OpShuffle:
		v = b.ShuffleXor(nd.Type, e.get(nd.A), nd.Imm)
	case OpLocal:
		x := e.get(nd.A)
		if e.n.Work {
			v = b.Sub(ir.U32, x, b.Const(ir.U32, 0))
		} else {
			v = b.Sub(ir.U32, x, b.Mul(ir.U32, b.CTAID(), b.NTID()))
		}
	case OpAdd, OpSub, OpMul, OpDiv, OpRem, OpMin, OpMax, OpShl, OpShr, OpAnd, OpXor:
		x, y := e.get(nd.A), e.get(nd.B)
		v = binary(b, nd.Op, nd.Type, x, y)
	default:
		e.fail("node %d: op %d", i, nd.Op)
		return 0
	}
	if nd.Op != OpIter && nd.Op != OpFold {
		e.val[i] = v
	}
	return v
}

// flatten is coordinates ax of buffer bf's shape as its flat index.
func (e *lowerer) flatten(bf int, ax []int) ir.Value {
	b, sh := e.b, e.n.Bufs[bf].Shape
	ix := e.get(ax[0])
	for d := 1; d < len(ax); d++ {
		ix = b.Add(ir.U32, b.Mul(ir.U32, ix, b.Const(ir.U32, int64(sh[d]))), e.get(ax[d]))
	}
	return ix
}

func (e *lowerer) storeIndex(st Store) ir.Value {
	if st.Ax != nil {
		return e.flatten(st.Buf, st.Ax)
	}
	return e.get(st.Ix)
}

func (e *lowerer) thread() ir.Value {
	if e.tid == 0 {
		e.tid = e.b.TID()
	}
	return e.tid
}

// groupReduce is x combined over the workgroup: a butterfly in each
// subgroup, each subgroup's result to its slot, and a butterfly over the
// slots.
func (e *lowerer) groupReduce(i int, x ir.Value, max bool) ir.Value {
	b := e.b
	comb := func(p, q ir.Value) ir.Value {
		if max {
			return b.Max(ir.F32, p, q)
		}
		return b.Add(ir.F32, p, q)
	}
	butterfly := func(v ir.Value) ir.Value {
		for m := ir.SubgroupLanes / 2; m >= 1; m /= 2 {
			v = comb(v, b.ShuffleXor(ir.F32, v, int64(m)))
		}
		return v
	}
	sh, tid := e.shared[i], e.thread()
	b.Store(sh, b.Shr(ir.U32, tid, b.Const(ir.U32, 5)), butterfly(x), 0)
	b.Barrier()
	return butterfly(b.Load(ir.F32, sh, b.And(ir.U32, tid, b.Const(ir.U32, ir.SubgroupLanes-1)), 0))
}

// exchange is a thread's value through its workgroup slot, combined over
// its group's slots.
func (e *lowerer) exchange(i int) ir.Value {
	b, nd := e.b, e.n.Nodes[i]
	x := e.get(nd.A)
	sh := e.shared[i]
	b.Store(sh, e.thread(), x, 0)
	b.Barrier()
	base := e.get(nd.B)
	var acc ir.Value
	if nd.C >= 0 {
		acc = e.get(nd.C)
	}
	for s := 0; s < int(nd.Imm); s++ {
		v := b.Load(ir.F32, sh, base, int64(s))
		if acc == 0 {
			acc = v
			continue
		}
		acc = binary(b, Op(nd.At), ir.F32, acc, v)
	}
	return acc
}

func binary(b *ir.Builder, op Op, t ir.Type, x, y ir.Value) ir.Value {
	switch op {
	case OpAdd:
		return b.Add(t, x, y)
	case OpSub:
		return b.Sub(t, x, y)
	case OpMul:
		return b.Mul(t, x, y)
	case OpDiv:
		return b.Div(t, x, y)
	case OpRem:
		return b.Rem(t, x, y)
	case OpMin:
		return b.Min(t, x, y)
	case OpShl:
		return b.Shl(t, x, y)
	case OpShr:
		return b.Shr(t, x, y)
	case OpAnd:
		return b.And(t, x, y)
	case OpXor:
		return b.Xor(t, x, y)
	}
	return b.Max(t, x, y)
}

func (e *lowerer) innermost() int {
	if len(e.open) == 0 {
		return -1
	}
	return e.open[len(e.open)-1]
}

// lowerLoop is loop c: what its trips read that does not change from trip to
// trip first, then the loop with its folds as phis and its stores.
func (e *lowerer) lowerLoop(c int) {
	n, b := e.n, e.b
	folds := n.foldsOf(c)
	for _, i := range e.body(c) {
		if h := e.home[i]; !e.inside(h, c) {
			e.get(i)
		}
	}
	inits := make([]ir.Value, len(folds))
	for j, f := range folds {
		inits[j] = e.get(n.Nodes[f].B)
	}
	l := n.Loops[c]
	var count ir.Value
	if l.Count > 0 {
		count = e.get(l.Count - 1)
	}
	start, step := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
	if l.Count > 0 {
		b.LoopN(count)
	} else if l.Spread {
		start, step = e.thread(), b.Const(ir.U32, int64(n.Group))
		b.Loop(int64((l.Trip + n.Group - 1) / n.Group))
	} else {
		b.Loop(int64(l.Trip))
	}
	it := b.Phi(ir.U32, start)
	e.iter[c] = it
	for j, f := range folds {
		e.phi[f] = b.Phi(n.Nodes[f].Type, inits[j])
	}
	e.open = append(e.open, c)
	saved := append([]ir.Value(nil), e.val...)
	steps := make([]ir.Value, len(folds))
	for j, f := range folds {
		steps[j] = e.get(n.Nodes[f].A)
	}
	for _, st := range n.Stores {
		if st.At == c && !st.Run && !st.Barrier {
			ix, v := e.storeIndex(st), e.get(st.Val)
			b.Store(e.params[st.Buf], ix, v, 0)
		}
	}
	for j, f := range folds {
		b.SetPhi(e.phi[f], steps[j])
	}
	b.SetPhi(it, b.Add(ir.U32, it, step))
	b.EndLoop()
	e.open = e.open[:len(e.open)-1]
	for i := range e.val {
		if e.inside(e.home[i], c) {
			e.val[i] = saved[i]
		}
	}
	e.done[c] = true
}

// inside reports whether loop h is c or sits in it.
func (e *lowerer) inside(h, c int) bool { return e.n.within(h, c) }

// body is every node loop c's trips compute: what its folds' steps and its
// stores read, through inner loops' starts and steps.
func (e *lowerer) body(c int) []int {
	n := e.n
	seen := map[int]bool{}
	var out []int
	var walk func(i int)
	walk = func(i int) {
		if seen[i] {
			return
		}
		seen[i] = true
		nd := n.Nodes[i]
		if nd.Op == OpFold {
			if nd.At != c && n.within(nd.At, c) {
				walk(nd.A)
				walk(nd.B)
			}
		} else {
			for _, a := range n.args(&nd) {
				walk(*a)
			}
		}
		out = append(out, i)
	}
	for _, f := range n.foldsOf(c) {
		walk(n.Nodes[f].A)
	}
	for _, st := range n.Stores {
		if !st.Barrier && !st.Run && n.within(st.At, c) {
			for _, p := range n.storeArgs(&st) {
				walk(*p)
			}
		}
	}
	if l := n.Loops[c]; l.Count > 0 {
		walk(l.Count - 1)
	}
	return out
}
