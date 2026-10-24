package jit

import (
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/jit/ir"
)

// gpuLowering lowers a kernel to the GPU IR. Poison writes NaN into every
// workgroup array before the kernel's first use of it (RULE 13): workgroup
// memory holds whatever earlier work left there, so a read of a slot no
// thread wrote is reproducible only poisoned.
type gpuLowering struct {
	Poison bool
}

func (g gpuLowering) Lower(n *Kernel, d Device) (Lowered, error) {
	if n.Group <= 0 {
		return Lowered{}, fmt.Errorf("no workgroup: the profile binds none")
	}
	e := &lowerer{n: n, b: ir.New(entryName(n.Name), [3]int{n.Group, 1, 1}), poison: g.Poison}
	k, err := e.run()
	if err == nil {
		// Held to the IR's invariants where it is formed, not first where a
		// backend compiles it.
		err = k.Validate()
	}
	return Lowered{GPU: k}, err
}

// entryName is a kernel's name as a backend's entry point can carry it:
// letters, digits and underscores, every other character an underscore,
// and not led by a digit.
func entryName(name string) string {
	out := strings.Map(func(r rune) rune {
		if r == '_' || 'a' <= r && r <= 'z' || 'A' <= r && r <= 'Z' || '0' <= r && r <= '9' {
			return r
		}
		return '_'
	}, name)
	if out == "" || '0' <= out[0] && out[0] <= '9' {
		out = "k" + out
	}
	return out
}

type lowerer struct {
	n      *Kernel
	b      *ir.Builder
	poison bool
	params []ir.Value
	val    []ir.Value
	home   []int
	coords []ir.Value
	flat   ir.Value
	shared map[int]ir.Value
	open   []int
	iter   map[int]ir.Value
	phi    map[int]ir.Value
	done   []bool
	// stored marks each store emitted: one no loop reached is a kernel
	// the lowering cannot place, never a store left out.
	stored []bool
	err    error
}

func (e *lowerer) run() (*ir.Kernel, error) {
	n, b := e.n, e.b
	e.val = make([]ir.Value, len(n.Nodes))
	e.done = make([]bool, len(n.Loops))
	e.stored = make([]bool, len(n.Stores))
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
	if e.poison {
		e.poisonShared()
	}
	var flat ir.Value
	if n.Work && len(n.Space) > 1 {
		e.coords = e.workCoords()
	} else if n.Work {
		flat = b.Min(ir.U32, b.CTAID(), b.Const(ir.U32, int64(n.Elements()-1)))
	} else {
		flat = b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
		e.flat = flat
		if n.Clamp > 0 {
			flat = b.Min(ir.U32, flat, b.Sub(ir.U32, e.get(n.Clamp-1), b.Const(ir.U32, 1)))
		} else {
			flat = b.Min(ir.U32, flat, b.Const(ir.U32, int64(n.Elements()-1)))
		}
	}
	if e.coords == nil {
		e.coords = make([]ir.Value, len(n.Space))
		rest := flat
		lead := make([]int, len(n.Space)+1)
		lead[0] = 1
		for d, x := range n.Space {
			lead[d+1] = lead[d] * x
		}
		for d := len(n.Space) - 1; d > 0; d-- {
			if n.Space[d] == 1 {
				e.coords[d] = b.Const(ir.U32, 0)
				continue
			}
			if lead[d] == 1 {
				e.coords[d] = rest
				rest = b.Const(ir.U32, 0)
				continue
			}
			ext := b.Const(ir.U32, int64(n.Space[d]))
			e.coords[d] = b.Rem(ir.U32, rest, ext)
			rest = b.Div(ir.U32, rest, ext)
		}
		if len(n.Space) > 0 {
			e.coords[0] = rest
			if n.Space[0] == 1 {
				e.coords[0] = b.Const(ir.U32, 0)
			}
		}
	}
	for si, st := range n.Stores {
		switch {
		case st.At >= 0:
			// A store in a nested loop is its outermost loop's.
			if c := e.childOf(st.At, -1); !e.done[c] {
				e.lowerLoop(c)
			}
		case st.Barrier:
			b.Barrier()
		default:
			ix, v := e.storeIndex(st), e.get(st.Val)
			b.Store(e.params[st.Buf], ix, v, 0)
			e.stored[si] = true
		}
	}
	for si, st := range n.Stores {
		if !st.Barrier && !st.Run && !e.stored[si] {
			e.fail("store %d (at loop %d) was never emitted", si, st.At)
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
			if n.Nodes[i].Op == OpFold && n.Nodes[i].At == c || !n.Varies(i, c) {
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
		p := e.n.Loops[h].Parent
		switch {
		case nd.Op == OpFold && e.done[h] && (p < 0 || e.isOpen(p)):
		case nd.Op == OpFold && p == e.innermost():
			e.lowerLoop(h)
		default:
			e.fail("node %d read outside its loop %d", i, h)
			return 0
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
	case OpDot4:
		v = b.Dot4(e.get(nd.A), e.get(nd.B), e.get(nd.C))
	case OpDot4U:
		e.fail("node %d: a byte dot of unsigned bytes, which no device profile here asks of a GPU", i)
	case OpWidenHalf:
		v = b.CvtF16H(e.get(nd.A))
	case OpHalf:
		v = b.PackF16(e.get(nd.A), b.Const(ir.F32, 0))
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
	case OpAdd, OpSub, OpMul, OpDiv, OpRem, OpMin, OpMax, OpShl, OpShr, OpAnd, OpXor, OpOr:
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

// thread is the thread's index in its workgroup, read where it is used: a
// copy kept from the first use would be one a scope made, which the threads
// that skip the scope never set.
func (e *lowerer) thread() ir.Value { return e.b.TID() }

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
	if e.n.Group <= ir.SubgroupLanes {
		return butterfly(x)
	}
	sh, tid := e.shared[i], e.thread()
	b.Store(sh, b.Shr(ir.U32, tid, b.Const(ir.U32, 5)), butterfly(x), 0)
	b.Barrier()
	lane := b.And(ir.U32, tid, b.Const(ir.U32, ir.SubgroupLanes-1))
	warps := (e.n.Group + ir.SubgroupLanes - 1) / ir.SubgroupLanes
	if warps >= ir.SubgroupLanes {
		return butterfly(b.Load(ir.F32, sh, lane, 0))
	}
	// Only the workgroup's warps wrote a slot: a lane past them reads the
	// last one written (a maximum's identity) or adds nothing.
	v := b.Load(ir.F32, sh, b.Min(ir.U32, lane, b.Const(ir.U32, int64(warps-1))), 0)
	if !max {
		v = b.Select(ir.F32, b.Lt(ir.U32, lane, b.Const(ir.U32, int64(warps))), v, b.Const(ir.F32, 0))
	}
	return butterfly(v)
}

// poisonShared has every thread write NaN to its slots of each workgroup
// array, then waits for the workgroup.
func (e *lowerer) poisonShared() {
	b, n := e.b, e.n
	fill := func(sh ir.Value, elem ir.Type, size int) {
		for j := 0; j < size; j += n.Group {
			at := b.Min(ir.U32, b.Add(ir.U32, b.TID(), b.Const(ir.U32, int64(j))), b.Const(ir.U32, int64(size-1)))
			b.Store(sh, at, b.Bitcast(elem, b.Const(ir.U32, 0x7fc00000)), 0)
		}
	}
	poisoned := false
	for i, bf := range n.Bufs {
		if bf.Kind == BufShared {
			fill(e.params[i], bf.Elem, bf.Size)
			poisoned = true
		}
	}
	for i := range n.Nodes {
		if sh, ok := e.shared[i]; ok {
			fill(sh, ir.F32, n.Group)
			poisoned = true
		}
	}
	if poisoned {
		b.Barrier()
	}
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
	case OpOr:
		return b.Or(t, x, y)
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
// trip first, then the loop with its folds as phis and its stores. A
// run-once loop every thread of the workgroup takes or skips together -- a
// share's guard -- is a scope instead: what its body reads is made inside it
// where the body first asks, and forgotten when it closes, so nothing it
// reads lives across the loops beside it, and a value of the thread's or the
// workgroup's index alone stays one the device can make again where it uses
// it.
func (e *lowerer) lowerLoop(c int) {
	n, b := e.n, e.b
	folds := n.foldsOf(c)
	l := n.Loops[c]
	scope := l.Uniform && l.Kept && l.Trip == 1 && len(folds) == 0
	for _, i := range e.body(c) {
		if h := e.home[i]; !e.inside(h, c) && !scope {
			e.get(i)
		}
	}
	inits := make([]ir.Value, len(folds))
	for j, f := range folds {
		inits[j] = e.get(n.Nodes[f].B)
	}
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
	for si, st := range n.Stores {
		switch {
		case st.At != c && st.At >= 0 && n.within(st.At, c):
			// A store in a loop inside c: that loop, where its first
			// store falls in c's trip.
			if d := e.childOf(st.At, c); !e.done[d] {
				e.lowerLoop(d)
			}
		case st.At != c || st.Run:
		case st.Barrier:
			// Only where every thread of the workgroup takes the trip.
			if l.Uniform {
				b.Barrier()
			}
		default:
			ix, v := e.storeIndex(st), e.get(st.Val)
			b.Store(e.params[st.Buf], ix, v, 0)
			e.stored[si] = true
		}
	}
	for j, f := range folds {
		b.SetPhi(e.phi[f], steps[j])
	}
	b.SetPhi(it, b.Add(ir.U32, it, step))
	b.EndLoop()
	e.open = e.open[:len(e.open)-1]
	for i := range e.val {
		if e.inside(e.home[i], c) || scope && saved[i] == 0 {
			e.val[i] = saved[i]
		}
	}
	e.done[c] = true
}

// childOf is the loop holding loop h that sits directly in c (-1: at the
// top), h itself where it does.
func (e *lowerer) childOf(h, c int) int {
	for e.n.Loops[h].Parent != c {
		h = e.n.Loops[h].Parent
	}
	return h
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
			for _, a := range n.Operands(&nd) {
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
