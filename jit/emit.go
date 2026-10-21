package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/ir"
)

func emit(n *Nest, t Target) (*ir.Kernel, error) {
	if t.Device != DeviceGPU {
		return nil, fmt.Errorf("no emitter for device %d", t.Device)
	}
	e := &emitter{n: n, b: ir.New(n.Name, [3]int{n.Group, 1, 1})}
	return e.run()
}

type emitter struct {
	n      *Nest
	b      *ir.Builder
	params []ir.Value
	val    []ir.Value
	home   []int
	coords []ir.Value
	open   []int
	iter   map[int]ir.Value
	phi    map[int]ir.Value
	done   []bool
	err    error
}

func (e *emitter) run() (*ir.Kernel, error) {
	n, b := e.n, e.b
	e.val = make([]ir.Value, len(n.Nodes))
	e.done = make([]bool, len(n.Loops))
	e.iter, e.phi = map[int]ir.Value{}, map[int]ir.Value{}
	e.homes()
	for i, bf := range n.Bufs {
		e.params = append(e.params, b.Param(fmt.Sprintf("p%d", i), bf.Elem))
	}
	flat := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	flat = b.Min(ir.U32, flat, b.Const(ir.U32, int64(n.elements()-1)))
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
	for _, st := range n.Stores {
		if st.At >= 0 {
			continue
		}
		ix, v := e.get(st.Ix), e.get(st.Val)
		b.Store(e.params[st.Buf], ix, v, 0)
	}
	if e.err != nil {
		return nil, e.err
	}
	return b.Done()
}

func (e *emitter) fail(format string, args ...any) {
	if e.err == nil {
		e.err = fmt.Errorf(format, args...)
	}
}

// homes is each node's innermost loop: the deepest loop whose iteration or
// fold it reads.
func (e *emitter) homes() {
	n := e.n
	e.home = make([]int, len(n.Nodes))
	depth := func(c int) int {
		d := 0
		for ; c >= 0; c = n.Loops[c].Parent {
			d++
		}
		return d
	}
	deeper := func(a, b int) int {
		if depth(b) > depth(a) {
			return b
		}
		return a
	}
	for i := range n.Nodes {
		e.home[i] = -2
	}
	var visit func(i int) int
	visit = func(i int) int {
		if e.home[i] != -2 {
			return e.home[i]
		}
		e.home[i] = -1
		nd := n.Nodes[i]
		h := -1
		switch nd.Op {
		case OpIter, OpFold:
			h = nd.At
		case OpConst, OpCoord:
		default:
			for _, a := range operands(nd) {
				h = deeper(h, visit(a))
			}
		}
		e.home[i] = h
		return h
	}
	for i := range n.Nodes {
		visit(i)
	}
}

func operands(nd Node) []int {
	switch nd.Op {
	case OpConst, OpCoord, OpIter, OpFold:
		return nil
	case OpLoad, OpBitcast:
		return []int{nd.A}
	case OpFma:
		return []int{nd.A, nd.B, nd.C}
	}
	return []int{nd.A, nd.B}
}

func (e *emitter) isOpen(c int) bool {
	for _, o := range e.open {
		if o == c {
			return true
		}
	}
	return false
}

// get is node i's value where the emitter stands, its loop emitted first
// where it is a fold read after the loop.
func (e *emitter) get(i int) ir.Value {
	nd := e.n.Nodes[i]
	if h := e.home[i]; h >= 0 && !e.isOpen(h) {
		if nd.Op != OpFold || e.n.Loops[h].Parent != e.innermost() {
			e.fail("node %d read outside its loop %d", i, h)
			return 0
		}
		if !e.done[h] {
			e.emitLoop(h)
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
	case OpIter:
		v = e.iter[nd.At]
	case OpFold:
		v = e.phi[i]
	case OpLoad:
		v = b.Load(nd.Type, e.params[nd.Buf], e.get(nd.A), nd.Imm)
	case OpBitcast:
		v = b.Bitcast(nd.Type, e.get(nd.A))
	case OpFma:
		x, y, z := e.get(nd.A), e.get(nd.B), e.get(nd.C)
		v = b.Fma(x, y, z)
	case OpAdd, OpSub, OpMul, OpDiv, OpRem, OpMin, OpMax:
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
	}
	return b.Max(t, x, y)
}

func (e *emitter) innermost() int {
	if len(e.open) == 0 {
		return -1
	}
	return e.open[len(e.open)-1]
}

// emitLoop is loop c: what its trips read that does not change from trip to
// trip first, then the loop with its folds as phis.
func (e *emitter) emitLoop(c int) {
	n, b := e.n, e.b
	var folds []int
	for i, nd := range n.Nodes {
		if nd.Op == OpFold && nd.At == c {
			folds = append(folds, i)
		}
	}
	body := e.body(c, folds)
	for _, i := range body {
		if h := e.home[i]; !e.inside(h, c) {
			e.get(i)
		}
	}
	inits := make([]ir.Value, len(folds))
	for j, f := range folds {
		inits[j] = e.get(n.Nodes[f].B)
	}
	b.Loop(int64(n.Loops[c].Trip))
	it := b.Phi(ir.U32, b.Const(ir.U32, 0))
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
	for j, f := range folds {
		b.SetPhi(e.phi[f], steps[j])
	}
	b.SetPhi(it, b.Add(ir.U32, it, b.Const(ir.U32, 1)))
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
func (e *emitter) inside(h, c int) bool {
	for ; h >= 0; h = e.n.Loops[h].Parent {
		if h == c {
			return true
		}
	}
	return false
}

// body is every node loop c's folds' steps read, through its own nodes.
func (e *emitter) body(c int, folds []int) []int {
	seen := map[int]bool{}
	var out []int
	var walk func(i int)
	walk = func(i int) {
		if seen[i] {
			return
		}
		seen[i] = true
		nd := e.n.Nodes[i]
		if nd.Op == OpFold && nd.At != c && e.inside(nd.At, c) {
			walk(nd.A)
			walk(nd.B)
		}
		for _, a := range operands(nd) {
			walk(a)
		}
		out = append(out, i)
	}
	for _, f := range folds {
		walk(e.n.Nodes[f].A)
	}
	return out
}
