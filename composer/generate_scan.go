package composer

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/composer/ir"
)

// generateScan is a scan's naive IR: a workgroup of one subgroup's threads
// a state row, each thread holding the components lane, lane+G, ..., a
// component past the row's last clamped to it and left out of every sum.
// The runs are a loop over the descriptor's runs, each a loop over its rows
// in order; the components are folds of the row loop, started from Init at
// the run's start and stored to Final after its last row. A step's row and
// run are read from the descriptor, so the trips are the step's own.
func generateScan(p Prim) (*Kernel, error) {
	sc := p.Scan
	if sc == nil || len(p.Shape) < 2 || sc.Desc < 0 || sc.Desc >= len(p.In) {
		return nil, fmt.Errorf("jit: a scan with no descriptor")
	}
	U := ir.U32
	rows, K := p.Shape[:len(p.Shape)-1], p.Shape[len(p.Shape)-1]
	G := ir.SubgroupLanes
	m := (K + G - 1) / G
	k := &Kernel{Name: "scan", Kind: PrimScan, Space: append([]int(nil), rows...), Work: true, Group: G}
	g := &generator{k: k, p: &p, flat: map[Operand]int{}, shaped: map[Operand]int{}}
	coords := make([]int, len(rows))
	for i := range coords {
		coords[i] = k.add(Node{Op: OpCoord, A: i, Type: U})
	}
	word := func(ix int) int { return k.load(g.buffer(g.flat, sc.Desc, nil), ix) }
	add := func(a, b int) int { return k.bin(OpAdd, U, a, b) }
	mul := func(a, b int) int { return k.bin(OpMul, U, a, b) }
	n := word(k.u32(0))
	// The runs are the last entry's run and those before it, none in a step
	// of no rows.
	n1 := k.bin(OpMax, U, n, k.u32(1))
	last := word(add(add(k.u32(1), mul(n1, k.u32(4))), k.bin(OpSub, U, n1, k.u32(1))))
	runs := k.bin(OpMin, U, add(word(add(add(k.u32(1), mul(n1, k.u32(6))), last)), k.u32(1)), n)
	outer := k.loop(0, -1)
	k.Loops[outer].Count, k.Loops[outer].Uniform = runs+1, true
	run := k.add(Node{Op: OpIter, At: outer, Type: U})
	twoRun := add(run, run)
	start := word(add(add(k.u32(1), mul(n, k.u32(2))), twoRun))
	count := word(add(add(k.u32(2), mul(n, k.u32(2))), twoRun))
	inner := k.loop(0, outer)
	k.Loops[inner].Count, k.Loops[inner].Uniform = count+1, true
	row := word(add(add(k.u32(1), mul(n, k.u32(4))), add(start, k.add(Node{Op: OpIter, At: inner, Type: U}))))

	lane := k.add(Node{Op: OpThread, Type: U})
	comp, live := make([]int, m), make([]int, m)
	for c := range m {
		at := add(lane, k.u32(c*G))
		comp[c] = k.bin(OpMin, U, at, k.u32(K-1))
		live[c] = k.add(Node{Op: OpLess, Type: ir.Pred, A: at, B: k.u32(K)})
	}
	ev := scanEval{g: g, coords: coords, row: row, run: run, comp: comp, live: live}
	fold := make([]int, m)
	for c := range m {
		init, err := ev.at(sc.Init, c)
		if err != nil {
			return nil, err
		}
		fold[c] = k.add(Node{Op: OpFold, At: inner, Type: ir.F32, B: init})
	}
	ev.state = fold
	ev.memo = map[[2]int]int{}
	for c := range m {
		next, err := ev.at(sc.Next, c)
		if err != nil {
			return nil, err
		}
		k.Nodes[fold[c]].A = next
	}
	for _, s := range p.Stores {
		v, err := ev.at(s.Value, 0)
		if err != nil {
			return nil, err
		}
		b := k.buf(Buf{Kind: BufOut, Elem: elemOf(s.Out.Elem), Operand: s.Out, Shape: append([]int(nil), s.Ext...)})
		st := Store{Buf: b, Val: v, At: inner}
		if s.Ax != nil {
			for _, a := range s.Ax {
				x, err := ev.at(a, 0)
				if err != nil {
					return nil, err
				}
				st.Ax = append(st.Ax, x)
			}
		} else {
			if st.Ix, err = ev.at(s.Index, 0); err != nil {
				return nil, err
			}
			k.Bufs[b].Extent = s.Extent
		}
		k.Stores = append(k.Stores, st)
	}
	fb := k.buf(Buf{Kind: BufOut, Elem: elemOf(sc.Final.Elem), Operand: sc.Final, Extent: sc.Extent})
	ev.memo = map[[2]int]int{}
	for c := range m {
		ix, err := ev.at(sc.Index, c)
		if err != nil {
			return nil, err
		}
		k.Stores = append(k.Stores, Store{Buf: fb, Ix: ix, Val: fold[c], At: outer})
	}
	return k, nil
}

// scanEval instances a scan's expressions: once for a node that does not
// read a component, once a component for one that does.
type scanEval struct {
	g                 *generator
	coords            []int
	row, run          int
	comp, live, state []int
	memo              map[[2]int]int
	perComp           map[int]bool
}

// at is node i of the scan's expressions for component c.
func (e *scanEval) at(i, c int) (int, error) {
	if e.memo == nil {
		e.memo = map[[2]int]int{}
	}
	if !e.reads(i) {
		c = -1
	}
	key := [2]int{i, c}
	if v, ok := e.memo[key]; ok {
		return v, nil
	}
	k, x := e.g.k, e.g.p.Expr[i]
	var v int
	switch x.Op {
	case ExprCoord:
		switch d := len(e.coords); {
		case x.A < d:
			v = e.coords[x.A]
		case x.A == d:
			v = e.comp[c]
		case x.A == d+1:
			v = e.row
		case x.A == d+2:
			v = e.run
		default:
			return 0, fmt.Errorf("jit: a scan's coordinate %d", x.A)
		}
	case ExprState:
		if e.state == nil {
			return 0, fmt.Errorf("jit: a scan's start reads its state")
		}
		v = e.state[c]
	case ExprLoad:
		ix, err := e.at(x.A, c)
		if err != nil {
			return 0, err
		}
		v = k.load(e.g.buffer(e.g.flat, x.In, nil), ix)
	case ExprAt:
		ax := make([]int, len(x.Ax))
		for j, a := range x.Ax {
			var err error
			if ax[j], err = e.at(a, c); err != nil {
				return 0, err
			}
		}
		b := e.g.buffer(e.g.shaped, x.In, x.Ext)
		v = k.add(Node{Op: OpLoad, Type: k.Bufs[b].Elem, Buf: b, Ax: ax})
	case OpConst:
		v = k.add(Node{Op: OpConst, Type: x.Type, Imm: x.Imm})
	case OpGroupSum:
		sum := -1
		for cc := range e.comp {
			t, err := e.at(x.A, cc)
			if err != nil {
				return 0, err
			}
			t = k.add(Node{Op: OpSelect, Type: ir.F32, A: e.live[cc], B: t, C: k.f32(math.Float32bits(0))})
			if sum < 0 {
				sum = t
				continue
			}
			sum = k.bin(OpAdd, ir.F32, sum, t)
		}
		v = k.un(OpGroupSum, ir.F32, sum)
	default:
		nd := Node{Op: x.Op, Type: x.Type}
		for j, a := range []int{x.A, x.B, x.C}[:arity(x.Op)] {
			o, err := e.at(a, c)
			if err != nil {
				return 0, err
			}
			switch j {
			case 0:
				nd.A = o
			case 1:
				nd.B = o
			default:
				nd.C = o
			}
		}
		v = k.add(nd)
	}
	e.memo[key] = v
	return v, nil
}

// reads reports whether node i reads a component: its coordinate or its
// state, past no sum over them.
func (e *scanEval) reads(i int) bool {
	if e.perComp == nil {
		e.perComp = map[int]bool{}
	}
	if r, ok := e.perComp[i]; ok {
		return r
	}
	x := e.g.p.Expr[i]
	r := false
	switch x.Op {
	case ExprCoord:
		r = x.A == len(e.coords)
	case ExprState:
		r = true
	case OpGroupSum, OpConst:
	case ExprLoad:
		r = e.reads(x.A)
	case ExprAt:
		for _, a := range x.Ax {
			r = r || e.reads(a)
		}
	default:
		for _, a := range []int{x.A, x.B, x.C}[:arity(x.Op)] {
			r = r || e.reads(a)
		}
	}
	e.perComp[i] = r
	return r
}

// generateScanLoops is a scan's naive IR on a device whose element is a
// thread with memory beside it (Scan.Loops): an element a state row, its
// components loops rather than lanes. The step's rows are one loop over the
// order list, run after run; a run's first row reads its start in the
// table the state is written to (From), each later row the state the row
// before wrote there (Index), so the state carries from row to row in
// memory. A sum over the
// components is a fold of a loop of its own, after the loops whose sums it
// reads; the state is written in the loop of the update (Next), which
// every sum reading the state must share or precede.
func generateScanLoops(p Prim) (*Kernel, error) {
	sc := p.Scan
	if sc == nil || len(p.Shape) < 2 || sc.Desc < 0 || sc.Desc >= len(p.In) {
		return nil, fmt.Errorf("jit: a scan with no descriptor")
	}
	U := ir.U32
	rows, K := p.Shape[:len(p.Shape)-1], p.Shape[len(p.Shape)-1]
	// The state rows are one axis, a row's coordinates its quotients.
	all := 1
	for _, r := range rows {
		all *= r
	}
	k := &Kernel{Name: "scan", Kind: PrimScan, Space: []int{all}}
	g := &generator{k: k, p: &p, flat: map[Operand]int{}, shaped: map[Operand]int{}}
	coords := make([]int, len(rows))
	at := k.add(Node{Op: OpCoord, Type: U})
	for i := len(rows) - 1; i > 0; i-- {
		r := k.u32(rows[i])
		q := k.bin(OpDiv, U, at, r)
		coords[i] = k.bin(OpSub, U, at, k.bin(OpMul, U, q, r))
		at = q
	}
	coords[0] = at
	desc := g.buffer(g.flat, sc.Desc, nil)
	n := k.load(desc, k.u32(0))
	word := func(a, b, ix int) int {
		return k.load(desc, k.bin(OpAdd, U, k.bin(OpAdd, U, k.u32(a), k.bin(OpMul, U, n, k.u32(b))), ix))
	}
	fb := k.buf(Buf{Kind: BufOut, Elem: elemOf(sc.Final.Elem), Operand: sc.Final, Extent: sc.Extent})
	held := k.buf(Buf{Kind: BufIn, Elem: elemOf(sc.Final.Elem), Operand: sc.Final, Extent: sc.Extent})
	entries := k.loop(0, -1)
	k.Loops[entries].Count = n + 1
	e := k.add(Node{Op: OpIter, At: entries, Type: U})
	row := word(1, 4, e)
	run := word(1, 6, row)
	// A run's first entry reads its start (From), its others the state the
	// row before wrote (Index): first is 1 at the first, 0 past it.
	first := k.bin(OpSub, U, k.u32(1), k.bin(OpMin, U, k.bin(OpSub, U, e, word(1, 2, k.bin(OpAdd, U, run, run))), k.u32(1)))
	ev := &loopEval{g: g, coords: coords, row: row, run: run, first: first, held: held, parent: entries, K: K,
		memo: map[[2]int]int{}, perComp: map[int]bool{}, levels: map[int]int{}}
	M := ev.level(sc.Next)
	next, err := ev.at(sc.Next, ev.inst(M))
	if err != nil {
		return nil, err
	}
	ix, err := ev.at(sc.Index, ev.inst(M))
	if err != nil {
		return nil, err
	}
	k.Stores = append(k.Stores, Store{Buf: fb, Ix: ix, Val: next, At: ev.loops[M]})
	for _, s := range p.Stores {
		if ev.reads(s.Value) {
			return nil, fmt.Errorf("jit: a scan's row store reads its state past its update")
		}
		v, err := ev.at(s.Value, -1)
		if err != nil {
			return nil, err
		}
		b := k.buf(Buf{Kind: BufOut, Elem: elemOf(s.Out.Elem), Operand: s.Out, Shape: append([]int(nil), s.Ext...)})
		st := Store{Buf: b, Val: v, At: entries}
		if s.Ax != nil {
			for _, a := range s.Ax {
				x, err := ev.at(a, -1)
				if err != nil {
					return nil, err
				}
				st.Ax = append(st.Ax, x)
			}
		} else {
			if st.Ix, err = ev.at(s.Index, -1); err != nil {
				return nil, err
			}
			k.Bufs[b].Extent = s.Extent
		}
		k.Stores = append(k.Stores, st)
	}
	if ev.late {
		return nil, fmt.Errorf("jit: a scan's sum reads its state past its update")
	}
	return k, nil
}

// loopEval instances a scan's expressions in component loops: once for a
// node that does not read a component, once a loop for one that does.
type loopEval struct {
	g               *generator
	coords          []int
	row, run, first int
	held, parent, K int
	loops, comps    []int
	memo            map[[2]int]int
	perComp         map[int]bool
	levels          map[int]int
	late            bool
}

// inst is the component loop of level l, made where none is: a loop of the
// row's components after the loops of the levels below it.
func (e *loopEval) inst(l int) int {
	k := e.g.k
	for len(e.loops) <= l {
		c := k.loop(e.K, e.parent)
		e.loops = append(e.loops, c)
		e.comps = append(e.comps, k.add(Node{Op: OpIter, At: c, Type: ir.U32}))
	}
	return l
}

// level is the component loop node i can be made in: past every loop whose
// sum it reads.
func (e *loopEval) level(i int) int {
	if l, ok := e.levels[i]; ok {
		return l
	}
	x := e.g.p.Expr[i]
	l := 0
	switch x.Op {
	case OpGroupSum:
		l = e.level(x.A) + 1
	case ExprCoord, ExprState, OpConst:
	case ExprLoad:
		l = e.level(x.A)
	case ExprAt:
		for _, a := range x.Ax {
			l = max(l, e.level(a))
		}
	default:
		for _, a := range []int{x.A, x.B, x.C}[:arity(x.Op)] {
			l = max(l, e.level(a))
		}
	}
	e.levels[i] = l
	return l
}

// readsState reports whether node i reads the state outside a sum.
func (e *loopEval) readsState(i int) bool {
	x := e.g.p.Expr[i]
	switch x.Op {
	case ExprState:
		return true
	case ExprCoord, OpConst:
		return false
	case OpGroupSum:
		return e.readsState(x.A)
	case ExprLoad:
		return e.readsState(x.A)
	case ExprAt:
		for _, a := range x.Ax {
			if e.readsState(a) {
				return true
			}
		}
		return false
	}
	for _, a := range []int{x.A, x.B, x.C}[:arity(x.Op)] {
		if e.readsState(a) {
			return true
		}
	}
	return false
}

// at is node i of the scan's expressions in component loop c (-1 outside
// them).
func (e *loopEval) at(i, c int) (int, error) {
	if !e.reads(i) {
		c = -1
	} else if c < 0 {
		return 0, fmt.Errorf("jit: a scan's component read outside its loops")
	}
	key := [2]int{i, c}
	if v, ok := e.memo[key]; ok {
		return v, nil
	}
	k, x := e.g.k, e.g.p.Expr[i]
	var v int
	switch x.Op {
	case ExprCoord:
		switch d := len(e.coords); {
		case x.A < d:
			v = e.coords[x.A]
		case x.A == d:
			v = e.comps[c]
		case x.A == d+1 && e.row >= 0:
			v = e.row
		case x.A == d+2:
			v = e.run
		default:
			return 0, fmt.Errorf("jit: a scan's coordinate %d", x.A)
		}
	case ExprState:
		ix, err := e.at(e.g.p.Scan.Index, c)
		if err != nil {
			return 0, err
		}
		from, err := e.at(e.g.p.Scan.From, c)
		if err != nil {
			return 0, err
		}
		U := ir.U32
		v = k.load(e.held, k.bin(OpAdd, U, ix, k.bin(OpMul, U, k.bin(OpSub, U, from, ix), e.first)))
	case ExprLoad:
		ix, err := e.at(x.A, c)
		if err != nil {
			return 0, err
		}
		v = k.load(e.g.buffer(e.g.flat, x.In, nil), ix)
	case ExprAt:
		ax := make([]int, len(x.Ax))
		for j, a := range x.Ax {
			var err error
			if ax[j], err = e.at(a, c); err != nil {
				return 0, err
			}
		}
		b := e.g.buffer(e.g.shaped, x.In, x.Ext)
		v = k.add(Node{Op: OpLoad, Type: k.Bufs[b].Elem, Buf: b, Ax: ax})
	case OpConst:
		v = k.add(Node{Op: OpConst, Type: x.Type, Imm: x.Imm})
	case OpGroupSum:
		l := e.level(x.A)
		if l > e.level(e.g.p.Scan.Next) && e.readsState(x.A) {
			e.late = true
		}
		cc := e.inst(l)
		t, err := e.at(x.A, cc)
		if err != nil {
			return 0, err
		}
		fold := k.add(Node{Op: OpFold, At: e.loops[cc], Type: ir.F32, B: k.f32(math.Float32bits(0))})
		k.Nodes[fold].A = k.bin(OpAdd, ir.F32, fold, t)
		v = fold
	default:
		nd := Node{Op: x.Op, Type: x.Type}
		for j, a := range []int{x.A, x.B, x.C}[:arity(x.Op)] {
			o, err := e.at(a, c)
			if err != nil {
				return 0, err
			}
			switch j {
			case 0:
				nd.A = o
			case 1:
				nd.B = o
			default:
				nd.C = o
			}
		}
		v = k.add(nd)
	}
	e.memo[key] = v
	return v, nil
}

// reads reports whether node i reads a component: its coordinate or its
// state, past no sum over them.
func (e *loopEval) reads(i int) bool {
	if r, ok := e.perComp[i]; ok {
		return r
	}
	x := e.g.p.Expr[i]
	r := false
	switch x.Op {
	case ExprCoord:
		r = x.A == len(e.coords)
	case ExprState:
		r = true
	case OpGroupSum, OpConst:
	case ExprLoad:
		r = e.reads(x.A)
	case ExprAt:
		for _, a := range x.Ax {
			r = r || e.reads(a)
		}
	default:
		for _, a := range []int{x.A, x.B, x.C}[:arity(x.Op)] {
			r = r || e.reads(a)
		}
	}
	e.perComp[i] = r
	return r
}
