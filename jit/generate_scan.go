package jit

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/jit/ir"
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
