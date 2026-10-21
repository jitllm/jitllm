package jit

import (
	"math"

	"github.com/jitllm/jitllm/jit/ir"
)

// SplitKeys splits a normalized exponential reduction over a range of
// positions -- a maximum over terms, the terms' exp less it, their sum and
// their weighted sum, the latter divided by the former, and what computes
// the terms -- into partitions of each row's live span: the range's axis
// becomes a partition and a position in it, a partition its share of the
// live span rounded up to Align, at most the chunk a workgroup holds, the
// live end the partition's. Each partition's maximum, sum and weighted
// sums go into a partial, the weighted sums padded to Pad then Copies
// copies of the maximum and of the sum; a merge weights each partition by
// exp of its maximum less the greatest.
type SplitKeys struct {
	Align, Pad, Copies int
}

func (SplitKeys) Name() string { return "splitkeys" }

func (s SplitKeys) Apply(prog []*Nest, t Target) ([]*Nest, error) {
	var out []*Nest
	for i := 0; i < len(prog); i++ {
		if used, ns, ok := s.split(prog[i:], t); ok {
			for _, n := range ns {
				n.Passes = append(n.Passes, s.Name())
			}
			out = append(out, ns...)
			i += used - 1
			continue
		}
		out = append(out, prog[i])
	}
	return out, nil
}

// xscope is where expressions are instanced as expressions.
type xscope struct {
	elem   int
	coords []int
	load   func(j, ix int) int
	at     func(j int, ax, ext []int) int
	live   func() int
}

func (x *exprs) instance(es []Expr, sc xscope) int {
	vals := make([]int, len(es))
	for i, e := range es {
		switch e.Op {
		case ExprElem:
			vals[i] = sc.elem
		case ExprCoord:
			vals[i] = sc.coords[e.A]
		case ExprLoad:
			vals[i] = sc.load(e.In, vals[e.A])
		case ExprAt:
			ax := make([]int, len(e.Ax))
			for k, a := range e.Ax {
				ax[k] = vals[a]
			}
			vals[i] = sc.at(e.In, ax, e.Ext)
		case ExprLive:
			vals[i] = sc.live()
		default:
			n := e
			n.A, n.B, n.C = vals[e.A], vals[e.B], vals[e.C]
			vals[i] = x.add(n)
		}
	}
	return vals[len(vals)-1]
}

func (x *exprs) flatten(c, ext []int) int {
	ix := c[0]
	for d := 1; d < len(ext); d++ {
		ix = x.op(OpAdd, ir.U32, x.op(OpMul, ir.U32, ix, x.u(uint32(ext[d]))), c[d])
	}
	return ix
}

// coordsOf is flat index e's coordinates in a space of extents ext.
func (x *exprs) coordsOf(e int, ext []int) []int {
	c := make([]int, len(ext))
	for d := len(ext) - 1; d > 0; d-- {
		c[d] = x.op(OpRem, ir.U32, e, x.u(uint32(ext[d])))
		e = x.op(OpDiv, ir.U32, e, x.u(uint32(ext[d])))
	}
	if len(ext) > 0 {
		c[0] = e
	}
	return c
}

// splitting is one split: its partitions and their capacity, and the
// members split, by their outputs.
type splitting struct {
	S, J    int
	members map[Operand]*Prim
}

// bounds is partition part of p's range at coordinates coords: its first
// position, the first past it, and the range's aligned start.
func (sp splitting) bounds(x *exprs, p *Prim, coords []int, part int) (lo, hi, base int) {
	U, A := ir.U32, uint32(p.Range.Align)
	at := func(es []Expr) int {
		return x.instance(es, xscope{coords: coords, load: func(j, ix int) int { return x.load(j, ix) }})
	}
	first, end := at(p.Range.First), at(p.Range.End)
	base = x.op(OpMul, U, x.op(OpDiv, U, first, x.u(A)), x.u(A))
	span := x.sel(U, x.op(OpLess, ir.Pred, first, end), x.op(OpSub, U, end, base), x.u(0))
	share := x.op(OpDiv, U, x.op(OpAdd, U, span, x.u(uint32(sp.S-1))), x.u(uint32(sp.S)))
	per := x.op(OpMul, U, x.op(OpDiv, U, x.op(OpAdd, U, share, x.u(A-1)), x.u(A)), x.u(A))
	lo = x.op(OpMin, U, end, x.op(OpAdd, U, base, x.op(OpMul, U, part, per)))
	hi = x.op(OpMin, U, end, x.op(OpAdd, U, lo, per))
	return lo, hi, base
}

// partition is member p over the partitions: its range's axis split into a
// partition and a position in it, a read of a member's output at the
// position's coordinate read at the partition's, the live end the
// partition's.
func (sp splitting) partition(p *Prim) (Prim, bool) {
	U := ir.U32
	a := p.Range.Axis
	shape := append(append(append([]int(nil), p.Shape[:a]...), sp.S, sp.J), p.Shape[a+1:]...)
	var x exprs
	coords := make([]int, len(shape))
	for i := range coords {
		coords[i] = x.coord(i)
	}
	part, j := coords[a], coords[a+1]
	lo, hi, base := sp.bounds(&x, p, coords, part)
	key := x.op(OpAdd, U, x.op(OpSub, U, lo, base), j)
	old := make([]int, len(p.Shape))
	for i := range old {
		switch {
		case i < a:
			old[i] = coords[i]
		case i == a:
			old[i] = key
		default:
			old[i] = coords[i+1]
		}
	}
	ok := true
	sc := xscope{coords: old, live: func() int { return hi }}
	sc.load = func(jj, ix int) int { return x.load(jj, ix) }
	sc.at = func(jj int, ax, ext []int) int {
		q, member := sp.members[p.In[jj]]
		if !member {
			return x.load(jj, x.flatten(ax, ext))
		}
		qa := q.Range.Axis
		var nax, next []int
		switch {
		case q.Kind == PrimReduce && qa == len(q.Shape)-1:
			nax = append(append([]int(nil), ax...), part)
			next = append(append([]int(nil), ext...), sp.S)
		case qa < len(ax) && ax[qa] == key:
			nax = append(append(append([]int(nil), ax[:qa]...), part, j), ax[qa+1:]...)
			next = append(append(append([]int(nil), ext[:qa]...), sp.S, sp.J), ext[qa+1:]...)
		default:
			ok = false
			return x.u(0)
		}
		return x.add(Expr{Op: ExprAt, In: jj, Ax: nax, Ext: next})
	}
	x.instance(p.Expr, sc)
	if !ok {
		return Prim{}, false
	}
	q := *p
	q.Shape, q.Expr, q.Range = shape, x.e, nil
	q.Part = &Part{Axis: a + 1, Split: a, Align: p.Range.Align, End: p.Range.End}
	if q.Kind == PrimReduce {
		q.N, q.K = q.N/p.Shape[a]*sp.S, sp.J
	} else {
		q.N = q.N / p.Shape[a] * sp.S * sp.J
	}
	var lx exprs
	lc := make([]int, len(shape))
	for i := range lc {
		lc[i] = lx.coord(i)
	}
	llo, lhi, _ := sp.bounds(&lx, p, lc, lc[a])
	lx.op(OpSub, U, lhi, llo)
	q.Span = lx.e
	return q, true
}

func (s SplitKeys) split(ch []*Nest, t Target) (int, []*Nest, bool) {
	var mem []*Prim
	for _, n := range ch {
		if n.Prim == nil || n.Prim.Range == nil || n.Prim.Shape == nil {
			break
		}
		mem = append(mem, n.Prim)
	}
	if len(mem) == 0 || len(mem) == len(ch) || ch[len(mem)].Prim == nil {
		return 0, nil, false
	}
	out := ch[len(mem)].Prim
	var mx, wt, sm, vl *Prim
	for _, p := range mem {
		if p.Kind == PrimReduce && p.Combine == OpMax && p.Range.Axis == len(p.Shape)-1 {
			mx = p
		}
	}
	if mx == nil || mx.Heads == nil {
		return 0, nil, false
	}
	for _, p := range mem {
		switch {
		case p.Kind == PrimMap && reads(p, mx.Out) && hasOp(p.Expr, OpExp):
			wt = p
		case wt != nil && p.Kind == PrimReduce && (p.Combine == OpAdd || p.Combine == OpFma) && reads(p, wt.Out) && reads(out, p.Out):
			if len(p.outShape()) == len(out.Shape) {
				vl = p
			} else {
				sm = p
			}
		}
	}
	if wt == nil || sm == nil || vl == nil || out.Kind != PrimMap || out.Shape == nil {
		return 0, nil, false
	}
	h := *mx.Heads
	C := mx.Shape[mx.Range.Axis]
	g := flashGeometry(h)
	S := g.splits(h, C, t.Slots)
	if S < 2 {
		return 0, nil, false
	}
	sp := splitting{S: S, J: g.chunk, members: map[Operand]*Prim{}}
	for _, p := range mem {
		sp.members[p.Out] = p
	}
	U, F := ir.U32, ir.F32
	O, D := h.Rows*h.Query, h.Dim
	stride := (D+s.Pad-1)/s.Pad*s.Pad + 2*s.Copies
	part := Operand{Kind: OperandScratch, Name: out.Out.Name + ".part"}
	minus := int64(math.Float32bits(math.Float32frombits(0xff7fffff)))
	// at is the partial's word off of (row and head o, partition p).
	at := func(x *exprs, o, p, off int) int {
		return x.op(OpAdd, U, x.op(OpMul, U, x.op(OpAdd, U, x.op(OpMul, U, o, x.u(uint32(S))), p), x.u(uint32(stride))), x.u(uint32(off)))
	}
	rowHead := func(x *exprs) int {
		return x.op(OpAdd, U, x.op(OpMul, U, x.coord(0), x.u(uint32(h.Query))), x.coord(1))
	}

	var parts []Prim
	for _, p := range mem {
		q, ok := sp.partition(p)
		if !ok {
			return 0, nil, false
		}
		if p == vl {
			q.Out = part
			var ix exprs
			ix.op(OpAdd, U, at(&ix, rowHead(&ix), ix.coord(3), 0), ix.coord(2))
			q.Index = ix.e
		}
		parts = append(parts, q)
	}

	// The copies: each partition's maximum and sum, Copies times each.
	var cx exprs
	heads := []int{h.Rows, h.Query, S}
	rh, cp, c := rowHead(&cx), cx.coord(2), cx.coord(3)
	mv := cx.add(Expr{Op: ExprAt, In: 0, Ax: []int{cx.coord(0), cx.coord(1), cp}, Ext: heads})
	lv := cx.add(Expr{Op: ExprAt, In: 1, Ax: []int{cx.coord(0), cx.coord(1), cp}, Ext: heads})
	mat := cx.op(OpAdd, U, at(&cx, rh, cp, stride-2*s.Copies), c)
	lat := cx.op(OpAdd, U, at(&cx, rh, cp, stride-s.Copies), c)
	copies := Prim{Kind: PrimMap, Shape: []int{h.Rows, h.Query, S, s.Copies}, N: O * S * s.Copies, In: []Operand{mx.Out, sm.Out},
		Expr: cx.e, Stores: []PrimStore{{Out: part, Index: mat, Value: mv}, {Out: part, Index: lat, Value: lv}}, Heads: mx.Heads}

	// The merge: the partitions weighted by exp of their maximum less the
	// greatest, the original normalization over the weighted sums.
	var mb exprs
	me := mb.add(Expr{Op: ExprElem})
	mrh, md := mb.op(OpDiv, U, me, mb.u(uint32(D))), mb.op(OpRem, U, me, mb.u(uint32(D)))
	big := mb.add(Expr{Op: OpConst, Type: F, Imm: minus})
	maxes, sums := make([]int, S), make([]int, S)
	for p := 0; p < S; p++ {
		maxes[p] = mb.load(0, at(&mb, mrh, mb.u(uint32(p)), stride-2*s.Copies))
		sums[p] = mb.load(0, at(&mb, mrh, mb.u(uint32(p)), stride-s.Copies))
		big = mb.op(OpMax, F, big, maxes[p])
	}
	L, N := mb.f(0), mb.f(0)
	for p := 0; p < S; p++ {
		e := mb.un(OpExp, F, mb.op(OpSub, F, maxes[p], big))
		L = mb.add(Expr{Op: OpFma, Type: F, A: sums[p], B: e, C: L})
		num := mb.load(0, mb.op(OpAdd, U, at(&mb, mrh, mb.u(uint32(p)), 0), md))
		N = mb.add(Expr{Op: OpFma, Type: F, A: num, B: e, C: N})
	}
	mb.instance(out.Expr, xscope{
		coords: mb.coordsOf(me, out.Shape),
		at: func(j int, ax, ext []int) int {
			if out.In[j] == sm.Out {
				return L
			}
			return N
		},
	})
	merge := Prim{Kind: PrimMap, N: O * D, Out: out.Out, In: []Operand{part}, Expr: mb.e}

	var ns []*Nest
	for _, p := range append(parts, copies, merge) {
		lowered, err := lower(p)
		if err != nil {
			return 0, nil, false
		}
		for _, x := range lowered {
			x.Kind = p.Kind
			q := p
			x.Prim = &q
			ns = append(ns, x)
		}
	}
	return len(mem) + 1, ns, true
}

func reads(p *Prim, o Operand) bool {
	for _, x := range p.In {
		if x == o {
			return true
		}
	}
	return false
}

func hasOp(es []Expr, op Op) bool {
	for _, e := range es {
		if e.Op == op {
			return true
		}
	}
	return false
}
