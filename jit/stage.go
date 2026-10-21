package jit

import "github.com/jitllm/jitllm/jit/ir"

// Stage forms a chain of primitives over a split range of positions into
// one workgroup a (row, group of query heads, partition): the group's heads
// computed together, what the chain passes along the positions staged in a
// plane of workgroup memory, each member in a phase of its own bound to the
// threads by its structure.
//   - A reduction over an axis other than the positions, the positions in
//     its output: a position a thread, lanes over consecutive positions, the
//     reduction within the thread in Chains interleaved chains summed
//     pairwise, its results to the plane.
//   - A map over the positions: a position a thread, its result to the
//     plane in place of what it reads there; a reduction of its results
//     over the positions folds into its trips, then over the lanes by a
//     butterfly, each warp's lanes to a workgroup array and the warps'
//     first lanes combined.
//   - A reduction over the positions with an axis of its own in its output:
//     lanes over that axis, warps over the positions Unroll a trip, Mask
//     setting a staged factor past the live span to zero; the warps' sums
//     through the plane, combined by a thread an element.
//   - A map of what the chain reduced: a thread an element.
//
// Where Runs is set, a position's page is found once a run of positions: the
// lanes' positions where the lanes run over them, else a trip's, the run's
// start held to the last aligned run of the range, and where the lanes run
// over the positions a position's place its run's plus its lane.
type Stage struct {
	Unroll, Chains int
	Runs, Mask     bool
}

func (Stage) Name() string { return "stage" }

func (s Stage) Apply(prog []*Nest, t Target) ([]*Nest, error) {
	var out []*Nest
	for i := 0; i < len(prog); i++ {
		if used, n, ok := s.form(prog[i:], t); ok {
			n.Passes = append(n.Passes, s.Name())
			out = append(out, n)
			i += used - 1
			continue
		}
		out = append(out, prog[i])
	}
	return out, nil
}

type stager struct {
	s                   Stage
	n                   *Nest
	R, H, S, G, W, L, J int
	r, sp, hbase        int
	tid, lane, warp     int
	plane               int
	bufs                map[Operand]int
	inPlane             map[Operand]*Prim
	regs                map[Operand][]int
	ok                  bool
}

func (s Stage) form(ch []*Nest, t Target) (int, *Nest, bool) {
	var mem []*Prim
	for _, n := range ch {
		p := n.Prim
		if p == nil || p.Part == nil || p.Heads == nil || p.Shape == nil {
			break
		}
		mem = append(mem, p)
	}
	if len(mem) < 2 || t.Device != DeviceGPU || t.Lanes <= 0 {
		return 0, nil, false
	}
	h := *mem[0].Heads
	g := flashGeometry(h)
	S := mem[0].Shape[mem[0].Part.Split]
	written := map[Operand]bool{}
	for _, p := range mem {
		written[p.Out] = true
		if p.Shape[p.Part.Split] != S || p.Shape[p.Part.Axis] != g.chunk {
			return 0, nil, false
		}
	}
	used := len(mem)
	var extra []*Prim
	for _, n := range ch[used:] {
		p := n.Prim
		if p == nil || p.Shape == nil || len(p.Shape) != 4 || p.Shape[2] != S || len(p.Stores) == 0 {
			break
		}
		for _, o := range p.In {
			if !written[o] {
				return 0, nil, false
			}
		}
		extra = append(extra, p)
	}
	used += len(extra)
	if g.wd != 1 || h.Dim > t.Lanes {
		return 0, nil, false
	}
	n := &Nest{Name: "stage", Space: []int{h.Rows, h.Query / g.group, S}, Work: true, Group: g.warps * t.Lanes, Kind: PrimReduce}
	st := &stager{s: s, n: n, R: h.Rows, H: h.Query, S: S, G: g.group, W: g.warps, L: t.Lanes, J: g.chunk,
		bufs: map[Operand]int{}, inPlane: map[Operand]*Prim{}, regs: map[Operand][]int{}, ok: true}
	st.r = n.add(Node{Op: OpCoord, A: 0, Type: ir.U32})
	gc := n.add(Node{Op: OpCoord, A: 1, Type: ir.U32})
	st.sp = n.add(Node{Op: OpCoord, A: 2, Type: ir.U32})
	st.hbase = n.bin(OpMul, ir.U32, gc, n.u32(st.G))
	st.tid = n.add(Node{Op: OpThread, Type: ir.U32})
	st.lane = n.bin(OpRem, ir.U32, st.tid, n.u32(st.L))
	st.warp = n.bin(OpDiv, ir.U32, st.tid, n.u32(st.L))
	st.plane = n.buf(Buf{Kind: BufShared, Elem: ir.F32, Size: st.G * st.J})

	for i := 0; i < len(mem); i++ {
		p := mem[i]
		switch {
		case p.Kind == PrimReduce && p.Part.Axis != len(p.Shape)-1:
			st.lanes(p)
		case p.Kind == PrimMap && p.Part.Axis == len(p.Shape)-1:
			var folds []*Prim
			for i+1 < len(mem) && mem[i+1].Kind == PrimReduce && mem[i+1].Part.Axis == len(mem[i+1].Shape)-1 && len(mem[i+1].Shape) == len(p.Shape) && reads(mem[i+1], p.Out) {
				folds = append(folds, mem[i+1])
				i++
			}
			st.spread(p, folds)
		case p.Kind == PrimReduce && len(p.Shape) == 5:
			st.warps(p)
		default:
			return 0, nil, false
		}
		if !st.ok {
			return 0, nil, false
		}
	}
	for _, p := range extra {
		st.each(p)
	}
	if !st.ok {
		return 0, nil, false
	}
	return used, n, true
}

func (st *stager) head(hl int) int { return st.n.bin(OpAdd, ir.U32, st.hbase, st.n.u32(hl)) }

func (st *stager) planeAt(hl, j int) int {
	n := st.n
	return n.bin(OpAdd, ir.U32, n.bin(OpMin, ir.U32, j, n.u32(st.J-1)), n.u32(hl*st.J))
}

// scope is member p's scope for head hl: external operands as buffers, the
// chain's own where the chain holds them.
func (st *stager) scope(p *Prim, hl int, coords []int) scope {
	n := st.n
	var sc scope
	sc.coords = coords
	sc.load = func(j, ix int) int { return n.load(st.buffer(p.In[j], BufIn), ix) }
	sc.at = func(j int, ax, ext []int) int {
		o := p.In[j]
		if q, ok := st.inPlane[o]; ok {
			return n.load(st.plane, st.planeAt(hl, ax[q.Part.Axis]))
		}
		if r, ok := st.regs[o]; ok {
			return r[hl]
		}
		return n.load(st.buffer(o, BufIn), n.flatten(ax, ext))
	}
	return sc
}

func (st *stager) buffer(o Operand, kind BufKind) int {
	if b, ok := st.bufs[o]; ok {
		return b
	}
	elem := storageElem(o)
	if kind == BufOut {
		elem = elemOf(o.Elem)
	}
	b := st.n.buf(Buf{Kind: kind, Elem: elem, Operand: o})
	st.bufs[o] = b
	return b
}

func (st *stager) value(p *Prim, es []Expr, sc scope) int {
	vals := st.n.instance(*p, es, sc)
	return vals[len(vals)-1]
}

// span is p's live positions in the workgroup's partition.
func (st *stager) span(p *Prim) int {
	c := make([]int, len(p.Shape))
	for i := range c {
		c[i] = st.n.u32(0)
	}
	c[0], c[p.Part.Split] = st.r, st.sp
	return st.value(p, p.Span, scope{coords: c, load: func(j, ix int) int { return st.n.load(st.buffer(p.In[j], BufIn), ix) }})
}

// trips is a loop over a span step positions a trip.
func (st *stager) trips(span, step int) int {
	n := st.n
	c := n.loop(0, -1)
	cnt := n.bin(OpDiv, ir.U32, n.bin(OpAdd, ir.U32, span, n.u32(step-1)), n.u32(step))
	n.Loops[c].Count = cnt + 1
	return c
}

func (st *stager) coords(p *Prim, hl int, at map[int]int) []int {
	split := 2
	if p.Part != nil {
		split = p.Part.Split
	}
	c := make([]int, len(p.Shape))
	c[0], c[1], c[split] = st.r, st.head(hl), st.sp
	for a, v := range at {
		c[a] = v
	}
	return c
}

// runs is the scope's hook finding a held position's page once a run:
// a position held as min(max(x, first), last) and then divided or reduced
// by a page of a multiple of Align, x of the position's coordinate.
func (st *stager) runs(p *Prim, sc *scope, start, lane int) {
	if !st.s.Runs {
		return
	}
	n, es, key, A := st.n, p.Expr, p.Part.Axis, p.Part.Align
	held := func(i int) (x, last int, ok bool) {
		m := es[i]
		if m.Op != OpMin || es[m.A].Op != OpMax || !depends(es, es[m.A].A, key) {
			return 0, 0, false
		}
		return es[m.A].A, m.B, true
	}
	base := *sc
	sc.node = func(i int, vals []int) (int, bool) {
		e := es[i]
		if e.Op != OpDiv && e.Op != OpRem || es[e.B].Op != OpConst {
			return 0, false
		}
		page := int(es[e.B].Imm)
		x, last, ok := held(e.A)
		if !ok || page%A != 0 {
			return 0, false
		}
		if e.Op == OpRem && lane < 0 {
			return 0, false
		}
		rc := append([]int(nil), base.coords...)
		rc[key] = start
		rs := base
		rs.coords = rc
		xr := n.instance(*p, es[:x+1], rs)[x]
		aligned := n.bin(OpMul, ir.U32, n.bin(OpDiv, ir.U32, vals[last], n.u32(A)), n.u32(A))
		h := n.bin(OpMin, ir.U32, xr, aligned)
		v := n.bin(e.Op, ir.U32, h, vals[e.B])
		if e.Op == OpRem {
			v = n.bin(OpAdd, ir.U32, v, lane)
		}
		return v, true
	}
}

func depends(es []Expr, i, axis int) bool {
	e := es[i]
	switch e.Op {
	case ExprCoord:
		return e.A == axis
	case OpConst, ExprElem, ExprLive:
		return false
	case ExprAt:
		for _, a := range e.Ax {
			if depends(es, a, axis) {
				return true
			}
		}
		return false
	case ExprLoad:
		return depends(es, e.A, axis)
	}
	for k, a := range []int{e.A, e.B, e.C} {
		if k < arity(e.Op) && depends(es, a, axis) {
			return true
		}
	}
	return false
}

// lanes is a reduction within a thread over an axis of its own, a position
// a thread.
func (st *stager) lanes(p *Prim) {
	n := st.n
	inner := len(p.Shape) - 1
	K := p.Shape[inner]
	c := st.trips(st.span(p), st.W*st.L)
	it := n.add(Node{Op: OpIter, At: c, Type: ir.U32})
	run := n.bin(OpMul, ir.U32, n.bin(OpAdd, ir.U32, n.bin(OpMul, ir.U32, it, n.u32(st.W)), st.warp), n.u32(st.L))
	j := n.bin(OpAdd, ir.U32, run, st.lane)
	t := elemOf(p.Type)
	chains := max(1, st.s.Chains)
	for hl := 0; hl < st.G; hl++ {
		acc := make([]int, chains)
		for k := range acc {
			acc[k] = n.add(Node{Op: OpConst, Type: t, Imm: p.Init})
		}
		for k := 0; k < K; k++ {
			sc := st.scope(p, hl, st.coords(p, hl, map[int]int{p.Part.Axis: j, inner: n.u32(k)}))
			st.runs(p, &sc, run, st.lane)
			acc[k%chains] = n.combine(p.Combine, t, acc[k%chains], st.value(p, p.Expr, sc))
		}
		sum := OpAdd
		if p.Combine != OpFma {
			sum = p.Combine
		}
		for len(acc) > 1 {
			var next []int
			for k := 0; k+1 < len(acc); k += 2 {
				next = append(next, n.bin(sum, t, acc[k], acc[k+1]))
			}
			if len(acc)%2 == 1 {
				next = append(next, acc[len(acc)-1])
			}
			acc = next
		}
		n.Stores = append(n.Stores, Store{Buf: st.plane, Ix: st.planeAt(hl, j), Val: acc[0], At: c})
	}
	st.inPlane[p.Out] = p
	n.Stores = append(n.Stores, Store{Barrier: true, At: -1})
}

// spread is a map over the positions, a position a thread, and the
// reductions that fold its results.
func (st *stager) spread(p *Prim, folds []*Prim) {
	n := st.n
	c := st.trips(st.span(p), n.Group)
	it := n.add(Node{Op: OpIter, At: c, Type: ir.U32})
	j := n.bin(OpAdd, ir.U32, n.bin(OpMul, ir.U32, it, n.u32(n.Group)), st.tid)
	fold := make([][]int, len(folds))
	for hl := 0; hl < st.G; hl++ {
		v := st.value(p, p.Expr, st.scope(p, hl, st.coords(p, hl, map[int]int{p.Part.Axis: j})))
		n.Stores = append(n.Stores, Store{Buf: st.plane, Ix: st.planeAt(hl, j), Val: v, At: c})
		for fi, f := range folds {
			t := elemOf(f.Type)
			acc := n.add(Node{Op: OpFold, At: c, Type: t, B: n.add(Node{Op: OpConst, Type: t, Imm: f.Init})})
			sc := st.scope(f, hl, st.coords(f, hl, map[int]int{f.Part.Axis: j}))
			at := sc.at
			sc.at = func(jj int, ax, ext []int) int {
				if f.In[jj] == p.Out {
					return v
				}
				return at(jj, ax, ext)
			}
			n.Nodes[acc].A = n.combine(f.Combine, t, acc, st.value(f, f.Expr, sc))
			fold[fi] = append(fold[fi], acc)
		}
	}
	st.inPlane[p.Out] = p
	if len(folds) == 0 {
		n.Stores = append(n.Stores, Store{Barrier: true, At: -1})
		return
	}
	for fi, f := range folds {
		t := elemOf(f.Type)
		x := n.buf(Buf{Kind: BufShared, Elem: t, Size: st.G * st.W * st.L})
		for hl := 0; hl < st.G; hl++ {
			v := fold[fi][hl]
			for m := 1; m < st.L; m *= 2 {
				v = n.bin(f.Combine, t, v, n.add(Node{Op: OpShuffle, Type: t, A: v, Imm: int64(m)}))
			}
			slot := n.bin(OpAdd, ir.U32, n.bin(OpMul, ir.U32, n.bin(OpAdd, ir.U32, n.u32(hl*st.W), st.warp), n.u32(st.L)), st.lane)
			n.Stores = append(n.Stores, Store{Buf: x, Ix: slot, Val: v, At: -1})
		}
		regs := make([]int, st.G)
		for hl := range regs {
			acc := n.add(Node{Op: OpConst, Type: t, Imm: f.Init})
			for w := 0; w < st.W; w++ {
				acc = n.bin(f.Combine, t, acc, n.load(x, n.u32((hl*st.W+w)*st.L)))
			}
			regs[hl] = acc
		}
		st.regs[f.Out] = regs
	}
	n.Stores = append(n.Stores, Store{Barrier: true, At: -1})
}

// warps is a reduction over the positions with an axis of its own: lanes
// over the axis, warps over the positions.
func (st *stager) warps(p *Prim) {
	n := st.n
	axis := -1
	for a := range p.Shape[:len(p.Shape)-1] {
		if a != 0 && a != 1 && a != p.Part.Split {
			axis = a
		}
	}
	D := p.Shape[axis]
	U := max(1, st.s.Unroll)
	if axis < 0 || D > st.L || st.G*st.W*D > st.G*st.J {
		st.ok = false
		return
	}
	span := st.span(p)
	c := st.trips(span, U*st.W)
	it := n.add(Node{Op: OpIter, At: c, Type: ir.U32})
	d := n.bin(OpMin, ir.U32, n.bin(OpAdd, ir.U32, st.lane, n.u32(0)), n.u32(D-1))
	t := elemOf(p.Type)
	acc := make([]int, st.G)
	last := make([]int, st.G)
	for hl := range acc {
		acc[hl] = n.add(Node{Op: OpFold, At: c, Type: t, B: n.add(Node{Op: OpConst, Type: t, Imm: p.Init})})
		last[hl] = acc[hl]
	}
	for u := 0; u < U; u++ {
		j := n.bin(OpAdd, ir.U32, n.bin(OpMul, ir.U32, n.bin(OpAdd, ir.U32, n.bin(OpMul, ir.U32, it, n.u32(U)), n.u32(u)), n.u32(st.W)), st.warp)
		for hl := 0; hl < st.G; hl++ {
			sc := st.scope(p, hl, st.coords(p, hl, map[int]int{axis: d, p.Part.Axis: j}))
			if st.s.Mask {
				at := sc.at
				sc.at = func(jj int, ax, ext []int) int {
					v := at(jj, ax, ext)
					if _, staged := st.inPlane[p.In[jj]]; staged {
						live := n.bin(OpLess, ir.Pred, j, span)
						v = n.add(Node{Op: OpSelect, Type: ir.F32, A: live, B: v, C: n.f32(0)})
					}
					return v
				}
			}
			run := n.bin(OpMul, ir.U32, n.bin(OpMul, ir.U32, it, n.u32(U)), n.u32(st.W))
			st.runs(p, &sc, run, -1)
			last[hl] = n.combine(p.Combine, t, last[hl], st.value(p, p.Expr, sc))
		}
	}
	for hl := range acc {
		n.Nodes[acc[hl]].A = last[hl]
	}
	n.Stores = append(n.Stores, Store{Run: true, At: c}, Store{Barrier: true, At: -1})
	for hl := 0; hl < st.G; hl++ {
		slot := n.bin(OpAdd, ir.U32, n.bin(OpMul, ir.U32, n.bin(OpAdd, ir.U32, n.bin(OpMul, ir.U32, st.warp, n.u32(st.G)), n.u32(hl)), n.u32(D)), d)
		n.Stores = append(n.Stores, Store{Buf: st.plane, Ix: slot, Val: acc[hl], At: -1})
	}
	n.Stores = append(n.Stores, Store{Barrier: true, At: -1})
	d2 := n.bin(OpMin, ir.U32, n.bin(OpAdd, ir.U32, st.tid, n.u32(0)), n.u32(D-1))
	sum := OpAdd
	if p.Combine != OpFma {
		sum = p.Combine
	}
	out := st.buffer(p.Out, BufOut)
	for hl := 0; hl < st.G; hl++ {
		v := n.add(Node{Op: OpConst, Type: t, Imm: p.Init})
		for w := 0; w < st.W; w++ {
			v = n.bin(sum, t, v, n.load(st.plane, n.bin(OpAdd, ir.U32, n.u32((w*st.G+hl)*D), d2)))
		}
		oc := make([]int, len(p.Shape)-1)
		oc[0], oc[1], oc[p.Part.Split], oc[axis] = st.r, st.head(hl), st.sp, d2
		ix := st.value(p, p.Index, st.scope(p, hl, oc))
		n.Stores = append(n.Stores, Store{Buf: out, Ix: ix, Val: v, At: -1})
	}
}

// each is a map of what the chain reduced, a thread an element.
func (st *stager) each(p *Prim) {
	n := st.n
	C := p.Shape[3]
	if n.Group%C != 0 {
		st.ok = false
		return
	}
	c := n.bin(OpRem, ir.U32, st.tid, n.u32(C))
	for hl := 0; hl < st.G; hl++ {
		vals := n.instance(*p, p.Expr, st.scope(p, hl, st.coords(p, hl, map[int]int{3: c})))
		for _, s := range p.Stores {
			n.Stores = append(n.Stores, Store{Buf: st.buffer(s.Out, BufOut), Ix: vals[s.Index], Val: vals[s.Value], At: -1})
		}
	}
}
