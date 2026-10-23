package jit

import "github.com/jitllm/jitllm/jit/ir"

// Bind deals a workgroup's phase loops to its threads by their structure:
//   - a loop over positions holding a reduction of an axis of its own: a
//     position a thread, lanes over consecutive positions and warps over
//     runs of them; where the axis is wider than a warp, main's slices:
//     the warps over slices of the axis (a warp per 32 of it, as many as
//     divide the warps) times runs of positions, each slice's partial a
//     row of its own that what reads the reduction sums;
//   - a loop over positions with nothing nested: a position a thread, its
//     sums combined over the lanes by a butterfly, each warp's to a
//     workgroup array a lane, and the warps' first lanes combined;
//   - a loop over an axis of its own holding a reduction over positions:
//     lanes over the axis, held to it, and warps over the positions, the
//     warps' sums through workgroup memory combined by a thread an element
//     of the axis;
//   - a loop of stores alone: lanes over its trips.
type Bind struct{}

func (Bind) Name() string { return "bind" }

func (p Bind) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if !k.Work || k.Group <= 0 || d.Lanes <= 0 || k.Group%d.Lanes != 0 {
			return nil, false, nil
		}
		m := k.clone()
		b := binder{k: m, L: d.Lanes, W: k.Group / d.Lanes}
		b.tid = m.add(Node{Op: OpThread, Type: ir.U32})
		b.lane = m.bin(OpRem, ir.U32, b.tid, m.u32(b.L))
		b.warp = m.bin(OpDiv, ir.U32, b.tid, m.u32(b.L))
		changed := false
		for c := range m.Loops {
			l := m.Loops[c]
			if l.Parent >= 0 || l.Bound != Unbound || len(m.itersOf(c)) == 0 && len(m.foldsOf(c)) == 0 {
				continue
			}
			inner := m.innerLoops(c)
			switch {
			case l.Range && len(inner) > 0 && !m.Loops[inner[0]].Range:
				if wd, ok := b.slices(c, inner); ok {
					b.runsSliced(c, inner[0], wd)
				} else {
					b.runs(c)
				}
			case l.Range && len(inner) == 0:
				b.spread(c)
			case !l.Range && len(inner) == 1 && m.Loops[inner[0]].Range && l.Trip <= b.L:
				b.warps(c, inner[0])
			case !l.Range && len(inner) == 1 && m.Loops[inner[0]].Range && l.Trip <= b.W*b.L &&
				m.Loops[inner[0]].Parent == c:
				b.warpsWide(c, inner[0])
			case !l.Range && len(inner) == 0 && len(m.foldsOf(c)) == 0 && m.Group%l.Trip == 0:
				b.lanes(c)
			default:
				continue
			}
			changed = true
		}
		if !changed {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

type binder struct {
	k               *Kernel
	L, W            int
	tid, lane, warp int
}

func (k *Kernel) itersOf(c int) []int {
	var out []int
	for i, nd := range k.Nodes {
		if nd.Op == OpIter && nd.At == c {
			out = append(out, i)
		}
	}
	return out
}

func (k *Kernel) innerLoops(c int) []int {
	var out []int
	for i, l := range k.Loops {
		if l.Parent == c {
			out = append(out, i)
		}
	}
	return out
}

// setIter makes every iteration node of loop c stand for node at, a fresh
// iteration it of c beneath it.
func (b *binder) setIter(c int, at func(it int) int) {
	k := b.k
	olds := k.itersOf(c)
	it := k.add(Node{Op: OpIter, At: c, Type: ir.U32})
	x := at(it)
	for _, o := range olds {
		k.redirect(o, x, x)
	}
}

// runs is lanes over consecutive positions, warps over runs of them.
func (b *binder) runs(c int) {
	k := b.k
	T := b.W * b.L
	b.setIter(c, func(it int) int {
		run := k.bin(OpMul, ir.U32, k.bin(OpAdd, ir.U32, k.bin(OpMul, ir.U32, it, k.u32(b.W)), b.warp), k.u32(b.L))
		return k.bin(OpAdd, ir.U32, run, b.lane)
	})
	k.Loops[c].Trip = (k.Loops[c].Trip + T - 1) / T
	k.Loops[c].Unit, k.Loops[c].Bound = T, BoundRuns
}

// slices is how many slices of the axis loop c's reduction runs over: a
// warp per lane-width of it, as many as divide the warps, where that is
// more than one, the slices whole, and loop c stores only rows of the
// workgroup's temporaries -- what a slice's partial can be a row of.
func (b *binder) slices(c int, inner []int) (int, bool) {
	k := b.k
	if len(inner) != 1 || len(k.foldsOf(inner[0])) == 0 || len(k.innerLoops(inner[0])) != 0 {
		return 0, false
	}
	D := k.Loops[inner[0]].Trip
	wd := min(b.W, (D+b.L-1)/b.L)
	for wd > 0 && b.W%wd != 0 {
		wd++
	}
	if wd <= 1 || D%wd != 0 {
		return 0, false
	}
	for _, s := range k.Stores {
		if s.At != c || s.Barrier || s.Run {
			continue
		}
		if bf := k.Bufs[s.Buf]; !bf.Operand.Temp || len(bf.Shape) != 4 || len(s.Ax) != 4 {
			return 0, false
		}
	}
	return wd, true
}

// runsSliced is runs with loop in, the reduction's axis, cut into wd
// slices: lanes over consecutive positions, the warps over the slices times
// runs of positions; each slice's partial stored as a row of its own, the
// rows a slice of each head's, and every read of the stored rows their sum.
func (b *binder) runsSliced(c, in, wd int) {
	k := b.k
	wk := b.W / wd
	ds := k.Loops[in].Trip / wd
	kg := k.bin(OpRem, ir.U32, b.warp, k.u32(wk))
	slice := k.bin(OpDiv, ir.U32, b.warp, k.u32(wk))
	T := wk * b.L
	b.setIter(c, func(it int) int {
		run := k.bin(OpMul, ir.U32, k.bin(OpAdd, ir.U32, k.bin(OpMul, ir.U32, it, k.u32(wk)), kg), k.u32(b.L))
		return k.bin(OpAdd, ir.U32, run, b.lane)
	})
	k.Loops[c].Trip = (k.Loops[c].Trip + T - 1) / T
	k.Loops[c].Unit, k.Loops[c].Bound = T, BoundRuns
	b.setIter(in, func(it int) int {
		return k.bin(OpAdd, ir.U32, k.bin(OpMul, ir.U32, slice, k.u32(ds)), it)
	})
	k.Loops[in].Trip = ds
	sliced := map[Operand]bool{}
	for i, s := range k.Stores {
		if s.At != c || s.Barrier || s.Run {
			continue
		}
		sliced[k.Bufs[s.Buf].Operand] = true
		ax := append([]int(nil), s.Ax...)
		ax[1] = k.bin(OpAdd, ir.U32, k.bin(OpMul, ir.U32, ax[1], k.u32(wd)), slice)
		k.Stores[i].Ax = ax
	}
	for i := range k.Bufs {
		if sliced[k.Bufs[i].Operand] {
			sh := append([]int(nil), k.Bufs[i].Shape...)
			sh[1] *= wd
			k.Bufs[i].Shape = sh
		}
	}
	var loads []int
	for i, nd := range k.Nodes {
		if nd.Op == OpLoad && sliced[k.Bufs[nd.Buf].Operand] {
			loads = append(loads, i)
		}
	}
	for _, i := range loads {
		nd := k.Nodes[i]
		sum := -1
		for s := range wd {
			ax := append([]int(nil), nd.Ax...)
			ax[1] = k.bin(OpAdd, ir.U32, k.bin(OpMul, ir.U32, nd.Ax[1], k.u32(wd)), k.u32(s))
			ld := k.add(Node{Op: OpLoad, Type: nd.Type, Buf: nd.Buf, Ax: ax})
			if sum < 0 {
				sum = ld
				continue
			}
			sum = k.bin(OpAdd, nd.Type, sum, ld)
		}
		k.redirect(i, sum, sum)
	}
}

// spread is a position a thread over the workgroup, and the loop's sums
// combined over the workgroup.
func (b *binder) spread(c int) {
	k := b.k
	T := b.W * b.L
	b.setIter(c, func(it int) int {
		return k.bin(OpAdd, ir.U32, k.bin(OpMul, ir.U32, it, k.u32(T)), b.tid)
	})
	k.Loops[c].Trip = (k.Loops[c].Trip + T - 1) / T
	k.Loops[c].Unit, k.Loops[c].Bound = T, BoundSpread
	folds := k.foldsOf(c)
	if len(folds) == 0 {
		return
	}
	start := len(k.Nodes)
	x := k.buf(Buf{Kind: BufShared, Elem: k.Nodes[folds[0]].Type, Size: len(folds) * b.W * b.L})
	at := k.lastAt(c)
	var add []Store
	totals := make([]int, len(folds))
	for i, f := range folds {
		t := k.Nodes[f].Type
		comb := k.Nodes[k.Nodes[f].A].Op
		v := f
		for m := 1; m < b.L; m *= 2 {
			v = k.bin(comb, t, v, k.add(Node{Op: OpShuffle, Type: t, A: v, Imm: int64(m)}))
		}
		slot := k.bin(OpAdd, ir.U32, k.bin(OpMul, ir.U32, k.bin(OpAdd, ir.U32, k.u32(i*b.W), b.warp), k.u32(b.L)), b.lane)
		add = append(add, Store{Buf: x, Ix: slot, Val: v, At: -1})
		acc := k.add(k.Nodes[k.Nodes[f].B])
		for w := 0; w < b.W; w++ {
			acc = k.bin(comb, t, acc, k.load(x, k.u32((i*b.W+w)*b.L)))
		}
		totals[i] = acc
	}
	add = append(add, Store{Barrier: true, At: -1})
	k.Stores = append(k.Stores[:at+1:at+1], append(add, k.Stores[at+1:]...)...)
	for i, f := range folds {
		k.redirectOutside(f, totals[i], c, start)
	}
}

// warps is lanes over the loop's axis and warps over the positions of the
// loop within it, the warps' sums combined through workgroup memory by a
// thread an element of the axis.
func (b *binder) warps(c, in int) {
	k := b.k
	D := k.Loops[c].Trip
	d := k.bin(OpMin, ir.U32, k.bin(OpAdd, ir.U32, b.lane, k.u32(0)), k.u32(D-1))
	d2 := k.bin(OpMin, ir.U32, k.bin(OpAdd, ir.U32, b.tid, k.u32(0)), k.u32(D-1))
	olds := k.itersOf(c)
	for _, o := range olds {
		k.redirect(o, d, d)
	}
	b.setIter(in, func(it int) int {
		return k.bin(OpAdd, ir.U32, k.bin(OpMul, ir.U32, it, k.u32(b.W)), b.warp)
	})
	k.Loops[in].Trip = (k.Loops[in].Trip + b.W - 1) / b.W
	k.Loops[in].Unit, k.Loops[in].Bound = b.W, BoundWarps
	k.Loops[in].Parent = -1
	k.Loops[c].Trip = 1
	folds := k.foldsOf(in)
	G := len(folds)
	plane := k.buf(Buf{Kind: BufShared, Elem: ir.F32, Size: b.W * G * D, Plane: true})
	var add []Store
	add = append(add, Store{Run: true, At: in}, Store{Barrier: true, At: -1})
	totals := make([]int, G)
	for i, f := range folds {
		slot := k.bin(OpAdd, ir.U32, k.bin(OpMul, ir.U32, k.bin(OpAdd, ir.U32, k.bin(OpMul, ir.U32, b.warp, k.u32(G)), k.u32(i)), k.u32(D)), d)
		add = append(add, Store{Buf: plane, Ix: slot, Val: f, At: -1})
		t := k.Nodes[f].Type
		acc := k.add(k.Nodes[k.Nodes[f].B])
		for w := 0; w < b.W; w++ {
			acc = k.bin(OpAdd, t, acc, k.load(plane, k.bin(OpAdd, ir.U32, k.u32((w*G+i)*D), d2)))
		}
		totals[i] = acc
	}
	add = append(add, Store{Barrier: true, At: -1})
	var out []Store
	placed := false
	for _, s := range k.Stores {
		if s.At == c {
			if !placed {
				out = append(out, add...)
				placed = true
			}
			if s.Run {
				continue
			}
			s.At = -1
			r := replica{k: k, sub: map[int]int{d: d2}, memo: map[int]int{}}
			for i, f := range folds {
				r.sub[f] = totals[i]
			}
			s.Val = r.copy(s.Val)
			s.Ix = r.copy(s.Ix)
		}
		out = append(out, s)
	}
	k.Stores = out
}

// warpsWide is warps for an axis wider than a warp: the axis a loop of
// ceil(D/L) trips, a lane an element of it a trip, the loop within it over
// warps inside it; each trip's sums to workgroup memory, combined after the
// loop by a thread an element of the axis, which the workgroup's threads
// cover.
func (b *binder) warpsWide(c, in int) {
	k := b.k
	D := k.Loops[c].Trip
	var d int
	b.setIter(c, func(it int) int {
		d = k.bin(OpMin, ir.U32, k.bin(OpAdd, ir.U32, k.bin(OpMul, ir.U32, it, k.u32(b.L)), b.lane), k.u32(D-1))
		return d
	})
	k.Loops[c].Trip = (D + b.L - 1) / b.L
	b.setIter(in, func(it int) int {
		return k.bin(OpAdd, ir.U32, k.bin(OpMul, ir.U32, it, k.u32(b.W)), b.warp)
	})
	k.Loops[in].Trip = (k.Loops[in].Trip + b.W - 1) / b.W
	k.Loops[in].Unit, k.Loops[in].Bound = b.W, BoundWarps
	d2 := k.bin(OpMin, ir.U32, b.tid, k.u32(D-1))
	folds := k.foldsOf(in)
	G := len(folds)
	// Each trip's sums are written while later trips still read what the
	// loop reads: they take workgroup memory of their own, not the plane
	// Stage shares with the temporaries.
	plane := k.buf(Buf{Kind: BufShared, Elem: ir.F32, Size: b.W * G * D})
	var add []Store
	totals := make([]int, G)
	for i, f := range folds {
		slot := k.bin(OpAdd, ir.U32, k.bin(OpMul, ir.U32, k.bin(OpAdd, ir.U32, k.bin(OpMul, ir.U32, b.warp, k.u32(G)), k.u32(i)), k.u32(D)), d)
		add = append(add, Store{Buf: plane, Ix: slot, Val: f, At: c})
		t := k.Nodes[f].Type
		acc := k.add(k.Nodes[k.Nodes[f].B])
		for w := 0; w < b.W; w++ {
			acc = k.bin(OpAdd, t, acc, k.load(plane, k.bin(OpAdd, ir.U32, k.u32((w*G+i)*D), d2)))
		}
		totals[i] = acc
	}
	add = append(add, Store{Barrier: true, At: -1})
	var out []Store
	placed := false
	for _, s := range k.Stores {
		if s.At == c {
			if !placed {
				out = append(out, add...)
				placed = true
			}
			if s.Run {
				continue
			}
			s.At = -1
			r := replica{k: k, sub: map[int]int{d: d2}, memo: map[int]int{}}
			for i, f := range folds {
				r.sub[f] = totals[i]
			}
			s.Val = r.copy(s.Val)
			s.Ix = r.copy(s.Ix)
		}
		out = append(out, s)
	}
	k.Stores = out
}

// lanes is the loop's trips over the workgroup's threads.
func (b *binder) lanes(c int) {
	k := b.k
	C := k.Loops[c].Trip
	x := k.bin(OpRem, ir.U32, b.tid, k.u32(C))
	for _, o := range k.itersOf(c) {
		k.redirect(o, x, x)
	}
	for i := range k.Stores {
		if k.Stores[i].At == c {
			k.Stores[i].At = -1
		}
	}
	var keep []Store
	for _, s := range k.Stores {
		if !(s.Run && s.At == -1) {
			keep = append(keep, s)
		}
	}
	k.Stores = keep
}

// lastAt is the index of the last store loop c runs.
func (k *Kernel) lastAt(c int) int {
	at := -1
	for i, s := range k.Stores {
		if s.At == c {
			at = i
		}
	}
	return at
}

// redirectOutside points every reader of fold f outside loop c, made before
// node from, at node to.
func (k *Kernel) redirectOutside(f, to, c, from int) {
	inside := map[int]bool{}
	var walk func(i int)
	walk = func(i int) {
		if inside[i] {
			return
		}
		inside[i] = true
		nd := k.Nodes[i]
		if nd.Op == OpFold && nd.At != c {
			return
		}
		for _, a := range k.Operands(&nd) {
			walk(*a)
		}
	}
	for _, g := range k.foldsOf(c) {
		walk(k.Nodes[g].A)
	}
	for _, s := range k.Stores {
		if s.At == c && !s.Barrier && !s.Run {
			for _, a := range k.storeArgs(&s) {
				walk(*a)
			}
		}
	}
	for i := range k.Nodes {
		if inside[i] || i >= from {
			continue
		}
		for _, a := range k.Operands(&k.Nodes[i]) {
			if *a == f {
				*a = to
			}
		}
	}
	for i := range k.Stores {
		s := &k.Stores[i]
		if s.Barrier || s.Run || s.At == c {
			continue
		}
		for _, a := range k.storeArgs(s) {
			if *a == f {
				*a = to
			}
		}
	}
}
