package compose

import (
	"math"

	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// Flash is a paged decode attention run whole in one workgroup a (row, KV
// head's query-head group, key split): the scores of a tile of keys a lane
// over a slice of the dimensions a warp, the row's maximum and sum by
// butterflies and workgroup arrays, the weights, the values a key a warp
// and a dimension a lane, and the splits' partials or the normalised row.
type Flash struct {
	Heads, KVHeads, Dim, Rows, Page int
	// Splits are the key partitions, Group the query heads a workgroup
	// serves (0 all of a KV head's), Warps its width (0 one a 32
	// dimensions, four to eight).
	Splits, Group, Warps int
	Scale, Softcap       float32
	// Sink is a per-head logit with no value; KImm addresses a tile's K as
	// one base and constant offsets; VecV loads V as 2- or 4-wide vectors
	// where a row divides into them across a warp.
	Sink, KImm, VecV bool
}

func (s Flash) warps() int {
	if s.Warps > 0 {
		return s.Warps
	}
	return min(8, max(4, (s.Dim+31)/32))
}

// Width is the workgroup's threads.
func (s Flash) Width() int { return 32 * s.warps() }

// geom is the warps (w), their slices of the dimensions (wd, ds each) and
// their groups of key tiles (wk).
func (s Flash) geom() (w, wd, wk, ds int) {
	w = s.warps()
	wd = min(w, (s.Dim+31)/32)
	for w%wd != 0 {
		wd++
	}
	return w, wd, w / wd, (s.Dim + wd - 1) / wd
}

// group is the query heads a workgroup serves: a KV head's, or a divisor
// of them whose partials and queries fit 32 KiB and 96 registers.
func (s Flash) group() int {
	gqa := max(1, s.Heads/max(1, s.KVHeads))
	if s.Group > 0 && s.Group < gqa {
		gqa = s.Group
	}
	w, _, _, ds := s.geom()
	g := gqa
	for g > 1 && (gqa%g != 0 || g*(w*s.Dim+64*w) > (32<<10)/4 || g*ds > 96) {
		g--
	}
	return g
}

// Chunk is the keys a workgroup's score plane holds.
func (s Flash) Chunk() int {
	gqa, T := s.group(), s.Width()
	w, wd, _, _ := s.geom()
	c := min(2048, ((32<<10)/4-2*gqa*w*32)/(gqa*wd)) / T * T
	return max(c, T, (w*s.Dim/wd+T-1)/T*T)
}

// Groups is the workgroups of a launch.
func (s Flash) Groups() int { return max(1, s.Rows) * s.Heads / s.group() * max(1, s.Splits) }

// vecV is the V elements a lane loads at once.
func (s Flash) vecV() int {
	if !s.VecV {
		return 1
	}
	for _, w := range []int{4, 2} {
		if s.Dim%(32*w) == 0 && (s.KVHeads*s.Dim)%w == 0 {
			return w
		}
	}
	return 1
}

// PagedFlash is s over the group (Width() elements, Groups() rows, a row a
// workgroup): q the rows' query heads, k and v the paged pools ([dim][page]
// and [page][dim] a page), tab the page ids, desc the row descriptors, out
// the normalised rows or the splits' partials (the merge's layout), sink
// the per-head logits (-1 for none).
func (g *Row) PagedFlash(s Flash, q, k, v, tab, desc, out, sink int) {
	u, ub := g.UConst, g.UBin
	add := func(a, b int) int { return ub(OpAdd, a, b) }
	mul := func(a, b int) int { return ub(OpMul, a, b) }
	div := func(a, b int) int { return ub(OpDiv, a, b) }
	rem := func(a, b int) int { return ub(OpRem, a, b) }
	umin := func(a, b int) int { return ub(OpMin, a, b) }
	umax := func(a, b int) int { return ub(OpMax, a, b) }
	T := s.Width()
	W, Wd, Wk, ds := s.geom()
	gqa, splits, C, P := s.group(), max(1, s.Splits), s.Chunk(), s.Page
	slices := s.Heads / s.KVHeads / gqa
	minus, zero := g.Const(-math.MaxFloat32), g.Const(0)
	sS := g.Shared(gqa*Wd*C, false)
	sMax := g.Shared(gqa*W*32, false)
	sSum := g.Shared(gqa*W*32, false)
	tid := g.Column()
	wl, warp := rem(tid, u(32)), div(tid, u(32))
	grp := g.RowIdx()
	row, qrow := u(0), -1
	if s.Rows > 1 {
		per := s.Heads / gqa * splits
		row = umin(div(grp, u(per)), u(s.Rows-1))
		grp = rem(grp, u(per))
		qrow = mul(row, u(s.Heads))
	}
	h0 := mul(umin(div(grp, u(splits)), u(s.Heads/gqa-1)), u(gqa))
	kvh := div(h0, u(gqa*slices))
	sp := rem(grp, u(splits))
	descAt := func(w int) int { return g.LoadAt(desc, add(mul(row, u(kernels.PRowWords)), u(w))) }
	tabOff, keyStart, count := descAt(kernels.PRowTab), descAt(kernels.PRowStart), descAt(kernels.PRowEnd)
	start := mul(div(keyStart, u(32)), u(32))
	span := g.Select(g.Less(keyStart, count), ub(OpSub, count, start), u(0))
	per := div(add(span, u(splits-1)), u(splits))
	per = mul(div(add(per, u(31)), u(32)), u(32))
	st := umin(count, add(start, mul(sp, per)))
	en := umin(count, add(st, per))
	keys := ub(OpSub, en, st)
	trips := div(add(keys, u(T-1)), u(T))
	lastKey := ub(OpSub, umax(count, u(1)), u(1))
	kvbase := mul(kvh, u(s.Dim))
	at := func(gi int, slot int) int { return add(slot, u(gi*Wd*C)) }
	kvRow := s.KVHeads * s.Dim
	lastTile := mul(div(lastKey, u(32)), u(32))
	pageOf := func(t int) int { return g.LoadAt(tab, add(tabOff, div(t, u(P)))) }
	butterfly := func(x int, op Op) int {
		for m := 1; m < 32; m *= 2 {
			x = g.Bin(op, x, g.Shuffle(x, m))
		}
		return x
	}

	// 1. Partial scores: a 32-key tile a warp's lanes, the warps over the
	// dimensions' slices and groups of tiles, the query slice in registers.
	wd, wk := rem(warp, u(Wd)), div(warp, u(Wd))
	d0 := mul(wd, u(ds))
	qr := make([][]int, gqa)
	for gi := range qr {
		qr[gi] = make([]int, ds)
		for i := range qr[gi] {
			di := add(d0, u(i))
			qh := add(h0, u(gi))
			if qrow >= 0 {
				qh = add(qh, qrow)
			}
			qv := g.LoadAt(q, add(mul(qh, u(s.Dim)), umin(di, u(s.Dim-1))))
			if ds*Wd != s.Dim {
				qv = g.Select(g.Less(di, u(s.Dim)), qv, zero)
			}
			qr[gi][i] = qv
		}
	}
	tiles := div(add(keys, u(32*Wk-1)), u(32*Wk))
	lastDim := u(s.Dim - 1)
	tileAt := func(j int) int { return umin(add(st, mul(add(mul(j, u(Wk)), wk), u(32))), lastTile) }
	const unroll = 4
	block := 32%(unroll*W) == 0
	blockAt := func(it int) int { return umin(add(st, mul(it, u(unroll*W))), lastTile) }
	c1 := g.LoopN(tiles)
	j := g.Iter(c1)
	pid := g.Fold(c1, pageOf(tileAt(u(0))))
	g.SetFold(pid, pageOf(tileAt(add(j, u(1)))))
	slot := add(mul(add(mul(j, u(Wk)), wk), u(32)), wl)
	kp := add(add(mul(pid, u(P*kvRow)), rem(tileAt(j), u(P))), wl)
	kd := make([]int, ds)
	if s.KImm && ds*Wd == s.Dim {
		kb := add(kp, mul(add(kvbase, d0), u(P)))
		for i := range kd {
			kd[i] = g.LoadAt(k, add(kb, u(i*P)))
		}
	} else {
		for i := range kd {
			kd[i] = g.LoadAt(k, add(kp, mul(add(kvbase, umin(add(d0, u(i)), lastDim)), u(P))))
		}
	}
	cslot := umin(slot, u(C-1))
	for gi := range qr {
		var dot [4]int
		for c := range dot {
			dot[c] = zero
		}
		for i := range kd {
			dot[i%4] = g.fma(qr[gi][i], kd[i], dot[i%4])
		}
		sum := g.Bin(OpAdd, g.Bin(OpAdd, dot[0], dot[1]), g.Bin(OpAdd, dot[2], dot[3]))
		g.StoreAt(sS, sum, add(at(gi, cslot), mul(wd, u(C))))
	}

	// 2. The scores: each thread folds its keys' partials, scales, caps and
	// masks them over the partials' first row.
	scores := g.SharedOver(sS)
	c2 := g.LoopN(trips)
	j1 := g.Iter(c2)
	mx := make([]int, gqa)
	slot1 := add(mul(j1, u(T)), tid)
	valid := g.Less(ub(OpSub, add(st, slot1), keyStart), ub(OpSub, en, keyStart))
	cslot1 := umin(slot1, u(C-1))
	for gi := range mx {
		mx[gi] = g.Fold(c2, minus)
		dotv := g.LoadAt(sS, at(gi, cslot1))
		for w := 1; w < Wd; w++ {
			dotv = g.Bin(OpAdd, dotv, g.LoadAt(sS, add(at(gi, cslot1), u(w*C))))
		}
		score := g.Bin(OpMul, dotv, g.Const(s.Scale))
		if s.Softcap > 0 {
			e := g.exp(g.Bin(OpMul, g.Const(2/s.Softcap), score))
			score = g.Bin(OpMul, g.Const(s.Softcap), g.Bin(OpSub, g.Const(1), g.Bin(OpDiv, g.Const(2), g.Bin(OpAdd, e, g.Const(1)))))
		}
		score = g.Select(valid, score, minus)
		g.StoreAt(scores, score, at(gi, cslot1))
		g.SetFold(mx[gi], g.Bin(OpMax, mx[gi], score))
	}

	// 3. The maximum: a butterfly a warp, then the warps' in turn.
	m := make([]int, gqa)
	for gi := range mx {
		g.StoreAt(sMax, butterfly(mx[gi], OpMax), add(mul(add(u(gi*W), warp), u(32)), wl))
	}
	for gi := range m {
		m[gi] = minus
		for w := 0; w < W; w++ {
			m[gi] = g.Bin(OpMax, m[gi], g.LoadAt(sMax, u((gi*W+w)*32)))
		}
	}

	// 4. The weights over the same keys, and their sums.
	weights := g.SharedOver(scores)
	c4 := g.LoopN(trips)
	j2 := g.Iter(c4)
	sum := make([]int, gqa)
	slot2 := add(mul(j2, u(T)), tid)
	valid2 := g.Less(add(st, slot2), en)
	cslot2 := umin(slot2, u(C-1))
	for gi := range sum {
		sum[gi] = g.Fold(c4, zero)
		a := add(cslot2, u(gi*Wd*C))
		w := g.Select(valid2, g.exp(g.Bin(OpSub, g.LoadAt(scores, a), m[gi])), zero)
		g.StoreAt(weights, w, a)
		g.SetFold(sum[gi], g.Bin(OpAdd, sum[gi], w))
	}
	l := make([]int, gqa)
	for gi := range sum {
		g.StoreAt(sSum, butterfly(sum[gi], OpAdd), add(mul(add(u(gi*W), warp), u(32)), wl))
	}
	for gi := range l {
		l[gi] = zero
		for w := 0; w < W; w++ {
			l[gi] = g.Bin(OpAdd, l[gi], g.LoadAt(sSum, u((gi*W+w)*32)))
		}
	}

	// 5. The values: a key a warp every W, the lanes over the dimensions,
	// unroll keys an iteration; the warps' sums meet in the score plane.
	nd := (s.Dim + 31) / 32
	vw := s.vecV()
	dims := make([]int, nd)
	for r := range dims {
		if vw > 1 {
			dims[r] = add(mul(add(wl, u(r/vw*32)), u(vw)), u(r%vw))
		} else {
			dims[r] = umin(add(wl, u(r*32)), u(s.Dim-1))
		}
	}
	pairs := div(add(keys, u(unroll*W-1)), u(unroll*W))
	lastIn := ub(OpSub, umax(en, u(1)), u(1))
	c5 := g.LoopN(pairs)
	it := g.Iter(c5)
	vpid := -1
	if block {
		vpid = g.Fold(c5, pageOf(blockAt(u(0))))
		g.SetFold(vpid, pageOf(blockAt(add(it, u(1)))))
	}
	acc := make([][]int, gqa)
	for gi := range acc {
		acc[gi] = make([]int, nd)
		for r := range acc[gi] {
			acc[gi][r] = g.Fold(c5, zero)
		}
	}
	next := make([][]int, gqa)
	for gi := range next {
		next[gi] = append([]int(nil), acc[gi]...)
	}
	type keyLoads struct{ ws, vv []int }
	loads := make([]keyLoads, unroll)
	for half := range loads {
		kk := add(mul(add(mul(it, u(unroll)), u(half)), u(W)), warp)
		live := g.Less(kk, keys)
		t := umin(umax(add(st, kk), keyStart), lastIn)
		vp := vpid
		if !block {
			vp = pageOf(t)
		}
		vb := add(mul(add(mul(vp, u(P)), rem(t, u(P))), u(kvRow)), kvbase)
		if vw > 1 {
			for r := 0; r < nd; r += vw {
				va := add(vb, mul(add(wl, u(r/vw*32)), u(vw)))
				for c := 0; c < vw; c++ {
					loads[half].vv = append(loads[half].vv, g.LoadVec(v, va, vw, c))
				}
			}
		} else {
			for _, dim := range dims {
				loads[half].vv = append(loads[half].vv, g.LoadAt(v, add(vb, dim)))
			}
		}
		wslot := umin(kk, u(C-1))
		for gi := 0; gi < gqa; gi++ {
			loads[half].ws = append(loads[half].ws, g.Select(live, g.LoadAt(weights, add(wslot, u(gi*Wd*C))), zero))
		}
	}
	for _, ld := range loads {
		for r := range dims {
			for gi := range next {
				next[gi][r] = g.fma(ld.ws[gi], ld.vv[r], next[gi][r])
			}
		}
	}
	for gi := range acc {
		for r := range acc[gi] {
			g.SetFold(acc[gi][r], next[gi][r])
		}
	}
	sums := g.SharedAfter(weights)
	for gi := range acc {
		for r, dim := range dims {
			g.StoreAt(sums, acc[gi][r], add(mul(add(mul(warp, u(gqa)), u(gi)), u(s.Dim)), dim))
		}
	}
	nt := (s.Dim + T - 1) / T
	odims := make([]int, nt)
	for r := range odims {
		odims[r] = umin(add(tid, u(r*T)), u(s.Dim-1))
	}
	fin := make([][]int, gqa)
	for gi := range fin {
		fin[gi] = make([]int, nt)
		for r, dim := range odims {
			x := zero
			for w := 0; w < W; w++ {
				x = g.Bin(OpAdd, x, g.LoadAt(sums, add(u((w*gqa+gi)*s.Dim), dim)))
			}
			fin[gi][r] = x
		}
	}

	// 6. Out: the merge's partials, or the row normalised (by the sink too).
	for gi := 0; gi < gqa; gi++ {
		head := add(h0, u(gi))
		if qrow >= 0 {
			head = add(head, qrow)
		}
		if splits > 1 {
			padded := (s.Dim + 31) / 32 * 32
			base := mul(add(mul(head, u(splits)), sp), u(padded+64))
			for r, dim := range odims {
				g.StoreAt(out, fin[gi][r], add(base, dim))
			}
			g.StoreAt(out, m[gi], add(add(base, wl), u(padded)))
			g.StoreAt(out, l[gi], add(add(base, wl), u(padded+32)))
			continue
		}
		factor, denom := g.Const(1), l[gi]
		if s.Sink && sink >= 0 {
			sh := head
			if qrow >= 0 {
				sh = add(h0, u(gi))
			}
			sv := g.LoadAt(sink, sh)
			final := g.Bin(OpMax, m[gi], sv)
			factor = g.exp(g.Bin(OpSub, m[gi], final))
			denom = g.fma(l[gi], factor, g.exp(g.Bin(OpSub, sv, final)))
		}
		factor = g.Bin(OpDiv, factor, g.Bin(OpMax, denom, g.Const(1e-30)))
		for r, dim := range odims {
			g.StoreAt(out, g.Bin(OpMul, fin[gi][r], factor), add(mul(head, u(s.Dim)), dim))
		}
	}
}
