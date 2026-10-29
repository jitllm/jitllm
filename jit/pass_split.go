package jit

import (
	"cmp"
	"slices"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/ir"
)

// Split cuts a sum over a loop into contiguous segments, each element's
// segments threads of their own writing partial planes, and a combine that
// sums the planes. The segments double while they stay under Max, the
// threads under twice Slots (the device's where Slots is 0), and the trips
// divide into Block-wide blocks a segment. Where Parts is set the segments
// are instead the most, at most Parts and a power of two, that divide the
// trips and keep the elements' threads within the device's slots: elements
// that fill the device alone -- a vocabulary's rows -- are not split. Where
// Group is set and the elements allow, a row's segments are
// instead one workgroup's (main's GroupSplit), the most segments that
// divide the trips, any count: 32 rows by the segments,
// 128 to 1024 threads, each segment's sum through workgroup memory and the
// row written whole, with no partial planes and no combine. Where Matrix is
// set it cuts only a sum over a weight matrix -- a projection's. A form of a
// capacity whose threads each sum a tile of rows is always cut within a
// workgroup, by the segments one unit of its rows takes (groupSplitRows).
type Split struct {
	Slots, Max, Block int
	Parts             int
	Group, Matrix     bool
	// Fill is the share of the device's resident threads a workgroup split's threads stay under.
	Fill float64
	// Width, where set, forces a workgroup split's threads.
	Width int
	// Few, where set, forms beside a tiled kernel's split (groupSplitRows),
	// or beside the tiled kernel where its rows are wide enough to take no
	// split, its untiled form, each row's sums cut and combined in the tiled
	// one's order, launched in its place at a call of at most Few rows: a row
	// computed alone rather than with the tile's copies of it. Only on a
	// device whose float arithmetic rounds every operation alone
	// (Device.ExactArith), where the two kernels write a row's same bits.
	Few   int
	Kinds []PrimKind
	// Drop adds every float sum's segments but the last: a gate's violation.
	Drop bool
}

func (Split) Name() string { return "split" }

func (s Split) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	widths := s.siblingWidths(ks, d)
	return eachKernel(s.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) { return s.apply(k, d, widths[k]) })
}

func (s Split) groupThreads(d Device) int {
	if s.Fill > 0 && d.Slots > 0 {
		return int(s.Fill * float64(d.Slots))
	}
	slots := s.Slots
	if slots == 0 {
		slots = d.Slots
	}
	return 4 * slots
}

func (s Split) groupSegments(n *Kernel, d Device) (S, G int, ok bool) {
	if !s.Group || s.Parts > 0 || n.Few > 0 || n.Many > 0 || n.Work || n.Tile > 1 && n.Capacity > 0 || !covers(s.Kinds, n.Kind) || s.Matrix && !readsMatrix(n) {
		return 0, 0, false
	}
	c, _, ok := sums(n)
	if !ok {
		return 0, 0, false
	}
	T, E, limit := n.Loops[c].Trip, n.cutElements(), s.groupThreads(d)
	for g := s.Max; g > 1; g-- {
		if G := groupWidth(g); T%g == 0 && E*g < limit && groupFits(n.Elements(), T, g, G) {
			return g, G, true
		}
	}
	return 0, 0, false
}

func groupWidth(S int) int { return min(max(32*S, 128), 1024) }

func groupFits(E, T, S, G int) bool { return G%S == 0 && E%(G/S) == 0 && T%S == 0 }

// siblingWidths gives kernels reading one activation the narrowest of their
// workgroups, so Siblings runs them as one launch; a segment keeps eight rows,
// a 32-byte sector.
func (s Split) siblingWidths(ks []*Kernel, d Device) map[*Kernel]int {
	if s.Width > 0 {
		return nil
	}
	var out map[*Kernel]int
	for i := 0; i < len(ks); {
		S0, G, ok := s.groupSegments(ks[i], d)
		if !ok {
			i++
			continue
		}
		segs := []int{S0}
		j := i + 1
		for ; j < len(ks); j++ {
			Sj, Gj, ok := s.groupSegments(ks[j], d)
			if !ok || !sharesActivation(ks[i], ks[j]) || writesInto(ks[i], ks[j]) || writesInto(ks[j], ks[i]) {
				break
			}
			segs, G = append(segs, Sj), min(G, Gj)
		}
		fits := j-i > 1
		for x, S := range segs {
			k := ks[i+x]
			c, _, _ := sums(k)
			fits = fits && G >= 8*S && groupFits(k.Elements(), k.Loops[c].Trip, S, G)
		}
		if fits {
			if out == nil {
				out = map[*Kernel]int{}
			}
			for _, k := range ks[i:j] {
				out[k] = G
			}
		}
		i = j
	}
	return out
}

func (s Split) apply(n *Kernel, d Device, width int) ([]*Kernel, bool, error) {
	if n.Few > 0 || n.Many > 0 {
		// A pair's two forms are cut alike or not at all.
		return nil, false, nil
	}
	if s.Parts == 0 && n.Tile > 1 && n.Capacity > 0 && !n.Work && covers(s.Kinds, n.Kind) && (!s.Matrix || readsMatrix(n)) {
		if gs, ok := s.groupSplitRows(n, d); ok {
			return gs, true, nil
		}
		return nil, false, nil
	}
	c, f, term, st, ok := splittable(n)
	if !ok {
		c, folds, ok := sums(n)
		if !ok || n.Work || !s.Group || s.Parts > 0 || !covers(s.Kinds, n.Kind) || s.Matrix && !readsMatrix(n) {
			return nil, false, nil
		}
		S, G, ok := s.groupSegments(n, d)
		switch {
		case !ok:
			T, E, slots := n.Loops[c].Trip, n.cutElements(), cmp.Or(s.Slots, d.Slots)
			for S = 1; S < s.Max && E*S < 2*slots && T%(s.Block*S*2) == 0; S *= 2 {
			}
			G = groupWidth(S)
		case s.Width > 0:
			G = s.Width
		case width > 0:
			G = width
		}
		if S < 2 {
			return nil, false, nil
		}
		if k, ok := groupSplitOf(n, c, folds, -1, 0, S, G, s.Drop); ok {
			return []*Kernel{k}, true, nil
		}
		return nil, false, nil
	}
	if n.Work || !covers(s.Kinds, n.Kind) || s.Matrix && !readsMatrix(n) {
		return nil, false, nil
	}
	T, E := n.Loops[c].Trip, n.cutElements()
	slots := s.Slots
	if slots == 0 {
		slots = d.Slots
	}
	S := 1
	if s.Parts > 0 {
		S = s.Parts
		for S > 1 && (T%S != 0 || E*S > slots) {
			S /= 2
		}
	}
	if g, G, ok := s.groupSegments(n, d); ok {
		switch {
		case s.Width > 0:
			G = s.Width
		case width > 0:
			G = width
		}
		if k, ok := groupSplit(n, c, f, term, st, g, G, s.Drop); ok {
			return []*Kernel{k}, true, nil
		}
	}
	for s.Parts == 0 && S < s.Max && E*S < 2*slots && T%(s.Block*S*2) == 0 {
		S *= 2
	}
	if S == 1 {
		return nil, false, nil
	}
	if s.Group {
		if g, ok := groupSplit(n, c, f, term, st, S, groupWidth(S), s.Drop); ok {
			return []*Kernel{g}, true, nil
		}
	}
	out := n.Bufs[n.Stores[st].Buf].Operand
	planes := Operand{Kind: OperandScratch, Name: out.Name + ".planes", Of: out.Of}
	at := segmentAxis(n)
	shape := slices.Insert(slices.Clone(n.Space), at, S)
	p := n.clone()
	p.Name = n.Name + ".part"
	p.Space = shape
	for i := range p.Nodes {
		if p.Nodes[i].Op == OpCoord && p.Nodes[i].A >= at {
			p.Nodes[i].A++
		}
	}
	seg := p.add(Node{Op: OpCoord, A: at, Type: ir.U32})
	p.Loops[c].Trip = T / S
	if it := p.iterOf(c); it >= 0 {
		k := p.bin(OpAdd, ir.U32, p.bin(OpMul, ir.U32, seg, p.u32(T/S)), it)
		p.redirect(it, k, k)
	}
	// The planes hold the sum's own type: an integer dot's sum is summed
	// as integers.
	t := n.Nodes[f].Type
	zero := func(k *Kernel) int { return k.add(Node{Op: OpConst, Type: t}) }
	b := p.buf(Buf{Kind: BufOut, Elem: t, Operand: planes, Shape: shape})
	val := f
	if term >= 0 {
		first := p.add(Node{Op: OpLess, Type: ir.Pred, A: seg, B: p.u32(1)})
		val = p.bin(OpAdd, t, f, p.add(Node{Op: OpSelect, Type: t, A: first, B: term, C: zero(p)}))
	}
	ax := slices.Insert(slices.Clone(p.Stores[st].Ax), at, seg)
	p.Stores[st] = Store{Buf: b, Val: val, At: -1, Ax: ax}

	m := &Kernel{Name: n.Name + ".combine", Kind: n.Kind, Space: append([]int(nil), n.Space...), Group: n.Group, Part: n.Part, Apart: n.Apart, Of: n.Of,
		Capacity: n.Capacity}
	coords := make([]int, len(m.Space))
	for i := range coords {
		coords[i] = m.add(Node{Op: OpCoord, A: i, Type: ir.U32})
	}
	in := m.buf(Buf{Kind: BufIn, Elem: t, Operand: planes, Shape: shape})
	ob := m.buf(n.Bufs[n.Stores[st].Buf])
	l := m.loop(kept(S, t, s.Drop), -1)
	it := m.add(Node{Op: OpIter, At: l, Type: ir.U32})
	acc := m.add(Node{Op: OpFold, At: l, Type: t, B: zero(m)})
	x := m.add(Node{Op: OpLoad, Type: t, Buf: in, Ax: slices.Insert(slices.Clone(coords), at, it)})
	m.Nodes[acc].A = m.bin(OpAdd, t, acc, x)
	m.Stores = append(m.Stores, Store{Buf: ob, Val: acc, At: -1, Ax: coords})
	return []*Kernel{p, m}, true, nil
}

// kept is how many of S segments a split of a sum of type t adds: a float
// sum drops its last where drop, an integer one (a count, an offset) none.
func kept(S int, t ir.Type, drop bool) int {
	if drop && t == ir.F32 {
		return S - 1
	}
	return S
}

// segmentAxis is the axis a cut's segments take in a kernel's space: the
// front, or behind the leading axis of a form of a capacity, which stays the
// step's rows for the passes after (CallRows holds a call to the rows of the
// leading axis).
func segmentAxis(n *Kernel) int {
	if n.Capacity > 0 && len(n.Space) > 0 {
		return 1
	}
	return 0
}

// cutElements is the elements a split of n's sums is decided by: its own,
// or where its elements are rows a selection picks (Kernel.Pick), the form's
// over every row, so a picked row's sum is cut as the whole form cuts it and
// is that row's value to the bit.
func (n *Kernel) cutElements() int {
	e := n.Elements()
	if p := n.Pick; p != nil && p.Rows > 0 && e%p.Rows == 0 {
		return e / p.Rows * p.Of
	}
	return e
}

// groupSplit is n's sum over loop c in S segments within one workgroup:
// G = 32*S threads (128 to 1024), rpg = G/S elements a workgroup, thread t
// segment t/rpg of element t%rpg, or where a trip reads one element of a
// verbatim weight's row segment t%S of element t/S, its trips s, s+S, ...,
// so adjacent threads read adjacent words; each segment's partial to workgroup
// memory, a barrier, and every thread of an element its segments' sum
// stored at the element. False where the elements do not fill whole
// workgroups.
func groupSplit(n *Kernel, c, f, term, st, S, G int, drop bool) (*Kernel, bool) {
	return groupSplitOf(n, c, []int{f}, term, st, S, G, drop)
}

// groupSplitOf is groupSplit of a value made of several sums over loop c.
func groupSplitOf(n *Kernel, c int, folds []int, term, st, S, G int, drop bool) (*Kernel, bool) {
	E, T := n.Elements(), n.Loops[c].Trip
	if !groupFits(E, T, S, G) {
		return nil, false
	}
	rpg := G / S
	g := n.clone()
	g.Name = n.Name + ".group"
	g.Work, g.Group, g.Space = true, G, []int{E / rpg}
	U := ir.U32
	tid := g.add(Node{Op: OpThread, Type: U})
	seg := g.bin(OpDiv, U, tid, g.u32(rpg))
	rl := g.bin(OpRem, U, tid, g.u32(rpg))
	slot := func(i int) int { return g.bin(OpAdd, U, rl, g.u32(i*rpg)) }
	wide := elementTrips(n)
	if wide {
		seg, rl = g.bin(OpRem, U, tid, g.u32(S)), g.bin(OpDiv, U, tid, g.u32(S))
		slot = func(i int) int { return g.bin(OpAdd, U, g.bin(OpMul, U, rl, g.u32(S)), g.u32(i)) }
	}
	var old []int
	for i, nd := range n.Nodes {
		if nd.Op == OpCoord {
			old = append(old, i)
		}
	}
	e := g.bin(OpAdd, U, g.bin(OpMul, U, g.add(Node{Op: OpCoord, A: 0, Type: U}), g.u32(rpg)), rl)
	coords := g.decode(e, n.Space)
	for _, i := range old {
		a := n.Nodes[i].A
		g.redirect(i, coords[a], coords[a])
	}
	g.Loops[c].Trip = T / S
	if it := g.iterOf(c); it >= 0 {
		if wide {
			m := g.bin(OpMul, U, it, g.u32(S))
			g.redirect(it, g.bin(OpAdd, U, m, seg), m)
		} else {
			k := g.bin(OpAdd, U, g.bin(OpMul, U, seg, g.u32(T/S)), it)
			g.redirect(it, k, k)
		}
	}
	f := folds[0]
	t := n.Nodes[f].Type
	val := f
	if term >= 0 {
		first := g.add(Node{Op: OpLess, Type: ir.Pred, A: seg, B: g.u32(1)})
		val = g.bin(OpAdd, t, f, g.add(Node{Op: OpSelect, Type: t, A: first, B: term, C: g.add(Node{Op: OpConst, Type: t})}))
	}
	sh := g.buf(Buf{Kind: BufShared, Elem: t, Size: G * len(folds)})
	final := g.Stores[st]
	at := func(ix, j int) int {
		if j == 0 {
			return ix
		}
		return g.bin(OpAdd, U, ix, g.u32(j*G))
	}
	g.Stores = nil
	total := map[int]int{}
	for j, fj := range folds {
		v := fj
		if j == 0 {
			v = val
		}
		g.Stores = append(g.Stores, Store{Buf: sh, Ix: at(tid, j), Val: v, At: -1})
	}
	g.Stores = append(g.Stores, Store{Barrier: true, At: -1})
	for j, fj := range folds {
		sum := -1
		for i := range kept(S, t, drop) {
			x := g.add(Node{Op: OpLoad, Type: t, Buf: sh, A: at(slot(i), j)})
			if sum < 0 {
				sum = x
				continue
			}
			sum = g.bin(OpAdd, t, sum, x)
		}
		total[fj] = sum
	}
	if len(folds) == 1 {
		final.Val = total[f]
	} else {
		final.Val = copyInto(g, g, final.Val, func(i int, nd Node) (int, bool) {
			v, ok := total[i]
			return v, ok
		})
	}
	g.Stores = append(g.Stores, final)
	return g, true
}

// sums is splittable's loop and sums for a store made of one or more sums.
func sums(n *Kernel) (c int, folds []int, ok bool) {
	if c, f, _, _, ok := splittable(n); ok {
		return c, []int{f}, true
	}
	if len(n.Stores) != 1 || n.Stores[0].At != -1 || !n.storesOwn(n.Stores[0]) {
		return 0, nil, false
	}
	c, ok = -1, true
	seen := map[int]bool{}
	var walk func(i int)
	walk = func(i int) {
		if seen[i] || !ok {
			return
		}
		seen[i] = true
		nd := n.Nodes[i]
		switch {
		case nd.Op == OpFold:
			if c >= 0 && nd.At != c || n.Loops[nd.At].Parent != -1 || !n.sumFold(i) {
				ok = false
				return
			}
			c = nd.At
			folds = append(folds, i)
			return
		case nd.Op == OpIter, (nd.Op == OpLoad || nd.Op == OpTileLoad) && nd.At > 0:
			ok = false
			return
		}
		for _, a := range n.Operands(&nd) {
			walk(*a)
		}
	}
	walk(n.Stores[0].Val)
	if !ok || len(folds) < 2 || len(n.foldsOf(c)) != len(folds) {
		return 0, nil, false
	}
	slices.Sort(folds)
	for _, f := range folds {
		if n.Nodes[f].Type != n.Nodes[folds[0]].Type {
			return 0, nil, false
		}
	}
	return c, folds, true
}

// elementTrips reports whether n reads a weight and every weight it reads
// is verbatim: a trip of its sum reads one element of a row.
func elementTrips(n *Kernel) bool {
	reads := false
	for _, b := range n.Bufs {
		if b.Kind == BufIn && b.Operand.Kind == OperandWeight {
			if !jlm.Verbatim(b.Operand.Format) {
				return false
			}
			reads = true
		}
	}
	return reads
}

// splitShared is the workgroup memory, in words, a split of a tile of rows
// holds its partial sums in: within every device's 48 KiB with room beside.
const splitShared = 8 << 10

// groupSplitRows is groupSplit for a form of a capacity whose threads each
// sum a tile of rows (RowTiles, Kernel.Tile): its stores each a sum over one
// loop. The segments are those one unit of the leading axis alone would
// take -- the unit's elements by S -- so a call of one row runs as many
// threads as a form of one row, and a row's sum is cut alike whatever the
// capacity or the call's rows. A workgroup is S segments of rpg elements of
// one unit: each thread's partials of every row of its tile to workgroup
// memory, a barrier, and then thread (s, e) sums the segments of element e's
// rows s, s+S, ... in order and stores them -- a row past the call's the
// call's last, its sum from that row's partials, so no row's value depends
// on which of the tile's copies of it a driver computed. The launch keeps
// the leading axis (Space: units, then the unit's workgroups), so a call
// holds its rows' units whole (CallRows).
func (s Split) groupSplitRows(n *Kernel, d Device) ([]*Kernel, bool) {
	T := n.Tile
	if T < 2 || len(n.Stores) != T || n.TileFirst == 0 || n.TileBase == 0 {
		return nil, false
	}
	c, _, _, ok := rowsSplittable(n)
	if !ok {
		return nil, false
	}
	slots := s.Slots
	if slots == 0 {
		slots = d.Slots
	}
	K, U := n.Loops[c].Trip, n.Space[0]
	E := n.Elements() / U
	S := 1
	for S < s.Max && E*S < 2*slots && K%(s.Block*S*2) == 0 && min(max(32*S*2, 128), 1024)*T <= splitShared {
		S *= 2
	}
	if S == 1 {
		// Rows wide enough to fill the device whole: no split, and the
		// untiled form for few rows runs a row's sum as the tile does.
		if s.Few == 0 || !d.ExactArith {
			return nil, false
		}
		u, ok := untiled(n)
		if !ok {
			return nil, false
		}
		g := n.clone()
		g.Many, u.Few = s.Few, s.Few
		u.Name = n.Name + ".few"
		return []*Kernel{g, u}, true
	}
	// A workgroup's segments hold whole elements of one unit: fewer
	// elements than a 128-thread workgroup's share take a narrower one.
	G := min(max(32*S, 128), 1024)
	for G > 32*S && E%(G/S) != 0 {
		G /= 2
	}
	g, ok := splitTile(n, S, G, s.Drop)
	if !ok {
		return nil, false
	}
	if s.Few == 0 || !d.ExactArith {
		return []*Kernel{g}, true
	}
	u, ok := untiled(n)
	if !ok {
		return []*Kernel{g}, true
	}
	f, ok := splitTile(u, S, fewGroup(G, S, E, d), s.Drop)
	if !ok {
		return []*Kernel{g}, true
	}
	g.Many, f.Few = s.Few, s.Few
	f.Name = n.Name + ".few.group"
	return []*Kernel{g, f}, true
}

// untiled is a tiled kernel n (RowTiles) whose threads each sum one row: its
// first row's stores alone, the tile's first row the leading coordinate, the
// leading axis the rows before tiling. A row's sum is the tiled kernel's
// row's, node for node.
func untiled(n *Kernel) (*Kernel, bool) {
	base := n.Nodes[n.TileBase-1]
	if n.TileRows == 0 || base.Op != OpMul || n.Nodes[base.A].Op != OpCoord || n.Nodes[base.A].A != 0 {
		return nil, false
	}
	m := n.clone()
	m.redirect(n.TileBase-1, base.A, -1)
	m.Stores = m.Stores[:len(n.Stores)/n.Tile]
	m.Space[0] = n.TileRows
	m.Tile, m.TileBase, m.TileRows = 1, base.A+1, 0
	m.compact()
	return m, true
}

// fewGroup is the workgroup of a form for few rows: the tiled form's G
// threads, halved while a row's E elements fill fewer workgroups than the
// device has multiprocessors, down to one warp. Its segments stay S, so a
// row's sums are the tiled form's; only how many elements a workgroup holds
// moves.
func fewGroup(G, S, E int, d Device) int {
	for d.Units > 0 && E*S/G < d.Units && G/2 >= 32 && G/2 >= S && E%(G/2/S) == 0 {
		G /= 2
	}
	return G
}

// splitTile is n, whose threads each sum a tile of n.Tile rows (one where
// untiled), split in S segments within a workgroup of G threads
// (groupSplitRows).
func splitTile(n *Kernel, S, G int, drop bool) (*Kernel, bool) {
	T := n.Tile
	if len(n.Stores) != T || n.TileFirst == 0 || n.TileBase == 0 {
		return nil, false
	}
	c, folds, terms, ok := rowsSplittable(n)
	if !ok {
		return nil, false
	}
	K, U := n.Loops[c].Trip, n.Space[0]
	E := n.Elements() / U
	if S == 1 || G%S != 0 || E%(G/S) != 0 || K%S != 0 || G*T > splitShared {
		return nil, false
	}
	rpg := G / S
	first := n.TileFirst - 1
	dep := make([]bool, len(n.Nodes))
	for changed := true; changed; {
		changed = false
		for i := range n.Nodes {
			if dep[i] {
				continue
			}
			d := i == first
			for _, a := range n.Operands(&n.Nodes[i]) {
				d = d || dep[*a]
			}
			if d {
				dep[i], changed = true, true
			}
		}
	}
	g := n.clone()
	g.Name = n.Name + ".group"
	g.Work, g.Group, g.Space = true, G, []int{U, E / rpg}
	g.Tile, g.TileBase, g.TileLast, g.TileFirst, g.TileRows = 0, 0, 0, 0, 0
	I := ir.U32
	var old []int
	for i, nd := range n.Nodes {
		if nd.Op == OpCoord && nd.A > 0 {
			old = append(old, i)
		}
	}
	tid := g.add(Node{Op: OpThread, Type: I})
	seg := g.bin(OpDiv, I, tid, g.u32(rpg))
	rl := g.bin(OpRem, I, tid, g.u32(rpg))
	e := g.bin(OpAdd, I, g.bin(OpMul, I, g.add(Node{Op: OpCoord, A: 1, Type: I}), g.u32(rpg)), rl)
	coords := g.decode(e, n.Space[1:])
	for _, i := range old {
		a := n.Nodes[i].A - 1
		g.redirect(i, coords[a], coords[a])
	}
	g.Loops[c].Trip = K / S
	if it := g.iterOf(c); it >= 0 {
		k := g.bin(OpAdd, I, g.bin(OpMul, I, seg, g.u32(K/S)), it)
		g.redirect(it, k, k)
	}
	t := n.Nodes[folds[0]].Type
	for _, f := range folds {
		if n.Nodes[f].Type != t {
			return nil, false
		}
	}
	store0 := g.Stores[0]
	sh := g.buf(Buf{Kind: BufShared, Elem: t, Size: G * T})
	lead := g.add(Node{Op: OpLess, Type: ir.Pred, A: seg, B: g.u32(1)})
	g.Stores = nil
	for j, f := range folds {
		val := f
		if terms[j] >= 0 {
			val = g.bin(OpAdd, t, f, g.add(Node{Op: OpSelect, Type: t, A: lead, B: terms[j], C: g.add(Node{Op: OpConst, Type: t})}))
		}
		g.Stores = append(g.Stores, Store{Buf: sh, Ix: g.bin(OpAdd, I, tid, g.u32(j*G)), Val: val, At: -1})
	}
	g.Stores = append(g.Stores, Store{Barrier: true, At: -1})
	for k := 0; k*S < T; k++ {
		jj := g.bin(OpMin, I, g.bin(OpAdd, I, seg, g.u32(k*S)), g.u32(T-1))
		if n.TileLast > 0 {
			jj = g.bin(OpMin, I, jj, g.bin(OpSub, I, n.TileLast-1, first))
		}
		at := g.bin(OpAdd, I, g.bin(OpMul, I, jj, g.u32(G)), rl)
		sum := -1
		for i := range kept(S, t, drop) {
			x := g.add(Node{Op: OpLoad, Type: t, Buf: sh, A: g.bin(OpAdd, I, at, g.u32(i*rpg))})
			if sum < 0 {
				sum = x
				continue
			}
			sum = g.bin(OpAdd, t, sum, x)
		}
		row := g.bin(OpAdd, I, n.TileBase-1, jj)
		if n.TileLast > 0 {
			row = g.bin(OpMin, I, row, n.TileLast-1)
		}
		leaf := func(i int, _ Node) (int, bool) {
			switch {
			case i == first:
				return row, true
			case i < len(dep) && !dep[i]:
				return i, true
			}
			return 0, false
		}
		st := store0
		st.Val = sum
		if store0.Ax == nil {
			st.Ix = copyInto(g, g, store0.Ix, leaf)
		} else {
			st.Ax = make([]int, len(store0.Ax))
			for a, x := range store0.Ax {
				st.Ax[a] = copyInto(g, g, x, leaf)
			}
		}
		g.Stores = append(g.Stores, st)
	}
	return g, true
}

// rowsSplittable finds each store's sum in a kernel whose stores are each a
// sum over one top-level loop, alone or with a term beside it the loop does
// not change (-1 for none): a tile of rows' sums (RowTiles).
func rowsSplittable(n *Kernel) (c int, folds, terms []int, ok bool) {
	c = -1
	for _, st := range n.Stores {
		if st.At != -1 || st.Barrier || st.Run {
			return 0, nil, nil, false
		}
		f, term := st.Val, -1
		if v := n.Nodes[f]; v.Op == OpAdd {
			switch {
			case n.Nodes[v.A].Op == OpFold:
				f, term = v.A, v.B
			case n.Nodes[v.B].Op == OpFold:
				f, term = v.B, v.A
			}
		}
		nd := n.Nodes[f]
		if nd.Op != OpFold || n.Loops[nd.At].Parent != -1 || c >= 0 && nd.At != c || !n.sumFold(f) ||
			term >= 0 && n.Varies(term, nd.At) || slices.Contains(folds, f) {
			return 0, nil, nil, false
		}
		c = nd.At
		folds, terms = append(folds, f), append(terms, term)
	}
	if c < 0 || len(n.foldsOf(c)) != len(folds) {
		return 0, nil, nil, false
	}
	return c, folds, terms, true
}

// splittable finds the element's sum: a top-level loop with one fold, a sum,
// which the element's one store writes at its own place, alone or with a
// term beside it the loop does not change (-1 for none).
func splittable(n *Kernel) (c, f, term, st int, ok bool) {
	if len(n.Stores) != 1 || n.Stores[0].At != -1 || !n.storesOwn(n.Stores[0]) {
		return 0, 0, 0, 0, false
	}
	f, term = n.Stores[0].Val, -1
	if v := n.Nodes[f]; v.Op == OpAdd {
		switch {
		case n.Nodes[v.A].Op == OpFold:
			f, term = v.A, v.B
		case n.Nodes[v.B].Op == OpFold:
			f, term = v.B, v.A
		}
	}
	nd := n.Nodes[f]
	if nd.Op != OpFold || n.Loops[nd.At].Parent != -1 || len(n.foldsOf(nd.At)) != 1 || !n.sumFold(f) ||
		term >= 0 && n.Varies(term, nd.At) {
		return 0, 0, 0, 0, false
	}
	return nd.At, f, term, 0, true
}

// Blocks runs a sum loop Size trips at a time: each block's trips unrolled
// as one chain from zero, the chain added into the sum once a block.
type Blocks struct {
	Size  int
	Kinds []PrimKind
}

func (Blocks) Name() string { return "blocks" }

func (b Blocks) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(b.Name(), ks, b.apply)
}

func (b Blocks) apply(n *Kernel) ([]*Kernel, bool, error) {
	B := b.Size
	// A form for few rows sums as its tiled form does, which blocks none
	// of its several sums a loop.
	if !covers(b.Kinds, n.Kind) || n.Few > 0 || n.Many > 0 {
		return nil, false, nil
	}
	var r *Kernel
	for c := range n.Loops {
		folds := n.foldsOf(c)
		it := n.iterOf(c)
		if B <= 1 || n.Loops[c].Count > 0 || n.Loops[c].Trip%B != 0 || len(folds) != 1 || it < 0 || !n.sumFold(folds[0]) {
			continue
		}
		if r == nil {
			r = n.clone()
		}
		f := folds[0]
		step := r.Nodes[f].A
		r.Loops[c].Trip /= B
		base := r.bin(OpMul, ir.U32, it, r.u32(B))
		chain := r.f32(0)
		for j := 0; j < B; j++ {
			k := base
			if j > 0 {
				k = r.bin(OpAdd, ir.U32, base, r.u32(j))
			}
			chain = r.copyOf(step, map[int]int{f: chain, it: k})
		}
		r.Nodes[f].A = r.bin(OpAdd, r.Nodes[f].Type, f, chain)
	}
	if r == nil {
		return nil, false, nil
	}
	return []*Kernel{r}, true, nil
}

func covers(kinds []PrimKind, k PrimKind) bool {
	for _, x := range kinds {
		if x == k {
			return true
		}
	}
	return false
}
