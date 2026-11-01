package jit

import (
	"math"

	"github.com/jitllm/jitllm/jit/ir"
)

// Flash forms a normalized exponential reduction over a range of positions
// -- an attention as decomposed: the dots of a row's factor and a
// position's, the scores, their maximum, the exponentials, their sum and
// weighted sum of a second position factor, the division -- whose rows are
// one sequence's (Heads.Seqs), on the device's matrix unit. A workgroup
// takes a tile of the sequence's consecutive rows and one head, a warp each
// Matrix[0] of them, and runs the positions the tile attends in blocks: the
// position factors of a block staged in workgroup memory once for every row
// of the tile, the dots of the block on the matrix unit, each row's scores,
// running maximum, exponentials and running sum by a thread a row, the
// exponentials' products with the second factor on the matrix unit, and the
// weighted sums carried in registers, rescaled by exp of the old maximum
// less the new. No buffer holds a row's scores, and the blocks run to the
// tile's last live position, read from the row descriptors at run time.
//
// The matrix unit multiplies binary16, so the form is taken only where Half
// admits binary16 operands: the row factor, the position factors and the
// exponentials rounded, one product a block of dots and one of weighted
// sums, whatever the history holds.
//
// Rows, Keys and Shared bound a workgroup's rows, a block's positions and
// the workgroup memory it takes, in bytes; fewer rows or positions are
// taken where the head is too wide for them. A head too wide for half Rows
// within Shared takes the unit's memory where the device states more
// (Device.Shared): two warps a workgroup stage the history for few rows and
// hide no latency.
type Flash struct {
	Rows, Keys, Shared int
	Half               bool
}

func (Flash) Name() string { return "flash" }

func (p Flash) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	if !p.Half || d.Matrix[0] == 0 || d.warp() != ir.SubgroupLanes {
		return ks, nil
	}
	var out []*Kernel
	for i := 0; i < len(ks); i++ {
		if ch, ok := rangeChain(ks[i:]); ok {
			if k, ok := p.form(ch, d); ok {
				k.Passes = append(append([]string(nil), ch.last.Passes...), p.Name())
				out = append(out, k)
				i += ch.n - 1
				continue
			}
		}
		out = append(out, ks[i])
	}
	return out, nil
}

// expChain is a normalized exponential reduction over a range of positions
// as decomposed: its members, among them the maximum, the exponentials, the
// sum and the weighted sum, then last, the normalization; n kernels in all.
type expChain struct {
	mem            []member
	mx, wt, sm, vl *member
	last           *Kernel
	n              int
}

// rangeChain is the normalized exponential reduction ch starts with.
func rangeChain(ch []*Kernel) (expChain, bool) {
	var c expChain
	for _, k := range ch {
		p := k.Positions
		if p == nil || k.Heads == nil || len(k.Stores) != 1 || p.Align <= 1 {
			break
		}
		c.mem = append(c.mem, member{k: k, axis: p.Axis, loop: p.Loop, out: k.Bufs[k.Stores[0].Buf].Operand})
	}
	if len(c.mem) == 0 || len(c.mem) == len(ch) {
		return c, false
	}
	c.last = ch[len(c.mem)]
	if c.last.Kind != PrimMap || len(c.last.Loops) != 0 || len(c.last.Stores) != 1 {
		return c, false
	}
	for i := range c.mem {
		m := &c.mem[i]
		if f, ok := m.k.fold(); ok && m.loop >= 0 && m.k.Nodes[m.k.Nodes[f].A].Op == OpMax {
			c.mx = m
		}
	}
	if c.mx == nil {
		return c, false
	}
	for i := range c.mem {
		m := &c.mem[i]
		f, isFold := m.k.fold()
		switch {
		case m.loop < 0 && m.k.reads(c.mx.out) && hasNode(m.k, OpExp):
			c.wt = m
		case c.wt != nil && isFold && m.loop >= 0 && m.k.reads(c.wt.out) && c.last.reads(m.out) && m.k.sumFold(f):
			if len(m.k.Space) == len(c.last.Space) {
				c.vl = m
			} else {
				c.sm = m
			}
		}
	}
	c.n = len(c.mem) + 1
	return c, c.wt != nil && c.sm != nil && c.vl != nil
}

// flashSide is a member's reads at a row, a head and a position: the
// position's node, the node it adds the key's coordinate to, and the key's
// coordinate -- the range's axis or the loop's iteration.
type flashSide struct {
	m               *member
	pos, base       int
	inner           int
	subs            map[Operand]int
	row, head, kpos int
}

// keyOf is member m's position node read as its base plus the key's
// coordinate.
func keyOf(m *member) (pos, base int, ok bool) {
	ps := m.k.Positions
	if len(ps.Pos) != 1 {
		return 0, 0, false
	}
	pos = ps.Pos[0]
	nd := m.k.Nodes[pos]
	if nd.Op != OpAdd {
		return 0, 0, false
	}
	isKey := func(i int) bool {
		x := m.k.Nodes[i]
		return m.axis >= 0 && x.Op == OpCoord && x.A == m.axis || m.loop >= 0 && x.Op == OpIter && x.At == m.loop
	}
	switch {
	case isKey(nd.B):
		return pos, nd.A, true
	case isKey(nd.A):
		return pos, nd.B, true
	}
	return 0, 0, false
}

// at copies node i of side s's member into m at the side's row, head and
// position: coordinate 0 the row, 1 the head, the key's coordinate the
// position less the base the row's descriptor gives, and inner the other
// coordinate or iteration, where the member has one; a load of an operand
// subs names is the node it names.
func (s *flashSide) at(m *Kernel, i int) int {
	k := s.m.k
	var rel int
	plain := func(_ int, nd Node) (int, bool) {
		switch {
		case nd.Op == OpCoord && nd.A == 0:
			return s.row, true
		case nd.Op == OpCoord && nd.A == 1:
			return s.head, true
		}
		return 0, false
	}
	leaf := func(j int, nd Node) (int, bool) {
		if v, ok := plain(j, nd); ok {
			return v, true
		}
		switch {
		case j == s.pos:
			return s.kpos, true
		case s.m.axis >= 0 && nd.Op == OpCoord && nd.A == s.m.axis, s.m.loop >= 0 && nd.Op == OpIter && nd.At == s.m.loop:
			if rel == 0 {
				rel = m.bin(OpSub, ir.U32, s.kpos, copyInto(m, k, s.base, plain)) + 1
			}
			return rel - 1, true
		case nd.Op == OpCoord || nd.Op == OpIter:
			return s.inner, true
		case nd.Op == OpLoad:
			if v, ok := s.subs[k.Bufs[nd.Buf].Operand]; ok {
				return v, true
			}
		}
		return 0, false
	}
	return copyInto(m, k, i, leaf)
}

// products is the two factors of member m's sum of products, its fold's
// start zero.
func products(m *member) (a, b int, ok bool) {
	k := m.k
	f, ok := k.fold()
	if !ok {
		return 0, 0, false
	}
	fd := k.Nodes[f]
	if fd.Type != ir.F32 || k.Nodes[fd.B].Op != OpConst || k.Nodes[fd.B].Imm != 0 {
		return 0, 0, false
	}
	st := k.Nodes[fd.A]
	switch {
	case st.Op == OpFma && st.C == f:
		return st.A, st.B, true
	case st.Op == OpAdd && st.A == f && k.Nodes[st.B].Op == OpMul:
		return k.Nodes[st.B].A, k.Nodes[st.B].B, true
	}
	return 0, 0, false
}

// flashGeom is a workgroup's tile: TR rows, KB positions a block, and its
// workgroup memory's regions in words -- the first the row factor's halves
// before the blocks, then each block's dots and then its weighted sums --
// and the words in all.
type flashGeom struct {
	TR, KB                              int
	sq, sv, sp                          int
	aOff, kOff, vOff, pOff, sOff, total int
}

func flashLayout(TR, KB, Dk, Dv int) flashGeom {
	g := flashGeom{TR: TR, KB: KB, sq: Dk + 4, sv: Dv + 4, sp: KB + 4}
	up := func(n int) int { return (n + 3) / 4 * 4 }
	g.kOff = up(max(TR*g.sq/2, TR*KB, TR*Dv))
	g.vOff = g.kOff + up(KB*g.sq/2)
	g.pOff = g.vOff + up(KB*g.sv/2)
	g.sOff = g.pOff + up((TR+1)*g.sp/2)
	g.total = g.sOff + 3*(TR+1)
	return g
}

func (p Flash) form(c expChain, d Device) (*Kernel, bool) {
	tm, tn, tk := d.Matrix[0], d.Matrix[1], d.Matrix[2]
	h := c.mx.k.Heads
	if h == nil || h.Seqs != 1 || len(c.mem) != 6 {
		return nil, false
	}
	var dt, sc *member
	for i := range c.mem {
		if m := &c.mem[i]; m != c.mx && m.loop < 0 && c.mx.k.reads(m.out) {
			sc = m
		}
	}
	if sc == nil {
		return nil, false
	}
	for i := range c.mem {
		if m := &c.mem[i]; m != sc && m.loop < 0 && m.k.Kind == PrimContract && sc.k.reads(m.out) {
			dt = m
		}
	}
	if dt == nil || len(dt.k.Loops) != 1 || len(c.vl.k.Space) != 3 || len(c.last.Space) != 3 {
		return nil, false
	}
	for _, m := range []*member{dt, sc, c.wt, c.vl} {
		if _, _, ok := keyOf(m); !ok {
			return nil, false
		}
	}
	windowed := false
	for _, m := range c.mem {
		for _, b := range m.k.Bufs {
			windowed = windowed || b.Operand == tableWindows
		}
	}
	if dt.k.Nodes[dt.k.Positions.First].Op != OpLoad || windowed && dt.k.Nodes[dt.k.Positions.End].Op != OpLoad {
		return nil, false
	}
	qx, kx, ok := products(dt)
	if !ok {
		return nil, false
	}
	keyed := dt.k.dependsOn(func(nd Node) bool { return nd.Op == OpCoord && nd.A == dt.axis })
	if keyed[qx] {
		qx, kx = kx, qx
	}
	if keyed[qx] || !keyed[kx] {
		return nil, false
	}
	wx, vx, ok := products(c.vl)
	if !ok {
		return nil, false
	}
	weighted := c.vl.k.dependsOn(func(nd Node) bool { return nd.Op == OpLoad && c.vl.k.Bufs[nd.Buf].Operand == c.wt.out })
	if weighted[vx] {
		wx, vx = vx, wx
	}
	if weighted[vx] || !weighted[wx] {
		return nil, false
	}
	for _, m := range []*member{c.mx, c.sm} {
		f, _ := m.k.fold()
		st := m.k.Nodes[m.k.Nodes[f].A]
		term := st.B
		if term == f {
			term = st.A
		}
		if t := m.k.Nodes[term]; t.Op != OpLoad || len(t.Ax) == 0 {
			return nil, false
		}
	}
	R, H, Dk, Dv := h.Rows, h.Query, dt.k.Loops[0].Trip, h.Dim
	// A warp takes tm rows, so a row is two threads of the workgroup
	// (rowsPhase) where tm is sixteen.
	if tm != 16 || Dk%tk != 0 || Dk%4 != 0 || Dv%tn != 0 || Dv%4 != 0 || R < tm {
		return nil, false
	}
	limit := p.Shared
	if d.Shared > 0 {
		limit = min(limit, d.Shared)
	}
	g := p.layout(R, Dk, Dv, limit, tm, tn, tk)
	if g.TR < p.Rows/2 && d.Shared > limit {
		if w := p.layout(R, Dk, Dv, d.Shared, tm, tn, tk); w.TR > g.TR {
			g = w
		}
	}
	if g.TR == 0 {
		return nil, false
	}
	b := flashBuild{c: c, dt: dt, sc: sc, g: g, Dk: Dk, Dv: Dv, R: R, H: H, tm: tm, tn: tn, tk: tk, qx: qx, kx: kx, vx: vx,
		tq: map[int]int{}, tr: map[int]int{}, scale: d.ScaleRows, windowed: windowed}
	return b.kernel(), true
}

// layout is the flash form of the most rows a workgroup, then the most
// positions a block, whose workgroup memory is at most limit bytes: no more
// rows than R take, at least a tile's. Its TR is 0 where none fits.
func (p Flash) layout(R, Dk, Dv, limit, tm, tn, tk int) flashGeom {
	top := p.Rows
	for top/2 >= tm && top/2 >= R {
		top /= 2
	}
	for TR := top; TR >= tm; TR /= 2 {
		for KB := p.Keys; KB >= max(tn, tk) && KB%2 == 0; KB /= 2 {
			if TR%tm == 0 && KB%tn == 0 && KB%tk == 0 && KB%4 == 0 && 4*flashLayout(TR, KB, Dk, Dv).total <= limit {
				return flashLayout(TR, KB, Dk, Dv)
			}
		}
	}
	return flashGeom{}
}

// flashBuild is a Flash kernel as it is built.
type flashBuild struct {
	c               expChain
	dt, sc          *member
	g               flashGeom
	Dk, Dv, R, H    int
	tm, tn, tk      int
	qx, kx, vx      int
	m               *Kernel
	st, t, hd, last int
	r0, rl, P, k0   int
	first, warpRow  int
	minus, zero     int
	tq, tr          map[int]int
	// scale keeps the weighted sums in accumulators (Device.ScaleRows).
	scale    bool
	sums     []int
	windowed bool
	own0     int
}

func (b *flashBuild) side(m *member, row, kpos, inner int, subs map[Operand]int) *flashSide {
	pos, base, _ := keyOf(m)
	return &flashSide{m: m, pos: pos, base: base, row: row, head: b.hd, kpos: kpos, inner: inner, subs: subs}
}

// word is two float32 values as the binary16 pair of their roundings.
func (b *flashBuild) word(x0, x1 int) int {
	U, m := ir.U32, b.m
	return m.bin(OpOr, U, m.un(OpHalf, U, x0), m.bin(OpShl, U, m.un(OpHalf, U, x1), m.u32(16)))
}

// store writes v at word ix of the workgroup memory, in loop at.
func (b *flashBuild) store(ix, v, at int) {
	b.m.Stores = append(b.m.Stores, Store{Buf: b.st, Ix: ix, Val: v, At: at})
}

// load is the workgroup memory's word ix as type t, read in loop at.
func (b *flashBuild) load(t ir.Type, ix, at int) int {
	return b.m.add(Node{Op: OpLoad, Type: t, Buf: b.st, A: ix, At: at + 1})
}

// split is the thread's u-th of n elements dealt over the workgroup, a
// surplus thread's the last, as its quotient and remainder by d. Where the
// workgroup's width is a multiple of d, or d of it, they are the thread's
// own quotient and remainder, made once, or the thread itself, beside
// constants: what a trip reads is not a pair of registers an element.
func (b *flashBuild) split(u, n, d int) (q, r int) {
	U, m, G := ir.U32, b.m, b.m.Group
	switch {
	case n%G == 0 && G%d == 0:
		if b.tq[d] == 0 {
			b.tq[d], b.tr[d] = m.bin(OpDiv, U, b.t, m.u32(d))+1, m.bin(OpRem, U, b.t, m.u32(d))+1
		}
		return m.bin(OpAdd, U, b.tq[d]-1, m.u32(u*G/d)), b.tr[d] - 1
	case n%G == 0 && d%G == 0:
		return m.u32(u * G / d), m.bin(OpAdd, U, b.t, m.u32(u*G%d))
	}
	w := m.bin(OpAdd, U, b.t, m.u32(u*G))
	if n%G != 0 {
		w = m.bin(OpMin, U, w, m.u32(n-1))
	}
	return m.bin(OpDiv, U, w, m.u32(d)), m.bin(OpRem, U, w, m.u32(d))
}

func (b *flashBuild) tile(t ir.TileType, off, stride int, col bool, at int) int {
	imm := int64(2)
	if col {
		imm |= 1
	}
	return b.m.add(Node{Op: OpTileLoad, Type: ir.Void, Buf: b.st, A: off, B: b.m.u32(stride), Imm: imm, At: at + 1, Tile: &t})
}

func (b *flashBuild) kernel() *Kernel {
	U := ir.U32
	g, c := b.g, b.c
	last := c.last
	m := &Kernel{Name: "flash", Kind: PrimMatrix, Work: true, Group: 32 * g.TR / b.tm, Space: []int{(b.R + g.TR - 1) / g.TR, b.H},
		Part: last.Part, Of: last.Of, Apart: last.Apart, Capacity: last.Capacity, RowsHeld: last.Capacity > 0}
	b.m = m
	b.st = m.buf(Buf{Kind: BufShared, Elem: U, Size: g.total, Operand: Operand{Kind: OperandScratch, Name: c.vl.out.Name + ".flash"}})
	b.t = m.add(Node{Op: OpThread, Type: U})
	b.hd = m.add(Node{Op: OpCoord, A: 1, Type: U})
	b.last = m.u32(b.R - 1)
	if last.Capacity > 0 {
		b.last = m.bin(OpSub, U, m.liveRows(last.Capacity), m.u32(1))
	}
	b.r0 = m.bin(OpMul, U, m.add(Node{Op: OpCoord, A: 0, Type: U}), m.u32(g.TR))
	b.rl = m.bin(OpMin, U, m.bin(OpAdd, U, b.r0, m.u32(g.TR-1)), b.last)
	rowAt := func(rloc int) int { return m.bin(OpMin, U, m.bin(OpAdd, U, b.r0, rloc), b.last) }
	b.minus = m.f32(math.Float32bits(math.Float32frombits(0xff7fffff)))
	b.warpRow = m.bin(OpMul, U, m.bin(OpShr, U, b.t, m.u32(5)), m.u32(b.tm))

	// The row factor's halves, staged once.
	half := b.Dk / 2
	n := g.TR * half
	for u := 0; u < (n+m.Group-1)/m.Group; u++ {
		rloc, dp := b.split(u, n, half)
		s := b.side(b.dt, rowAt(rloc), 0, 0, nil)
		q := func(e int) int {
			s.inner = m.bin(OpAdd, U, m.bin(OpMul, U, dp, m.u32(2)), m.u32(e))
			return s.at(m, b.qx)
		}
		ix := m.bin(OpAdd, U, m.bin(OpMul, U, rloc, m.u32(g.sq/2)), dp)
		b.store(m.bin(OpAdd, U, ix, m.u32(g.aOff)), b.word(q(0), q(1)), -1)
	}
	m.Stores = append(m.Stores, Store{Barrier: true, At: -1})

	// The blocks: from the tile's first row's first position to its last
	// row's end.
	ps := b.dt.k.Positions
	first := b.side(b.dt, m.bin(OpMin, ir.U32, b.r0, b.last), 0, 0, nil).at(m, ps.First)
	end := b.side(b.dt, b.rl, 0, 0, nil).at(m, ps.End)
	if b.windowed {
		// The first row's own position, its descriptor's write word.
		r := m.bin(OpMin, ir.U32, b.r0, b.last)
		own := m.Nodes[b.side(b.dt, r, 0, 0, nil).at(m, ps.End)]
		own.A = m.bin(OpAdd, U, m.bin(OpMul, U, r, m.u32(rowWords)), m.u32(rowWrite))
		b.own0 = m.add(own)
	}
	span := m.add(Node{Op: OpSelect, Type: U, A: m.add(Node{Op: OpLess, Type: ir.Pred, A: first, B: end}), B: m.bin(OpSub, U, end, first), C: m.u32(0)})
	blocks := m.bin(OpMax, U, m.bin(OpDiv, U, m.bin(OpAdd, U, span, m.u32(g.KB-1)), m.u32(g.KB)), m.u32(1))
	P := m.loop(1, -1)
	m.Loops[P].Count, m.Loops[P].Uniform, m.Loops[P].StepsLast = blocks+1, true, true
	b.P = P
	kb := m.add(Node{Op: OpIter, At: P, Type: U})
	b.k0 = m.bin(OpAdd, U, first, m.bin(OpMul, U, kb, m.u32(g.KB)))
	b.first = m.add(Node{Op: OpLess, Type: ir.Pred, A: kb, B: m.u32(1)})
	b.stageBlock()
	b.dots()
	b.rowsPhase()
	b.weighted()
	b.output()
	// The copies of one read -- a position's page, a row's first position --
	// made once.
	if m.cse(false) {
		m.compact()
	}
	return m
}

// stageBlock stages a block's position factors: the dots' halves a
// position a row, the weighted sum's a position a row, the last row's
// descriptor addressing the history.
func (b *flashBuild) stageBlock() {
	U, m, g := ir.U32, b.m, b.g
	if b.scale {
		// The trip before multiplied its block after its last barrier.
		m.Stores = append(m.Stores, Store{Barrier: true, At: b.P})
	}
	half := b.Dk / 2
	n := g.KB * half
	for u := 0; u < (n+m.Group-1)/m.Group; u++ {
		dp, key := b.split(u, n, g.KB)
		kp := m.bin(OpAdd, U, b.k0, key)
		s := b.side(b.dt, b.stageRow(kp), kp, 0, nil)
		k := func(e int) int {
			s.inner = m.bin(OpAdd, U, m.bin(OpMul, U, dp, m.u32(2)), m.u32(e))
			return s.at(m, b.kx)
		}
		ix := m.bin(OpAdd, U, m.bin(OpMul, U, key, m.u32(g.sq/2)), dp)
		b.store(m.bin(OpAdd, U, ix, m.u32(g.kOff)), b.word(k(0), k(1)), b.P)
	}
	vhalf := b.Dv / 2
	n = g.KB * vhalf
	for u := 0; u < (n+m.Group-1)/m.Group; u++ {
		key, dp := b.split(u, n, vhalf)
		kp := m.bin(OpAdd, U, b.k0, key)
		s := b.side(b.c.vl, b.stageRow(kp), kp, 0, nil)
		v := func(e int) int {
			s.inner = m.bin(OpAdd, U, m.bin(OpMul, U, dp, m.u32(2)), m.u32(e))
			return s.at(m, b.vx)
		}
		ix := m.bin(OpAdd, U, m.bin(OpMul, U, key, m.u32(g.sv/2)), dp)
		b.store(m.bin(OpAdd, U, ix, m.u32(g.vOff)), b.word(v(0), v(1)), b.P)
	}
	m.Stores = append(m.Stores, Store{Barrier: true, At: b.P})
}

// stageRow is the row whose descriptor addresses position kpos: the
// tile's last, or in a windowed chain the row at position kpos, the first
// row for a position before it and the last past it. Not the row whose
// range ends at kpos: a bidirectional run's rows all end at the run's end.
func (b *flashBuild) stageRow(kpos int) int {
	if !b.windowed {
		return b.rl
	}
	U, m := ir.U32, b.m
	own := b.own0
	past := m.add(Node{Op: OpSelect, Type: U, A: m.add(Node{Op: OpLess, Type: ir.Pred, A: own, B: kpos}), B: m.bin(OpSub, U, kpos, own), C: m.u32(0)})
	return m.bin(OpMin, U, m.bin(OpAdd, U, b.r0, m.bin(OpMin, U, past, m.u32(b.g.TR-1))), b.last)
}

// dots is a warp's rows' dots with the block's positions, on the matrix
// unit, into the first region.
func (b *flashBuild) dots() {
	U, m, g := ir.U32, b.m, b.g
	aT := ir.TileType{Rows: b.tm, Cols: b.tk, Elem: ir.TileF16, Use: ir.TileA}
	bT := ir.TileType{Rows: b.tk, Cols: b.tn, Elem: ir.TileF16, Use: ir.TileB}
	cT := ir.TileType{Rows: b.tm, Cols: b.tn, Elem: ir.TileF32, Use: ir.TileAcc}
	b.zero = m.add(Node{Op: OpTileSplat, Type: ir.Void, A: m.f32(0), Tile: &cT})
	at := func(base, scale, off int) int {
		return m.bin(OpAdd, U, m.bin(OpMul, U, base, m.u32(scale)), m.u32(off))
	}
	steps := b.Dk / b.tk
	qh := make([]int, steps)
	for s := range steps {
		qh[s] = b.tile(aT, at(b.warpRow, g.sq, 2*g.aOff+s*b.tk), g.sq, false, -1)
	}
	for j := 0; j < g.KB/b.tn; j++ {
		acc := b.zero
		for s := range steps {
			off := 2*g.kOff + j*b.tn*g.sq + s*b.tk
			kh := b.tile(bT, m.u32(off), g.sq, true, b.P)
			acc = m.add(Node{Op: OpTileMMA, Type: ir.Void, A: qh[s], B: kh, C: acc, Tile: &cT})
		}
		m.Stores = append(m.Stores, Store{Buf: b.st, Ix: at(b.warpRow, g.KB, g.aOff+j*b.tn), Val: acc, At: b.P, Tile: true, Stride: m.u32(g.KB)})
	}
	m.Stores = append(m.Stores, Store{Barrier: true, At: b.P})
}

// rowsPhase is two threads a row of the tile, adjacent lanes, each half
// the block's positions: the row's scores, the running maximum, the
// exponentials -- their binary16 roundings staged for the weighted sums --
// the running sum, and the factor the weighted sums so far are rescaled
// by, the pair's halves combined by a shuffle. The even lane writes the
// row's running values, the odd one a slot of its own.
func (b *flashBuild) rowsPhase() {
	U, F, m, g := ir.U32, ir.F32, b.m, b.g
	half := m.bin(OpAnd, U, b.t, m.u32(1))
	tr := m.bin(OpShr, U, b.t, m.u32(1))
	slot := m.add(Node{Op: OpSelect, Type: U, A: m.add(Node{Op: OpLess, Type: ir.Pred, A: half, B: m.u32(1)}), B: tr, C: m.u32(g.TR)})
	row := m.bin(OpMin, U, m.bin(OpAdd, U, b.r0, tr), b.last)
	mOff, lOff, aOff := g.sOff, g.sOff+g.TR+1, g.sOff+2*(g.TR+1)
	at := func(x, off int) int { return m.bin(OpAdd, U, x, m.u32(off)) }
	mOld := m.add(Node{Op: OpSelect, Type: F, A: b.first, B: b.minus, C: b.load(F, at(tr, mOff), b.P)})
	lOld := m.add(Node{Op: OpSelect, Type: F, A: b.first, B: m.f32(0), C: b.load(F, at(tr, lOff), b.P)})
	n := g.KB / 2
	j0 := m.bin(OpMul, U, half, m.u32(n))
	scores, kpos := make([]int, n), make([]int, n)
	mx := b.minus
	for i := range n {
		j := at(j0, i)
		kpos[i] = m.bin(OpAdd, U, b.k0, j)
		sv := b.load(F, m.bin(OpAdd, U, m.bin(OpMul, U, tr, m.u32(g.KB)), at(j, g.aOff)), b.P)
		scores[i] = b.side(b.sc, row, kpos[i], 0, map[Operand]int{b.dt.out: sv}).at(m, b.sc.k.Stores[0].Val)
		mx = m.bin(OpMax, F, mx, scores[i])
	}
	mx = m.bin(OpMax, F, mx, m.add(Node{Op: OpShuffle, Type: F, A: mx, Imm: 1}))
	mNew := m.bin(OpMax, F, mOld, mx)
	sum := m.f32(0)
	pw := m.bin(OpAdd, U, m.bin(OpMul, U, tr, m.u32(g.sp/2)), m.bin(OpMul, U, half, m.u32(n/2)))
	for i := 0; i < n; i += 2 {
		var pe [2]int
		for e := range pe {
			pe[e] = b.side(b.c.wt, row, kpos[i+e], 0, map[Operand]int{b.sc.out: scores[i+e], b.c.mx.out: mNew}).at(m, b.c.wt.k.Stores[0].Val)
			sum = m.bin(OpAdd, F, sum, pe[e])
		}
		b.store(at(pw, g.pOff+i/2), b.word(pe[0], pe[1]), b.P)
	}
	sum = m.bin(OpAdd, F, sum, m.add(Node{Op: OpShuffle, Type: F, A: sum, Imm: 1}))
	alpha := m.un(OpExp, F, m.bin(OpSub, F, mOld, mNew))
	b.store(at(slot, mOff), mNew, b.P)
	b.store(at(slot, lOff), m.add(Node{Op: OpFma, Type: F, A: lOld, B: alpha, C: sum}), b.P)
	b.store(at(slot, aOff), alpha, b.P)
	m.Stores = append(m.Stores, Store{Barrier: true, At: b.P})
}

// weighted is a warp's rows' exponentials times the block's second
// factor, on the matrix unit, into the first region, and each thread's
// share of the tile's weighted sums carried in registers: rescaled, and
// the block's added.
func (b *flashBuild) weighted() {
	U, F, m, g := ir.U32, ir.F32, b.m, b.g
	aT := ir.TileType{Rows: b.tm, Cols: b.tk, Elem: ir.TileF16, Use: ir.TileA}
	bT := ir.TileType{Rows: b.tk, Cols: b.tn, Elem: ir.TileF16, Use: ir.TileB}
	cT := ir.TileType{Rows: b.tm, Cols: b.tn, Elem: ir.TileF32, Use: ir.TileAcc}
	at := func(base, scale, off int) int {
		return m.bin(OpAdd, U, m.bin(OpMul, U, base, m.u32(scale)), m.u32(off))
	}
	steps := g.KB / b.tk
	ph := make([]int, steps)
	for s := range steps {
		ph[s] = b.tile(aT, at(b.warpRow, g.sp, 2*g.pOff+s*b.tk), g.sp, false, b.P)
	}
	if b.scale {
		alpha := m.bin(OpAdd, U, b.warpRow, m.u32(g.sOff+2*(g.TR+1)))
		var sums []int
		for j := 0; j < b.Dv/b.tn; j++ {
			f := m.add(Node{Op: OpFold, Type: ir.Void, At: b.P, B: b.zero, Tile: &cT})
			acc := m.add(Node{Op: OpTileScale, Type: ir.Void, A: f, B: alpha, Buf: b.st, Tile: &cT})
			for s := range steps {
				v := b.tile(bT, m.u32(2*g.vOff+s*b.tk*g.sv+j*b.tn), g.sv, false, b.P)
				acc = m.add(Node{Op: OpTileMMA, Type: ir.Void, A: ph[s], B: v, C: acc, Tile: &cT})
			}
			m.Nodes[f].A = acc
			sums = append(sums, f)
		}
		m.Stores = append(m.Stores, Store{Barrier: true, At: -1})
		for j, f := range sums {
			m.Stores = append(m.Stores, Store{Buf: b.st, Ix: at(b.warpRow, b.Dv, g.aOff+j*b.tn), Val: f, At: -1, Tile: true, Stride: m.u32(b.Dv)})
		}
		m.Stores = append(m.Stores, Store{Barrier: true, At: -1})
		for u := 0; u < g.TR*b.Dv/m.Group; u++ {
			rloc, dd := b.split(u, g.TR*b.Dv, b.Dv)
			b.sums = append(b.sums, b.load(F, m.bin(OpAdd, U, m.bin(OpMul, U, rloc, m.u32(b.Dv)), m.bin(OpAdd, U, dd, m.u32(g.aOff))), -1))
		}
		return
	}
	for j := 0; j < b.Dv/b.tn; j++ {
		acc := b.zero
		for s := range steps {
			v := b.tile(bT, m.u32(2*g.vOff+s*b.tk*g.sv+j*b.tn), g.sv, false, b.P)
			acc = m.add(Node{Op: OpTileMMA, Type: ir.Void, A: ph[s], B: v, C: acc, Tile: &cT})
		}
		m.Stores = append(m.Stores, Store{Buf: b.st, Ix: at(b.warpRow, b.Dv, g.aOff+j*b.tn), Val: acc, At: b.P, Tile: true, Stride: m.u32(b.Dv)})
	}
	m.Stores = append(m.Stores, Store{Barrier: true, At: b.P})
	n := g.TR * b.Dv
	for u := 0; u < n/m.Group; u++ {
		f := m.add(Node{Op: OpFold, Type: F, At: b.P, B: m.f32(0)})
		m.Nodes[f].A = b.rescaled(f, u)
		b.sums = append(b.sums, f)
	}
}

// rescaled is the thread's u-th weighted sum, f, rescaled by its row's
// factor and the block's added, both read from workgroup memory after the
// trip's last barrier (the loop's StepsLast).
func (b *flashBuild) rescaled(f, u int) int {
	U, F, m, g := ir.U32, ir.F32, b.m, b.g
	rloc, dd := b.split(u, g.TR*b.Dv, b.Dv)
	e := m.bin(OpAdd, U, m.bin(OpMul, U, rloc, m.u32(b.Dv)), dd)
	alpha := b.load(F, m.bin(OpAdd, U, rloc, m.u32(g.sOff+2*(g.TR+1))), b.P)
	return m.add(Node{Op: OpFma, Type: F, A: f, B: alpha, C: b.load(F, m.bin(OpAdd, U, e, m.u32(g.aOff)), b.P)})
}

// output is the normalization at each of the thread's elements: the
// weighted sum its register's, the sum and the maximum the row's.
func (b *flashBuild) output() {
	U, F, m, g, last := ir.U32, ir.F32, b.m, b.g, b.c.last
	st := last.Stores[0]
	for u, f := range b.sums {
		rloc, dd := b.split(u, g.TR*b.Dv, b.Dv)
		row := m.bin(OpMin, U, m.bin(OpAdd, U, b.r0, rloc), b.last)
		subs := map[Operand]int{
			b.c.vl.out: f,
			b.c.sm.out: b.load(F, m.bin(OpAdd, U, rloc, m.u32(g.sOff+g.TR+1)), -1),
			b.c.mx.out: b.load(F, m.bin(OpAdd, U, rloc, m.u32(g.sOff)), -1),
		}
		leaf := func(_ int, nd Node) (int, bool) {
			switch {
			case nd.Op == OpCoord && nd.A == 0:
				return row, true
			case nd.Op == OpCoord && nd.A == 1:
				return b.hd, true
			case nd.Op == OpCoord && nd.A == 2:
				return dd, true
			case nd.Op == OpLoad:
				if v, ok := subs[last.Bufs[nd.Buf].Operand]; ok {
					return v, true
				}
			}
			return 0, false
		}
		ns := Store{Buf: m.bufLike(last.Bufs[st.Buf]), Val: copyInto(m, last, st.Val, leaf), At: -1}
		for _, a := range st.Ax {
			ns.Ax = append(ns.Ax, copyInto(m, last, a, leaf))
		}
		if st.Ax == nil {
			ns.Ix = copyInto(m, last, st.Ix, leaf)
		}
		m.Stores = append(m.Stores, ns)
	}
}
