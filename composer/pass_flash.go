package composer

import (
	"math"

	"github.com/jitllm/jitllm/composer/ir"
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
// second factor rounded, one product a block of dots. The exponentials are
// staged as their halves and their remainders' halves, two products a
// block of weighted sums, so a weight carries some 22 bits: rounded to
// eleven, the weights of synth-apertus's near one-hot rows (q/k-normed
// heads) put its prompt 1.3e-4 from the CPU with an argmax flip, 1.2e-7
// with the remainders, for 4-22% of the kernel's time on sm_70.
//
// A head wide enough that a warp's share of the row factor and of the
// weighted sums would not fit its registers is sliced: S warps take each
// warp's rows, each its S-th of the head's dots and of its weighted sums,
// the dots' partials summed in workgroup memory before the scores, 2S
// threads a row. Where the second factor is the first's rows (a latent
// history, its values its keys'), a block is staged once for both.
//
// Rows, Keys and Shared bound a workgroup's rows, a block's positions and
// the workgroup memory it takes, in bytes; fewer rows or positions are
// taken where the head is too wide for them. A head too wide for half Rows
// within Shared takes the unit's memory where the device states more
// (Device.Shared): two warps a workgroup stage the history for few rows and
// hide no latency.
//
// Split, past 1, deals a tile's blocks over that many workgroups, each its
// share in order: each leaves its weighted sums, maximum and sum in a
// partial, and the normalization becomes the partials' merge (SplitRange's),
// each weighted by exp of its maximum less the greatest. A tile with fewer
// blocks than workgroups leaves the rest's partials empty. 1 splits none.
// 0 forms both ways where the device's waves leave slots idle (flashSplit):
// the form unsplit at a call's history of up to the positions where the
// model prices the split less (crossing), split past them. The model is a
// block's products on a unit (a tile's rows by a block's positions by both
// heads' widths, by the workgroups the unit holds at once) by each
// workgroup's blocks by the waves its grid spans, and the merge's Launch
// and Plane a word its partials move on one unit (written, read, and the
// output written), in the same products. A windowed history
// is not split.
type Flash struct {
	Rows, Keys, Shared int
	Half               bool
	Split              int
	Launch, Plane      int
}

func (Flash) Name() string { return "flash" }

func (p Flash) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	if !p.Half || d.Matrix[0] == 0 || d.warp() != ir.SubgroupLanes {
		return ks, nil
	}
	var out []*Kernel
	for i := 0; i < len(ks); i++ {
		if ch, ok := rangeChain(ks[i:]); ok {
			if fs, ok := p.form(ch, d); ok {
				for _, k := range fs {
					k.Passes = append(append([]string(nil), ch.last.Passes...), p.Name())
				}
				out = append(out, fs...)
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
	if c.wt == nil || c.sm == nil || c.vl == nil {
		return c, false
	}
	// The normalization reads the chain's own maximum, sum and weighted
	// sum: a map merging another reduction's partial beside them (an
	// attention over two histories) is not this chain's division.
	for _, nd := range c.last.Nodes {
		if nd.Op != OpLoad {
			continue
		}
		if o := c.last.Bufs[nd.Buf].Operand; o.Kind == OperandScratch && o != c.mx.out && o != c.sm.out && o != c.vl.out {
			return c, false
		}
	}
	return c, true
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
// and the words in all; plOff is the exponentials' remainders past their
// halves.
type flashGeom struct {
	TR, KB, S                                  int
	sq, sv, sp                                 int
	aOff, kOff, vOff, pOff, plOff, sOff, total int
	// vt stages the second factor a column a row at stride sv, unless it
	// is the first's rows.
	vt bool
}

// flashLayout is the tile of TR rows, KB positions a block and S warps a
// warp's rows; vik stages the second factor with the first (its rows), vt
// a column a row.
func flashLayout(TR, KB, Dk, Dv, S int, vik, vt, held bool) flashGeom {
	g := flashGeom{TR: TR, KB: KB, S: S, sq: Dk + 4, sv: Dv + 4, sp: KB + 4}
	up := func(n int) int { return (n + 3) / 4 * 4 }
	sums := TR * Dv
	if held {
		sums = 0
	}
	g.kOff = up(max(TR*g.sq/2, S*TR*KB, sums))
	g.vOff = g.kOff + up(KB*g.sq/2)
	g.pOff = g.vOff + up(KB*g.sv/2)
	if vt && !vik {
		g.vt, g.sv = true, KB+4
		g.pOff = g.vOff + up(Dv*g.sv/2)
	}
	if vik {
		g.vOff, g.sv = g.kOff, g.sq
		g.pOff = g.kOff + up(KB*g.sq/2)
	}
	g.plOff = g.pOff + up((TR+1)*g.sp/2)
	g.sOff = g.plOff + up((TR+1)*g.sp/2)
	if held {
		g.sOff = max(g.sOff, up(TR*Dv))
	}
	g.total = g.sOff + 3*(TR+1)
	return g
}

func (p Flash) form(c expChain, d Device) ([]*Kernel, bool) {
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
			windowed = windowed || b.Operand == TableWindows.op()
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
	vik := sameLoad(dt.k, kx, c.vl.k, vx)
	g, G := p.geom(R, H, h.Key, Dk, Dv, vik, d)
	if g.TR == 0 {
		return nil, false
	}
	b := flashBuild{c: c, dt: dt, sc: sc, g: g, G: G, Dk: Dk, Dv: Dv, R: R, H: H, tm: tm, tn: tn, tk: tk, qx: qx, kx: kx, vx: vx,
		tq: map[int]int{}, tr: map[int]int{}, scale: d.ScaleRows, scaleAdd: d.ScaleAdd && !d.ScaleRows, windowed: windowed, NC: 1}
	k := b.kernel()
	nc, past := p.Split, 0
	if nc == 0 {
		nc = flashSplit(k, d)
		if nc > 1 {
			past = p.crossing(k, g, nc, R*H*Dv, Dk+Dv, d)
		}
	}
	if nc <= 1 || windowed || past < 0 {
		return []*Kernel{k}, true
	}
	whole := k
	b.m, b.tq, b.tr, b.sums = nil, map[int]int{}, map[int]int{}, nil
	b.NC = nc
	no := c.last.Bufs[c.last.Stores[0].Buf].Operand
	b.part = Operand{Kind: OperandScratch, Name: no.Name + ".flashpart", Of: no.Of}
	b.stride = flashParts().stride(Dv)
	k = b.kernel()
	hm := *h
	hm.Rows, hm.Query, hm.Dim = R, H, Dv
	mg := SplitRange{Work: flashParts()}.merge(c.last, hm, b.NC, b.stride, b.part, c.vl.out, c.sm.out, c.mx.out)
	mg.Of, mg.Capacity = c.last.Of, c.last.Capacity
	if past == 0 {
		return []*Kernel{k, mg}, true
	}
	whole.Keys = Band{Upto: past}
	k.Keys, mg.Keys = Band{Past: past}, Band{Past: past}
	return []*Kernel{whole, k, mg}, true
}

// crossing is the history positions past which the model prices whole's
// blocks dealt over nc workgroups a tile, with the merge of out outputs,
// less than whole's: -1 where it prices the split no less at any history.
// width is a position's products a row, both heads'.
func (p Flash) crossing(whole *Kernel, g flashGeom, nc, out, width int, d Device) int {
	slots, W := flashSlots(whole, d), whole.groups()
	if slots == 0 {
		return -1
	}
	per := slots / d.Units
	// A wave of n workgroups: its count, and the products a unit runs a
	// block of it.
	waves := func(n int) int { return (n + slots - 1) / slots }
	trip := func(n int) int { return min(per, (n+d.Units-1)/d.Units) * g.TR * g.KB * width }
	merge := p.Launch + p.Plane*(2*nc+1)*out/d.Units
	// Each price grows by its waves' products a block, so past the blocks
	// where the split's grows the slower it stays the less.
	a, b := waves(W)*trip(W), waves(W*nc)*trip(W*nc)
	if b >= a*nc {
		return -1
	}
	for blocks := 1; blocks <= 1<<20; blocks++ {
		if b*((blocks+nc-1)/nc)+merge < a*blocks {
			return (blocks - 1) * g.KB
		}
	}
	return -1
}

// flashSplit is the workgroups a tile's blocks are dealt over where the
// form states none: the fewest whose grid fills 95% of the slots its waves
// span, at the workgroups of k a unit holds (their threads, their workgroup
// memory, flashThreadRegs registers a thread). None where the launch holds
// every workgroup at once already (resident) or the unsplit grid fills its
// waves. On a V100 at a 12800-position chunk of 512 rows: 3 for Llama's
// 128-wide heads (5121 us to 4475 with the merge; 5 reads 5040) and 256-wide
// (6178 to 5415), none for 64-wide, whose unsplit grid is one wave of two a
// unit (2550 against 3138 split).
func flashSplit(k *Kernel, d Device) int {
	if d.Units <= 0 || d.Slots <= 0 || k.Group <= 0 || resident(k, d) > 0 {
		return 1
	}
	slots, W := flashSlots(k, d), k.groups()
	if slots == 0 {
		return 1
	}
	for nc := 1; nc <= 8; nc++ {
		w := W * nc
		if 20*w >= 19*((w+slots-1)/slots*slots) {
			return nc
		}
	}
	return 1
}

// flashSlots is the workgroups of k the device holds at once: those a unit
// holds by their threads, their workgroup memory and flashThreadRegs
// registers a thread, on every unit; 0 where a unit holds none.
func flashSlots(k *Kernel, d Device) int {
	per := d.Slots / d.Units / k.Group
	if sb := sharedBytes(k); d.Shared > 0 && sb > 0 {
		per = min(per, d.Shared/sb)
	}
	if d.Regs > 0 {
		per = min(per, d.Regs*(d.Slots/d.Units)/(flashThreadRegs*k.Group))
	}
	return max(per, 0) * d.Units
}

// flashThreadRegs is the registers a flash thread takes, the most a CUDA
// thread holds: the forms measured on sm_70 take 253 to 255 where their
// grid sets no bound.
const flashThreadRegs = 255

// flashParts is a split flash's partial: a row and head's weighted sums,
// then its maximum and its sum, a slot each workgroup of the tile.
func flashParts() Workgroups { return Workgroups{Pad: 1, Copies: 1} }

// geom is the layout of a form over R rows of H heads, in groups of the
// heads sharing one of Hk key heads, Dk and Dv wide (vik: the values the
// keys' rows) on d, and the heads a workgroup takes together: a TR of 0
// where none fits the device's workgroup memory.
func (p Flash) geom(R, H, Hk, Dk, Dv int, vik bool, d Device) (flashGeom, int) {
	tm, tn, tk := d.Matrix[0], d.Matrix[1], d.Matrix[2]
	limit := p.Shared
	if d.Shared > 0 {
		limit = min(limit, d.Shared)
	}
	S := flashSlices(Dk/tk, Dv/tn)
	G := flashGroup(H, Hk)
	held := d.ScaleRows || d.ScaleAdd
	g := p.layout(R*G, Dk, Dv, limit, tm, tn, tk, S, vik, held)
	if g.TR < p.Rows/2 && d.Shared > limit {
		if w := p.layout(R*G, Dk, Dv, d.Shared, tm, tn, tk, S, vik, held); w.TR > g.TR {
			g = w
		}
	}
	if g.TR%G != 0 {
		g = flashGeom{}
	}
	return g, G
}

// layout is the flash form of the most rows a workgroup, then the most
// positions a block, whose workgroup memory is at most limit bytes: no more
// rows than R take, at least a tile's, and no more threads than a
// workgroup holds. At those rows and positions the second factor is staged
// a column a row where that fits, its sums held past the loop (held) in the
// regions the loop leaves. Its TR is 0 where none fits.
func (p Flash) layout(R, Dk, Dv, limit, tm, tn, tk, S int, vik, held bool) flashGeom {
	top := p.Rows
	for top/2 >= tm && top/2 >= R {
		top /= 2
	}
	for TR := top; TR >= tm; TR /= 2 {
		if 32*TR*S/tm > 1024 {
			continue
		}
		for KB := p.Keys; KB >= max(tn, tk) && KB%2 == 0; KB /= 2 {
			if TR%tm != 0 || KB%tn != 0 || KB%tk != 0 || KB%(4*S) != 0 {
				continue
			}
			if g := flashLayout(TR, KB, Dk, Dv, S, vik, false, false); 4*g.total > limit {
				continue
			}
			if g := flashLayout(TR, KB, Dk, Dv, S, vik, true, held); 4*g.total <= limit {
				return g
			}
			return flashLayout(TR, KB, Dk, Dv, S, vik, false, false)
		}
	}
	return flashGeom{}
}

// flashRegs is the registers a warp's share of the row factor's tiles
// (four a tile of k) and of its weighted sums' (eight a tile) may take
// before the head is sliced (flashSlices).
const flashRegs = 104

// flashSlices is the warps a warp's rows are sliced over at steps tiles of
// k and jv of the weighted sums: the fewest, a power of two dividing both,
// whose share fits flashRegs, else the most that divide both.
func flashSlices(steps, jv int) int {
	S := 1
	for 4*steps/S+8*jv/S > flashRegs && S < 8 && steps%(2*S) == 0 && jv%(2*S) == 0 {
		S *= 2
	}
	return S
}

// sameLoad reports whether node i of a and node j of b read the same
// operand: each a load, or what a load is widened or cut to.
func sameLoad(a *Kernel, i int, b *Kernel, j int) bool {
	oa, ok := loadOperand(a, i)
	if !ok {
		return false
	}
	ob, ok := loadOperand(b, j)
	return ok && oa == ob
}

func loadOperand(k *Kernel, i int) (Operand, bool) {
	for range 8 {
		nd := k.Nodes[i]
		switch nd.Op {
		case OpLoad:
			return k.Bufs[nd.Buf].Operand, true
		case OpWidenHalf, OpBitcast, OpAnd, OpShr, OpCvtF32:
			i = nd.A
		default:
			return Operand{}, false
		}
	}
	return Operand{}, false
}

// flashBuild is a Flash kernel as it is built.
type flashBuild struct {
	c            expChain
	dt, sc       *member
	g            flashGeom
	Dk, Dv, R, H int
	// G is the query heads a tile's rows take of one position, sharing a
	// kv head (flashGroup): tile row r is position r/G's head r%G of the
	// workgroup's G.
	G               int
	tm, tn, tk      int
	qx, kx, vx      int
	m               *Kernel
	st, t, hd, last int
	r0, rl, P, k0   int
	first, warpRow  int
	minus, zero     int
	tq, tr          map[int]int
	// slice is the warp's slice of its rows' head (flashGeom.S), -1 where
	// the head is not sliced.
	slice int
	// scale keeps the weighted sums in accumulators (Device.ScaleRows), and
	// scaleAdd keeps them in tiles each block's products are added to
	// (Device.ScaleAdd).
	scale    bool
	scaleAdd bool
	sums     []int
	windowed bool
	own0     int
	// NC is the workgroups a tile's blocks are dealt over (Flash.Split),
	// part their partial, stride a slot's words, and empty, where NC is past
	// 1, whether the workgroup's share holds no block.
	NC, stride int
	part       Operand
	empty      int
}

func (b *flashBuild) side(m *member, row, kpos, inner int, subs map[Operand]int) *flashSide {
	return b.sideAt(m, row, b.head0(), kpos, inner, subs)
}

func (b *flashBuild) sideAt(m *member, row, head, kpos, inner int, subs map[Operand]int) *flashSide {
	pos, base, _ := keyOf(m)
	return &flashSide{m: m, pos: pos, base: base, row: row, head: head, kpos: kpos, inner: inner, subs: subs}
}

// head0 is the workgroup's first query head: every head of its G reads the
// same kv head and the same positions.
func (b *flashBuild) head0() int {
	if b.G == 1 {
		return b.hd
	}
	return b.m.bin(OpMul, ir.U32, b.hd, b.m.u32(b.G))
}

// rowOf is tile row rloc's row of the call, clamped to the last.
func (b *flashBuild) rowOf(rloc int) int {
	U, m := ir.U32, b.m
	if b.G > 1 {
		rloc = m.bin(OpShr, U, rloc, m.u32(log2(b.G)))
	}
	return m.bin(OpMin, U, m.bin(OpAdd, U, b.r0, rloc), b.last)
}

// headOf is tile row rloc's query head.
func (b *flashBuild) headOf(rloc int) int {
	if b.G == 1 {
		return b.hd
	}
	U, m := ir.U32, b.m
	return m.bin(OpAdd, U, b.head0(), m.bin(OpAnd, U, rloc, m.u32(b.G-1)))
}

// flashGroup is the query heads a tile takes of each position: the most,
// a power of two up to eight, that divide the heads sharing a kv head.
func flashGroup(heads, kv int) int {
	if kv <= 0 || heads%kv != 0 {
		return 1
	}
	G := 1
	for (heads/kv)%(2*G) == 0 && 2*G <= 8 {
		G *= 2
	}
	return G
}

// word is two float32 values as the binary16 pair of their roundings.
func (b *flashBuild) word(x0, x1 int) int {
	U, m := ir.U32, b.m
	return m.bin(OpOr, U, m.un(OpHalf, U, x0), m.bin(OpShl, U, m.un(OpHalf, U, x1), m.u32(16)))
}

// rest is two float32 values' remainders past their binary16 roundings, as
// the binary16 pair of theirs: a value's word and rest sum to it within some
// 22 bits, where its word alone holds 11.
func (b *flashBuild) rest(x0, x1 int) int {
	U, F, m := ir.U32, ir.F32, b.m
	r := func(x int) int { return m.bin(OpSub, F, x, m.un(OpWidenHalf, F, m.un(OpHalf, U, x))) }
	return b.word(r(x0), r(x1))
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

// sliced is the warp's slice times n, -1 where the head is not sliced.
func (b *flashBuild) sliced(n int) int {
	if b.slice < 0 {
		return -1
	}
	return b.m.bin(OpMul, ir.U32, b.slice, b.m.u32(n))
}

// plus is x plus node y, x itself where y is -1.
func (b *flashBuild) plus(x, y int) int {
	if y < 0 {
		return x
	}
	return b.m.bin(OpAdd, ir.U32, x, y)
}

func log2(n int) int {
	l := 0
	for n > 1 {
		n, l = n/2, l+1
	}
	return l
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
	space := []int{(b.R + g.TR/b.G - 1) / (g.TR / b.G), b.H / b.G}
	if b.NC > 1 {
		space = append(space, b.NC)
	}
	m := &Kernel{Name: "flash", Kind: PrimMatrix, Work: true, Group: 32 * g.TR * g.S / b.tm, Space: space,
		Part: last.Part, Of: last.Of, Apart: last.Apart, Capacity: last.Capacity, RowsHeld: last.Capacity > 0}
	b.m = m
	b.st = m.buf(Buf{Kind: BufShared, Elem: U, Size: g.total, Operand: Operand{Kind: OperandScratch, Name: c.vl.out.Name + ".flash"}})
	b.t = m.add(Node{Op: OpThread, Type: U})
	b.hd = m.add(Node{Op: OpCoord, A: 1, Type: U})
	b.last = m.u32(b.R - 1)
	if last.Capacity > 0 {
		b.last = m.bin(OpSub, U, m.liveRows(last.Capacity), m.u32(1))
	}
	b.r0 = m.bin(OpMul, U, m.add(Node{Op: OpCoord, A: 0, Type: U}), m.u32(g.TR/b.G))
	b.rl = m.bin(OpMin, U, m.bin(OpAdd, U, b.r0, m.u32(g.TR/b.G-1)), b.last)
	b.minus = m.f32(math.Float32bits(math.Float32frombits(0xff7fffff)))
	b.slice = -1
	if g.S > 1 {
		w := m.bin(OpShr, U, b.t, m.u32(5))
		b.slice = m.bin(OpAnd, U, w, m.u32(g.S-1))
		b.warpRow = m.bin(OpMul, U, m.bin(OpShr, U, w, m.u32(log2(g.S))), m.u32(b.tm))
	} else {
		b.warpRow = m.bin(OpMul, U, m.bin(OpShr, U, b.t, m.u32(5)), m.u32(b.tm))
	}

	// The row factor's halves, staged once.
	half := b.Dk / 2
	n := g.TR * half
	for u := 0; u < (n+m.Group-1)/m.Group; u++ {
		rloc, dp := b.split(u, n, half)
		s := b.sideAt(b.dt, b.rowOf(rloc), b.headOf(rloc), 0, 0, nil)
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
	// A split's workgroup runs its share of the blocks; one whose share is
	// empty runs the tile's last block, and leaves an empty partial.
	trips, from := blocks, m.u32(0)
	if b.NC > 1 {
		c := m.add(Node{Op: OpCoord, A: 2, Type: U})
		per := m.bin(OpDiv, U, m.bin(OpAdd, U, blocks, m.u32(b.NC-1)), m.u32(b.NC))
		lo := m.bin(OpMul, U, c, per)
		b.empty = m.add(Node{Op: OpLess, Type: ir.Pred, A: m.bin(OpSub, U, blocks, m.u32(1)), B: lo})
		trips = m.add(Node{Op: OpSelect, Type: U, A: b.empty, B: m.u32(1), C: m.bin(OpMin, U, per, m.bin(OpSub, U, blocks, lo))})
		from = m.add(Node{Op: OpSelect, Type: U, A: b.empty, B: m.bin(OpSub, U, blocks, m.u32(1)), C: lo})
	}
	P := m.loop(1, -1)
	m.Loops[P].Count, m.Loops[P].Uniform, m.Loops[P].StepsLast = trips+1, true, true
	b.P = P
	kb := m.add(Node{Op: OpIter, At: P, Type: U})
	b.k0 = m.bin(OpAdd, U, first, m.bin(OpMul, U, m.bin(OpAdd, U, from, kb), m.u32(g.KB)))
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
// position a row, the weighted sum's a column a row (vt: a tile of them
// loads its positions adjacent, as the dots' does, where a position a row
// takes a load an element), the last row's descriptor addressing the
// history.
func (b *flashBuild) stageBlock() {
	U, m, g := ir.U32, b.m, b.g
	if b.scale || b.scaleAdd {
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
	if g.vOff == g.kOff {
		n = 0
	}
	if g.vt {
		n /= 2
	}
	for u := 0; u < (n+m.Group-1)/m.Group; u++ {
		key, dp := b.split(u, n, vhalf)
		if g.vt {
			key = m.bin(OpMul, U, key, m.u32(2))
		}
		v := func(key, e int) int {
			kp := m.bin(OpAdd, U, b.k0, key)
			s := b.side(b.c.vl, b.stageRow(kp), kp, 0, nil)
			s.inner = m.bin(OpAdd, U, m.bin(OpMul, U, dp, m.u32(2)), m.u32(e))
			return s.at(m, b.vx)
		}
		if !g.vt {
			ix := m.bin(OpAdd, U, m.bin(OpMul, U, key, m.u32(g.sv/2)), dp)
			b.store(m.bin(OpAdd, U, ix, m.u32(g.vOff)), b.word(v(key, 0), v(key, 1)), b.P)
			continue
		}
		key1 := m.bin(OpAdd, U, key, m.u32(1))
		for e := range 2 {
			col := m.bin(OpAdd, U, m.bin(OpMul, U, dp, m.u32(2)), m.u32(e))
			ix := m.bin(OpAdd, U, m.bin(OpMul, U, col, m.u32(g.sv/2)), m.bin(OpShr, U, key, m.u32(1)))
			b.store(m.bin(OpAdd, U, ix, m.u32(g.vOff)), b.word(v(key, e), v(key1, e)), b.P)
		}
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
	return m.bin(OpMin, U, m.bin(OpAdd, U, b.r0, m.bin(OpMin, U, past, m.u32(b.g.TR/b.G-1))), b.last)
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
	steps := b.Dk / b.tk / g.S
	ks := b.sliced(steps * b.tk)
	qh := make([]int, steps)
	for s := range steps {
		qh[s] = b.tile(aT, b.plus(at(b.warpRow, g.sq, 2*g.aOff+s*b.tk), ks), g.sq, false, -1)
	}
	part := b.sliced(g.TR * g.KB)
	for j := 0; j < g.KB/b.tn; j++ {
		acc := b.zero
		for s := range steps {
			off := 2*g.kOff + j*b.tn*g.sq + s*b.tk
			kh := b.tile(bT, b.plus(m.u32(off), ks), g.sq, true, b.P)
			acc = m.add(Node{Op: OpTileMMA, Type: ir.Void, A: qh[s], B: kh, C: acc, Tile: &cT})
		}
		m.Stores = append(m.Stores, Store{Buf: b.st, Ix: b.plus(at(b.warpRow, g.KB, g.aOff+j*b.tn), part), Val: acc, At: b.P, Tile: true, Stride: m.u32(g.KB)})
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
	L := 2 * g.S
	half := m.bin(OpAnd, U, b.t, m.u32(L-1))
	tr := m.bin(OpShr, U, b.t, m.u32(log2(L)))
	slot := m.add(Node{Op: OpSelect, Type: U, A: m.add(Node{Op: OpLess, Type: ir.Pred, A: half, B: m.u32(1)}), B: tr, C: m.u32(g.TR)})
	row, head := b.rowOf(tr), b.headOf(tr)
	mOff, lOff, aOff := g.sOff, g.sOff+g.TR+1, g.sOff+2*(g.TR+1)
	at := func(x, off int) int { return m.bin(OpAdd, U, x, m.u32(off)) }
	mOld := m.add(Node{Op: OpSelect, Type: F, A: b.first, B: b.minus, C: b.load(F, at(tr, mOff), b.P)})
	lOld := m.add(Node{Op: OpSelect, Type: F, A: b.first, B: m.f32(0), C: b.load(F, at(tr, lOff), b.P)})
	n := g.KB / L
	j0 := m.bin(OpMul, U, half, m.u32(n))
	scores, kpos := make([]int, n), make([]int, n)
	mx := b.minus
	for i := range n {
		j := at(j0, i)
		kpos[i] = m.bin(OpAdd, U, b.k0, j)
		ix := m.bin(OpAdd, U, m.bin(OpMul, U, tr, m.u32(g.KB)), at(j, g.aOff))
		sv := b.load(F, ix, b.P)
		for p := 1; p < g.S; p++ {
			sv = m.bin(OpAdd, F, sv, b.load(F, at(ix, p*g.TR*g.KB), b.P))
		}
		scores[i] = b.sideAt(b.sc, row, head, kpos[i], 0, map[Operand]int{b.dt.out: sv}).at(m, b.sc.k.Stores[0].Val)
		mx = m.bin(OpMax, F, mx, scores[i])
	}
	for x := 1; x < L; x *= 2 {
		mx = m.bin(OpMax, F, mx, m.add(Node{Op: OpShuffle, Type: F, A: mx, Imm: int64(x)}))
	}
	mNew := m.bin(OpMax, F, mOld, mx)
	sum := m.f32(0)
	pw := m.bin(OpAdd, U, m.bin(OpMul, U, tr, m.u32(g.sp/2)), m.bin(OpMul, U, half, m.u32(n/2)))
	for i := 0; i < n; i += 2 {
		var pe [2]int
		for e := range pe {
			pe[e] = b.sideAt(b.c.wt, row, head, kpos[i+e], 0, map[Operand]int{b.sc.out: scores[i+e], b.c.mx.out: mNew}).at(m, b.c.wt.k.Stores[0].Val)
			sum = m.bin(OpAdd, F, sum, pe[e])
		}
		b.store(at(pw, g.pOff+i/2), b.word(pe[0], pe[1]), b.P)
		b.store(at(pw, g.plOff+i/2), b.rest(pe[0], pe[1]), b.P)
	}
	for x := 1; x < L; x *= 2 {
		sum = m.bin(OpAdd, F, sum, m.add(Node{Op: OpShuffle, Type: F, A: sum, Imm: int64(x)}))
	}
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
	ph, pl := make([]int, steps), make([]int, steps)
	for s := range steps {
		ph[s] = b.tile(aT, at(b.warpRow, g.sp, 2*g.pOff+s*b.tk), g.sp, false, b.P)
		pl[s] = b.tile(aT, at(b.warpRow, g.sp, 2*g.plOff+s*b.tk), g.sp, false, b.P)
	}
	// A block's products: the exponentials' halves by the second factor,
	// then their remainders' halves by it.
	js := b.sliced(b.Dv / g.S)
	jt := b.sliced(b.Dv / g.S * g.sv)
	jv := b.Dv / b.tn / g.S
	products := func(acc, s, j int) int {
		var v int
		if g.vt {
			v = b.tile(bT, b.plus(m.u32(2*g.vOff+j*b.tn*g.sv+s*b.tk), jt), g.sv, true, b.P)
		} else {
			v = b.tile(bT, b.plus(m.u32(2*g.vOff+s*b.tk*g.sv+j*b.tn), js), g.sv, false, b.P)
		}
		acc = m.add(Node{Op: OpTileMMA, Type: ir.Void, A: ph[s], B: v, C: acc, Tile: &cT})
		return m.add(Node{Op: OpTileMMA, Type: ir.Void, A: pl[s], B: v, C: acc, Tile: &cT})
	}
	if b.scale || b.scaleAdd {
		alpha := m.bin(OpAdd, U, b.warpRow, m.u32(g.sOff+2*(g.TR+1)))
		var sums []int
		for j := 0; j < jv; j++ {
			f := m.add(Node{Op: OpFold, Type: ir.Void, At: b.P, B: b.zero, Tile: &cT})
			var acc int
			if b.scale {
				acc = m.add(Node{Op: OpTileScale, Type: ir.Void, A: f, B: alpha, Buf: b.st, Tile: &cT})
				for s := range steps {
					acc = products(acc, s, j)
				}
			} else {
				blk := b.zero
				for s := range steps {
					blk = products(blk, s, j)
				}
				acc = m.add(Node{Op: OpTileScaleAdd, Type: ir.Void, A: f, B: alpha, C: blk, Buf: b.st, Tile: &cT})
			}
			m.Nodes[f].A = acc
			sums = append(sums, f)
		}
		m.Stores = append(m.Stores, Store{Barrier: true, At: -1})
		for j, f := range sums {
			m.Stores = append(m.Stores, Store{Buf: b.st, Ix: b.plus(at(b.warpRow, b.Dv, g.aOff+j*b.tn), js), Val: f, At: -1, Tile: true, Stride: m.u32(b.Dv)})
		}
		m.Stores = append(m.Stores, Store{Barrier: true, At: -1})
		for u := 0; u < g.TR*b.Dv/m.Group; u++ {
			rloc, dd := b.split(u, g.TR*b.Dv, b.Dv)
			b.sums = append(b.sums, b.load(F, m.bin(OpAdd, U, m.bin(OpMul, U, rloc, m.u32(b.Dv)), m.bin(OpAdd, U, dd, m.u32(g.aOff))), -1))
		}
		return
	}
	for j := 0; j < jv; j++ {
		acc := b.zero
		for s := range steps {
			acc = products(acc, s, j)
		}
		m.Stores = append(m.Stores, Store{Buf: b.st, Ix: b.plus(at(b.warpRow, b.Dv, g.aOff+j*b.tn), js), Val: acc, At: b.P, Tile: true, Stride: m.u32(b.Dv)})
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
	if b.NC > 1 {
		b.partials()
		return
	}
	for u, f := range b.sums {
		rloc, dd := b.split(u, g.TR*b.Dv, b.Dv)
		row, head := b.rowOf(rloc), b.headOf(rloc)
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
				return head, true
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

// partials is a split's output: each of the thread's weighted sums into its
// row and head's slot of the workgroup's share, and beside them the row's
// maximum and sum, each element's thread writing the row's same two. An
// empty share's slot holds no weight: its maximum the least, its sums 0.
func (b *flashBuild) partials() {
	U, F, m, g := ir.U32, ir.F32, b.m, b.g
	pb := m.buf(Buf{Kind: BufOut, Elem: F, Operand: b.part, Extent: b.R * b.H * b.NC * b.stride})
	c := m.add(Node{Op: OpCoord, A: 2, Type: U})
	pick := func(none, v int) int { return m.add(Node{Op: OpSelect, Type: F, A: b.empty, B: none, C: v}) }
	zero := m.f32(0)
	for u, f := range b.sums {
		rloc, dd := b.split(u, g.TR*b.Dv, b.Dv)
		row, head := b.rowOf(rloc), b.headOf(rloc)
		slot := m.bin(OpAdd, U, m.bin(OpMul, U, m.bin(OpAdd, U, m.bin(OpMul, U, row, m.u32(b.H)), head), m.u32(b.NC)), c)
		at := func(off int) int {
			return m.bin(OpAdd, U, m.bin(OpMul, U, slot, m.u32(b.stride)), off)
		}
		mx := b.load(F, m.bin(OpAdd, U, rloc, m.u32(g.sOff)), -1)
		sm := b.load(F, m.bin(OpAdd, U, rloc, m.u32(g.sOff+g.TR+1)), -1)
		m.Stores = append(m.Stores,
			Store{Buf: pb, Ix: at(dd), Val: pick(zero, f), At: -1},
			Store{Buf: pb, Ix: at(m.u32(b.stride - 2)), Val: pick(b.minus, mx), At: -1},
			Store{Buf: pb, Ix: at(m.u32(b.stride - 1)), Val: pick(zero, sm), At: -1})
	}
}
