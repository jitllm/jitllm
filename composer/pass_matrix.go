package composer

import (
	"fmt"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/format/jlm"
)

// Matrix forms a reduction over several rows on the device's matrix unit
// (Device.Matrix): a reduction of one loop whose term is the product of a
// factor of the row and the reduced coordinate alone -- a projection's
// input -- and a factor of the output and the reduced coordinate alone -- its
// decoded weight. The row's factor is written first as binary16 pairs by a
// kernel of its own (halves), rounded once rather than once a workgroup of
// outputs. A workgroup of Warps[0]*Warps[1] warps takes Rows rows by Cols
// outputs. Each trip stages Depth elements of k of both factors in
// workgroup memory, the row's as the words it was written in and the
// output's decoded to binary16 pairs, a thread eight consecutive elements
// of a row at a time, so the elements of one packed word share its load and
// its scales once Affine has proved the shared address (headroom); a row of
// the stage is padded, A's so that a fragment's run of eight elements is
// one aligned access, and so that the rows a fragment load reads fall in
// different banks. The stage is two: trip t
// multiplies one while it fills the other with trip t+1's factors, whose
// words Prefetch has loaded a trip before, and one barrier ends the trip.
// Each warp multiplies its share of the block in the device's tiles into
// float32 accumulators carried across the trips; and the block's sums pass
// through workgroup memory -- the stage's own, a row of every warp's tiles
// at a time, behind barriers -- to the kernel's own stores, an element a
// thread. A row past the call's (Kernel.Capacity) or the step's, and an
// output past the last, is the last one's, so the kernel reads and writes
// nothing the step does not state. A weight held as float32 is left as it
// is: binary16 would round it.
//
// Resident is how many workgroups a unit is to hold at once, where their
// threads and workgroup memory fit it: one's barrier is another's issue.
// Tiles is the other forms the device's model weighs per projection
// (cost); Latency is a round of a unit's resident workgroups' fixed cost and
// Decode a weight row's staging, each in products of one output element a
// k. Where the workgroups leave units idle, k is cut into Split segments
// (splits): each writes its sums to planes, and a combine sums them and
// stores, at Launch for its launch and Plane a word it moves on one unit.
// The stage converts the row's factor itself where the words the
// workgroups read of it cost less than a halves kernel's launch and words
// (rowCost); Convert, where set, states it (1) or not (-1).
type Matrix struct {
	Rows, Cols, Depth int
	Warps             [2]int
	Resident          int
	Split             int
	Convert           int
	Tiles             []MatrixTile
	Latency, Decode   int
	Launch, Plane     int
}

// MatrixTile is one form of Matrix: a workgroup's rows, outputs and warps,
// the elements of k a trip stages, the workgroups a unit holds, and the
// segments k is cut in (0: the model's), and whether the stage converts the
// row's factor (Matrix.Convert).
type MatrixTile struct {
	Rows, Cols, Depth int
	Warps             [2]int
	Resident          int
	Split             int
	Convert           int
}

// Tile is p's own form.
func (p Matrix) Tile() MatrixTile {
	return MatrixTile{Rows: p.Rows, Cols: p.Cols, Depth: p.Depth, Warps: p.Warps, Resident: p.Resident, Split: p.Split, Convert: p.Convert}
}

func (p Matrix) with(t MatrixTile) Matrix {
	p.Rows, p.Cols, p.Depth, p.Warps, p.Resident, p.Split, p.Convert = t.Rows, t.Cols, t.Depth, t.Warps, t.Resident, t.Split, t.Convert
	return p
}

// chosen is p at the tile d states, else at the form of p's the device's
// model costs least for k (cost), p's own on a tie.
func (p Matrix) chosen(k *Kernel, d Device) Matrix {
	if d.Tile.Rows > 0 {
		return p.with(d.Tile)
	}
	if d.Units == 0 || len(p.Tiles) == 0 {
		return p
	}
	best, least := p, 0
	for i, t := range append([]MatrixTile{p.Tile()}, p.Tiles...) {
		q := p.with(t)
		f, ok := q.bilinear(k, d)
		if !ok || f.block > 0 {
			if i == 0 {
				return p
			}
			continue
		}
		if _, c := q.splits(k, f.depth, d); i == 0 || c < least {
			best, least = q, c
		}
	}
	return best
}

// cost is the model's time for R rows by N outputs over K at p, k cut in S
// segments: each unit runs its share of the workgroups in rounds of those it
// holds at once, which share its matrix unit, a round paying Latency besides
// their products and the weight rows they decode, a k; and a cut's combine
// its launch and the planes' words, written and read, and the sums'.
func (p Matrix) cost(R, N, K, S, depth int, d Device) int {
	groups := (R + p.Rows - 1) / p.Rows * ((N + p.Cols - 1) / p.Cols) * S
	share := (groups + d.Units - 1) / d.Units
	held := min(share, p.resident(depth, d))
	c := (share + held - 1) / held * (p.Latency + held*(p.Rows*p.Cols+p.Decode*p.Cols)) * (K / S)
	if S > 1 {
		c += p.Launch + p.Plane*(2*S+1)*R*N/d.Units
	}
	return c
}

// splits is the segments k of k is cut in at p, and the model's cost: the
// tile's where it states them, else the least costly of the powers of two
// leaving each segment two trips or more.
func (p Matrix) splits(k *Kernel, depth int, d Device) (int, int) {
	R, N, K := k.Space[0], k.Space[1], k.Loops[0].Trip
	if p.Split > 0 {
		return p.Split, p.cost(R, N, K, p.Split, depth, d)
	}
	best, least := 1, p.cost(R, N, K, 1, depth, d)
	for S := 2; p.Plane > 0 && K%(depth*S) == 0 && K/(depth*S) >= 2; S *= 2 {
		if c := p.cost(R, N, K, S, depth, d); c < least {
			best, least = S, c
		}
	}
	return best, least
}

// resident is the workgroups of p a unit holds at once on d: p.Resident
// where their threads and stage fit it, else one.
func (p Matrix) resident(depth int, d Device) int {
	threads := 32 * p.Warps[0] * p.Warps[1]
	stage := 4 * matrixStage(p.Rows, p.Cols, depth, p.Warps[0], d)
	if p.Resident > 1 && p.Resident*threads <= d.Slots/max(d.Units, 1) && (d.Shared == 0 || p.Resident*stage <= d.Shared) {
		return p.Resident
	}
	return 1
}

func (Matrix) Name() string { return "matrix" }

func (p Matrix) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	if d.Matrix[0] == 0 {
		return ks, nil
	}
	// A row factor's cost both ways, by what a halves kernel of it
	// computes: projections of one row factor (q, k and v of one normed
	// input) read one.
	type planned struct {
		q   Matrix
		f   bilinear
		sig string
	}
	plans := map[*Kernel]planned{}
	costs := map[string]*rowCost{}
	for _, k := range ks {
		q, f, ok := p.plan(k, d)
		if !ok {
			continue
		}
		h, _ := halves(k, f)
		sig := h.computes()
		c := costs[sig]
		if c == nil {
			c = newRowCost(k, f, d)
			costs[sig] = c
		}
		c.add(k, q.Cols)
		plans[k] = planned{q, f, sig}
	}
	made := map[string]Buf{}
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		pl, ok := plans[k]
		if !ok {
			return nil, false, nil
		}
		q := pl.q
		if q.Convert == 0 {
			q.Convert = -1
			if c := costs[pl.sig]; c.convert < c.halves && q.regs(pl.f.depth, d)+c.regs(q, pl.f.depth) <= q.regBudget(d) {
				q.Convert = 1
			}
		}
		return q.form(k, d, pl.f, made), true, nil
	})
}

// plan is k's form at p on d and its parts, false where the matrix unit
// does not take it.
func (p Matrix) plan(k *Kernel, d Device) (Matrix, bilinear, bool) {
	p = p.chosen(k, d)
	f, ok := p.bilinear(k, d)
	if !ok {
		return p, f, false
	}
	q := p
	if f.block > 0 && f.block < p.Rows {
		// A workgroup's rows within one block: as many rows fewer, as
		// many warps fewer, each warp's share of the block the same, and
		// as many more workgroups a unit. Its outputs stay the
		// profile's: a wider workgroup would leave more of the last
		// columns of a bank's sheets idle.
		s := p.Rows / f.block
		q.Rows, q.Warps, q.Resident = f.block, [2]int{p.Warps[0] / s, p.Warps[1]}, max(p.Resident, 1)*s
		if q.Warps[0] == 0 || q.Rows%(q.Warps[0]*d.Matrix[0]) != 0 || q.Cols%(q.Warps[1]*d.Matrix[1]) != 0 {
			return p, f, false
		}
	}
	// A device whose workgroup memory does not hold the stage takes
	// half the rows a workgroup, and half the warps along them, until
	// it does.
	for d.Shared > 0 && 4*matrixStage(q.Rows, q.Cols, f.depth, q.Warps[0], d) > d.Shared &&
		q.Warps[0] > 1 && q.Warps[0]%2 == 0 && q.Rows%2 == 0 {
		q.Rows, q.Warps[0] = q.Rows/2, q.Warps[0]/2
	}
	if d.Shared > 0 && 4*matrixStage(q.Rows, q.Cols, f.depth, q.Warps[0], d) > d.Shared {
		return p, f, false
	}
	switch {
	case q.Split > 0 && k.Loops[0].Trip%(f.depth*q.Split) != 0:
		q.Split = 1
	case q.Split == 0 && d.Units > 0:
		q.Split, _ = q.splits(k, f.depth, d)
	case q.Split == 0:
		q.Split = 1
	}
	return q, f, true
}

// rowCost is the words a row factor costs its projections: written as
// halves by a kernel of its own -- a launch, its inputs read once and its
// halves written, then each workgroup reading its rows' halves -- or
// converted in each workgroup's stage, each reading its rows' inputs.
type rowCost struct {
	halves, convert int
	// elem is the words an element of the factor reads along the row and
	// k, along those along k.
	elem, along int
}

func newRowCost(k *Kernel, f bilinear, d Device) *rowCost {
	row := k.dependsOn(func(nd Node) bool { return nd.Op == OpCoord && nd.A == 0 })
	iter := k.dependsOn(func(nd Node) bool { return nd.Op == OpIter && nd.At == 0 })
	c := &rowCost{}
	for i, r := range k.reaches(f.row) {
		if !r || k.Nodes[i].Op != OpLoad || !iter[i] {
			continue
		}
		c.along++
		if row[i] {
			c.elem++
		}
	}
	RK := k.Space[0] * k.Loops[0].Trip
	if f.index > 0 {
		RK = f.rows * k.Loops[0].Trip
	}
	c.halves = d.Launch + RK*c.elem + RK/2
	return c
}

// add is a projection of k, cols outputs a workgroup, reading the factor.
func (c *rowCost) add(k *Kernel, cols int) {
	RK, groups := k.Space[0]*k.Loops[0].Trip, (k.Space[1]+cols-1)/cols
	c.halves += RK / 2 * groups
	c.convert += RK * c.elem * groups
}

// regs is the registers a thread of q holds besides regs' where its stage
// converts the factor: the inputs' words a trip ahead, not its halves.
func (c *rowCost) regs(q Matrix, depth int) int {
	threads := 32 * q.Warps[0] * q.Warps[1]
	return max(q.Rows*depth*(2*c.along-1)/(2*threads), 0)
}

// maxThreadRegs is the 32-bit registers a GPU thread addresses at most.
const maxThreadRegs = 255

// regs is a thread's registers at p over depth elements of k a trip: its
// warp's accumulator tiles, one trip's fragments of them (one eight of k's
// in the fragment form), and the words it stages a trip ahead.
func (p Matrix) regs(depth int, d Device) int {
	tm, tn, tk := d.Matrix[0], d.Matrix[1], d.Matrix[2]
	mi, nj := p.Rows/p.Warps[0]/tm, p.Cols/p.Warps[1]/tn
	threads := 32 * p.Warps[0] * p.Warps[1]
	if fragmentForm(d) {
		return mi*nj*tm*tn/32 + (mi+nj)*tm*tk/64 + (p.Rows+p.Cols)*depth/(2*threads)
	}
	return mi*nj*tm*tn/32 + (mi+nj)*(depth/tk)*tm*tk/32 + (p.Rows+p.Cols)*depth/(2*threads)
}

// regBudget is the registers a thread of p's workgroup holds on d.
func (p Matrix) regBudget(d Device) int {
	if d.Regs == 0 || d.Units == 0 {
		return maxThreadRegs
	}
	return min(maxThreadRegs, d.Regs*d.Slots/d.Units/(32*p.Warps[0]*p.Warps[1]))
}

// matrixStage is the words of a matrix workgroup's stage: two stages of
// TR rows of A and TN of B at a depth of KC, padded (form), or after the
// trips the sums, WR warps' rows of a tile of TN; the fragment form's two
// stages unpadded.
func matrixStage(TR, TN, KC, WR int, d Device) int {
	if fragmentForm(d) {
		return max(TR*KC+TN*KC, WR*16*(TN+sumsPad(TN)))
	}
	return max(2*(TR*(KC+8)/2+TN*(KC+4)/2), WR*d.Matrix[0]*TN)
}

// fragmentForm reports whether Matrix lays out d's own instruction's lanes.
func fragmentForm(d Device) bool { return d.Fragment == ir.MMAVolta() }

// bilinear is a reduction's parts the matrix unit takes: its sum, its
// factor of the row and its factor of the output, and the elements of k a
// trip stages.
type bilinear struct {
	fold, row, out, depth int
	// block, where set, is the rows of the blocks the output's factor reads
	// the row in: it is a factor of the block of rows alone.
	block int
	// index, where set, is the node the row's factor reads its rows at,
	// of rows of them (rowIndex).
	index, rows int
	// fma is a term fused with its step.
	fma bool
}

func (p Matrix) bilinear(k *Kernel, d Device) (bilinear, bool) {
	var f bilinear
	tm, tn, tk := d.Matrix[0], d.Matrix[1], d.Matrix[2]
	if k.Work || k.Group > 0 || k.Clamp != 0 || k.Tile != 0 || k.Positions != nil || k.Heads != nil ||
		k.Quarter != nil || k.Kind != PrimReduce && k.Kind != PrimContract || len(k.Space) != 2 || len(k.Loops) != 1 ||
		k.Space[0] < tm || p.Rows%(p.Warps[0]*tm) != 0 || p.Cols%(p.Warps[1]*tn) != 0 {
		return f, false
	}
	if l := k.Loops[0]; l.Count != 0 || l.Range || l.Spread || l.Lanes != 0 {
		return f, false
	}
	for depth := p.Depth; depth >= tk; depth /= 2 {
		if depth%tk == 0 && k.Loops[0].Trip%depth == 0 && p.regs(depth, d) <= p.regBudget(d) {
			f.depth = depth
			break
		}
	}
	if f.depth == 0 {
		return f, false
	}
	g, ok := k.bilinearOf(false)
	if !ok || k.readsFloatWeight(g.out) {
		return f, false
	}
	g.depth = f.depth
	if g.block > 0 && g.block < d.Matrix[0] {
		return f, false
	}
	return g, true
}

// bilinearOf is k's one fold over its one loop whose term is a product of
// a factor of the row and one of the output, each of k alone.
func (k *Kernel) bilinearOf(fused bool) (bilinear, bool) {
	var f bilinear
	folds := k.foldsOf(0)
	if len(folds) != 1 {
		return f, false
	}
	f.fold = folds[0]
	fd := k.Nodes[f.fold]
	if fd.Type != ir.F32 || k.Nodes[fd.B].Op != OpConst || k.Nodes[fd.B].Imm != 0 {
		return f, false
	}
	step := k.Nodes[fd.A]
	var x, y int
	if f.fma = fused && step.Op == OpFma && step.Type == ir.F32 && step.C == f.fold; f.fma {
		x, y = step.A, step.B
	} else {
		term := step.B
		if step.B == f.fold {
			term = step.A
		} else if step.A != f.fold {
			return f, false
		}
		if step.Op != OpAdd || step.Type != ir.F32 || k.Nodes[term].Op != OpMul || k.Nodes[term].Type != ir.F32 {
			return f, false
		}
		x, y = k.Nodes[term].A, k.Nodes[term].B
	}
	row := k.dependsOn(func(nd Node) bool { return nd.Op == OpCoord && nd.A == 0 })
	// blockOf is the rows of the block node i reads the row in -- the row
	// shifted right by a constant -- 0 where it is not one.
	blockOf := func(i int) int {
		nd := k.Nodes[i]
		if nd.Op != OpShr || nd.Type != ir.U32 || k.Nodes[nd.A].Op != OpCoord || k.Nodes[nd.A].A != 0 || k.Nodes[nd.B].Op != OpConst || k.Nodes[nd.B].Imm > 16 {
			return 0
		}
		return 1 << k.Nodes[nd.B].Imm
	}
	// byRow is what reaches the row other than through a block of it.
	byRow := k.dependsOnCut(func(nd Node) bool { return nd.Op == OpCoord && nd.A == 0 }, func(i int) bool { return blockOf(i) > 0 })
	out := k.dependsOn(func(nd Node) bool { return nd.Op == OpCoord && nd.A == 1 })
	folded := k.dependsOn(func(nd Node) bool { return nd.Op == OpFold })
	if row[y] {
		x, y = y, x
	}
	if !row[x] || out[x] || !out[y] || byRow[y] || folded[x] || folded[y] {
		return f, false
	}
	for _, s := range k.Stores {
		if s.At != -1 || s.Barrier || s.Run || s.Tile {
			return f, false
		}
	}
	f.row, f.out = x, y
	f.index, f.rows = k.rowIndex(x)
	if row[y] {
		reach := k.reaches(y)
		for i := range k.Nodes {
			if b := blockOf(i); b > 0 && reach[i] && (f.block == 0 || b < f.block) {
				f.block = b
			}
		}
		if f.block == 0 {
			return f, false
		}
	}
	return f, true
}

// dependsOn marks the nodes of k that reach a node pred holds for.
func (k *Kernel) dependsOn(pred func(Node) bool) []bool {
	return k.dependsOnCut(pred, func(int) bool { return false })
}

// reaches marks the nodes node i reads, i among them.
func (k *Kernel) reaches(i int) []bool {
	r := make([]bool, len(k.Nodes))
	var walk func(i int)
	walk = func(i int) {
		if r[i] {
			return
		}
		r[i] = true
		nd := k.Nodes[i]
		for _, a := range k.Operands(&nd) {
			walk(*a)
		}
	}
	walk(i)
	return r
}

// dependsOnCut marks the nodes of k that reach a node pred holds for other
// than through a node cut holds for.
func (k *Kernel) dependsOnCut(pred func(Node) bool, cut func(int) bool) []bool {
	dep := make([]bool, len(k.Nodes))
	for changed := true; changed; {
		changed = false
		for i := range k.Nodes {
			if dep[i] {
				continue
			}
			if cut(i) {
				continue
			}
			nd := k.Nodes[i]
			d := pred(nd)
			for _, a := range k.Operands(&nd) {
				d = d || dep[*a]
			}
			if d {
				dep[i], changed = true, true
			}
		}
	}
	return dep
}

// readsFloatWeight reports whether node i reaches a load of a weight held
// as float32.
func (k *Kernel) readsFloatWeight(i int) bool {
	reach := k.dependsOn(func(nd Node) bool {
		if nd.Op != OpLoad {
			return false
		}
		o := k.Bufs[nd.Buf].Operand
		return o.Kind == OperandWeight && o.Format == jlm.TypeF32
	})
	return reach[i]
}

// rowIndex is the node row factor x reads the rows of a buffer at, and
// the buffer's rows, where everything of x's the row reaches goes through
// it and it is not the row itself: a grouped row's token. 0, 0 where not.
func (k *Kernel) rowIndex(x int) (int, int) {
	isRow := func(nd Node) bool { return nd.Op == OpCoord && nd.A == 0 }
	row := k.dependsOn(isRow)
	for i, r := range k.reaches(x) {
		nd := k.Nodes[i]
		sh := k.Bufs[nd.Buf].Shape
		if !r || nd.Op != OpLoad || len(nd.Ax) < 2 || len(sh) != len(nd.Ax) || !row[nd.Ax[0]] || isRow(k.Nodes[nd.Ax[0]]) {
			continue
		}
		ri := nd.Ax[0]
		if !k.dependsOnCut(isRow, func(i int) bool { return i == ri })[x] {
			return ri, sh[0]
		}
	}
	return 0, 0
}

// halves is a kernel writing the factor of the row of k as binary16 pairs,
// a word of a row a thread, into a buffer of R rows of K/2 words, and that
// buffer: the factor is rounded once, not once a workgroup of outputs, and a
// trip stages it as words.
func halves(k *Kernel, f bilinear) (*Kernel, Buf) {
	U := ir.U32
	R, W := k.Space[0], k.Loops[0].Trip/2
	if f.index > 0 {
		R = f.rows
	}
	// Named by what k stores, as a split names its planes: two
	// projections of one form hold their own.
	out := k.Bufs[k.Stores[0].Buf].Operand
	b := Buf{Kind: BufOut, Elem: U, Shape: []int{R, W},
		Operand: Operand{Kind: OperandScratch, Name: out.Name + ".halves", Elem: U, Temp: true, Of: out.Of}}
	h := &Kernel{Name: k.Name + ".halves", Kind: PrimMap, Space: []int{R, W}, Part: k.Part, Apart: k.Apart, Of: k.Of, Capacity: k.Capacity}
	if k.Rows > 0 && f.index == 0 {
		h.Rows = copyInto(h, k, k.Rows-1, func(int, Node) (int, bool) { return 0, false }) + 1
	}
	r := h.add(Node{Op: OpCoord, A: 0, Type: U})
	w := h.add(Node{Op: OpCoord, A: 1, Type: U})
	elem := func(at int) int {
		return h.un(OpHalf, U, copyInto(h, k, f.row, func(i int, nd Node) (int, bool) {
			switch {
			case f.index > 0 && i == f.index:
				return r, true
			case nd.Op == OpCoord && nd.A == 0:
				return r, true
			case nd.Op == OpIter && nd.At == 0:
				return at, true
			}
			return 0, false
		}))
	}
	e := h.bin(OpMul, U, w, h.u32(2))
	val := h.bin(OpOr, U, elem(e), h.bin(OpShl, U, elem(h.bin(OpAdd, U, e, h.u32(1))), h.u32(16)))
	h.Stores = append(h.Stores, Store{Buf: h.buf(b), Val: val, At: -1, Ax: []int{r, w}})
	return h, b
}

// form is k as the matrix unit's workgroups, the factor of its row
// written as binary16 pairs first.
func (p Matrix) form(k *Kernel, d Device, f bilinear, made map[string]Buf) []*Kernel {
	U, F := ir.U32, ir.F32
	tm, tn, tk := d.Matrix[0], d.Matrix[1], d.Matrix[2]
	TR, TN, KC := p.Rows, p.Cols, f.depth
	WR, WN := p.Warps[0], p.Warps[1]
	R, N := k.Space[0], k.Space[1]
	m := k.clone()
	m.Name, m.Kind = k.Name+".matrix", PrimMatrix
	m.Matrix = [2]int{N, k.Loops[0].Trip}
	m.Nodes, m.Loops, m.Stores, m.Rows = nil, nil, nil, 0
	S, NB := max(p.Split, 1), (N+TN-1)/TN
	m.Space = []int{(R + TR - 1) / TR, NB * S}
	m.Work, m.Group = true, 32*WR*WN
	shared := func(name string, e ir.Type, n int) int {
		return m.buf(Buf{Kind: BufShared, Elem: e, Size: n, Operand: Operand{Kind: OperandScratch, Name: k.Name + "." + name}})
	}
	// SKA and SKB are a staged row's elements of A and of B, padded; G the
	// elements a thread stages at a time. A's rows start eight elements
	// apart, so a lane's run of eight is one access, and the rows one
	// access of its fragment reads fall in different banks; B's start four
	// apart, which keeps the rows n and n+8 a fragment load reads together
	// in different banks.
	const G = 8
	SKA, SKB := KC+8, KC+4
	frag := fragmentForm(d)
	if frag {
		SKA, SKB = KC, KC
	}
	// A stage holds A's rows, then B's at bAt words; the array holds two,
	// trip t multiplying stage t&1 while it fills the other with trip
	// t+1's factors, and after the trips the sums.
	bAt := TR * SKA / 2
	half := bAt + TN*SKB/2
	st := shared("stage", U, matrixStage(TR, TN, KC, WR, d))
	if units := max(d.Units, 1); p.Resident*m.Group <= d.Slots/units && (d.Shared == 0 || p.Resident*4*m.Bufs[st].Size <= d.Shared) {
		m.Resident = p.Resident
	}

	last := m.u32(R - 1)
	if k.Capacity > 0 {
		last = m.bin(OpSub, U, m.liveRows(k.Capacity), m.u32(1))
	}
	lastOut := m.u32(N - 1)
	r0 := m.bin(OpMul, U, m.add(Node{Op: OpCoord, A: 0, Type: U}), m.u32(TR))
	c1 := m.add(Node{Op: OpCoord, A: 1, Type: U})
	T := k.Loops[0].Trip / KC / S
	n0, k0 := m.bin(OpMul, U, c1, m.u32(TN)), m.u32(0)
	if S > 1 {
		n0 = m.bin(OpMul, U, m.bin(OpRem, U, c1, m.u32(NB)), m.u32(TN))
		k0 = m.bin(OpMul, U, m.bin(OpDiv, U, c1, m.u32(NB)), m.u32(T*KC))
	}
	thread := m.add(Node{Op: OpThread, Type: U})
	warp := m.bin(OpShr, U, thread, m.u32(5))
	wr0 := m.bin(OpMul, U, m.bin(OpDiv, U, warp, m.u32(WN)), m.u32(TR/WR))
	wn0 := m.bin(OpMul, U, m.bin(OpRem, U, warp, m.u32(WN)), m.u32(TN/WN))
	// spread is a loop dealing n trips over the workgroup, and its trip,
	// a surplus thread's held to the last.
	spread := func(n, parent int) (int, int) {
		l := m.loop(n, parent)
		m.Loops[l].Spread, m.Loops[l].Uniform = true, n%m.Group == 0
		it := m.add(Node{Op: OpIter, At: l, Type: U})
		if n%m.Group != 0 {
			it = m.bin(OpMin, U, it, m.u32(n-1))
		}
		return l, it
	}

	// stage stores, at loop l, the rows of factor at coordinate axis base +
	// the row, held to most, from element kc of k into the stage at word
	// into: G elements a share, binary16 pairs a word. A thread's shares
	// are its own, not a loop's, so what a trip's shares load is a value of
	// the trip a later pass can load a trip ahead.
	W := k.Loops[0].Trip / 2
	var h *Kernel
	hw := -1
	if p.Convert < 0 {
		var hb Buf
		h, hb = halves(k, f)
		sig := h.computes()
		if b, ok := made[sig]; ok {
			h, hb = nil, b
		} else {
			made[sig] = hb
		}
		hb.Kind, hb.Shape, hb.Extent = BufIn, nil, hb.Shape[0]*hb.Shape[1]
		hw = m.buf(hb)
	}
	stage := func(l, kc, into, rows, axis, base, most, factor int) {
		SK := SKA
		if axis == 1 {
			SK = SKB
		}
		n := rows * KC / G
		for j := 0; j*m.Group < n; j++ {
			w := m.bin(OpAdd, U, thread, m.u32(j*m.Group))
			whole := (j+1)*m.Group <= n
			if !whole {
				w = m.bin(OpMin, U, w, m.u32(n-1))
			}
			row, g := m.share(thread, w, j*m.Group, KC/G, whole)
			if frag && axis == 1 {
				// One row's runs of k a thread: its scales decode once.
				g, row = m.share(thread, w, j*m.Group, rows, whole)
			}
			c := m.bin(OpMin, U, m.bin(OpAdd, U, base, row), most)
			kg := m.bin(OpAdd, U, kc, m.bin(OpMul, U, g, m.u32(G)))
			if frag {
				g = m.bin(OpXor, U, g, m.swizzle(row, KC/G))
			}
			word := m.bin(OpAdd, U, m.bin(OpMul, U, row, m.u32(SK/2)), m.bin(OpAdd, U, m.bin(OpMul, U, g, m.u32(G/2)), into))
			copyAt := func(x, at int) int {
				return copyInto(m, k, x, func(i int, nd Node) (int, bool) {
					switch {
					case nd.Op == OpCoord && nd.A == axis:
						return c, true
					case nd.Op == OpCoord && nd.A == 0:
						// The output's factor reads the row in its block:
						// the workgroup's.
						return r0, true
					case nd.Op == OpIter && nd.At == 0:
						return at, true
					}
					return 0, false
				})
			}
			elem := func(at int) int { return m.un(OpHalf, U, copyAt(factor, at)) }
			if axis == 0 && hw >= 0 {
				ri := c
				if f.index > 0 {
					ri = copyAt(f.index, kg)
				}
				at := m.bin(OpAdd, U, m.bin(OpMul, U, ri, m.u32(W)), m.bin(OpShr, U, kg, m.u32(1)))
				for e := 0; e < G/2; e++ {
					v := m.add(Node{Op: OpLoad, Type: U, Buf: hw, A: m.bin(OpAdd, U, at, m.u32(e))})
					m.Stores = append(m.Stores, Store{Buf: st, Ix: m.bin(OpAdd, U, word, m.u32(e)), Val: v, At: l})
				}
				continue
			}
			for e := 0; e < G; e += 2 {
				lo := elem(kg)
				if e > 0 {
					lo = elem(m.bin(OpAdd, U, kg, m.u32(e)))
				}
				hi := m.bin(OpShl, U, elem(m.bin(OpAdd, U, kg, m.u32(e+1))), m.u32(16))
				ix := m.bin(OpAdd, U, word, m.u32(e/2))
				m.Stores = append(m.Stores, Store{Buf: st, Ix: ix, Val: m.bin(OpOr, U, lo, hi), At: l})
			}
		}
	}
	stage(-1, k0, m.u32(0), TR, 0, r0, last, f.row)
	stage(-1, k0, m.u32(bAt), TN, 1, n0, lastOut, f.out)
	m.Stores = append(m.Stores, Store{Barrier: true, At: -1})
	P := m.loop(T, -1)
	m.Loops[P].Uniform = true
	if k.Rows > 0 {
		// A workgroup whose rows are all past the live ones (Kernel.Rows)
		// multiplies nothing: its sums are rows no one reads.
		live := copyInto(m, k, k.Rows-1, func(int, Node) (int, bool) { return 0, false })
		trips := m.add(Node{Op: OpSelect, Type: U, A: m.add(Node{Op: OpLess, Type: ir.Pred, A: r0, B: live}), B: m.u32(T), C: m.u32(0)})
		m.Loops[P].Count = trips + 1
	}
	it := m.add(Node{Op: OpIter, At: P, Type: U})
	next := m.bin(OpAdd, U, it, m.u32(1))
	kn := m.bin(OpAdd, U, m.bin(OpMul, U, m.bin(OpMin, U, next, m.u32(T-1)), m.u32(KC)), k0)
	into := m.bin(OpMul, U, m.bin(OpAnd, U, next, m.u32(1)), m.u32(half))
	stage(P, kn, into, TR, 0, r0, last, f.row)
	stage(P, kn, m.bin(OpAdd, U, into, m.u32(bAt)), TN, 1, n0, lastOut, f.out)
	m.Stores = append(m.Stores, Store{Barrier: true, At: P})
	// emit is k's stores of sum at row r and output n, in loop at; or
	// where k is cut, the sum to its segment's plane.
	var planes Operand
	pb := -1
	if S > 1 {
		out := k.Bufs[k.Stores[0].Buf].Operand
		planes = Operand{Kind: OperandScratch, Name: out.Name + ".segments", Of: out.Of}
		pb = m.buf(Buf{Kind: BufOut, Elem: F, Operand: planes, Shape: []int{S, R, N}})
	}
	emit := func(r, n, sum, at int) {
		if pb >= 0 {
			m.Stores = append(m.Stores, Store{Buf: pb, Val: sum, At: at, Ax: []int{m.bin(OpDiv, U, c1, m.u32(NB)), r, n}})
			return
		}
		k.storesOf(m, f.fold, r, n, sum, at)
	}
	formed := []*Kernel{m}
	if h != nil {
		formed = []*Kernel{h, m}
	}
	if S > 1 {
		formed = append(formed, combine(k, planes, S))
	}
	if frag {
		p.fragments(m, f, st, P, it, half, bAt, r0, n0, last, lastOut, wr0, wn0, thread, emit)
		return formed
	}
	// from is the stage trip t multiplies, in binary16 elements.
	from := m.bin(OpMul, U, m.bin(OpAnd, U, it, m.u32(1)), m.u32(2*half))

	aT := ir.TileType{Rows: tm, Cols: tk, Elem: ir.TileF16, Use: ir.TileA}
	bT := ir.TileType{Rows: tk, Cols: tn, Elem: ir.TileF16, Use: ir.TileB}
	cT := ir.TileType{Rows: tm, Cols: tn, Elem: ir.TileF32, Use: ir.TileAcc}
	mi, nj, ks := TR/WR/tm, TN/WN/tn, KC/tk
	at := func(base, i, unit, stride, s int) int {
		return m.bin(OpAdd, U, m.bin(OpMul, U, m.bin(OpAdd, U, base, m.u32(i*unit)), m.u32(stride)), m.u32(s))
	}
	// A staged view's rows start four elements apart and its offsets are
	// multiples of four, A's of eight: aligned, A's to eight.
	tile := func(t ir.TileType, off int, col bool) int {
		imm, sk := int64(2|4), SKA
		if col {
			imm, sk = 2|1, SKB
		}
		return m.add(Node{Op: OpTileLoad, Type: ir.Void, Buf: st, A: off, B: m.u32(sk), Imm: imm, At: P + 1, Tile: &t})
	}
	a := make([][]int, mi)
	for i := range a {
		for s := 0; s < ks; s++ {
			a[i] = append(a[i], tile(aT, m.bin(OpAdd, U, at(wr0, i, tm, SKA, s*tk), from), false))
		}
	}
	b := make([][]int, nj)
	for j := range b {
		for s := 0; s < ks; s++ {
			b[j] = append(b[j], tile(bT, m.bin(OpAdd, U, at(wn0, j, tn, SKB, s*tk), m.bin(OpAdd, U, from, m.u32(2*bAt))), true))
		}
	}
	zero := m.add(Node{Op: OpTileSplat, Type: ir.Void, A: m.f32(0), Tile: &cT})
	sums := make([][]int, mi)
	for i := range sums {
		for j := 0; j < nj; j++ {
			fold := m.add(Node{Op: OpFold, Type: ir.Void, At: P, B: zero, Tile: &cT})
			acc := fold
			for s := 0; s < ks; s++ {
				acc = m.add(Node{Op: OpTileMMA, Type: ir.Void, A: a[i][s], B: b[j][s], C: acc, Tile: &cT})
			}
			m.Nodes[fold].A = acc
			sums[i] = append(sums[i], fold)
		}
	}
	// The sums pass through the stage a row of tiles at a time: every
	// warp's i-th, WR*tm rows of the block, behind a barrier, then the
	// kernel's own stores an element a thread.
	wq := m.bin(OpDiv, U, warp, m.u32(WN))
	for i := 0; i < mi; i++ {
		m.Stores = append(m.Stores, Store{Barrier: true, At: -1})
		for j := 0; j < nj; j++ {
			off := m.bin(OpAdd, U, m.bin(OpMul, U, wq, m.u32(tm*TN)), m.bin(OpAdd, U, wn0, m.u32(j*tn)))
			m.Stores = append(m.Stores, Store{Buf: st, Ix: off, Val: sums[i][j], At: -1, Tile: true, Stride: m.u32(TN)})
		}
		m.Stores = append(m.Stores, Store{Barrier: true, At: -1})
		E, e := spread(WR*tm*TN, -1)
		lr := m.bin(OpDiv, U, e, m.u32(TN))
		row := m.bin(OpAdd, U, m.bin(OpMul, U, m.bin(OpDiv, U, lr, m.u32(tm)), m.u32(TR/WR)), m.bin(OpAdd, U, m.bin(OpRem, U, lr, m.u32(tm)), m.u32(i*tm)))
		r := m.bin(OpMin, U, m.bin(OpAdd, U, r0, row), last)
		n := m.bin(OpMin, U, m.bin(OpAdd, U, n0, m.bin(OpRem, U, e, m.u32(TN))), lastOut)
		emit(r, n, m.add(Node{Op: OpLoad, Type: F, Buf: st, A: e}), E)
	}
	return formed
}

// share is w/per and w%per for w = thread + at, at a multiple of the
// workgroup, in terms of the thread alone where per and the workgroup
// divide one another, so every share of one row reads one address.
func (m *Kernel) share(thread, w, at, per int, whole bool) (int, int) {
	U := ir.U32
	switch {
	case whole && per%m.Group == 0:
		return m.u32(at / per), m.bin(OpAdd, U, thread, m.u32(at%per))
	case whole && m.Group%per == 0:
		return m.bin(OpAdd, U, m.bin(OpDiv, U, thread, m.u32(per)), m.u32(at/per)), m.bin(OpRem, U, thread, m.u32(per))
	}
	return m.bin(OpDiv, U, w, m.u32(per)), m.bin(OpRem, U, w, m.u32(per))
}

// storesOf copies k's stores into m at row r and output n, the value of
// k's fold the sum, in loop at.
func (k *Kernel) storesOf(m *Kernel, fold, r, n, sum, at int) {
	leaf := func(i int, nd Node) (int, bool) {
		switch {
		case i == fold:
			return sum, true
		case nd.Op == OpCoord && nd.A == 0:
			return r, true
		case nd.Op == OpCoord && nd.A == 1:
			return n, true
		}
		return 0, false
	}
	for _, s := range k.Stores {
		c := s
		c.At = at
		c.Val = copyInto(m, k, s.Val, leaf)
		if s.Ax == nil {
			c.Ix = copyInto(m, k, s.Ix, leaf)
		}
		c.Ax = make([]int, len(s.Ax))
		for x, ax := range s.Ax {
			c.Ax[x] = copyInto(m, k, ax, leaf)
		}
		m.Stores = append(m.Stores, c)
	}
}

// combine is k's stores of the sum of S planes of its sums, in order: a
// map, which no later pass cuts again.
func combine(k *Kernel, planes Operand, S int) *Kernel {
	U, F := ir.U32, ir.F32
	R, N := k.Space[0], k.Space[1]
	c := k.clone()
	c.Name, c.Kind, c.Nodes, c.Loops, c.Stores, c.Rows = k.Name+".combine", PrimMap, nil, nil, nil, 0
	if k.Rows > 0 {
		c.Rows = copyInto(c, k, k.Rows-1, func(int, Node) (int, bool) { return 0, false }) + 1
	}
	r := c.add(Node{Op: OpCoord, A: 0, Type: U})
	n := c.add(Node{Op: OpCoord, A: 1, Type: U})
	in := c.buf(Buf{Kind: BufIn, Elem: F, Operand: planes, Shape: []int{S, R, N}})
	sum := c.add(Node{Op: OpLoad, Type: F, Buf: in, Ax: []int{c.u32(0), r, n}})
	for s := 1; s < S; s++ {
		sum = c.bin(OpAdd, F, sum, c.add(Node{Op: OpLoad, Type: F, Buf: in, Ax: []int{c.u32(s), r, n}}))
	}
	k.storesOf(c, k.foldsOf(0)[0], r, n, sum, -1)
	return c
}

// swizzle is what a row xors its 16-byte chunk index with, so the rows one
// fragment or staging access names fall in different banks.
func (m *Kernel) swizzle(row, chunks int) int {
	U := ir.U32
	if nd := m.Nodes[row]; nd.Op == OpAdd && m.Nodes[nd.B].Op == OpConst && m.Nodes[nd.B].Imm%16 == 0 {
		row = nd.A
	}
	bit := func(b int) int { return m.bin(OpAnd, U, m.bin(OpShr, U, row, m.u32(b)), m.u32(1)) }
	bits := []int{bit(0), bit(1), m.bin(OpXor, U, bit(2), bit(3))}
	s := m.u32(0)
	for i, b := range bits[max(0, 3-bitsOf(chunks)):] {
		s = m.bin(OpOr, U, s, m.bin(OpShl, U, b, m.u32(i)))
	}
	return s
}

// fragments is form's trips and sums on ir.MMAVolta: blocks of 16 x 16,
// quad-pair q taking quarter (q>>1, q&1), the sums through the stage.
func (p Matrix) fragments(m *Kernel, f bilinear, st, P, it, half, bAt, r0, n0, last, lastOut, wr0, wn0, thread int, emit func(r, n, sum, at int)) {
	U, F := ir.U32, ir.F32
	TR, TN, KC := p.Rows, p.Cols, f.depth
	WR, WN := p.Warps[0], p.Warps[1]
	mi, nj, C := TR/WR/16, TN/WN/16, KC/8
	from := m.bin(OpMul, U, m.bin(OpAnd, U, it, m.u32(1)), m.u32(half))
	lane := m.bin(OpAnd, U, thread, m.u32(31))
	pr := m.bin(OpOr, U, m.bin(OpAnd, U, lane, m.u32(3)), m.bin(OpAnd, U, m.bin(OpShr, U, lane, m.u32(2)), m.u32(4)))
	rowA := m.bin(OpOr, U, m.bin(OpAnd, U, lane, m.u32(8)), pr)
	colB := m.bin(OpOr, U, m.bin(OpAnd, U, m.bin(OpShl, U, lane, m.u32(1)), m.u32(8)), pr)
	// Rows before a lane's own are multiples of 16: the swizzle is its own.
	swA, swB := m.swizzle(rowA, C), m.swizzle(colB, C)
	rowAt := func(base, lr int) int {
		return m.bin(OpMul, U, m.bin(OpAdd, U, base, lr), m.u32(KC/2))
	}
	aAt, bAt0 := rowAt(wr0, rowA), m.bin(OpAdd, U, rowAt(wn0, colB), m.u32(bAt))
	fragment := func(base, sw, i, ch int) [4]int {
		at := m.bin(OpAdd, U, m.bin(OpAdd, U, from, base), m.bin(OpMul, U, m.bin(OpXor, U, m.u32(ch), sw), m.u32(4)))
		var w [4]int
		for e := range w {
			w[e] = m.add(Node{Op: OpLoad, Type: U, Buf: st, A: at, Imm: int64(16*i*KC/2 + e)})
		}
		return w
	}
	zero := m.f32(0)
	acc := make([][][]int, mi)
	folds := make([][][]int, mi)
	for i := range acc {
		acc[i], folds[i] = make([][]int, nj), make([][]int, nj)
		for j := range acc[i] {
			for x := 0; x < 8; x++ {
				fd := m.add(Node{Op: OpFold, Type: F, At: P, B: zero})
				folds[i][j] = append(folds[i][j], fd)
			}
			acc[i][j] = folds[i][j]
		}
	}
	for ch := 0; ch < C; ch++ {
		a, b := make([][4]int, mi), make([][4]int, nj)
		for i := range a {
			a[i] = fragment(aAt, swA, i, ch)
		}
		for j := range b {
			b[j] = fragment(bAt0, swB, j, ch)
		}
		for x := 0; x < 2; x++ {
			for i := 0; i < mi; i++ {
				for j := 0; j < nj; j++ {
					ax := append([]int{a[i][2*x], a[i][2*x+1], b[j][2*x], b[j][2*x+1]}, acc[i][j]...)
					mma := m.add(Node{Op: OpMMA, Type: ir.Void, Ax: ax})
					out := make([]int, 8)
					for y := range out {
						out[y] = m.add(Node{Op: OpMMAGet, Type: F, A: mma, Imm: int64(y)})
					}
					acc[i][j] = out
				}
			}
		}
	}
	for i := range acc {
		for j := range acc[i] {
			for x, fd := range folds[i][j] {
				m.Nodes[fd].A = acc[i][j][x]
			}
		}
	}
	dr := m.bin(OpOr, U, m.bin(OpAnd, U, lane, m.u32(9)), m.bin(OpAnd, U, m.bin(OpShr, U, lane, m.u32(2)), m.u32(4)))
	dc := m.bin(OpOr, U, m.bin(OpAnd, U, lane, m.u32(2)), m.bin(OpAnd, U, m.bin(OpShl, U, lane, m.u32(1)), m.u32(8)))
	// Each warp's i-th row of blocks through the stage, then the kernel's
	// stores an element a thread, a row of outputs a run of threads.
	SN := TN + sumsPad(TN)
	warp := m.bin(OpShr, U, thread, m.u32(5))
	wq := m.bin(OpDiv, U, warp, m.u32(WN))
	lrow := m.bin(OpAdd, U, m.bin(OpMul, U, wq, m.u32(16)), dr)
	lcol := m.bin(OpAdd, U, wn0, dc)
	for i := 0; i < mi; i++ {
		m.Stores = append(m.Stores, Store{Barrier: true, At: -1})
		for j := 0; j < nj; j++ {
			for y := 0; y < 8; y++ {
				at := m.bin(OpAdd, U, m.bin(OpMul, U, m.bin(OpAdd, U, lrow, m.u32(2*((y>>1)&1))), m.u32(SN)),
					m.bin(OpAdd, U, lcol, m.u32(16*j+(y&1)+4*(y>>2))))
				m.Stores = append(m.Stores, Store{Buf: st, Ix: at, Val: folds[i][j][y], At: -1})
			}
		}
		m.Stores = append(m.Stores, Store{Barrier: true, At: -1})
		n := WR * 16 * TN
		E := m.loop(n, -1)
		m.Loops[E].Spread, m.Loops[E].Uniform = true, n%m.Group == 0
		e := m.add(Node{Op: OpIter, At: E, Type: U})
		if n%m.Group != 0 {
			e = m.bin(OpMin, U, e, m.u32(n-1))
		}
		lr, c := m.bin(OpDiv, U, e, m.u32(TN)), m.bin(OpRem, U, e, m.u32(TN))
		row := m.bin(OpAdd, U, m.bin(OpMul, U, m.bin(OpDiv, U, lr, m.u32(16)), m.u32(TR/WR)), m.bin(OpAdd, U, m.bin(OpRem, U, lr, m.u32(16)), m.u32(16*i)))
		r := m.bin(OpMin, U, m.bin(OpAdd, U, r0, row), last)
		nn := m.bin(OpMin, U, m.bin(OpAdd, U, n0, c), lastOut)
		sum := m.add(Node{Op: OpLoad, Type: F, Buf: st, A: m.bin(OpAdd, U, m.bin(OpMul, U, lr, m.u32(SN)), c)})
		emit(r, nn, sum, E)
	}
}

// sumsPad is the words a staged row of TN sums is padded by, the least
// that spreads one store of a fragment's sums over the most banks.
func sumsPad(TN int) int {
	best, least := 0, 33
	for pad := 0; pad < 32; pad++ {
		var banks [32]int
		worst := 0
		for l := 0; l < 32; l++ {
			r := l&9 | l>>2&4
			c := l&2 | l<<1&8
			b := (r*(TN+pad) + c) % 32
			banks[b]++
			worst = max(worst, banks[b])
		}
		if worst < least {
			best, least = pad, worst
		}
	}
	return best
}

// computes is what a halves kernel writes, its output aside: its space and
// rows, its nodes and the operands it reads.
func (h *Kernel) computes() string {
	in := make([]Operand, 0, len(h.Bufs))
	for i, b := range h.Bufs {
		if i != h.Stores[0].Buf {
			in = append(in, b.Operand)
		}
	}
	return fmt.Sprint(h.Space, h.Rows, h.Capacity, h.Part, h.Apart, h.Of, h.Nodes, in)
}
