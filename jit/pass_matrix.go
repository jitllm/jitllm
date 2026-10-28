package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/ir"
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
type Matrix struct {
	Rows, Cols, Depth int
	Warps             [2]int
	Resident          int
}

func (Matrix) Name() string { return "matrix" }

func (p Matrix) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	if d.Matrix[0] == 0 {
		return ks, nil
	}
	// made is the halves kernels formed, by what they compute: projections
	// of one row factor (q, k and v of one normed input) read one.
	made := map[string]Buf{}
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		f, ok := p.bilinear(k, d)
		if !ok {
			return nil, false, nil
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
				return nil, false, nil
			}
		}
		if d.Shared > 0 && 4*matrixStage(q.Rows, q.Cols, f.depth, d.Matrix[0], q.Warps[0]) > d.Shared {
			return nil, false, nil
		}
		return q.form(k, d, f, made), true, nil
	})
}

// matrixStage is the words of a matrix workgroup's stage: two stages of
// TR rows of A and TN of B at a depth of KC, padded (form), or after the
// trips the sums, WR warps' tm rows of TN.
func matrixStage(TR, TN, KC, tm, WR int) int {
	return max(2*(TR*(KC+8)/2+TN*(KC+4)/2), WR*tm*TN)
}

// bilinear is a reduction's parts the matrix unit takes: its sum, its
// factor of the row and its factor of the output, and the elements of k a
// trip stages.
type bilinear struct {
	fold, row, out, depth int
	// block, where set, is the rows of the blocks the output's factor reads
	// the row in: it is a factor of the block of rows alone.
	block int
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
		if depth%tk == 0 && k.Loops[0].Trip%depth == 0 {
			f.depth = depth
			break
		}
	}
	folds := k.foldsOf(0)
	if f.depth == 0 || len(folds) != 1 {
		return f, false
	}
	f.fold = folds[0]
	fd := k.Nodes[f.fold]
	if fd.Type != ir.F32 || k.Nodes[fd.B].Op != OpConst || k.Nodes[fd.B].Imm != 0 {
		return f, false
	}
	step := k.Nodes[fd.A]
	term := step.B
	if step.B == f.fold {
		term = step.A
	} else if step.A != f.fold {
		return f, false
	}
	if step.Op != OpAdd || step.Type != ir.F32 || k.Nodes[term].Op != OpMul || k.Nodes[term].Type != ir.F32 {
		return f, false
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
	x, y := k.Nodes[term].A, k.Nodes[term].B
	if row[y] {
		x, y = y, x
	}
	if !row[x] || out[x] || !out[y] || byRow[y] || folded[x] || folded[y] || k.readsFloatWeight(y) {
		return f, false
	}
	for _, s := range k.Stores {
		if s.At != -1 || s.Barrier || s.Run || s.Tile {
			return f, false
		}
	}
	f.row, f.out = x, y
	if row[y] {
		reach := k.reaches(y)
		for i := range k.Nodes {
			if b := blockOf(i); b > 0 && reach[i] && (f.block == 0 || b < f.block) {
				f.block = b
			}
		}
		if f.block < d.Matrix[0] {
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

// halves is a kernel writing the factor of the row of k as binary16 pairs,
// a word of a row a thread, into a buffer of R rows of K/2 words, and that
// buffer: the factor is rounded once, not once a workgroup of outputs, and a
// trip stages it as words.
func halves(k *Kernel, f bilinear) (*Kernel, Buf) {
	U := ir.U32
	R, W := k.Space[0], k.Loops[0].Trip/2
	// Named by what k stores, as a split names its planes: two
	// projections of one form hold their own.
	out := k.Bufs[k.Stores[0].Buf].Operand
	b := Buf{Kind: BufOut, Elem: U, Shape: []int{R, W},
		Operand: Operand{Kind: OperandScratch, Name: out.Name + ".halves", Elem: U, Temp: true, Of: out.Of}}
	h := &Kernel{Name: k.Name + ".halves", Kind: PrimMap, Space: []int{R, W}, Part: k.Part, Apart: k.Apart, Of: k.Of, Capacity: k.Capacity}
	if k.Rows > 0 {
		h.Rows = copyInto(h, k, k.Rows-1, func(int, Node) (int, bool) { return 0, false }) + 1
	}
	r := h.add(Node{Op: OpCoord, A: 0, Type: U})
	w := h.add(Node{Op: OpCoord, A: 1, Type: U})
	elem := func(at int) int {
		return h.un(OpHalf, U, copyInto(h, k, f.row, func(i int, nd Node) (int, bool) {
			switch {
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
	m.Nodes, m.Loops, m.Stores, m.Rows = nil, nil, nil, 0
	m.Space = []int{(R + TR - 1) / TR, (N + TN - 1) / TN}
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
	// A stage holds A's rows, then B's at bAt words; the array holds two,
	// trip t multiplying stage t&1 while it fills the other with trip
	// t+1's factors, and after the trips the sums.
	bAt := TR * SKA / 2
	half := bAt + TN*SKB/2
	st := shared("stage", U, matrixStage(TR, TN, KC, tm, WR))
	if units := max(d.Units, 1); p.Resident*m.Group <= d.Slots/units && (d.Shared == 0 || p.Resident*4*m.Bufs[st].Size <= d.Shared) {
		m.Resident = p.Resident
	}

	last := m.u32(R - 1)
	if k.Capacity > 0 {
		last = m.bin(OpSub, U, m.liveRows(k.Capacity), m.u32(1))
	}
	lastOut := m.u32(N - 1)
	r0 := m.bin(OpMul, U, m.add(Node{Op: OpCoord, A: 0, Type: U}), m.u32(TR))
	n0 := m.bin(OpMul, U, m.add(Node{Op: OpCoord, A: 1, Type: U}), m.u32(TN))
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
	h, hb := halves(k, f)
	sig := h.computes()
	if b, ok := made[sig]; ok {
		h, hb = nil, b
	} else {
		made[sig] = hb
	}
	W := k.Loops[0].Trip / 2
	hb.Kind, hb.Shape, hb.Extent = BufIn, nil, R*W
	hw := m.buf(hb)
	stage := func(l, kc, into, rows, axis, base, most, factor int) {
		SK := SKA
		if axis == 1 {
			SK = SKB
		}
		n := rows * KC / G
		for j := 0; j*m.Group < n; j++ {
			w := m.bin(OpAdd, U, thread, m.u32(j*m.Group))
			if (j+1)*m.Group > n {
				w = m.bin(OpMin, U, w, m.u32(n-1))
			}
			row := m.bin(OpDiv, U, w, m.u32(KC/G))
			g := m.bin(OpRem, U, w, m.u32(KC/G))
			c := m.bin(OpMin, U, m.bin(OpAdd, U, base, row), most)
			kg := m.bin(OpAdd, U, kc, m.bin(OpMul, U, g, m.u32(G)))
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
			if axis == 0 {
				at := m.bin(OpAdd, U, m.bin(OpMul, U, c, m.u32(W)), m.bin(OpShr, U, kg, m.u32(1)))
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
	T := k.Loops[0].Trip / KC
	stage(-1, m.u32(0), m.u32(0), TR, 0, r0, last, f.row)
	stage(-1, m.u32(0), m.u32(bAt), TN, 1, n0, lastOut, f.out)
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
	kn := m.bin(OpMul, U, m.bin(OpMin, U, next, m.u32(T-1)), m.u32(KC))
	into := m.bin(OpMul, U, m.bin(OpAnd, U, next, m.u32(1)), m.u32(half))
	stage(P, kn, into, TR, 0, r0, last, f.row)
	stage(P, kn, m.bin(OpAdd, U, into, m.u32(bAt)), TN, 1, n0, lastOut, f.out)
	m.Stores = append(m.Stores, Store{Barrier: true, At: P})
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
		sum := m.add(Node{Op: OpLoad, Type: F, Buf: st, A: e})
		leaf := func(i int, nd Node) (int, bool) {
			switch {
			case i == f.fold:
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
			c.At = E
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
	if h == nil {
		return []*Kernel{m}
	}
	return []*Kernel{h, m}
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
