package jit

import (
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/ir"
)

// Matrix forms a reduction over several rows on the device's matrix unit
// (Device.Matrix): a reduction of one loop whose term is the product of a
// factor of the row and the reduced coordinate alone -- a projection's
// input -- and a factor of the output and the reduced coordinate alone -- its
// decoded weight. A workgroup of Warps[0]*Warps[1] warps takes Rows rows by
// Cols outputs. Each trip stages Depth elements of k of both factors in
// workgroup memory as binary16 pairs, behind a barrier, a thread eight
// consecutive elements of a row at a time, so the elements of one packed
// word share its load and its scales once Affine has proved the shared
// address (headroom); a row of the stage is padded by four elements, which
// puts the sixteen rows a fragment load reads in different banks. Each warp multiplies
// its share of the block in the device's tiles into float32 accumulators
// carried across the trips; and the block's sums pass through workgroup
// memory -- the stage's own, behind a barrier -- to the kernel's own stores,
// an element a thread. A row past the
// call's (Kernel.Capacity) or the step's, and an output past the last, is
// the last one's, so the kernel reads and writes nothing the step does not
// state. A weight held as float32 is left as it is: binary16 would round it.
type Matrix struct {
	Rows, Cols, Depth int
	Warps             [2]int
}

func (Matrix) Name() string { return "matrix" }

func (p Matrix) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	if d.Matrix[0] == 0 {
		return ks, nil
	}
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		f, ok := p.bilinear(k, d)
		if !ok {
			return nil, false, nil
		}
		return []*Kernel{p.form(k, d, f)}, true, nil
	})
}

// bilinear is a reduction's parts the matrix unit takes: its sum, its
// factor of the row and its factor of the output, and the elements of k a
// trip stages.
type bilinear struct {
	fold, row, out, depth int
}

func (p Matrix) bilinear(k *Kernel, d Device) (bilinear, bool) {
	var f bilinear
	tm, tn, tk := d.Matrix[0], d.Matrix[1], d.Matrix[2]
	if k.Work || k.Group > 0 || k.Rows != 0 || k.Clamp != 0 || k.Tile != 0 || k.Positions != nil || k.Heads != nil ||
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
	out := k.dependsOn(func(nd Node) bool { return nd.Op == OpCoord && nd.A == 1 })
	folded := k.dependsOn(func(nd Node) bool { return nd.Op == OpFold })
	x, y := k.Nodes[term].A, k.Nodes[term].B
	if row[y] {
		x, y = y, x
	}
	if !row[x] || out[x] || !out[y] || row[y] || folded[x] || folded[y] || k.readsFloatWeight(y) {
		return f, false
	}
	for _, s := range k.Stores {
		if s.At != -1 || s.Barrier || s.Run || s.Tile {
			return f, false
		}
	}
	f.row, f.out = x, y
	return f, true
}

// dependsOn marks the nodes of k that reach a node pred holds for.
func (k *Kernel) dependsOn(pred func(Node) bool) []bool {
	dep := make([]bool, len(k.Nodes))
	for changed := true; changed; {
		changed = false
		for i := range k.Nodes {
			if dep[i] {
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

// form is k as the matrix unit's workgroups.
func (p Matrix) form(k *Kernel, d Device, f bilinear) *Kernel {
	U, F := ir.U32, ir.F32
	tm, tn, tk := d.Matrix[0], d.Matrix[1], d.Matrix[2]
	TR, TN, KC := p.Rows, p.Cols, f.depth
	WR, WN := p.Warps[0], p.Warps[1]
	R, N := k.Space[0], k.Space[1]
	m := k.clone()
	m.Name, m.Kind = k.Name+".matrix", PrimMatrix
	m.Nodes, m.Loops, m.Stores = nil, nil, nil
	m.Space = []int{(R + TR - 1) / TR, (N + TN - 1) / TN}
	m.Work, m.Group = true, 32*WR*WN
	shared := func(name string, e ir.Type, n int) int {
		return m.buf(Buf{Kind: BufShared, Elem: e, Size: n, Operand: Operand{Kind: OperandScratch, Name: k.Name + "." + name}})
	}
	// SK is a staged row's elements, padded; G the elements a thread stages
	// at a time.
	const pad, G = 4, 8
	SK := KC + pad
	// One array holds the stage of A, then B's at bAt words, and after the
	// trips the sums.
	bAt := TR * SK / 2
	st := shared("stage", U, max(bAt+TN*SK/2, TR*TN))

	last := m.u32(R - 1)
	if k.Capacity > 0 {
		last = m.bin(OpSub, U, m.liveRows(k.Capacity), m.u32(1))
	}
	lastOut := m.u32(N - 1)
	r0 := m.bin(OpMul, U, m.add(Node{Op: OpCoord, A: 0, Type: U}), m.u32(TR))
	n0 := m.bin(OpMul, U, m.add(Node{Op: OpCoord, A: 1, Type: U}), m.u32(TN))
	warp := m.bin(OpShr, U, m.add(Node{Op: OpThread, Type: U}), m.u32(5))
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

	P := m.loop(k.Loops[0].Trip/KC, -1)
	m.Loops[P].Uniform, m.Loops[P].StepsLast = true, true
	kc0 := m.bin(OpMul, U, m.add(Node{Op: OpIter, At: P, Type: U}), m.u32(KC))
	m.Stores = append(m.Stores, Store{Barrier: true, At: P})
	// stage is buf's rows of factor at coordinate axis base + the row, held
	// to most: a trip G elements of k, binary16 pairs a word.
	stage := func(at0, rows, axis, base, most, factor int) {
		l, w := spread(rows*KC/G, P)
		row := m.bin(OpDiv, U, w, m.u32(KC/G))
		g := m.bin(OpRem, U, w, m.u32(KC/G))
		c := m.bin(OpMin, U, m.bin(OpAdd, U, base, row), most)
		kg := m.bin(OpAdd, U, kc0, m.bin(OpMul, U, g, m.u32(G)))
		word := m.bin(OpAdd, U, m.bin(OpMul, U, row, m.u32(SK/2)), m.bin(OpAdd, U, m.bin(OpMul, U, g, m.u32(G/2)), m.u32(at0)))
		elem := func(at int) int {
			return m.un(OpHalf, U, copyInto(m, k, factor, func(i int, nd Node) (int, bool) {
				switch {
				case nd.Op == OpCoord && nd.A == axis:
					return c, true
				case nd.Op == OpIter && nd.At == 0:
					return at, true
				}
				return 0, false
			}))
		}
		for e := 0; e < G; e += 2 {
			lo := elem(m.bin(OpAdd, U, kg, m.u32(e)))
			hi := m.bin(OpShl, U, elem(m.bin(OpAdd, U, kg, m.u32(e+1))), m.u32(16))
			ix := m.bin(OpAdd, U, word, m.u32(e/2))
			m.Stores = append(m.Stores, Store{Buf: st, Ix: ix, Val: m.bin(OpOr, U, lo, hi), At: l})
		}
	}
	stage(0, TR, 0, r0, last, f.row)
	stage(bAt, TN, 1, n0, lastOut, f.out)
	m.Stores = append(m.Stores, Store{Barrier: true, At: P})

	aT := ir.TileType{Rows: tm, Cols: tk, Elem: ir.TileF16, Use: ir.TileA}
	bT := ir.TileType{Rows: tk, Cols: tn, Elem: ir.TileF16, Use: ir.TileB}
	cT := ir.TileType{Rows: tm, Cols: tn, Elem: ir.TileF32, Use: ir.TileAcc}
	mi, nj, ks := TR/WR/tm, TN/WN/tn, KC/tk
	at := func(base, i, unit, stride, s int) int {
		return m.bin(OpAdd, U, m.bin(OpMul, U, m.bin(OpAdd, U, base, m.u32(i*unit)), m.u32(stride)), m.u32(s))
	}
	// A staged view's rows start four elements apart and its offsets are
	// multiples of four: aligned.
	tile := func(t ir.TileType, off int, col bool) int {
		imm := int64(2)
		if col {
			imm |= 1
		}
		return m.add(Node{Op: OpTileLoad, Type: ir.Void, Buf: st, A: off, B: m.u32(SK), Imm: imm, At: P + 1, Tile: &t})
	}
	a := make([][]int, mi)
	for i := range a {
		for s := 0; s < ks; s++ {
			a[i] = append(a[i], tile(aT, at(wr0, i, tm, SK, s*tk), false))
		}
	}
	b := make([][]int, nj)
	for j := range b {
		for s := 0; s < ks; s++ {
			b[j] = append(b[j], tile(bT, m.bin(OpAdd, U, at(wn0, j, tn, SK, s*tk), m.u32(2*bAt)), true))
		}
	}
	zero := m.add(Node{Op: OpTileSplat, Type: ir.Void, A: m.f32(0), Tile: &cT})
	// The sums overwrite the stage the last trip's products read.
	m.Stores = append(m.Stores, Store{Barrier: true, At: -1})
	for i := 0; i < mi; i++ {
		for j := 0; j < nj; j++ {
			fold := m.add(Node{Op: OpFold, Type: ir.Void, At: P, B: zero, Tile: &cT})
			acc := fold
			for s := 0; s < ks; s++ {
				acc = m.add(Node{Op: OpTileMMA, Type: ir.Void, A: a[i][s], B: b[j][s], C: acc, Tile: &cT})
			}
			m.Nodes[fold].A = acc
			off := m.bin(OpAdd, U, at(wr0, i, tm, TN, 0), m.bin(OpAdd, U, wn0, m.u32(j*tn)))
			m.Stores = append(m.Stores, Store{Buf: st, Ix: off, Val: fold, At: -1, Tile: true, Stride: m.u32(TN)})
		}
	}
	m.Stores = append(m.Stores, Store{Barrier: true, At: -1})

	E, e := spread(TR*TN, -1)
	r := m.bin(OpMin, U, m.bin(OpAdd, U, r0, m.bin(OpDiv, U, e, m.u32(TN))), last)
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
	return m
}
