package jit

import (
	"github.com/jitllm/jitllm/jit/ir"
)

// FloatTiles forms a reduction over rows whose output's factor is a float32
// weight on the float units: a workgroup a tile of rows by outputs, a thread
// a micro-tile strided over it, k staged in workgroup memory two stages deep.
type FloatTiles struct {
	Depth int
	Tiles []FloatTile
}

type FloatTile struct {
	Rows, Cols int
	Micro      [2]int
}

func (FloatTiles) Name() string { return "floattiles" }

func (p FloatTiles) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		f, ok := p.takes(k)
		if !ok {
			return nil, false, nil
		}
		return []*Kernel{p.form(k, d, f)}, true, nil
	})
}

func (p FloatTiles) takes(k *Kernel) (bilinear, bool) {
	var f bilinear
	if k.Work || k.Group > 0 || k.Clamp != 0 || k.Tile != 0 || k.Positions != nil || k.Heads != nil ||
		k.Quarter != nil || k.Kind != PrimReduce && k.Kind != PrimContract || len(k.Space) != 2 || len(k.Loops) != 1 ||
		k.Space[0] < p.Tiles[len(p.Tiles)-1].Rows || k.Space[1] < p.Tiles[len(p.Tiles)-1].Cols {
		return f, false
	}
	if l := k.Loops[0]; l.Count != 0 || l.Range || l.Spread || l.Lanes != 0 || l.Trip%p.Depth != 0 {
		return f, false
	}
	for _, s := range k.Stores {
		if s.At != -1 || s.Barrier || s.Run || s.Tile {
			return f, false
		}
	}
	f, ok := k.bilinearOf(true)
	if !ok || f.block > 0 || f.index > 0 || !k.readsFloatWeight(f.out) || k.readsFloatWeight(f.row) {
		return f, false
	}
	f.depth = p.Depth
	return f, true
}

// tile is the first tile whose workgroups cover d's units, else the one of most.
func (p FloatTiles) tile(R, N int, d Device) FloatTile {
	var best FloatTile
	most := 0
	for _, t := range p.Tiles {
		if t.Rows > R || t.Cols > N {
			continue
		}
		g := ((R + t.Rows - 1) / t.Rows) * ((N + t.Cols - 1) / t.Cols)
		if g >= max(d.Units, 1) {
			return t
		}
		if g > most {
			best, most = t, g
		}
	}
	return best
}

func (p FloatTiles) form(k *Kernel, d Device, f bilinear) *Kernel {
	U, F := ir.U32, ir.F32
	R, N := k.Space[0], k.Space[1]
	t := p.tile(R, N, d)
	TR, TN, KC := t.Rows, t.Cols, p.Depth
	MR, MN := t.Micro[0], t.Micro[1]
	TY, TX := TR/MR, TN/MN
	m := k.clone()
	m.Name, m.Kind = k.Name+".tiles", PrimMatrix
	m.Nodes, m.Loops, m.Stores, m.Rows = nil, nil, nil, 0
	m.Space = []int{(R + TR - 1) / TR, (N + TN - 1) / TN}
	m.Work, m.Group = true, TY*TX
	SK := KC + 1
	bAt := TR * SK
	half := (TR + TN) * SK
	st := m.buf(Buf{Kind: BufShared, Elem: F, Size: 2 * half, Operand: Operand{Kind: OperandScratch, Name: k.Name + ".stage"}})

	last := m.u32(R - 1)
	if k.Capacity > 0 {
		last = m.bin(OpSub, U, m.liveRows(k.Capacity), m.u32(1))
	}
	lastOut := m.u32(N - 1)
	r0 := m.bin(OpMul, U, m.add(Node{Op: OpCoord, A: 0, Type: U}), m.u32(TR))
	n0 := m.bin(OpMul, U, m.add(Node{Op: OpCoord, A: 1, Type: U}), m.u32(TN))
	thread := m.add(Node{Op: OpThread, Type: U})
	tx, ty := m.bin(OpRem, U, thread, m.u32(TX)), m.bin(OpDiv, U, thread, m.u32(TX))
	T := k.Loops[0].Trip / KC

	stage := func(l, kc, into, rows, axis, base, most, factor int) {
		n := rows * KC
		for j := 0; j*m.Group < n; j++ {
			w := m.bin(OpAdd, U, thread, m.u32(j*m.Group))
			if (j+1)*m.Group > n {
				w = m.bin(OpMin, U, w, m.u32(n-1))
			}
			row, e := m.bin(OpDiv, U, w, m.u32(KC)), m.bin(OpRem, U, w, m.u32(KC))
			c := m.bin(OpMin, U, m.bin(OpAdd, U, base, row), most)
			at := m.bin(OpAdd, U, kc, e)
			v := copyInto(m, k, factor, func(i int, nd Node) (int, bool) {
				switch {
				case nd.Op == OpCoord && nd.A == axis:
					return c, true
				case nd.Op == OpIter && nd.At == 0:
					return at, true
				}
				return 0, false
			})
			ix := m.bin(OpAdd, U, m.bin(OpAdd, U, m.bin(OpMul, U, row, m.u32(SK)), e), into)
			m.Stores = append(m.Stores, Store{Buf: st, Ix: ix, Val: v, At: l})
		}
	}
	stage(-1, m.u32(0), m.u32(0), TR, 0, r0, last, f.row)
	stage(-1, m.u32(0), m.u32(bAt), TN, 1, n0, lastOut, f.out)
	m.Stores = append(m.Stores, Store{Barrier: true, At: -1})
	P := m.loop(T, -1)
	m.Loops[P].Uniform = true
	if k.Rows > 0 {
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

	from := m.bin(OpMul, U, m.bin(OpAnd, U, it, m.u32(1)), m.u32(half))
	aRow := make([]int, MR)
	for i := range aRow {
		aRow[i] = m.bin(OpAdd, U, from, m.bin(OpMul, U, m.bin(OpAdd, U, ty, m.u32(i*TY)), m.u32(SK)))
	}
	bRow := make([]int, MN)
	for j := range bRow {
		bRow[j] = m.bin(OpAdd, U, m.bin(OpAdd, U, from, m.u32(bAt)), m.bin(OpMul, U, m.bin(OpAdd, U, tx, m.u32(j*TX)), m.u32(SK)))
	}
	zero := m.f32(0)
	folds := make([]int, MR*MN)
	acc := make([]int, MR*MN)
	for x := range folds {
		folds[x] = m.add(Node{Op: OpFold, Type: F, At: P, B: zero})
		acc[x] = folds[x]
	}
	for e := 0; e < KC; e++ {
		a := make([]int, MR)
		for i := range a {
			a[i] = m.add(Node{Op: OpLoad, Type: F, Buf: st, A: m.bin(OpAdd, U, aRow[i], m.u32(e))})
		}
		for j := 0; j < MN; j++ {
			b := m.add(Node{Op: OpLoad, Type: F, Buf: st, A: m.bin(OpAdd, U, bRow[j], m.u32(e))})
			for i := range a {
				x := i*MN + j
				if f.fma {
					acc[x] = m.add(Node{Op: OpFma, Type: F, A: a[i], B: b, C: acc[x]})
				} else {
					acc[x] = m.bin(OpAdd, F, acc[x], m.bin(OpMul, F, a[i], b))
				}
			}
		}
	}
	for x := range folds {
		m.Nodes[folds[x]].A = acc[x]
	}
	for i := 0; i < MR; i++ {
		r := m.bin(OpMin, U, m.bin(OpAdd, U, r0, m.bin(OpAdd, U, ty, m.u32(i*TY))), last)
		for j := 0; j < MN; j++ {
			n := m.bin(OpMin, U, m.bin(OpAdd, U, n0, m.bin(OpAdd, U, tx, m.u32(j*TX))), lastOut)
			k.storesOf(m, f.fold, r, n, folds[i*MN+j], -1)
		}
	}
	return m
}
