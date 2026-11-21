package composer

import "github.com/jitllm/jitllm/composer/ir"

// RowTiles runs a projection over several rows a tile of rows a thread: the
// leading axis cut into tiles of T rows, each thread one tile's elements at
// the other coordinates, what does not depend on the row -- the weight's
// loads and its decode -- made once a trip and read by every row of the
// tile, each row its own sum and store. T is the device's statement alone:
// the most rows, a power of two, whose sums and terms, rowRegs registers a
// row, fit the registers a thread holds at the device's most resident
// threads (Rows rows where the device states no registers), no more than
// the form's rows, and past two no more than leave every unit four warps
// of threads: a router's 32 outputs over 512 rows in tiles of 16 were
// 1024 threads, eight workgroups on a V100's 80 units (92.9 ms of a
// 2048-token prompt against main's 12.7). The kernel's ops grow with T, never with the step's
// rows, and a call's rows need not fill the last tile: its rows past the
// call's (Kernel.Capacity) or the step's are guarded, each the last row's copy,
// so they read and write nothing the rows before them do not. A kernel of
// one row, or whose rows' work reads no weight it could share, is left as
// it is.
type RowTiles struct {
	Rows int
}

// rowRegs is the registers a tile's row holds through its trips: its sum and
// the term it adds.
const rowRegs = 2

func (RowTiles) Name() string { return "rowtiles" }

func (p RowTiles) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		t := p.tile(k, d)
		if t < 2 {
			return nil, false, nil
		}
		return []*Kernel{tileRows(k, t)}, true, nil
	})
}

// tile is the rows a tile of k holds, 1 where k is not tiled.
func (p RowTiles) tile(k *Kernel, d Device) int {
	if k.Work || k.Group > 0 || k.Rows != 0 || k.Clamp != 0 || k.Positions != nil || k.Heads != nil ||
		k.Kind != PrimReduce && k.Kind != PrimContract || len(k.Space) < 2 || len(k.Loops) != 1 || !readsMatrix(k) {
		return 1
	}
	if _, _, _, _, ok := splittable(k); !ok {
		return 1
	}
	dep := k.rowDependent()
	shared := false
	for i, nd := range k.Nodes {
		if nd.Op == OpLoad && !dep[i] && k.Bufs[nd.Buf].Operand.Kind == OperandWeight && k.Varies(i, 0) {
			shared = true
		}
	}
	if !shared {
		return 1
	}
	most := p.Rows
	if d.Regs > 0 {
		most = d.Regs / rowRegs
	}
	// More rows a thread share more of the weight's loads but leave fewer
	// threads: past the first two, a tile doubles only while the threads
	// left give every unit four warps.
	inner := k.Elements() / k.Space[0]
	t := 1
	for 2*t <= most && 2*t <= k.Space[0] && (t < 2 || d.Units == 0 || k.Space[0]/(2*t)*inner >= 4*d.Units*max(d.Lanes, 1)) {
		t *= 2
	}
	return t
}

// rowDependent marks the nodes of k whose value depends on the leading
// coordinate.
func (k *Kernel) rowDependent() []bool {
	dep := make([]bool, len(k.Nodes))
	for changed := true; changed; {
		changed = false
		for i := range k.Nodes {
			if dep[i] {
				continue
			}
			nd := k.Nodes[i]
			d := nd.Op == OpCoord && nd.A == 0
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

// tileRows is k with its leading axis in tiles of t rows: row j of a tile
// is the tile's coordinate times t plus j, each row's nodes copies of the
// first row's where they depend on the row, the same nodes where not.
func tileRows(k *Kernel, t int) *Kernel {
	m := k.clone()
	dep := m.rowDependent()
	var lead []int
	for i, nd := range m.Nodes {
		if nd.Op == OpCoord && nd.A == 0 {
			lead = append(lead, i)
		}
	}
	R := m.Space[0]
	m.Space[0] = (R + t - 1) / t
	base := m.bin(OpMul, ir.U32, m.add(Node{Op: OpCoord, A: 0, Type: ir.U32}), m.u32(t))
	last := -1
	switch {
	case m.Capacity > 0:
		last = m.bin(OpSub, ir.U32, m.liveRows(m.Capacity), m.u32(1))
	case R%t != 0:
		last = m.u32(R - 1)
	}
	stores := m.Stores
	for j := 1; j < t; j++ {
		row := m.bin(OpAdd, ir.U32, base, m.u32(j))
		if last >= 0 {
			row = m.bin(OpMin, ir.U32, row, last)
		}
		leaf := func(i int, nd Node) (int, bool) {
			switch {
			case nd.Op == OpCoord && nd.A == 0:
				return row, true
			case i < len(dep) && !dep[i]:
				return i, true
			}
			return 0, false
		}
		for _, s := range stores {
			c := s
			c.Val = copyInto(m, m, s.Val, leaf)
			c.Ax = make([]int, len(s.Ax))
			for a, x := range s.Ax {
				c.Ax[a] = copyInto(m, m, x, leaf)
			}
			m.Stores = append(m.Stores, c)
		}
	}
	first := base
	if last >= 0 {
		first = m.bin(OpMin, ir.U32, base, last)
	}
	for _, c := range lead {
		m.redirect(c, first, first)
	}
	m.Name = k.Name + ".tiled"
	m.Tile, m.TileBase, m.TileLast, m.TileFirst, m.TileRows = t, base+1, last+1, first+1, R
	return m
}
