package composer

import (
	"slices"

	"github.com/jitllm/jitllm/composer/ir"
)

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
	// Alias reads every output of a tile through the first output's weight
	// loads: a gate's violation.
	Alias bool
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
		return []*Kernel{tileRows(k, t, p.outputs(k, d, t), tileLanes(d), p.Alias)}, true, nil
	})
}

// tile is the rows a tile of k holds, 1 where k is not tiled.
func (p RowTiles) tile(k *Kernel, d Device) int {
	if !tileable(k) {
		return 1
	}
	most := p.most(d)
	if d.RowTile > 0 {
		t := 1
		for 2*t <= min(d.RowTile, most) && 2*t <= k.Space[0] {
			t *= 2
		}
		return t
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

// outputs is the outputs a tile of t rows of k holds, the device's
// statement (Device.OutTile) where k's outputs are its one axis past the
// rows, more than a group of the tile's lanes where they are not a
// multiple of it (tileOutputs), and its sums and terms fit the registers
// with the rows'; 1 else.
func (p RowTiles) outputs(k *Kernel, d Device, t int) int {
	if d.OutTile < 2 || len(k.Space) != 2 || len(k.Stores) != 1 || len(k.Stores[0].Ax) != 2 {
		return 1
	}
	N, L := k.Space[1], tileLanes(d)
	u := 1
	for 2*u <= d.OutTile && 2*u*t <= p.most(d) && outputsFit(N, 2*u, L) {
		u *= 2
	}
	return u
}

// outputsFit reports whether n outputs take tiles of u outputs lanes apart
// (tileOutputs): a multiple of u*lanes, or more.
func outputsFit(n, u, lanes int) bool { return n%(u*lanes) == 0 || n > u*lanes }

// tileable reports whether RowTiles takes k: a projection's one sum over
// several rows, whose weight's loads no row depends on.
func tileable(k *Kernel) bool {
	if k.Work || k.Group > 0 || k.Rows != 0 || k.Clamp != 0 || k.Positions != nil || k.Heads != nil ||
		k.Kind != PrimReduce && k.Kind != PrimContract || len(k.Space) < 2 || len(k.Loops) != 1 || !readsMatrix(k) {
		return false
	}
	if _, _, _, _, ok := splittable(k); !ok {
		return false
	}
	dep := k.rowDependent()
	for i, nd := range k.Nodes {
		if nd.Op == OpLoad && !dep[i] && k.Bufs[nd.Buf].Operand.Kind == OperandWeight && k.Varies(i, 0) {
			return true
		}
	}
	return false
}

// tileLanes is the outputs a tile's output copy lies apart from the
// next: the device's lanes, so that the lanes of a subgroup read
// consecutive outputs at every copy, or the IR's subgroup where the device
// guarantees none.
func tileLanes(d Device) int {
	if d.Lanes > 1 {
		return d.Lanes
	}
	return ir.SubgroupLanes
}

// most is the most rows a tile holds on d: as many as its registers
// hold sums and terms for, where known.
func (p RowTiles) most(d Device) int {
	if d.Regs > 0 {
		return d.Regs / rowRegs
	}
	return p.Rows
}

// rowDependent marks the nodes of k whose value depends on the leading
// coordinate.
func (k *Kernel) rowDependent() []bool { return k.coordDependent(0) }

// coordDependent marks the nodes of k whose value depends on coordinate a.
func (k *Kernel) coordDependent(a int) []bool {
	dep := make([]bool, len(k.Nodes))
	for changed := true; changed; {
		changed = false
		for i := range k.Nodes {
			if dep[i] {
				continue
			}
			nd := k.Nodes[i]
			d := nd.Op == OpCoord && nd.A == a
			for _, x := range k.Operands(&nd) {
				d = d || dep[*x]
			}
			if d {
				dep[i], changed = true, true
			}
		}
	}
	return dep
}

// copyStores appends to m a copy of each of stores whose nodes leaf maps,
// one copy of each node however many of the stores reach it.
func copyStores(m *Kernel, stores []Store, leaf func(i int, nd Node) (int, bool)) {
	roots := make([]int, 0, len(stores)*2)
	for _, s := range stores {
		roots = append(roots, s.Val)
		roots = append(roots, s.Ax...)
	}
	got := copyRoots(m, m, roots, leaf)
	for _, s := range stores {
		c := s
		c.Val, got = got[0], got[1:]
		c.Ax = slices.Clone(got[:len(s.Ax)])
		got = got[len(s.Ax):]
		m.Stores = append(m.Stores, c)
	}
}

// tileRows is k with its leading axis in tiles of t rows and its outputs in
// tiles of u: row j of a tile is the tile's coordinate times t plus j, each
// row's nodes copies of the first row's where they depend on the row, the
// same nodes where not; and output i of a tile the first output plus i
// times lanes, its nodes likewise copies of the first output's where they
// depend on the output. A thread's first outputs are lanes consecutive
// outputs apart from its neighbours', so the lanes of a subgroup read
// consecutive words of a weight laid out by outputs at every copy. A
// thread's stores run from its last row to its first, row j's output i
// store (t-1-j)*u+i: a row past the call's is the last live one's,
// computed by another copy, which a driver free to contract a copy's
// arithmetic its own way may round apart, and the row's own copy stores
// it last.
func tileRows(k *Kernel, t, u, lanes int, alias bool) *Kernel {
	m := k.clone()
	if u > 1 {
		tileOutputs(m, u, lanes, alias)
	}
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
	// The last live row is the kernel's own: the call's rows where its rows
	// are the step's, else the call's share of R -- a quarter of them under a
	// fold of four rows.
	last := -1
	switch {
	case m.Capacity == R:
		last = m.bin(OpSub, ir.U32, m.liveRows(m.Capacity), m.u32(1))
	case m.Capacity > 0:
		last = m.bin(OpSub, ir.U32, m.liveElements([]int{R}, m.Capacity), m.u32(1))
	case R%t != 0:
		last = m.u32(R - 1)
	}
	stores := m.Stores
	for j := 1; j < t; j++ {
		row := m.bin(OpAdd, ir.U32, base, m.u32(j))
		if last >= 0 {
			row = m.bin(OpMin, ir.U32, row, last)
		}
		copyStores(m, stores, func(i int, nd Node) (int, bool) {
			switch {
			case nd.Op == OpCoord && nd.A == 0:
				return row, true
			case i < len(dep) && !dep[i]:
				return i, true
			}
			return 0, false
		})
	}
	first := base
	if last >= 0 {
		first = m.bin(OpMin, ir.U32, base, last)
	}
	for _, c := range lead {
		m.redirect(c, first, first)
	}
	per := len(stores)
	rows := make([]Store, 0, len(m.Stores))
	for j := t - 1; j >= 0; j-- {
		rows = append(rows, m.Stores[j*per:(j+1)*per]...)
	}
	m.Stores = rows
	m.Name = k.Name + ".tiled"
	m.Tile, m.TileOut, m.TileBase, m.TileLast, m.TileFirst, m.TileRows = t, u, base+1, last+1, first+1, R
	return m
}

// tileOutputs is m's outputs, its second and last axis, in tiles of u
// outputs a thread: thread c's output i is c's group of lanes times u*lanes,
// plus i*lanes, plus c's lane in the group; past the last output, the group
// before's output i, which another thread computes by the same copy of the
// arithmetic and so to the same bits whatever a driver contracts. Its
// stores are u, output i's store i. m's outputs are more than u*lanes
// where they are not a multiple of them. With alias every output reads
// the first's weight (RowTiles.Alias).
func tileOutputs(m *Kernel, u, lanes int, alias bool) {
	dep := m.coordDependent(1)
	var old []int
	for i, nd := range m.Nodes {
		if nd.Op == OpCoord && nd.A == 1 {
			old = append(old, i)
		}
	}
	N := m.Space[1]
	m.Space[1] = (N + u*lanes - 1) / (u * lanes) * lanes
	c := m.add(Node{Op: OpCoord, A: 1, Type: ir.U32})
	e := m.bin(OpAdd, ir.U32, m.bin(OpMul, ir.U32, m.bin(OpDiv, ir.U32, c, m.u32(lanes)), m.u32(u*lanes)),
		m.bin(OpRem, ir.U32, c, m.u32(lanes)))
	out := func(i int) int {
		o := e
		if i > 0 {
			o = m.bin(OpAdd, ir.U32, e, m.u32(i*lanes))
		}
		if N%(u*lanes) != 0 {
			in := m.add(Node{Op: OpLess, Type: ir.Pred, A: o, B: m.u32(N)})
			o = m.add(Node{Op: OpSelect, Type: ir.U32, A: in, B: o, C: m.bin(OpSub, ir.U32, o, m.u32(u*lanes))})
		}
		return o
	}
	stores := m.Stores
	for i := 1; i < u; i++ {
		o := out(i)
		copyStores(m, stores, func(x int, nd Node) (int, bool) {
			switch {
			case nd.Op == OpCoord && nd.A == 1:
				return o, true
			case x < len(dep) && !dep[x]:
				return x, true
			case alias && nd.Op == OpLoad && m.Bufs[nd.Buf].Operand.Kind == OperandWeight:
				return x, true
			}
			return 0, false
		})
	}
	o := out(0)
	for _, x := range old {
		m.redirect(x, o, o)
	}
}
