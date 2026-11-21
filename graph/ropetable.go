package graph

// RopeTable is a step's rotary table: NRot/2 pairs a row, the table of the
// model's rotary Index (Attrs.Rotary).
type RopeTable struct {
	NRot  int
	Index int
	// Grid reads each row's position as a cell of the step's grid (the
	// request's first word its width, the step's live rows over it its
	// height) in the grid's nested order -- Merge x Merge merge windows in
	// Window x Window windows of them (OpPatchify's Width and Window) -- a
	// pair turning by the cell's row or its column, as the model's planes
	// name for the pair.
	Grid          bool
	Merge, Window int
	// Lead is the rows before a grid's first cell (a class row's), turned
	// as its first; Origin a cell's position on each axis at the grid's
	// first row and column. A row past the grid's cells (a class row after
	// them) is not turned.
	Lead, Origin int
	// Axes is each pair's position axis (Attrs.Axes): where set, a row has a
	// position on each axis and each pair turns by its axis's.
	Axes []uint8
}

// Rotary is one rotary a model turns its heads by, as its rotary ops state
// it: the width turned, the base, whether stored factors divide the
// angles, whether its positions are grid cells (RopeTable.Grid), and the
// position axis each of its pairs turns by (Attrs.Axes).
type Rotary struct {
	NRot          int
	Base          float64
	Freqs         bool
	Grid          bool
	Merge, Window int
	Lead          int
	// Origin is a grid cell's position on each axis at the grid's first
	// row and column (RopeTable.Origin).
	Origin int
	Axes   []uint8
}

// Build is the table's graph: each row's positions on the rotary axes from
// its sequence's marks (OpRopePos), then the table over them; a grid's rows
// read their cells as the step states them.
func (r *RopeTable) Build() *Graph {
	g := &Graph{Block: -1}
	cs := g.value("rope.cs", r.NRot)
	g.Out = []ValueID{cs}
	var in []ValueID
	if !r.Grid {
		pos := g.value("rope.pos", 4)
		g.Values[pos].Elem = ElemU32
		g.op(Op{Kind: OpRopePos, Out: []ValueID{pos}})
		in = []ValueID{pos}
	}
	g.op(Op{Kind: OpRopeTable, In: in, Out: []ValueID{cs}, Attr: Attrs{NRot: r.NRot, Rotary: r.Index, Grid: r.Grid, Axes: r.Axes,
		Width: r.Merge, Window: r.Window, Offset: r.Lead, Origin: r.Origin}})
	return g
}

// A rotary mark's words (OpRopePos), MarkWords a mark; a sequence's marks
// follow its count. A mark covers the cache positions from MarkAt to the
// next mark's. Its row at position p, j = p - MarkAt - MarkSkip, turns on
// each axis a at Base + Pos*p + Mul*(j/Div % Mod) (axis a's words, MarkAxis;
// Div 0 is one, Mod 0 no modulus) where 0 <= j < MarkCount (MarkCount 0:
// every row the mark covers past MarkSkip), and at p on every axis
// otherwise.
const (
	MarkAt = iota
	MarkSkip
	MarkCount
	markAxes
	MarkWords = markAxes + 4*MarkAxisWords
)

// An axis's words of a mark (MarkAxis).
const (
	MarkBase = iota
	MarkPos
	MarkMul
	MarkDiv
	MarkMod
	MarkAxisWords
)

// MarkAxis is the mark word w (MarkBase .. MarkMod) of axis a.
func MarkAxis(a, w int) int { return markAxes + a*MarkAxisWords + w }

// MarkPosition is axis a's position of a row at cache position p that mark m
// covers, as OpRopePos computes it.
func MarkPosition(m []uint32, a int, p uint32) uint32 {
	j := p - m[MarkAt]
	if p < m[MarkAt] || j < m[MarkSkip] || m[MarkCount] != 0 && j-m[MarkSkip] >= m[MarkCount] {
		return p
	}
	j -= m[MarkSkip]
	w := func(k int) uint32 { return m[MarkAxis(a, k)] }
	q := j / max(w(MarkDiv), 1)
	if w(MarkMod) != 0 {
		q %= w(MarkMod)
	}
	return w(MarkBase) + w(MarkPos)*p + w(MarkMul)*q
}
