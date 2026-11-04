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
	// as its first.
	Lead int
	// Sections, Interleaved, Cycle and CycleAxes are the rotary's position
	// axes (Attrs.Sections, Attrs.Interleaved, Attrs.Cycle): where Sections
	// or Cycle is set, a row has a position on each axis and each pair turns
	// by its axis's.
	Sections    [4]int
	Interleaved bool
	Cycle       int
	CycleAxes   [4]uint8
}

// Rotary is one rotary a model turns its heads by, as its rotary ops state
// it: the width turned, the base, whether stored factors divide the
// angles, whether its positions are grid cells (RopeTable.Grid), and the
// position axes its pairs turn by (Attrs.Sections, Attrs.Interleaved,
// Attrs.Cycle).
type Rotary struct {
	NRot          int
	Base          float64
	Freqs         bool
	Grid          bool
	Merge, Window int
	Lead          int
	Sections      [4]int
	Interleaved   bool
	Cycle         int
	CycleAxes     [4]uint8
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
	g.op(Op{Kind: OpRopeTable, In: in, Out: []ValueID{cs}, Attr: Attrs{NRot: r.NRot, Rotary: r.Index, Grid: r.Grid, Sections: r.Sections, Interleaved: r.Interleaved,
		Cycle: r.Cycle, CycleAxes: r.CycleAxes, Width: r.Merge, Window: r.Window, Offset: r.Lead}})
	return g
}

// A rotary mark's words (OpRopePos), MarkWords a mark; a sequence's marks
// follow its count.
const (
	// MarkAt is the first cache position the mark covers, MarkBase the
	// rotary position there, MarkKind how its rows turn.
	MarkAt = iota
	MarkBase
	MarkKind
	// MarkT, MarkH and MarkW are a grid's frames, rows and columns (under
	// XD-RoPE the columns hold a row's line ends), MarkEnds an XD-RoPE
	// picture's text-like rows before and after its grid, MarkIndex its
	// ordinal in the sequence.
	MarkT
	MarkH
	MarkW
	MarkEnds
	MarkIndex
	MarkWords
)

// How a mark's rows turn (MarkKind).
const (
	MarkText = iota
	MarkGrid
	MarkGridXD
)
