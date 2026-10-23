package graph

// RopeTable is a step's rotary table: NRot/2 pairs a row, the value named
// Name ("rope.cs" where empty).
type RopeTable struct {
	NRot int
	Name string
	// Grid reads each row's position as a cell of the step's grid (the
	// request's first word its width): a pair turns by the cell's row or
	// its column, as the model's planes name for the pair.
	Grid bool
}

// Rotary is one rotary a model turns its heads by, as its rotary ops state
// it: the width turned, the base, whether stored factors divide the
// angles, and whether its positions are grid cells (RopeTable.Grid).
type Rotary struct {
	NRot  int
	Base  float64
	Freqs bool
	Grid  bool
}

func (r *RopeTable) Build() *Graph {
	g := &Graph{Block: -1}
	name := r.Name
	if name == "" {
		name = "rope.cs"
	}
	cs := g.value(name, r.NRot)
	g.Out = []ValueID{cs}
	g.op(Op{Kind: OpRopeTable, Out: []ValueID{cs}, Attr: Attrs{NRot: r.NRot, Grid: r.Grid}})
	return g
}
