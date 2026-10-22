package graph

// RopeTable is a step's rotary table: NRot/2 pairs a row, the value named
// Name ("rope.cs" where empty).
type RopeTable struct {
	NRot int
	Name string
}

// Rotary is one rotary a model turns its heads by, as its rotary ops state
// it: the width turned, the base, and whether stored factors divide the
// angles.
type Rotary struct {
	NRot  int
	Base  float64
	Freqs bool
}

func (r *RopeTable) Build() *Graph {
	g := &Graph{Block: -1}
	name := r.Name
	if name == "" {
		name = "rope.cs"
	}
	cs := g.value(name, r.NRot)
	g.Out = []ValueID{cs}
	g.op(Op{Kind: OpRopeTable, Out: []ValueID{cs}, Attr: Attrs{NRot: r.NRot}})
	return g
}
