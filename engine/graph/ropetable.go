package graph

// RopeTable is a step's rotary table: NRot/2 pairs a row.
type RopeTable struct {
	NRot int
}

func (r *RopeTable) Build() *Graph {
	g := &Graph{Block: -1}
	cs := g.value("rope.cs", r.NRot)
	g.Out = []ValueID{cs}
	g.op(Op{Kind: OpRopeTable, Out: []ValueID{cs}, Attr: Attrs{NRot: r.NRot}})
	return g
}
