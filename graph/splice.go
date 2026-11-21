package graph

// Spliced is the input rows em with a picture's rows spliced in: an input
// of the projector's rows, pic wide (a row's projected row first, as
// TowerHead writes it), and the rows the step names taking them (OpSplice)
// in place of their tokens'.
func Spliced(em *Graph, pic int) *Graph {
	g := *em
	g.Values = append([]Value(nil), em.Values...)
	g.Ops = append([]Op(nil), em.Ops...)
	g.In = append([]ValueID(nil), em.In...)
	g.Out = append([]ValueID(nil), em.Out...)
	x := g.Out[0]
	w := g.Values[x].Width
	g.Values = append(g.Values, Value{Name: "picture", Width: pic})
	p := ValueID(len(g.Values) - 1)
	g.Values = append(g.Values, Value{Name: "x_spliced", Width: w})
	y := ValueID(len(g.Values) - 1)
	g.Ops = append(g.Ops, Op{Kind: OpSplice, In: []ValueID{x, p}, Out: []ValueID{y}})
	g.In = append(g.In, p)
	g.Out[0] = y
	return &g
}
