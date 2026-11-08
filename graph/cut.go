package graph

import "fmt"

// Cut is g as two graphs run one after the other, the region r's ops and
// every op that reads what they make in the second (rest), every other op
// in the first (pre): pre's one output is the values rest reads that pre
// makes or g is handed, joined end to end; rest reads that row, cut back
// into those values, and g's inputs past the first, and writes g's outputs.
// A placement runs pre and rest on two devices, the row crossing between
// them. Rest touches no state, and a crossing value is a float row of the
// step's rows: Cut refuses a region that breaks either.
func Cut(g *Graph, r Region) (pre, rest *Graph, err error) {
	if r.Lo < 0 || r.Hi > len(g.Ops) || r.Lo >= r.Hi {
		return nil, nil, fmt.Errorf("a region [%d, %d) of a graph of %d ops", r.Lo, r.Hi, len(g.Ops))
	}
	later := make([]bool, len(g.Values))
	inRest := make([]bool, len(g.Ops))
	for i, o := range g.Ops {
		in := i >= r.Lo && i < r.Hi
		for _, v := range o.In {
			in = in || later[v]
		}
		if i < r.Lo || !in {
			continue
		}
		if len(o.Reads) != 0 || len(o.Writes) != 0 {
			return nil, nil, fmt.Errorf("op %d (%v) of the second half touches a state", i, o.Kind)
		}
		inRest[i] = true
		for _, v := range o.Out {
			later[v] = true
		}
	}
	side := map[ValueID]int{}
	for k, v := range g.In[1:] {
		side[v] = k + 1
	}
	var cross []ValueID
	seen := map[ValueID]bool{}
	carry := func(v ValueID) error {
		if later[v] || seen[v] {
			return nil
		}
		if _, ok := side[v]; ok {
			return nil
		}
		if x := g.Values[v]; x.Elem != ElemF32 || x.Rows != 0 {
			return fmt.Errorf("value %s crosses the cut as a %v row of %d rows", x.Name, x.Elem, x.Rows)
		}
		seen[v] = true
		cross = append(cross, v)
		return nil
	}
	for i, o := range g.Ops {
		if !inRest[i] {
			continue
		}
		for _, v := range o.In {
			if err := carry(v); err != nil {
				return nil, nil, err
			}
		}
	}
	for _, v := range g.Out {
		if err := carry(v); err != nil {
			return nil, nil, err
		}
	}
	if len(cross) == 0 {
		return nil, nil, fmt.Errorf("nothing crosses a cut at [%d, %d)", r.Lo, r.Hi)
	}

	width := 0
	for _, v := range cross {
		width += g.Values[v].Width
	}
	pre = &Graph{Version: g.Version, Block: g.Block}
	pm := remap{g: g, to: pre, ids: map[ValueID]ValueID{}}
	for _, v := range g.In {
		pre.In = append(pre.In, pm.id(v))
	}
	for i, o := range g.Ops {
		if !inRest[i] {
			pm.op(o)
		}
	}
	var joined []ValueID
	for _, v := range cross {
		joined = append(joined, pm.id(v))
	}
	row := pre.value("cut", width)
	pre.op(Op{Kind: OpConcat, In: joined, Out: []ValueID{row}})
	pre.Out = []ValueID{row}

	rest = &Graph{Version: g.Version, Block: g.Block}
	rm := remap{g: g, to: rest, ids: map[ValueID]ValueID{}}
	in := rest.value("cut", width)
	rest.In = []ValueID{in}
	for _, v := range g.In[1:] {
		rest.In = append(rest.In, rm.id(v))
	}
	off := 0
	for _, v := range cross {
		x := g.Values[v]
		s := rest.value(x.Name, x.Width)
		rest.op(Op{Kind: OpSlice, In: []ValueID{in}, Out: []ValueID{s}, Attr: Attrs{Offset: off, Width: x.Width}})
		rm.ids[v] = s
		off += x.Width
	}
	for i, o := range g.Ops {
		if inRest[i] {
			rm.op(o)
		}
	}
	for _, v := range g.Out {
		rest.Out = append(rest.Out, rm.id(v))
	}
	return pre, rest, nil
}

// remap copies ops of g into another graph, each value g names given one
// of its own there at its first use.
type remap struct {
	g, to *Graph
	ids   map[ValueID]ValueID
}

func (m *remap) id(v ValueID) ValueID {
	if n, ok := m.ids[v]; ok {
		return n
	}
	m.to.Values = append(m.to.Values, m.g.Values[v])
	n := ValueID(len(m.to.Values) - 1)
	m.ids[v] = n
	return n
}

func (m *remap) op(o Op) {
	c := o
	c.In, c.Out = make([]ValueID, len(o.In)), make([]ValueID, len(o.Out))
	for i, v := range o.In {
		c.In[i] = m.id(v)
	}
	for i, v := range o.Out {
		c.Out[i] = m.id(v)
	}
	m.to.Ops = append(m.to.Ops, c)
}
