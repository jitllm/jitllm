package graph

// Copy is a row's Width elements moved from its input to its output: where a
// block's form reads its input in the kernel writing its output, the output
// is written apart from the input and copied over it after.
type Copy struct {
	Width int
}

func (c *Copy) Build() *Graph {
	b := New()
	in := b.In("in", c.Width)
	b.Out(b.Emit(OpSlice, []ValueID{in}, []Result{{Name: "out", Width: c.Width}}, nil, Attrs{Width: c.Width}, nil, nil)[0])
	return b.G
}

// TakeRow is one row of Width elements, the one the step's request names
// (OpTakeRows), moved from its input of the step's rows to its output of
// one: a chunk's last row into the row its head reads.
type TakeRow struct {
	Width int
	// Rows is how many rows the request names, 1 where it is 0.
	Rows int
}

func (c *TakeRow) Build() *Graph {
	b := New()
	in := b.In("in", c.Width)
	b.Out(b.Emit(OpTakeRows, []ValueID{in}, []Result{{Name: "row", Width: c.Width, Rows: max(c.Rows, 1)}}, nil, Attrs{}, nil, nil)[0])
	return b.G
}
