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
