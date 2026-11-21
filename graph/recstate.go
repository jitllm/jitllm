package graph

// RecurrentState is the block's recurrent state for one sequence, in floats,
// as its ops lay it out: conv the window an OpStateConv keeps in
// StateRecConv, state the slot of StateRecState -- an OpStateScan's matrices,
// or the Span its ops state where a tail or a convolution's window follows
// them. Both are 0 for a block that keeps no recurrent state.
func (g *Graph) RecurrentState() (conv, state int) {
	if g == nil {
		return 0, 0
	}
	for _, o := range g.Ops {
		switch o.Kind {
		case OpStateConv:
			if len(o.Reads) == 1 && o.Reads[0].State == StateRecState {
				state = max(state, o.Attr.Span)
				continue
			}
			conv = max(conv, (o.Attr.Taps-1)*max(1, o.Attr.Dil)*g.Values[o.In[0]].Width)
		case OpStateScan:
			state = max(state, g.Values[o.Out[0]].Width*o.Attr.HeadDim, o.Attr.Span)
		}
	}
	return conv, state
}
