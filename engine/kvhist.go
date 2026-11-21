package engine

// histPlane is history hist of a block past its rows' (history 0): one
// entry each period positions, of width floats, its values its keys where
// inKeys (its graph's append to it, graph.Graph.KeptHistories).
type histPlane struct {
	block, hist, period, width int
	inKeys                     bool
}

// histPlanes is every history past its rows' that the instance's blocks
// keep, each where its graph appends to it.
func (s *Instance) histPlanes() []histPlane {
	var out []histPlane
	for li := s.lo; li < s.hi; li++ {
		for _, k := range s.m.Composing().BlockGraph(li).KeptHistories() {
			out = append(out, histPlane{li, k.Hist, k.Period, k.Width, k.ValuesInKeys})
		}
	}
	return out
}
