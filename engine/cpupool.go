package engine

// An embedding model's readout on the CPU device (cpudev.Pool): the pooled,
// unit-length row (poolGraph) from what the composer forms for it, at the
// segment's rows, composed once at its placement (cpuPlacePool) and counted
// served or declined by name (CPUPoolSteps).

// poolWidth is the readout's width: the last dense head's rows, or the
// model's width where it has none.
func (s *Instance) poolWidth() int {
	if g := s.m.FrontGraphs().Pool; g != nil {
		return g.Values[g.Out[0]].Width
	}
	return s.c.NEmbd
}
