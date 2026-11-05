package engine

// allocRecurrent sizes the recurrent state, nil for every architecture that
// is not a hybrid.
//
// It is per sequence, like the KV cache: a batched session's rows share the
// weights and nothing else.
func (s *Instance) allocRecurrent(nseq int) {
	c := s.c
	if !c.Hybrid() || !s.linearIn() {
		return
	}
	g := c.Delta()
	s.dg = g
	v := s.Session.cpuView()
	for i := 0; i < c.NLayer; i++ {
		if c.LayerKind(i).Recurrent() {
			v.MakeRec(i, nseq*g.ConvStateLen(), nseq*g.StateStride())
		}
	}
}

// linearIn reports whether any block this instance runs is a linear one. A draft
// Instance over a prediction block runs attention alone, so it keeps no summary
// and recurrent() is false for it.
func (s *Instance) linearIn() bool {
	for li := s.lo; li < s.hi; li++ {
		if s.c.LayerKind(li).Recurrent() {
			return true
		}
	}
	return false
}

// ResetRecurrent clears one row's recurrent state, which is what retiring a
// sequence means for a linear layer: there is no per-position cache to
// invalidate, only a running summary to forget.
func (s *Instance) ResetRecurrent(row int) {
	for li := range s.view.Blocks() {
		if s.view.Conv(li) == nil {
			continue
		}
		n := s.dg.ConvStateLen()
		clear(s.view.Conv(li)[row*n : (row+1)*n])
		n = s.dg.StateStride()
		clear(s.view.State(li)[row*n : (row+1)*n])
	}
}
