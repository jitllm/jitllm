package engine

// A recurrent block's state is its device's: the CPU device keeps a
// session's in the session's view (cpudev.View) as a card keeps it on the
// card, and the instance reads or writes it only through the device the
// block runs on (nn.RecMigrator), into a buffer of its own (recStaged).

// recurrent reports that this session carries a linear block's running state,
// which is what makes re-running a block from the embeddings unsound.
func (s *Instance) recurrent() bool { return s.dg.ConvStateLen() > 0 || s.dg.StateStride() > 0 }

// keepsRec says block li keeps a recurrent state.
func (s *Instance) keepsRec(li int) bool {
	return s.recurrent() && li >= 0 && li < s.c.NLayer && s.c.LayerKind(li).Recurrent()
}

// recLens is a block's recurrent state's widths: its convolution rows' and
// its summary's, every sequence's.
func (s *Instance) recLens() (conv, state int) {
	n := max(s.nseq, 1)
	return n * s.dg.ConvStateLen(), n * s.dg.StateStride()
}

// recStaged is the instance's buffer for one block's state, kept so a warm
// prompt allocates nothing for it.
func (s *Instance) recStaged() (conv, state []float32) {
	nc, ns := s.recLens()
	if cap(s.recBuf) < nc+ns {
		s.recBuf = make([]float32, nc+ns)
	}
	b := s.recBuf[:nc+ns]
	return b[:nc], b[nc:]
}

// recIO moves block li's state between the device it runs on and conv and
// state: in writes it there.
func (s *Instance) recIO(li int, conv, state []float32, in bool) bool {
	r := s.routeAt(li)
	return r.recmig != nil && r.recmig.MigrateRec(li, conv, state, in)
}
