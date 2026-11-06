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

// recRoutes is ok over the session's routes that run a block keeping a
// state, each once: false at the first that is not.
func (s *Instance) recRoutes(ok func(r *route) bool) bool {
	var seen [len(s.routes)]bool
	for li := s.lo; li < min(s.hi, s.c.NLayer); li++ {
		if !s.keepsRec(li) {
			continue
		}
		i := s.routeOf(li)
		if seen[i] {
			continue
		}
		seen[i] = true
		if !ok(&s.routes[i]) {
			return false
		}
	}
	return true
}

// recMarkAll records where every device's recurrent state is (nn.RecRewinder),
// and recRewindAll returns each there.
func (s *Instance) recMarkAll() {
	s.recRoutes(func(r *route) bool {
		if r.rew != nil {
			r.rew.RecMark()
		}
		return true
	})
}

func (s *Instance) recRewindAll() bool {
	return s.recRoutes(func(r *route) bool { return r.rew != nil && r.rew.RecRewind() })
}

// recRowsKept says every device running a linear block keeps a call's rows'
// states (nn.RecRowKeeper).
func (s *Instance) recRowsKept() bool {
	return s.recRoutes(func(r *route) bool { return r.rowk != nil })
}

// recKeepRows has every such device keep the next calls' first rows rows,
// and recRestoreRow return each to its state after row.
func (s *Instance) recKeepRows(rows int) {
	s.recRoutes(func(r *route) bool {
		if r.rowk != nil {
			r.rowk.RecKeepRows(rows)
		}
		return true
	})
}

func (s *Instance) recRestoreRow(row int) bool {
	return s.recRoutes(func(r *route) bool { return r.rowk != nil && r.rowk.RecRestoreRow(row) })
}
