package model

// mixerChunk runs block li's recurrent mixer over a host prompt chunk: its
// normed rows in s.bh, its output rows into outRows (s.bh itself, or s.bhf
// on a block that attends as well). The recurrence is a per-row loop -- each
// row's state depends on the row before it, and a batched recurrence would
// be a different algorithm (a chunked scan) -- while its projections run as
// GEMMs over the chunk around the loop (linrows.go); the rows overwrite bh,
// so the output projection waits for the loop to finish.
func (s *State) mixerChunk(li int, l *layer, slot, base int, outRows []float32, n int) error {
	ne := s.c.NEmbd
	batched := s.linearBatched()
	if batched {
		if err := s.linearPre(l, s.bh, n); err != nil {
			return err
		}
	}
	ta := s.tick()
	for r := 0; r < n; r++ {
		h := s.bh[r*ne : (r+1)*ne]
		out := outRows[r*ne : (r+1)*ne]
		s.lin.r = r
		s.jit.NewInput()
		if err := s.linearAttn(li, l, slot, h, out); err != nil {
			s.lin.on = false
			return err
		}
		s.rows.snapRow(s, li, slot, base+r)
	}
	s.tock(opAttn, ta)
	if batched {
		return s.linearPost(li, l, outRows, n)
	}
	return nil
}

// mixerRagged is mixerChunk over a ragged step's rows: each row steps its own
// sequence's state (rowOwner), in position order (causal). Rows owned by
// other sessions keep the per-row matvec, the projections being this State's
// scratch.
func (s *State) mixerRagged(li int, l *layer, seq, causal []int, outRows []float32, n int) error {
	ne := s.c.NEmbd
	batched := s.linearBatched() && len(s.rowOwn) == 0
	if batched {
		if err := s.linearPre(l, s.bh, n); err != nil {
			return err
		}
	}
	t0 := s.tick()
	for _, r := range causal {
		h := s.bh[r*ne : (r+1)*ne]
		out := outRows[r*ne : (r+1)*ne]
		o, slot := s.rowOwner(r, seq)
		o.lin.r = r
		o.jit.NewInput()
		if err := o.linearAttn(li, l, slot, h, out); err != nil {
			s.lin.on = false
			return err
		}
	}
	s.tock(opAttn, t0)
	if batched {
		return s.linearPost(li, l, outRows, n)
	}
	return nil
}
