package engine

// growPLE sizes the per-layer embedding scratch. A model without them gets
// none.
func (s *Instance) growPLE() {
	c := s.c
	if c.SideWidth() == 0 || s.ple != nil {
		return
	}
	// A decode row's tap rows are zeros, which nothing writes over.
	s.ple = make([]float32, c.SideWidth())
}

// kvGroupOffered reports whether block li may be offered by offerRange(lo,
// hi): any block outside a KV-sharing group, or one whose every group member
// is on a device already or offered by the same call.
func (s *Instance) kvGroupOffered(li, lo, hi int) bool {
	c := s.c
	if c.NKVShared == 0 {
		return true
	}
	src := c.KVSource(li)
	for b := 0; b < c.NLayer; b++ {
		if b != src && c.KVSource(b) != src {
			continue
		}
		if !s.devAt(b) && (b < lo || b >= hi) {
			return false
		}
	}
	return true
}

// kvSpan is the blocks that must be offered with block li: li alone, or the
// span of its KV-sharing group, source to last reader.
func (s *Instance) kvSpan(li int) (lo, hi int) {
	c := s.c
	lo, hi = li, li+1
	if c.NKVShared == 0 {
		return lo, hi
	}
	src := c.KVSource(li)
	for b := 0; b < c.NLayer; b++ {
		if b == src || c.KVSource(b) == src {
			lo, hi = min(lo, b), max(hi, b+1)
		}
	}
	return lo, hi
}

// fixKVGroups keeps every KV-sharing group whole (Gemma 4's E2B/E4B): a block
// attends to its source's history, which lives on exactly one side, so the
// source and every block reading it must be placed together. A group split by
// the placement -- a device that took the source and declined a reader, or a
// relocation that took one member to the CPU -- moves to the CPU whole, with the source's
// history. The device declines a reader whose source it does not hold (tier)
// and offerRange offers a group only whole, so this is the third half.
func (s *Instance) fixKVGroups() {
	c := s.c
	if c.NKVShared == 0 || s.ld == nil {
		return
	}
	for li := c.NLayer - c.NKVShared; li < c.NLayer; li++ {
		src := c.KVSource(li)
		if s.devAt(li) == s.devAt(src) {
			continue
		}
		// Every block of src's group moves to the CPU.
		for b := 0; b < c.NLayer; b++ {
			if b != src && c.KVSource(b) != src {
				continue
			}
			if !s.devAt(b) {
				continue
			}
			if b == src && !s.migrateKV(b, s.pos, false) {
				continue // the history cannot move; the device keeps the block
			}
			s.ld.ReleaseLayers(b, b+1)
			s.unmarkOnDev(b)
		}
		s.gpuTarget = min(s.gpuTarget, s.gpuLayers)
	}
}

// growBPLE sizes a batched chunk's side inputs for n rows. Grows, never
// shrinks.
func (s *Instance) growBPLE(n int) {
	if w := n * s.c.SideWidth(); len(s.bple) < w {
		s.bple = make([]float32, w)
	}
}

// tokenSide is n token rows' side inputs, zero where they are tap rows (a
// token's have none); a model of per-layer inputs computes its own over
// them.
func (s *Instance) tokenSide(n int) []float32 {
	w := s.c.SideWidth()
	if w == 0 {
		return nil
	}
	s.growBPLE(n)
	side := s.bple[:n*w]
	if s.c.DeepTaps > 0 {
		clear(side)
	}
	return side
}

// fillDeep writes rows base..base+n-1 of src's tap rows into side
// (sideWidth a row): a given run's taps where its span carries them, zeros
// for every other row.
func (s *Instance) fillDeep(src rowSrc, base, n int, side []float32) {
	if s.c.DeepTaps == 0 {
		return
	}
	E, W := s.c.NEmbd, s.c.SideWidth()
	clear(side[:n*W])
	for _, g := range src.given {
		if g.deep == nil {
			continue
		}
		lo, hi := max(g.at, base), min(g.at+len(g.rows)/E, base+n)
		for i := lo; i < hi; i++ {
			at := g.n0 + i - g.at
			for k := range s.c.DeepTaps {
				copy(side[(i-base)*W+k*E:(i-base)*W+(k+1)*E], g.deep[(k*g.n+at)*E:])
			}
		}
	}
}
