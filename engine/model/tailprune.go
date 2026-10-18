package model

// A KV-sharing block (Config.KVShared) computes q alone and attends to its
// source's history, so it writes nothing a later row reads. The final run of
// them therefore only has to carry the rows something consumes: an
// intermediate prompt chunk's output is never read (the last chunk's last row
// becomes the decode residual), and the last chunk needs its last row alone.
//
// What consumes more rows is stated by the State, never inferred from the
// prompt: an embedding run (emb), a row tap (rowTap: MTP catch-up, a decision
// head), a speculation step's wanted rows (rows), a deepstack run reaching the
// tail, a bidirectional span. Any of them keeps every row. AltUp's and the
// other multi-stream residuals are laid out stream-major per chunk, so they
// are declined by name rather than compacted.

// tailPruneStats counts, over a State's prompts, the row-blocks a KV-sharing
// tail would have run unpruned (Would) and those it ran (Ran).
type tailPruneStats struct{ Would, Ran int }

// TailPruneRowBlocks reports the row-blocks the KV-sharing tail would have
// carried unpruned and the ones it carried.
func (s *State) TailPruneRowBlocks() (would, ran int) {
	return s.tailPrune.Would, s.tailPrune.Ran
}

// pruneTailAt is the first block of the prompt chunk's prunable tail, or -1.
func (s *State) pruneTailAt(base, n, total int, chunkBidir bool) int {
	c := s.c
	if s.m.opt.noTailPrune || c.NKVShared == 0 || c.AltUp != 0 || c.Streams() > 1 ||
		c.DSV4() || c.Q4E() || c.ResAttn() || s.emb != nil || s.rows != nil ||
		(s.rowTap != nil && !s.m.faults.tailPruneIgnoresDemand) || s.batched || s.recurrent() ||
		chunkBidir || len(s.bidir) > 0 || s.hi != int(c.NLayer) {
		return -1
	}
	start := max(s.lo, int(c.NLayer-c.NKVShared))
	if start >= s.hi || s.deepIn(start, base, n) {
		return -1
	}
	return start
}

// prunedTail runs blocks [start, s.hi) of the last chunk over its last row
// alone, leaving that row's output where the full chunk's would be. false with
// a nil error is a device that refused: the caller restarts the chunk from the
// embeddings with every row.
func (s *State) prunedTail(start, slot, p0, base, n, chunkRows int) (bool, error) {
	c := s.c
	last, E, rw, sw, w := n-1, c.NEmbd, c.RopeW(), c.NRotSWA, c.pleWidth()
	if len(s.tailSave) < rw+sw {
		s.tailSave = make([]float32, rw+sw)
	}
	save := s.tailSave
	copy(save[:rw], s.bcs[:rw])
	if s.bcsSWA != nil {
		copy(save[rw:], s.bcsSWA[:sw])
	}
	restore := func() {
		copy(s.bcs[:rw], save[:rw])
		if s.bcsSWA != nil {
			copy(s.bcsSWA[:sw], save[rw:])
		}
	}
	copy(s.bx[:E], s.bx[last*E:(last+1)*E])
	copy(s.bcs[:rw], s.bcs[last*rw:(last+1)*rw])
	if s.bcsSWA != nil {
		copy(s.bcsSWA[:sw], s.bcsSWA[last*sw:(last+1)*sw])
	}
	if w != 0 {
		copy(s.bple[:w], s.bple[last*w:(last+1)*w])
	}
	pl := p0 + last
	for li := start; li < s.hi; li++ {
		if s.devAt(li) {
			hi := li + 1
			for hi < s.hi && s.devAt(hi) {
				hi++
			}
			s.layerInputs(s.bple[:w])
			if !s.ld.Layers(li, hi, pl, 1, s.bx[:E], s.bcs[:rw], s.bswaTable(1, sw), nil) {
				restore()
				return false, nil
			}
			li = hi - 1
			continue
		}
		if err := s.hostRows(li, slot, base+last, pl, 1, chunkRows, false); err != nil {
			restore()
			return false, err
		}
	}
	copy(s.bx[last*E:(last+1)*E], s.bx[:E])
	restore()
	s.tailPrune.Would += n * (s.hi - start)
	s.tailPrune.Ran += s.hi - start
	return true, nil
}
