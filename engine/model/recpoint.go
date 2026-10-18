package model

import "github.com/jitllm/jitllm/engine/nn"

// A hybrid's restore points: where the prefix cache keeps a linear block's
// recurrent summary, so a prompt that shares a prefix with an earlier one
// resumes from it. A summary exists only at the instant the position is
// exactly there and cannot be sliced, so it is captured while the prefill
// steps the row loop (recRowPoint), every recEvery positions. The attention
// pages it pairs with are the shared store's, named by the chunks they hold
// (kvCache.aliasChunks). Restore probes those positions, largest first, and
// takes the first one whose pages and every summary are both there.
//
// Beside the spaced points, one at the position a prompt parts from the one
// the State ran before it (recMark): the end of what they share, which the
// prompt after this one is likely to share too -- a request's state before
// each of its questions, a conversation before its newest turn -- so that
// prompt resumes exactly there and not at the spaced point below it.

// recEvery is the spacing of a hybrid's restore points in positions: the
// smallest power-of-two multiple of kvChunk at which one point's summaries
// cost no more than the attention history the same span adds, capped at the
// page. On LFM2 (a short convolution's two rows a block) it is kvChunk; on a
// gated delta rule, whose state is megabytes a block, the page. Zero on a
// model with no recurrence or no pages.
func (s *State) recEvery() int {
	if s.recEv != 0 {
		return max(s.recEv, 0)
	}
	s.recEv = -1
	p := s.recurrentBoundary()
	if s.rconv == nil || p == 0 {
		return 0
	}
	var point, perPos uint64
	for li := range s.rconv {
		if s.rconv[li] != nil {
			point += uint64(len(s.rconv[li])+len(s.rstate[li])) * 4
		}
	}
	for li := s.kv.lo; li < len(s.kv.layers); li++ {
		if pg := &s.kv.layers[li]; pg.p > 0 {
			perPos += uint64(2*pg.pp/pg.p) * 4
		}
	}
	r := kvChunk
	for r < p && uint64(r)*perPos < point {
		r *= 2
	}
	s.recEv = min(r, p)
	return s.recEv
}

// recPointKey is the key and store index a summary at position pos is kept
// under: a page boundary's is the page's own key, one on a kvChunk boundary
// the chunk's (chunkKey, kept for the sequence where a fill's key is hashed
// afresh, so a restore probing every chunk hashes the prompt once), any other
// the partial page's named by its fill (pageKeyFill), as restoreKV reads
// them.
func (s *State) recPointKey(pos int) (string, int) {
	p := s.recurrentBoundary()
	al := s.attnLayer()
	if p == 0 || al < 0 || pos <= 0 {
		return "", 0
	}
	if pos%p == 0 {
		return s.kv.pageKey(al, pos/p-1), pos/p - 1
	}
	if pos%kvChunk == 0 {
		return s.kv.chunkKey(pos), pos / p
	}
	return s.kv.pageKeyFill(al, pos/p, pos%p), pos / p
}

// recPointsOn says this prefill keeps restore points: a single sequence
// under a cache namespace with a store, on a hybrid, outside a speculation
// step (whose rows may be rolled back).
func (s *State) recPointsOn() bool {
	if s.rconv == nil || s.kv.ns == "" || s.batched || s.rows != nil {
		return false
	}
	_, none := s.kv.store.(NoStore)
	return !none
}

// sealRecLayer stores layer li's summary as it is in rconv/rstate, as the
// restore point at pos. The two are written as one entry through a buffer
// kept on the State, so a warm prompt allocates nothing for it.
func (s *State) sealRecLayer(li, pos int) bool {
	key, idx := s.recPointKey(pos)
	if key == "" {
		return false
	}
	// A block whose device keeps its restore points: copied there, on the
	// card.
	if d := s.recDev(li); d != nil {
		return d.SnapRec(li, key)
	}
	nc, ns := len(s.rconv[li]), len(s.rstate[li])
	if cap(s.recBuf) < nc+ns {
		s.recBuf = make([]float32, nc+ns)
	}
	b := s.recBuf[:nc+ns]
	copy(b, s.rconv[li])
	copy(b[nc:], s.rstate[li])
	s.recRd.Reset(kvBytesOf(b))
	if err := s.kv.store.Set(key, s.recurrentLayer(li), idx, &s.recRd); err != nil {
		s.kv.failed++
		return false
	}
	s.kv.stored++
	return true
}

// recRowPoint is the host row loop's half: after the row that leaves the
// sequence at pos, a host linear block's summary is a restore point when pos
// is one.
func (s *State) recRowPoint(li, slot, pos int) {
	if slot != 0 || !s.recPointsOn() || !s.recPoint(pos) {
		return
	}
	s.sealRecLayer(li, pos)
}

// recPoint says pos is a restore point of this prefill: a multiple of
// recEvery, or the mark.
func (s *State) recPoint(pos int) bool {
	r := s.recEvery()
	return r > 0 && pos > 0 && (pos%r == 0 || pos == s.recMark)
}

// markParting sets recMark for a cached prompt named by key: where it parts
// from the last one, when that is inside it. The key is kept for the next.
func (s *State) markParting(key []int32) {
	s.recMark = 0
	if s.rconv == nil {
		return
	}
	c := 0
	for c < len(key) && c < len(s.recLast) && key[c] == s.recLast[c] {
		c++
	}
	// On the kvChunk grid, where restore looks for points (a mark from an
	// earlier prompt is another prompt's to find).
	if c = c / kvChunk * kvChunk; c > 0 && c < len(key) {
		s.recMark = c
	}
	s.recLast = append(s.recLast[:0], key...)
}

// recChunkPoint is a chunk's end: where it ends on a restore point, a placed
// linear block's summary is brought home from the device and stored as the
// point (its row loop runs there, out of recRowPoint's reach). pos is where
// the chunk ended.
func (s *State) recChunkPoint(slot, pos int) {
	if slot != 0 || !s.recPointsOn() || !s.recPoint(pos) || s.devCount() == 0 || s.ld == nil {
		return
	}
	for li := s.lo; li < len(s.onDev); li++ {
		if !s.onDev[li] || s.rconv[li] == nil || s.recDev(li) != nil {
			continue
		}
		// Brought home where the device keeps no points.
		if !s.migrateRecLayer(li, false) {
			s.kv.syncFailed++
			return
		}
		s.sealRecLayer(li, pos)
	}
	s.sealRecOnDevice(pos)
}

// sealRecOnDevice keeps the point at pos of every block whose device keeps
// its restore points, in one call (nn.RecShareDevice.SnapRecs).
func (s *State) sealRecOnDevice(pos int) bool {
	key, _ := s.recPointKey(pos)
	if key == "" {
		return false
	}
	s.recBatch = s.recBatch[:0]
	var d nn.RecShareDevice
	for li := range s.rconv {
		if s.rconv[li] == nil {
			continue
		}
		if x := s.recDev(li); x != nil {
			d = x
			s.recBatch = append(s.recBatch, int32(li))
		}
	}
	return d == nil || d.SnapRecs(s.recBatch, key)
}

// recPointAt reports whether every linear block's summary for position pos is
// in the store, and leaves them in rconv/rstate when it is: the probe a
// restore asks before it takes a fill.
func (s *State) recPointAt(pos int) bool {
	key, idx := s.recPointKey(pos)
	return key != "" && s.faultRecurrentAt(idx, key, false)
}

// aliasMark names the shared pages the mark falls inside by the mark's fill
// (pageKeyFill), once the prompt has run past it: the restore of exactly
// that prefix, which a spaced point below it would miss.
func (s *State) aliasMark(from, pos int) {
	m := s.recMark
	if !s.recPointsOn() || s.kv.share == nil || m <= from || m >= pos {
		return
	}
	for li := s.kv.lo; li < len(s.kv.layers); li++ {
		if pg := &s.kv.layers[li]; pg.p > 0 && m%pg.p != 0 {
			s.kv.aliasPage(li, m/pg.p, s.kv.pageKeyFill(li, m/pg.p, m%pg.p))
		}
	}
}

// recDev is the device keeping block li's restore points when the session
// shares through the prefix cache and the device keeps them
// (nn.RecShareDevice), nil otherwise.
func (s *State) recDev(li int) nn.RecShareDevice {
	if s.kv.share == nil || !s.devAt(li) {
		return nil
	}
	d, _ := s.ld.(nn.RecShareDevice)
	return d
}
