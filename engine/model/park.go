package model

import "fmt"

// Parking is preemption for a session between tokens: Park gives back what the
// session holds on the device and, with a store, its sealed host KV pages;
// Resume takes them back. It is built from the two moves the five principles
// already gate -- relocation (a block's KV pages and recurrent state come home
// from the device and go back with the same answer, SetGPULayers) and KV
// paging (a sealed page is evicted into the store and faulted back, trim and
// fault) -- so a parked session resumes byte for byte where it stopped.
//
// What stays: the session's recurrent state (one matrix per linear block, the
// size of a page or two, and never stored) and its unsealed tail page. A
// session with no store of its own parks into a private MemStore that Resume
// empties, so its pages are held once, in the store or in the cache, never in
// both. A parked State must not step: the caller parks between tokens, never
// in one.

// ParkStats counts what a Park moved.
type ParkStats struct {
	// Blocks is how many blocks came home from the device, KV and state with
	// them; PagesOut how many host KV pages (a key and a value page each count)
	// went to the store.
	Blocks, PagesOut int
}

// Park preempts the session. Parking a parked State does nothing.
func (s *State) Park() (ParkStats, error) {
	if err := s.planRefuses("parking a sequence (Park)"); err != nil {
		return ParkStats{}, err
	}
	return s.park()
}

// ParkInto is Park into st rather than the session's own store: every sealed
// page the session holds is offered to st -- the ones its own store already
// had too, since that store may drop them (a bounded one does) while the
// session is parked -- and Resume faults them back from st and gives the
// session its own store back. A server parks into a view of its model's
// memory cache that holds what it is given until Resume (MemStore.Pinned).
func (s *State) ParkInto(st KVStore) (ParkStats, error) {
	if s.parked {
		return ParkStats{}, nil
	}
	s.parkPrev = s.kv.store
	s.kv.store = st
	for li := s.kv.lo; li < len(s.kv.layers); li++ {
		s.kv.layers[li].lastSealed = 0
	}
	ps, err := s.park()
	if err != nil {
		s.kv.store, s.parkPrev = s.parkPrev, nil
	}
	return ps, err
}

func (s *State) park() (ParkStats, error) {
	var st ParkStats
	if s.parked {
		return st, nil
	}
	if n := s.devCount(); n > 0 {
		s.parkedSeam = s.gpuLayers
		if s.SetGPULayers(0); s.devCount() != 0 {
			return st, fmt.Errorf("model: park: %d block(s) stay on the device (pinned, or its "+
				"history would not come home)", s.devCount())
		}
		st.Blocks = n
	}
	if _, ok := s.kv.store.(NoStore); ok {
		s.parkStore = NewMemStore()
		s.kv.store = s.parkStore
	}
	{
		ev := s.kv.evicted
		// Seal first: a page the store does not hold is never evicted. The
		// tail page is being written and stays.
		s.kv.seal(s.pos)
		budget := s.kv.budget
		s.kv.budget = 1
		s.kv.trim(s.pos)
		s.kv.budget = budget
		st.PagesOut = int(s.kv.evicted - ev)
	}
	s.parked = true
	s.ParkedOut += int64(st.PagesOut)
	return st, nil
}

// Resume undoes Park: every page Park evicted comes back from the store, then
// the blocks it brought home are offered to the device again, their history
// with them. It returns how many pages came back.
func (s *State) Resume() (int, error) {
	if err := s.planRefuses("resuming a parked sequence (Resume)"); err != nil {
		return 0, err
	}
	if !s.parked {
		return 0, nil
	}
	in := 0
	for li := s.kv.lo; li < len(s.kv.layers); li++ {
		pg := &s.kv.layers[li]
		for n := range pg.out {
			if !pg.out[n] {
				continue
			}
			if err := s.kv.fault(li, n); err != nil {
				return in, fmt.Errorf("model: resume: layer %d page %d: %w", li, n, err)
			}
			in += 2
		}
	}
	s.parked = false
	s.ParkedIn += int64(in)
	if s.parkStore != nil {
		// Every page is back in the cache: the store and its copies go with
		// the last reference, and the session is storeless again. Nothing was
		// sealed into it before Park, so the next park starts the mark over.
		s.parkStore = nil
		s.kv.store = NoStore{}
		for li := s.kv.lo; li < len(s.kv.layers); li++ {
			s.kv.layers[li].lastSealed = 0
		}
	}
	if s.parkPrev != nil {
		// ParkInto's store held the pages while parked; the session's own
		// store takes them again from here, so the mark starts over.
		s.kv.store, s.parkPrev = s.parkPrev, nil
		for li := s.kv.lo; li < len(s.kv.layers); li++ {
			s.kv.layers[li].lastSealed = 0
		}
	}
	if s.parkedSeam > 0 {
		s.SetGPULayers(s.parkedSeam)
		s.parkedSeam = 0
	}
	return in, nil
}

// Parked reports whether the session is parked.
func (s *State) Parked() bool { return s.parked }

// HistoryBytes is what this session's history occupies wherever it lives --
// host pages and the device's alike -- at its current position, and
// PositionBytes what one more position adds. An admission budget is priced
// in them: KVBytes counts only the host's resident pages, which a session on
// the device or a parked one does not hold.
func (s *State) HistoryBytes() uint64 { return s.historyAt(s.pos) }

func (s *State) PositionBytes() uint64 { return s.historyAt(s.pos+1) - s.historyAt(s.pos) }

// historyAt is HistoryBytes at position pos: every attending layer's pages
// up to pos, a windowed layer's capped at its window and a page.
func (s *State) historyAt(pos int) uint64 {
	var b uint64
	for li := s.kv.lo; li < len(s.kv.layers); li++ {
		b += s.blockHistoryAt(li, pos)
	}
	return b
}
