package model

import "fmt"

// PlaceBlocks puts blocks [lo, hi) on the attached device with their history,
// wherever they sit in the stack: placement is a set (onDev), not a prefix,
// so a caller -- a user's move, a server making room or promoting a busy
// conversation -- names any range, not only the next blocks after the seam.
// A block already placed is left alone; one the device declines stays home
// and the others are still placed. It reports how many blocks it placed.
//
// Call it between tokens, like every placement call.
func (s *State) PlaceBlocks(lo, hi int) (int, error) {
	if s.ld == nil {
		return 0, fmt.Errorf("model: no device attached (SetDevice first)")
	}
	lo, hi = max(lo, s.lo), min(hi, s.hi)
	if lo >= hi {
		return 0, nil
	}
	defer s.followHost()
	defer s.fixKVGroups()
	s.moved = nil
	var fresh []int
	for li := lo; li < hi; li++ {
		if !s.devAt(li) {
			fresh = append(fresh, li)
		}
	}
	if len(fresh) == 0 {
		return 0, nil
	}
	s.offerRange(lo, hi)
	placed := 0
	for i, li := range fresh {
		if !s.devAt(li) {
			continue
		}
		if !s.migrateKV(li, s.pos, true) || !s.migrateRec(li, true) {
			// Resident without its history: it would attend over nothing.
			// Give it back, and every block placed after it.
			for _, lj := range fresh[i:] {
				if s.devAt(lj) {
					s.ld.ReleaseLayers(lj, lj+1)
					s.unmarkOnDev(lj)
				}
			}
			return placed, fmt.Errorf("model: block %d's history would not move to the device", li)
		}
		placed++
	}
	return placed, nil
}

// ReleaseBlocks brings blocks [lo, hi) home with their history, wherever they
// sit in the stack -- the host block between two device runs it leaves is a
// placement the steps run as they run relocation's. A pinned block refuses the
// call before anything moves; a block whose history will not come home stays
// where it is, and the call reports it. It reports how many blocks came home.
//
// Call it between tokens, like every placement call.
func (s *State) ReleaseBlocks(lo, hi int) (int, error) {
	if s.ld == nil {
		return 0, nil
	}
	lo, hi = max(lo, s.lo), min(hi, s.hi)
	var on []int
	for li := lo; li < hi; li++ {
		if s.devAt(li) {
			if s.pinnedOnDev(li) {
				return 0, fmt.Errorf("model: block %d is pinned to its device", li)
			}
			on = append(on, li)
		}
	}
	if len(on) == 0 {
		return 0, nil
	}
	defer s.followHost()
	defer s.fixKVGroups()
	s.moved = nil
	// The head goes first, as SetGPULayers sends it: it rides the last
	// placed block's submission, and that block may be one of these.
	s.head, s.headReady = nil, nil
	n := 0
	for _, li := range on {
		if !s.migrateKV(li, s.pos, false) || !s.migrateRec(li, false) {
			return n, fmt.Errorf("model: block %d's history would not come home", li)
		}
		s.ld.ReleaseLayers(li, li+1)
		s.unmarkOnDev(li)
		n++
	}
	return n, nil
}
