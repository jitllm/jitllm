package model

import (
	"fmt"
	"time"
)

// PlaceBlocks is MoveBlocks from the CPU with the tier choosing the card:
// the blocks of [lo, hi) this instance runs on the CPU go to whichever attached
// device takes them, with their history, wherever they sit in the stack. A
// block already on a device is left alone; one every device declines stays
// on the CPU and the others are still placed. It reports how many blocks it
// placed.
//
// Call it between tokens, like every placement call.
func (s *Instance) PlaceBlocks(lo, hi int) (int, error) {
	if s.ld == nil {
		return 0, fmt.Errorf("model: no device attached (SetDevice first)")
	}
	lo, hi = max(lo, s.lo), min(hi, s.hi)
	if lo >= hi {
		return 0, nil
	}
	placed, err := s.placeFromCPU(lo, hi, "")
	return len(placed), err
}

// placeFromCPU puts the CPU blocks of [lo, hi) on the device named to ("" the
// tier's choice) through the offer path, and their history after them. It
// reports the blocks placed.
func (s *Instance) placeFromCPU(lo, hi int, to string) ([]int, error) {
	if s.ld == nil {
		return nil, fmt.Errorf("model: no device attached (SetDevice first)")
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
		return nil, nil
	}
	was := s.placeErr
	s.offerOn, s.placeErr = to, nil
	s.offerRange(lo, hi)
	err := s.placeErr
	s.offerOn, s.placeErr = "", was
	var placed []int
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
		placed = append(placed, li)
	}
	return placed, err
}

// moveToCPU is MoveBlocks to the CPU: blocks [lo, hi) come off their devices
// with their history, wherever they sit in the stack -- the CPU block between
// two device runs it leaves is a placement the steps run as they run
// relocation's. A pinned block refuses the call before anything moves; a
// block whose history will not come to the CPU stays where it is, and the
// call reports it.
func (s *Instance) moveToCPU(lo, hi int) (mv BlockMove, err error) {
	if s.ld == nil {
		return mv, nil
	}
	var on []int
	for li := lo; li < hi; li++ {
		if s.devAt(li) {
			if s.pinnedOnDev(li) {
				return mv, fmt.Errorf("model: block %d is pinned to its device", li)
			}
			on = append(on, li)
		}
	}
	if len(on) == 0 {
		return mv, nil
	}
	t0 := time.Now()
	defer func() { mv.Pause = time.Since(t0) }()
	defer s.followHost()
	defer s.fixKVGroups()
	s.moved = nil
	// The head goes first, as SetGPULayers sends it: it rides the last
	// placed block's submission, and that block may be one of these.
	mv.Head = s.head != nil
	s.head, s.headReady = nil, nil
	for _, li := range on {
		if !s.migrateKV(li, s.pos, false) || !s.migrateRec(li, false) {
			return mv, fmt.Errorf("model: block %d's history would not come home", li)
		}
		s.ld.ReleaseLayers(li, li+1)
		s.unmarkOnDev(li)
		mv.Moved = append(mv.Moved, li)
		mv.KVBytes += s.blockHistoryAt(li, s.pos)
		mv.RecBytes += s.recBytesOf(li)
	}
	return mv, nil
}
