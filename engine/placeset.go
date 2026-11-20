package engine

import (
	"fmt"
	"time"

	"github.com/jitllm/jitllm/nn"
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
	defer s.settleHist()
	if s.ld == nil {
		return 0, fmt.Errorf("model: no device attached (SetDevice first)")
	}
	lo, hi = max(lo, s.lo), min(hi, s.hi)
	if lo >= hi {
		return 0, nil
	}
	placed, err := s.placeOnto(lo, hi, "")
	return len(placed), err
}

// placeOnto puts the CPU blocks of [lo, hi) on the device named to ("" the
// tier's choice) through the offer path, and their history after them: the
// card holds it, and the CPU device's pages of it go, but where a block on
// the CPU device reads it, whose copy they are now (histcross.go). It
// reports the blocks placed.
func (s *Instance) placeOnto(lo, hi int, to string) ([]int, error) {
	if s.ld == nil {
		return nil, fmt.Errorf("model: no device attached (SetDevice first)")
	}
	defer s.followSystem()
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
	// The offers are one placement, ended as an attach's is (nn.Placer): the
	// device forms what a prompt, a step of rows and a picture of each width
	// read over the blocks it now holds, so no step after the move composes.
	if pl, ok := s.ld.(nn.Placer); ok {
		pl.BeginPlacement()
		s.offerRange(lo, hi)
		pl.EndPlacement()
	} else {
		s.offerRange(lo, hi)
	}
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
		s.dropUnread(li)
		placed = append(placed, li)
	}
	return placed, err
}

// moveOffCards is MoveBlocks to the CPU: blocks [lo, hi) come off their devices
// with their history, wherever they sit in the stack -- the CPU block between
// two device runs it leaves is a placement the steps run as they run
// relocation's. A pinned block refuses the call before anything moves; a
// block whose history will not come to the CPU stays where it is, and the
// call reports it.
func (s *Instance) moveOffCards(lo, hi int) (mv BlockMove, err error) {
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
	// The projection stays on its card while a block of this instance is
	// left beside it, as a move between cards leaves it (Commit). With the
	// last of them it comes to the CPU, and the next move to a card takes it
	// there (headHome). Asked once fixKVGroups has brought every group home.
	nd, named := s.ld.(nn.NamedDevice)
	headWas, beside := "", false
	if named && s.head != nil {
		var on bool
		headWas, on = nd.HeadDeviceName()
		beside = on && s.anyBlockOn(nd, headWas)
	}
	defer func() {
		if s.head != nil && (!named || beside && !s.anyBlockOn(nd, headWas)) {
			s.head, s.headHome, mv.Head = nil, true, true
		}
	}()
	defer s.followSystem()
	defer s.fixKVGroups()
	s.moved = nil
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
