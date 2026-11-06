package engine

import (
	"slices"

	"github.com/jitllm/jitllm/nn"
)

// flatSlots is each row's place in the device's flat history, sequence q's
// position p at q*maxSeq+p (nn.RowsDevice).
func (s *Instance) flatSlots(seq, pos []int) []int {
	s.rrFlat = slices.Grow(s.rrFlat[:0], len(seq))[:len(seq)]
	slot := s.rrFlat
	for r, q := range seq {
		slot[r] = q*s.maxSeq + pos[r]
	}
	return slot
}

// onRowsDevice is the device a batch step can run on: every block and the head
// on one device that runs ragged rows. A final softcap rides the head there
// as it does in decode (the tier caps each row's logits before its argmax).
func (s *Instance) onRowsDevice() (nn.RowsDevice, bool) {
	return s.ldRows, s.rowsRefusal() == nil
}

// rowsRefusal is why onRowsDevice is false, nil where it is true.
func (s *Instance) rowsRefusal() error {
	switch {
	case s.ldRows == nil:
		return errStepNoRows
	case !s.headWithLastBlock():
		return errStepNoHead
	case s.devCount() != s.hi-s.lo:
		return errStepSplit
	}
	return nil
}

// headWithLastBlock is the head on a device and the last block on one, which
// is what a rows step takes the projection with (gpu.GPU.LayersRows): on the
// last block's device it rides that block's step, and on another -- the card
// holding the blocks had no room for it -- it is a step of its own there.
func (s *Instance) headWithLastBlock() bool {
	return s.head != nil && s.devAt(s.hi-1)
}
