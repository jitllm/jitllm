package model

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/engine/nn"
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

// rowsRun runs device blocks [lo, hi) of a split placement for the rows at
// (pos[i], slot[i]), whose residual is x; with head it also takes the
// projection and reads the first nlogit rows' logits into logits. The host runs
// the blocks around it through its own batched path.
func (s *Instance) rowsRun(lo, hi int, pos, slot []int, x, cs, csSWA []float32, head bool, logits []float32, nlogit int) error {
	rd := s.ldRows
	if rd == nil {
		return errBatchDevice{}
	}
	var h *nn.Head
	if head && nlogit > 0 {
		h = s.head
		keep := h.Logits
		h.RowLogits, h.Logits, h.LogitRows = true, logits, nlogit
		defer func() { h.RowLogits, h.Logits, h.LogitRows = false, keep, 0 }()
	}
	s.layerInputs(s.bple[:len(pos)*s.c.sideWidth()])
	if !rd.LayersRows(lo, hi, pos, slot, s.maxSeq, x, cs, csSWA, h) {
		why := "no reason given"
		if e, ok := s.ld.(nn.ErrReporter); ok {
			why = e.Err()
		}
		return fmt.Errorf("model: the device refused blocks [%d,%d) for %d rows: %s", lo, hi, len(pos), why)
	}
	return nil
}

// rowsSessions is rowsRun for a step across sessions (stepHost): blocks
// [lo, hi) on the device for every row at its own position, each row through
// its own session's view of the device (nn.SessionStepper), so its history
// there is its own session's.
func (s *Instance) rowsSessions(lo, hi int, pos, _ []int, x, cs, csSWA []float32, head bool, logits []float32, nlogit int) error {
	ss := s.ldStep
	if ss == nil {
		return errBatchDevice{}
	}
	var h *nn.Head
	if head && nlogit > 0 {
		h = s.head
		keep := h.Logits
		h.RowLogits, h.Logits, h.LogitRows = true, logits, nlogit
		defer func() { h.RowLogits, h.Logits, h.LogitRows = false, keep, 0 }()
	}
	views := s.stepSess[:0]
	for _, o := range s.rowOwn {
		views = append(views, o.ld)
	}
	defer func() { s.stepSess = clearDevs(views) }()
	s.layerInputs(s.bple[:len(pos)*s.c.sideWidth()])
	if !ss.LayersSessions(lo, hi, views, pos, x, cs, csSWA, h) {
		why := "no reason given"
		if e, ok := s.ld.(nn.ErrReporter); ok {
			why = e.Err()
		}
		return fmt.Errorf("model: the device refused blocks [%d,%d) for %d rows across sessions: %s", lo, hi, len(pos), why)
	}
	return nil
}
