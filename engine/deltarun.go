package engine

import (
	"fmt"

	"github.com/jitllm/jitllm/nn/history"
)

// allocRecurrent sizes the recurrent state, nil for every architecture that
// is not a hybrid.
//
// It is per sequence, like the KV cache: a batched session's rows share the
// weights and nothing else.
func (s *Instance) allocRecurrent() error {
	c := s.c
	if !c.Hybrid() || !s.linearIn() {
		return nil
	}
	g := s.m.Composing()
	first := -1
	for li := range c.NLayer {
		if !c.LayerKind(li).Recurrent() {
			continue
		}
		conv, state := g.BlockGraph(li).RecurrentState()
		if conv == 0 && state == 0 {
			// A block of a recurrent kind whose graph keeps nothing (a
			// bidirectional model's convolution over its rows) needs none.
			continue
		}
		if first < 0 {
			first, s.recConv, s.recState = li, conv, state
			continue
		}
		// The instance stages one block's state at a time in one buffer
		// (recStaged), sized once: a choice of this code, not of the model.
		if conv != s.recConv || state != s.recState {
			return fmt.Errorf("engine: block %d keeps %d window and %d state floats a sequence, block %d %d and %d: "+
				"an instance stages one size of recurrent state", li, conv, state, first, s.recConv, s.recState)
		}
	}
	return nil
}

// makeRec makes each recurrent block's state for nseq sequences in v, a
// session's history's view of the CPU device.
func (s *Instance) makeRec(v *history.View, nseq int) {
	for li := range s.c.NLayer {
		if s.keepsRec(li) {
			v.MakeRec(li, nseq*s.recConv, nseq*s.recState)
		}
	}
}

// linearIn reports whether any block this instance runs is a linear one. A draft
// Instance over a prediction block runs attention alone, so it keeps no summary
// and recurrent() is false for it.
func (s *Instance) linearIn() bool {
	for li := s.lo; li < s.hi; li++ {
		if s.c.LayerKind(li).Recurrent() {
			return true
		}
	}
	return false
}

// ResetRecurrent clears one row's recurrent state, which is what retiring a
// sequence means for a linear layer: there is no per-position cache to
// invalidate, only a running summary to forget. A device that holds a block's
// state zeroes the row where it is (nn.RecRowsDevice), once for every block
// it holds; any other's state comes up, is cleared and goes back. A device
// that cannot is the next call's error.
func (s *Instance) ResetRecurrent(row int) {
	nc, ns := s.recConv, s.recState
	conv, state := s.recStaged()
	var zeroed [len(s.routes)]bool
	s.recRow[0] = row
	for li := range s.c.NLayer {
		if !s.keepsRec(li) {
			continue
		}
		i := s.routeOf(li)
		if rr := s.routes[i].recrows; rr != nil {
			if !zeroed[i] {
				zeroed[i] = true
				if !rr.ResetRecRows(s.recRow[:]) {
					why := ""
					if er := s.routes[i].errs; er != nil && er.Err() != "" {
						why = ": " + er.Err()
					}
					s.retireErr = fmt.Errorf("model: the recurrent state on the device could not be reset%s", why)
				}
			}
			continue
		}
		if !s.recIO(li, conv, state, false) {
			continue
		}
		clear(conv[row*nc : (row+1)*nc])
		clear(state[row*ns : (row+1)*ns])
		s.recIO(li, conv, state, true)
	}
}
