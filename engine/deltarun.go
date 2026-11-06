package engine

import "github.com/jitllm/jitllm/engine/device/cpudev"

// allocRecurrent sizes the recurrent state, nil for every architecture that
// is not a hybrid.
//
// It is per sequence, like the KV cache: a batched session's rows share the
// weights and nothing else.
func (s *Instance) allocRecurrent(nseq int) {
	c := s.c
	if !c.Hybrid() || !s.linearIn() {
		return
	}
	s.dg = c.Delta()
}

// makeRec makes each recurrent block's state for nseq sequences in v, a
// session's history's view of the CPU device.
func (s *Instance) makeRec(v *cpudev.View, nseq int) {
	for li := range s.c.NLayer {
		if s.keepsRec(li) {
			v.MakeRec(li, nseq*s.dg.ConvStateLen(), nseq*s.dg.StateStride())
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
// invalidate, only a running summary to forget.
func (s *Instance) ResetRecurrent(row int) {
	nc, ns := s.dg.ConvStateLen(), s.dg.StateStride()
	conv, state := s.recStaged()
	for li := range s.c.NLayer {
		if !s.keepsRec(li) || !s.recIO(li, conv, state, false) {
			continue
		}
		clear(conv[row*nc : (row+1)*nc])
		clear(state[row*ns : (row+1)*ns])
		s.recIO(li, conv, state, true)
	}
}
