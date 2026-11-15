package engine

import (
	"fmt"

	"github.com/jitllm/jitllm/nn"
)

// resid is the one-row residual a device is handed: every stream.
func (s *Instance) resid() []float32 {
	if s.xa != nil {
		return s.xa
	}
	return s.x
}

// altRowsStep is rowsStep's device call for a model whose rows are
// embedded: the blocks on the device without the head, and the head on the
// host over the rows' residual, finished as the device's is (the final
// softcap; the caller adds the rest). With no logits wanted the rows' greedy
// tokens go to the head's Tokens, as the device's argmax would.
func (s *Instance) altRowsStep(rd nn.RowsDevice, pos, slot []int, tokens []int32, n int, logits []float32,
	nlogit int) error {
	c := s.m.Cfg
	x := s.bx[:n*c.ResidW()]
	if !rd.LayersRows(s.lo, s.hi, pos, slot, s.maxSeq, x, s.batchTables(n), nil) {
		why := "no reason given"
		if e, ok := s.ld.(nn.ErrReporter); ok {
			why = e.Err()
		}
		return fmt.Errorf("model: the device refused a %d-row step: %s", n, why)
	}
	greedy := logits == nil
	if greedy {
		nlogit = n
		if len(s.blogits) < n*c.NVocab {
			s.blogits = make([]float32, n*c.NVocab)
		}
		logits = s.blogits[:n*c.NVocab]
	}
	if nlogit == 0 {
		return nil
	}
	if c.ResidStreams() {
		s.rows.tapStreams(s, x, n)
	}
	if why := s.headRows(x, logits, nil, nlogit, s.plane(n)); why != nn.DeclineServes {
		return s.headDeclined(why)
	}
	if greedy {
		if len(s.head.Tokens) < n {
			s.head.Tokens = make([]int32, n)
		}
		for i := 0; i < n; i++ {
			tok, err := s.Sample(nil, logits[i*c.NVocab:(i+1)*c.NVocab])
			if err != nil {
				return err
			}
			s.head.Tokens[i] = tok
		}
	}
	return nil
}
