package engine

import (
	"github.com/jitllm/jitllm/nn"
)

// mtpEntry writes `rows` rows of z, the rows the prediction block reads,
// from the tokens toks and the rows of x (the trunk's hidden rows, every
// stream) on the device the instance places the entry on, or says why not.
func (s *Instance) mtpEntry(z, x []float32, toks []int32, rows int) nn.Decline {
	if s.entry == nil {
		return nn.DeclineCaller
	}
	return s.entry.Entry(z, x, toks, rows)
}
