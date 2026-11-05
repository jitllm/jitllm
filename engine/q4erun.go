package engine

import (
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// allocQ4E sizes a instance's buffers for one row, and its streams.
func (s *Instance) allocQ4E() {
	c := s.m.Cfg
	if !c.Q4E() {
		return
	}
	s.q4 = &model.Q4eScratch{}
	s.growQ4E(1)
}

// growQ4E sizes the per-row buffers for n rows.
func (s *Instance) growQ4E(n int) {
	g := s.q4
	if g == nil || g.Rows >= n {
		return
	}
	g.Rows = n
}

// q4eDraftStreams lays a prediction block's draft rows, which arrive with
// every stream (s.q4In, row-major, eh_proj's per-stream output), into a chunk
// of rows whose first is row base of the call.
func (s *Instance) q4eDraftStreams(x []float32, rows, base int) {
	c := s.m.Cfg
	w, d := c.ResidW(), c.NEmbd
	for r := 0; r < rows; r++ {
		in := s.q4In[(base+r)*w : (base+r+1)*w]
		for k := 0; k < c.HCMult; k++ {
			copy(c.Stream(x, k, rows, r), in[k*d:(k+1)*d])
		}
	}
}

// q4eCollapse is the head's mix for n rows of x, written into stream 0 of
// each row, which the projection reads with no norm: the composed collapse,
// a prediction block's draft through the block's own head mix.
//
// The rows' residual, every stream, is a speculation step's hidden state on
// this model (rowsOut.tapStreams): what a prediction block reads, and what a
// draft chains to its next step.
func (s *Instance) q4eCollapse(x []float32, n int) error {
	s.growQ4E(n)
	s.rows.tapStreams(s, x, n)
	if why := s.cpuCollapse(x, n); why != nn.DeclineServes {
		return s.streamsDeclined("collapse", why)
	}
	return nil
}
