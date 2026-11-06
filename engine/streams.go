package engine

import (
	"fmt"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// A model of several residual streams (Config.Streams): their expansion
// from the input rows and their collapse before the head.

// streamsPart is the tensors graph g binds past the head's own and the
// embedding, by role, as the CPU binds them with no block (weightSource.matrix,
// weightSource.vector): a device's for the streams' expansion in the input rows
// and their collapse before the head. nil where g is, or binds a tensor the
// model does not hold.
func (s *Instance) streamsPart(g *graph.Graph) *nn.RowsPart {
	if g == nil {
		return nil
	}
	p := &nn.RowsPart{Graph: g, Rows: 1, Weights: map[jlm.Role]nn.Weight{}, Vectors: map[jlm.Role][]float32{}, File: s.m.Container}
	for _, o := range g.Ops {
		for _, b := range o.Weight {
			switch b.Role {
			case jlm.RoleOutput, jlm.RoleTokenEmbd, jlm.RolePosEmbd:
				continue
			}
			if o.Kind != graph.OpEmbed && (b.Rows == 0 || b.Role.Expanded()) {
				v := s.ws.vector(-1, nil, b.Role)
				if len(v) == 0 {
					return nil
				}
				p.Vectors[b.Role] = v
				continue
			}
			t := s.ws.matrix(-1, nil, b.Role)
			if t == nil || t.E == nil {
				return nil
			}
			p.Weights[b.Role] = model.Wt(*t)
		}
	}
	return p
}

// allocStreams gives the decode token's residual every stream the model
// carries (xa), x its stream 0.
func (s *Instance) allocStreams() {
	c := s.m.Cfg
	if c.Streams() < 2 {
		return
	}
	s.xa = make([]float32, c.ResidW())
	s.x = s.xa[:c.NEmbd]
}

// expandStreams fills the streams past the first of n rows of x, the
// chunk's first row row base of the call, through the composed expansion of
// the embedding in stream 0: its copies or its predictions. A bank of
// checkpoints states none, each block writing its checkpoint before any
// block reads it; a prediction block's draft rows arrive with every stream
// (draftIn).
func (s *Instance) expandStreams(x []float32, n, base int) error {
	if s.draftIn != nil {
		s.draftStreams(x, n, base)
		return nil
	}
	if !s.m.ExpandsStreams() {
		return nil
	}
	if why := s.cpuExpand(x, n); why != nn.DeclineServes {
		return s.streamsDeclined("expansion", why)
	}
	return nil
}

// collapseStreams writes the head's input into stream 0 of n rows of x
// through the composed collapse.
//
// Over a residual of every stream the rows' residual is a speculation
// step's hidden state (rowsOut.tapStreams): what a prediction block reads,
// and what a draft chains to its next step.
func (s *Instance) collapseStreams(x []float32, n int) error {
	if s.c.ResidStreams() {
		s.rows.tapStreams(s, x, n)
	}
	if why := s.cpuCollapse(x, n); why != nn.DeclineServes {
		return s.streamsDeclined("collapse", why)
	}
	return nil
}

// streamsDeclined is the error of a streams' expansion or collapse no
// composed form serves.
func (s *Instance) streamsDeclined(what string, why nn.Decline) error {
	return fmt.Errorf("model: the streams' %s has no composed form on this instance (%s): %v %v",
		what, nn.DeclineName(int(why)), s.cpu.formErr, s.m.FrontGraphs().StreamsErr)
}

// draftStreams lays a prediction block's draft rows, which arrive with every
// stream (s.draftIn, row-major), into a chunk of rows whose first is row base
// of the call.
func (s *Instance) draftStreams(x []float32, rows, base int) {
	c := s.m.Cfg
	w, d := c.ResidW(), c.NEmbd
	for r := 0; r < rows; r++ {
		in := s.draftIn[(base+r)*w : (base+r+1)*w]
		for k := range c.Streams() {
			copy(c.Stream(x, k, rows, r), in[k*d:(k+1)*d])
		}
	}
}
