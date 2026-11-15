package engine

import (
	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// A residual of several planes (Config.Streams): a call of n rows holds
// plane k's rows from k*n*NEmbd on. The input rows' graph writes the planes
// it expands into and the head's graph reads every plane, so the runner
// hands both the residual as it is.

// streamsPart is the tensors graph g binds past the head's own and the
// embedding, by role, as the CPU binds them with no block (weightSource.matrix,
// weightSource.vector): a card's for the input rows' planes and the head's
// mix of them. nil where g is, or binds a tensor the model does not hold.
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

// allocStreams gives the decode token's residual every plane the model
// carries (xa), x its plane 0.
func (s *Instance) allocStreams() {
	c := s.m.Cfg
	if c.Streams() < 2 {
		return
	}
	s.xa = make([]float32, c.ResidW())
	s.x = s.xa[:c.NEmbd]
}

// plane is the floats between two planes of a residual of n rows.
func (s *Instance) plane(n int) int { return n * s.c.NEmbd }

// layDraft lays a prediction block's draft rows, which arrive with every
// plane (s.draftIn, row-major), into a chunk of rows whose first is row base
// of the call, over what the input rows wrote; nothing where the call is not
// a draft's.
func (s *Instance) layDraft(x []float32, rows, base int) {
	if s.draftIn == nil {
		return
	}
	c := s.m.Cfg
	w, d := c.ResidW(), c.NEmbd
	for r := 0; r < rows; r++ {
		in := s.draftIn[(base+r)*w : (base+r+1)*w]
		for k := range c.Streams() {
			copy(c.Stream(x, k, rows, r), in[k*d:(k+1)*d])
		}
	}
}

// devHeadReads says the device's head reads the instance's residual as it
// is: a model of one plane, or a device that mixes the planes itself
// (nn.Head.Collapses), the model's mix, not a prediction block's.
func (s *Instance) devHeadReads() bool {
	return s.c.Streams() == 1 || s.head != nil && s.head.Collapses && s.lo < s.c.NLayer
}
