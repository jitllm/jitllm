package engine

import (
	"fmt"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// A model of several residual streams (Config.Streams): their expansion
// from the input rows and their collapse before the head.

// streamsPart is the tensors graph g binds past the head's own and the
// embedding, by role, as the CPU binds them with no block (cpuRoleWeight,
// cpuRoleVector): a device's for the streams' expansion in the input rows
// and their collapse before the head. nil where g is, or binds a tensor the
// model does not hold.
func (s *Instance) streamsPart(g *graph.Graph) *nn.RowsPart {
	if g == nil {
		return nil
	}
	p := &nn.RowsPart{Graph: g, Rows: 1, Weights: map[jlm.Role]nn.Weight{}, Vectors: map[jlm.Role][]float32{}}
	for _, o := range g.Ops {
		for _, b := range o.Weight {
			switch b.Role {
			case jlm.RoleOutput, jlm.RoleTokenEmbd, jlm.RolePosEmbd:
				continue
			}
			if o.Kind != graph.OpEmbed && (b.Rows == 0 || b.Role.Expanded()) {
				v := s.cpuRoleVector(-1, nil, b.Role)
				if len(v) == 0 {
					return nil
				}
				p.Vectors[b.Role] = v
				continue
			}
			t := s.cpuRoleWeight(-1, nil, b.Role)
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
// (q4In).
func (s *Instance) expandStreams(x []float32, n, base int) error {
	if s.q4In != nil {
		s.q4eDraftStreams(x, n, base)
		return nil
	}
	if !s.m.ExpandsStreams() {
		return nil
	}
	if why := s.cpuExpand(x, n); why != composer.DeclineServes {
		return s.streamsDeclined("expansion", why)
	}
	return nil
}

// collapseStreams writes the head's input into stream 0 of n rows of x
// through the composed collapse.
func (s *Instance) collapseStreams(x []float32, n int) error {
	if s.c.Q4E() {
		return s.q4eCollapse(x, n)
	}
	if why := s.cpuCollapse(x, n); why != composer.DeclineServes {
		return s.streamsDeclined("collapse", why)
	}
	return nil
}

// streamsDeclined is the error of a streams' expansion or collapse no
// composed form serves.
func (s *Instance) streamsDeclined(what string, why composer.Decline) error {
	return fmt.Errorf("model: the streams' %s has no composed form on this instance (%s): %v %v",
		what, composer.DeclineName(int(why)), s.cpu.formErr, s.m.FrontGraphs().StreamsErr)
}
