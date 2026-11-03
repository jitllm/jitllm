package model

import (
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// A model of several residual streams (Config.Streams): their expansion
// from the input rows and their collapse before the head.

// streamHead says the head reads a collapse of several residual streams,
// which the host takes (AltUp's predictions, a residual bank, gated
// hyper-connections): a device runs the blocks and hands every stream back,
// and never folds the head into them.
func (c *Config) streamHead() bool { return c.PredStreams != 0 || c.ResAttn() || c.Q4E() }

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
			if b.Rows == 0 {
				v := s.cpuRoleVector(-1, nil, b.Role)
				if len(v) == 0 {
					return nil
				}
				p.Vectors[b.Role] = v
				continue
			}
			t := s.cpuRoleWeight(-1, nil, b.Role)
			if t == nil || t.e == nil {
				return nil
			}
			p.Weights[b.Role] = wt(*t)
		}
	}
	return p
}

// expandStreams fills every stream of n rows of x from its embedding in
// stream 0.
func (s *Instance) expandStreams(x []float32, n int) error {
	if s.m.Cfg.ResAttn() {
		s.k3Expand(x, n)
		return nil
	}
	if s.m.Cfg.Q4E() {
		s.q4eExpand(x, n)
		return nil
	}
	return s.altExpand(x, n)
}

// collapseStreams writes the head's input into stream 0 of n rows of x.
func (s *Instance) collapseStreams(x []float32, n int) error {
	if s.m.Cfg.ResAttn() {
		return s.k3Collapse(x, n)
	}
	if s.m.Cfg.Q4E() {
		return s.q4eCollapse(x, n)
	}
	return s.altCollapse(x, n)
}
