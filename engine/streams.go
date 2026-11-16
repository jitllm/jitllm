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
// embedding, by role, as the CPU binds them (weightSource.matrix,
// weightSource.vector): block li's first where li is a block the instance
// runs, the model's past its blocks otherwise -- a card's for the input rows'
// planes and the head's mix of them. nil where g is, or binds a tensor
// neither holds.
func (s *Instance) streamsPart(g *graph.Graph, li int) *nn.RowsPart {
	if g == nil {
		return nil
	}
	var l *model.Layer
	if li >= s.lo && li < s.hi {
		l = s.layer(li)
	} else {
		li = -1
	}
	vector := func(r jlm.Role) []float32 {
		if v := s.ws.vector(li, l, r); len(v) != 0 || li < 0 {
			return v
		}
		return s.ws.vector(-1, nil, r)
	}
	matrix := func(r jlm.Role) *model.Tensor {
		if t := s.ws.matrix(li, l, r); (t != nil && t.E != nil) || li < 0 {
			return t
		}
		return s.ws.matrix(-1, nil, r)
	}
	p := &nn.RowsPart{Graph: g, Rows: 1, Weights: map[jlm.Role]nn.Weight{}, Vectors: map[jlm.Role][]float32{}, File: s.m.Container}
	for _, o := range g.Ops {
		for _, b := range o.Weight {
			switch b.Role {
			case jlm.RoleOutput, jlm.RoleTokenEmbd, jlm.RolePosEmbd:
				continue
			}
			if o.Kind != graph.OpEmbed && (b.Rows == 0 || b.Role.Expanded()) {
				v := vector(b.Role)
				if len(v) == 0 {
					return nil
				}
				p.Vectors[b.Role] = v
				continue
			}
			t := matrix(b.Role)
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

// wantsResidual says the step's rows want their residual of every plane
// back as their hidden state (Config.ResidStreams), which a head folded into
// the blocks' call keeps on the device.
func (s *Instance) wantsResidual() bool { return s.rows != nil && s.c.ResidStreams() }

// ownHead is the head the instance projects through, on every device: the
// prediction block's own where the block it starts at states one (its graph
// over that block's mix of the streams), the model's otherwise; block is the
// block whose tensors the head binds before the model's, -1 for none.
func (s *Instance) ownHead() (head, hidden *graph.Graph, block int) {
	fg := s.m.FrontGraphs()
	if i := s.lo - s.c.NLayer; i >= 0 && i < len(fg.MtpHead) && fg.MtpHead[i] != nil {
		return fg.MtpHead[i], fg.MtpHidden[i], s.lo
	}
	return fg.CpuHead, fg.CpuHidden, -1
}
