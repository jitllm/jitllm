package gpu

import (
	"fmt"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/nn"
)

// The head over a residual of several planes (a model of several residual
// streams): its graph reads every plane, an input each, and binds the
// tensors of its mix past the head's own (nn.Head.Mix), held for the model
// as the picture rows' are (rowsWeight, rowsVector).

// headRole says r is a tensor the tier holds for every head: the
// projection, the norm and their biases.
func headRole(r jlm.Role) bool {
	switch r {
	case jlm.RoleOutput, jlm.RoleTokenEmbd, jlm.RoleOutputNorm, jlm.RoleOutputNormBias, jlm.RoleOutputBias:
		return true
	}
	return false
}

// mixHolds says the head's mix p holds the tensor of role r.
func mixHolds(p *nn.RowsPart, r jlm.Role) bool {
	if p == nil || headRole(r) {
		return false
	}
	if _, ok := p.Weights[r]; ok {
		return true
	}
	_, ok := p.Vectors[r]
	return ok
}

// mixParts is the formats, weights and vectors the head's mix p binds, held
// on the card.
func (g *devTier) mixParts(p *nn.RowsPart) (composedPart, error) {
	cp := composedPart{formats: map[jlm.Role]jlm.Type{}, weights: map[jlm.Role]*resident{}, vectors: map[jlm.Role]backend.Buf{}}
	for r, w := range p.Weights {
		if headRole(r) {
			continue
		}
		res, err := g.rowsWeight(p.File, w)
		if err != nil {
			return composedPart{}, fmt.Errorf("the head's mix: %v: %w", r, err)
		}
		t, _ := jlmType(res.t)
		cp.formats[r], cp.weights[r] = t, res
	}
	for r, v := range p.Vectors {
		if headRole(r) || len(v) == 0 {
			continue
		}
		buf, err := g.rowsVector(p.Graph.Block, r, v)
		if err != nil {
			return composedPart{}, fmt.Errorf("the head's mix: %v: %w", r, err)
		}
		cp.formats[r], cp.vectors[r] = jlm.TypeF32, buf
	}
	return cp, nil
}

// headIns is the head's inputs over bs's residual of rows rows: x itself
// for one plane, each plane's view of it otherwise.
func (g *devTier) headIns(bs *blockScratch, rows int) ([]backend.Buf, error) {
	if bs.p.Streams() == 1 {
		g.rows.headX[0] = bs.x
		return g.rows.headX[:], nil
	}
	_, out, err := bs.servedStreams(rows, bs.p.Streams())
	return out, err
}

// stagedIns is bs's staging rows (headIn) as the head's inputs: a plane of
// ragRows rows each.
func (bs *blockScratch) stagedIns() ([]backend.Buf, error) {
	if len(bs.headInPlanes) != 0 {
		return bs.headInPlanes, nil
	}
	n := bs.p.Streams()
	if n == 1 {
		bs.headInPlanes = []backend.Buf{bs.headIn}
		return bs.headInPlanes, nil
	}
	w := bs.ragRows * bs.p.NEmbd * 4
	out := make([]backend.Buf, n)
	for k := range out {
		v, err := backend.View(bs.headIn, k*w, w)
		if err != nil {
			return nil, fmt.Errorf("tier: plane %d of the head's staging rows: %w", k, err)
		}
		out[k] = v
	}
	bs.headInPlanes = out
	return out, nil
}
