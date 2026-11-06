package engine

import (
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// weightSource is where an instance's forms read their weights: the CPU
// device's bindings of them (cpudev.Bindings), each block's copy read as
// model.Layer holds it and the model's own tensors as modelTensors does --
// the head's, and the input rows' where they embed.
type weightSource struct {
	head, input modelTensors
}

// newWeightSource is the model's tensors as forms read them, the head's
// projected through outW and normed by outNorm.
func newWeightSource(m *model.Model, dense *model.Dense, outW *model.Tensor, outNorm []float32) weightSource {
	t := modelTensors{m: m, dense: dense, outW: outW, outNorm: outNorm}
	w := weightSource{head: t, input: t}
	w.input.input = true
	return w
}

// tensors is where block li's forms read their weights: l, its copy, or
// with no block the model's own.
func (w *weightSource) tensors(li int, l *model.Layer) nn.Tensors {
	switch {
	case l != nil:
		return l
	case li == cpuInputRows:
		return &w.input
	}
	return &w.head
}

// matrix is block li's (l's) matrix of role r, nil for one it does not
// hold; with no block, the model's.
func (w *weightSource) matrix(li int, l *model.Layer, r jlm.Role) *model.Tensor {
	switch {
	case l != nil:
		return l.Held(r)
	case li == cpuInputRows:
		return w.input.held(r)
	}
	return w.head.held(r)
}

// vector is block li's (l's) vector of role r, nil for one it does not
// hold; with no block, the model's.
func (w *weightSource) vector(li int, l *model.Layer, r jlm.Role) []float32 {
	return w.tensors(li, l).Floats(r)
}

// modelTensors is the model's own tensors as a form reads them
// (nn.Tensors): the dense ones and the head the instance projects
// through, and with input the input rows' embeddings.
type modelTensors struct {
	m       *model.Model
	dense   *model.Dense
	outW    *model.Tensor
	outNorm []float32
	input   bool
}

// held is the model's matrix of role r, nil for one it does not hold.
func (t *modelTensors) held(r jlm.Role) *model.Tensor {
	switch m := t.dense; {
	case t.input && r == jlm.RoleTokenEmbd:
		return &m.Embd
	case t.input && r == jlm.RolePosEmbd:
		return &m.PosEmbd
	case r == jlm.RoleOutput || r == jlm.RoleTokenEmbd:
		return t.outW
	case r == jlm.RolePLETokEmbd && m.PleTok.E != nil:
		return &m.PleTok
	case r == jlm.RolePLEModelProj && m.PleProj.E != nil:
		return &m.PleProj
	case r == jlm.RoleEmbdDense1:
		return &m.Dense1
	case r == jlm.RoleEmbdDense2:
		return &m.Dense2
	}
	for _, v := range t.m.VecMats {
		if v.Role == r {
			return v.T
		}
	}
	return t.dense.Matrix(r)
}

// Bytes is nn.Tensors': the model's matrix of role r as a form reads it.
func (t *modelTensors) Bytes(r jlm.Role) ([]byte, *nn.Packed, int) { return t.held(r).Bytes() }

// Floats is nn.Tensors': the model's vector of role r, nil for none.
func (t *modelTensors) Floats(r jlm.Role) []float32 {
	switch r {
	case jlm.RoleOutputNorm:
		return t.outNorm
	case jlm.RoleOutputNormBias:
		return t.m.OutNormB
	case jlm.RoleOutputBias:
		return t.m.OutB
	case jlm.RolePLEProjNorm:
		return t.m.PleNorm
	}
	return nn.VectorOf(t.m.Vecs, r)
}
