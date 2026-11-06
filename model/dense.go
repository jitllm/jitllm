package model

import (
	"slices"

	"github.com/jitllm/jitllm/format/jlm"
)

// Dense is a model's tensors outside its blocks -- the embedding, the
// position table, the output projection, an embedding model's heads, the
// per-layer and stream tables, a tower's entry and
// head and its taps -- as one instance binds them at the frames of the
// device running them (Model.BindDense). The model's own are descriptors:
// shapes and entries, no bytes.
type Dense struct {
	Embd, PosEmbd, Output    Tensor
	Dense1, Dense2           Tensor
	PleTok, PleProj, AltProj Tensor
	AltUnembd                []Tensor
	TowerDense               Layer
	// Mats is the model's other dense matrices, by role (Model.DenseMats).
	Mats      []roleTensor
	TowerTaps []Layer
}

// BindDense makes d an instance's own copy of the model's dense tensors,
// bound at fr: each holds its own packed planes, so binding one copy moves
// no other.
func (m *Model) BindDense(fr *jlm.Frames, d *Dense) error {
	d.Embd, d.PosEmbd, d.Output = m.Embd, m.PosEmbd, m.Output
	d.Dense1, d.Dense2 = m.Dense1, m.Dense2
	d.PleTok, d.PleProj, d.AltProj = m.PleTok, m.PleProj, m.AltProj
	d.AltUnembd = slices.Clone(m.AltUnembd)
	ts := []*Tensor{&d.Embd, &d.PosEmbd, &d.Output, &d.Dense1, &d.Dense2, &d.PleTok, &d.PleProj, &d.AltProj}
	for i := range d.AltUnembd {
		ts = append(ts, &d.AltUnembd[i])
	}
	d.Mats = make([]roleTensor, len(m.DenseMats))
	for i, x := range m.DenseMats {
		t := *x.T
		d.Mats[i] = roleTensor{x.Role, &t}
		ts = append(ts, &t)
	}
	for _, t := range ts {
		if t.Packed != nil {
			p := *t.Packed
			t.Packed = &p
		}
		bindOne(fr, t)
	}
	m.TowerDense.CloneTo(&d.TowerDense)
	if err := bindPacked(fr, m.Cfg, &d.TowerDense); err != nil {
		return err
	}
	d.TowerTaps = make([]Layer, len(m.TowerTaps))
	for j := range m.TowerTaps {
		m.TowerTaps[j].CloneTo(&d.TowerTaps[j])
		if err := bindPacked(fr, m.Cfg, &d.TowerTaps[j]); err != nil {
			return err
		}
	}
	return nil
}

// denseNamed reports that a field of Dense names role r's matrix.
func denseNamed(r jlm.Role) bool {
	switch r {
	case jlm.RoleTokenEmbd, jlm.RolePosEmbd, jlm.RoleOutput, jlm.RoleEmbdDense1, jlm.RoleEmbdDense2,
		jlm.RolePLETokEmbd, jlm.RolePLEModelProj, jlm.RoleStreamExpand:
		return true
	}
	return r >= jlm.RoleStreamCollapse1 && r <= jlm.RoleStreamCollapse3
}

// Matrix is the dense matrix of role r a field of d does not name, nil for
// none (Dense.Mats).
func (d *Dense) Matrix(r jlm.Role) *Tensor {
	for _, x := range d.Mats {
		if x.Role == r {
			return x.T
		}
	}
	return nil
}
