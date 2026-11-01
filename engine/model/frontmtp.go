package model

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// A model's prediction blocks (Config.NMTP of them, after the trunk's NLayer)
// are blocks the drafter steps (spec.go). Each one's body is stated as any
// block's; its entry's and head's tensors are not the body's to read.

// nextnRole reports whether r is a prediction block's entry or head tensor.
func nextnRole(r jlm.Role) bool {
	switch r {
	case jlm.RoleNextnEHProj, jlm.RoleNextnENorm, jlm.RoleNextnHNorm, jlm.RoleNextnHeadNorm, jlm.RoleNextnHead, jlm.RoleNextnEmbd:
		return true
	}
	return false
}

// mtpEntryGraph is prediction block li's entry: the token's embedding row e
// and the trunk's hidden row x, each RMS-normed by its own vector, side by
// side through eh_proj into the row the block's body reads.
func (m *Model) mtpEntryGraph(li int) (*graph.Graph, Uncovered, error) {
	c := m.Cfg
	if c.Streams() > 1 {
		return nil, UncoveredStreams, nil
	}
	roles := [...]jlm.Role{jlm.RoleNextnENorm, jlm.RoleNextnHNorm, jlm.RoleNextnEHProj}
	f := graph.Facts{Inventory: map[jlm.Role]graph.Shape{}}
	for i := range m.container.Entries() {
		e := &m.container.Entries()[i]
		if e.Block != int32(li) || e.Index >= 0 {
			continue
		}
		for _, r := range roles {
			if e.Role != r {
				continue
			}
			s := graph.Shape{K: e.K()}
			if e.NDim >= 2 {
				s.Rows = e.Rows()
			}
			f.Inventory[r] = s
		}
	}
	for _, r := range roles {
		if _, ok := f.Inventory[r]; !ok {
			return nil, UncoveredUnverified, fmt.Errorf("prediction block %d stores no %v", li, r)
		}
	}
	d := c.NEmbd
	b := graph.NewBlock(li)
	e := b.Norm(b.In("e", d), jlm.RoleNextnENorm, c.RMSEps)
	x := b.Norm(b.In("x", d), jlm.RoleNextnHNorm, c.RMSEps)
	eh := b.Emit(graph.OpConcat, []graph.ValueID{e, x}, []graph.Result{{Name: "eh", Width: 2 * d}}, nil, graph.Attrs{}, nil, nil)[0]
	b.Out(b.MatVec(eh, jlm.RoleNextnEHProj, d))
	return verified(b, f)
}
