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

// mtpEntryGraph is prediction block li's entry: each row's token's row of
// the block's table (its own, or the trunk's where it carries none; no
// scale) and the trunk's hidden row x, each RMS-normed by its own vector,
// side by side through eh_proj into the row the block's body reads. Over a
// trunk of several streams x is every stream, each normed by its own row of
// the hidden norm and through eh_proj beside the embedding's: the rows the
// block's streams start from, side by side.
func (m *Model) mtpEntryGraph(li int) (*graph.Graph, Uncovered, error) {
	c := m.Cfg
	roles := [...]jlm.Role{jlm.RoleNextnENorm, jlm.RoleNextnHNorm, jlm.RoleNextnEHProj, jlm.RoleNextnEmbd}
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
	for _, r := range roles[:3] {
		if _, ok := f.Inventory[r]; !ok {
			return nil, UncoveredUnverified, fmt.Errorf("prediction block %d stores no %v", li, r)
		}
	}
	d, ns := c.NEmbd, c.Streams()
	if _, ok := f.Inventory[jlm.RoleNextnEmbd]; !ok {
		// The trunk's table, which the block reads in place of its own.
		f.Inventory[jlm.RoleNextnEmbd] = graph.Shape{K: d, Rows: c.NVocab}
	}
	b := graph.NewBlock(li)
	tok := b.Emit(graph.OpEmbed, nil, []graph.Result{{Name: "e", Width: d}},
		[]graph.Binding{{Role: jlm.RoleNextnEmbd, K: d, Rows: c.NVocab}}, graph.Attrs{}, nil, nil)[0]
	e := b.Norm(tok, jlm.RoleNextnENorm, c.RMSEps)
	x := b.In("x", ns*d)
	concat := func(name string, w int, vs ...graph.ValueID) graph.ValueID {
		return b.Emit(graph.OpConcat, vs, []graph.Result{{Name: name, Width: w}}, nil, graph.Attrs{}, nil, nil)[0]
	}
	if ns == 1 {
		x = b.Norm(x, jlm.RoleNextnHNorm, c.RMSEps)
		b.Out(b.MatVec(concat("eh", 2*d, e, x), jlm.RoleNextnEHProj, d))
		return verified(b, f)
	}
	xn := b.Emit(graph.OpHeadNorm, []graph.ValueID{x}, []graph.Result{{Name: "h.normed", Width: ns * d}}, nil,
		graph.Attrs{Eps: c.RMSEps, HeadDim: d}, nil, nil)[0]
	xn = mapRow(b, "h.norm", graph.Attrs{Fn: graph.MapMul}, []graph.Binding{{Role: jlm.RoleNextnHNorm, K: ns * d}}, ns*d, xn)
	zs := make([]graph.ValueID, ns)
	for k := range zs {
		xk := b.Emit(graph.OpSlice, []graph.ValueID{xn}, []graph.Result{{Name: "h.stream", Width: d}}, nil,
			graph.Attrs{Offset: k * d, Width: d}, nil, nil)[0]
		zs[k] = b.MatVec(concat("eh", 2*d, e, xk), jlm.RoleNextnEHProj, d)
	}
	b.Out(concat("z", ns*d, zs...))
	return verified(b, f)
}
