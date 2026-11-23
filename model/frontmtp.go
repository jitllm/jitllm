package model

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// A model's prediction blocks (Config.NPredict of them, the blocks the
// container states past the trunk's NLayer) are blocks the drafter steps
// (spec.go). Each one's body is stated as any block's; its entry's and
// head's tensors are not the body's to read.

// predictRole reports whether r is a prediction block's entry or head tensor.
func predictRole(r jlm.Role) bool {
	switch r {
	case jlm.RolePredictProj, jlm.RolePredictTokenNorm, jlm.RolePredictHiddenNorm, jlm.RolePredictHeadNorm, jlm.RolePredictHead, jlm.RolePredictEmbd:
		return true
	}
	return false
}

// mtpEntryGraph is prediction block li's entry as the container states it:
// each row's token's row and the trunk's hidden row, each normed, side by
// side through the block's projection into the row its body reads; over a
// trunk of several streams every stream so, side by side. It reads the
// block's own entry tensors and the trunk's table where it carries none.
func (m *Model) mtpEntryGraph(li int) (*graph.Graph, Uncovered, error) {
	c := m.Cfg
	roles := [...]jlm.Role{jlm.RolePredictTokenNorm, jlm.RolePredictHiddenNorm, jlm.RolePredictProj, jlm.RolePredictEmbd}
	f := graph.Facts{Inventory: map[jlm.Role]graph.Shape{}}
	for i := range m.Container.Entries() {
		e := &m.Container.Entries()[i]
		if e.Block != int32(m.Container.PageOf(li)) || e.Index >= 0 {
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
	if _, ok := f.Inventory[jlm.RolePredictEmbd]; !ok {
		// The trunk's table, which the block reads in place of its own.
		f.Inventory[jlm.RolePredictEmbd] = graph.Shape{K: c.NEmbd, Rows: c.NVocab}
	}
	return m.statedPart(jlm.FrontPredict(jlm.FrontPredictEntry, li-c.NLayer), li, f)
}
