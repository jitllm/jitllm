package engine

import (
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/model"
)

// marksLayout is the four rotary positions of slot's n rows from cache
// position p0 as graph.OpRopePos computes them from the slot's marks: the
// layout a prompt's spans left, for a gate holding it to the reference.
func marksLayout(r *model.RopeMarks, slot, p0, n int) [][4]int {
	rows := make([][4]int, n)
	for i := range rows {
		p := p0 + i
		rows[i] = [4]int{p, p, p, p}
		for k := r.Count(slot) - 1; k >= 0; k-- {
			m := r.Mark(slot, k)
			at := int(m[graph.MarkAt])
			if at > p {
				continue
			}
			for ax := range rows[i] {
				rows[i][ax] = int(graph.MarkPosition(m, ax, uint32(p)))
			}
			break
		}
	}
	return rows
}

// spanRows is how many rows spans hold.
func spanRows(m *model.Model, spans []model.Span) int {
	n := 0
	for _, sp := range spans {
		n += sp.N(m.Cfg.NEmbd)
	}
	return n
}
