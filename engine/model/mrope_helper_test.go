package model

import "github.com/jitllm/jitllm/graph"

// marksLayout is the four rotary positions of slot's n rows from cache
// position p0 as graph.OpRopePos computes them from the slot's marks: the
// layout a prompt's spans left, for a gate holding it to the reference.
func marksLayout(r *ropeMarks, slot, p0, n int) [][4]int {
	rows := make([][4]int, n)
	for i := range rows {
		p := p0 + i
		rows[i] = [4]int{p, p, p, p}
		for k := r.count(slot) - 1; k >= 0; k-- {
			m := r.mark(slot, k)
			at := int(m[graph.MarkAt])
			if at > p {
				continue
			}
			j, base := p-at, int(m[graph.MarkBase])
			h, w := int(m[graph.MarkH]), int(m[graph.MarkW])
			switch m[graph.MarkKind] {
			case graph.MarkText:
				rows[i] = [4]int{j + base, j + base, j + base, j + base}
			case graph.MarkGrid:
				t := base + j/(h*w)
				rows[i] = [4]int{t, base + (j/w)%h, base + j%w, t}
			default:
				ends := int(m[graph.MarkEnds])
				if jj := j - ends; j >= ends && jj < h*w {
					rows[i] = [4]int{p, jj % w, jj / w, int(m[graph.MarkIndex])}
				}
			}
			break
		}
	}
	return rows
}

// spanRows is how many rows spans hold.
func spanRows(m *Model, spans []Span) int {
	n := 0
	for _, sp := range spans {
		n += sp.n(m.Cfg.NEmbd)
	}
	return n
}
