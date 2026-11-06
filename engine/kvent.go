package engine

import "github.com/jitllm/jitllm/graph"

// entPlane is a block's entries plane: one entry each period positions, of
// width floats (its graph's append to history 1, OpKVAppend's Span).
type entPlane struct{ block, period, width int }

// entPlanes is the entries planes of the instance's blocks, each where its
// graph appends to a history of its own beside its rows.
func (s *Instance) entPlanes() []entPlane {
	var out []entPlane
	for li := s.lo; li < s.hi; li++ {
		g := s.m.Composing().BlockGraph(li)
		if g == nil {
			continue
		}
		for _, o := range g.Ops {
			if o.Kind == graph.OpKVAppend && len(o.Writes) == 1 && o.Writes[0].Hist == 1 {
				out = append(out, entPlane{li, max(1, o.Attr.Span), g.Values[o.In[0]].Width})
				break
			}
		}
	}
	return out
}
