package engine

import "github.com/jitllm/jitllm/graph"

// histPlane is history hist of a block past its rows' (history 0): one
// entry each period positions, of width floats (its graph's append to it,
// OpKVAppend's Span).
type histPlane struct{ block, hist, period, width int }

// histPlanes is every history past its rows' that the instance's blocks
// keep, each where its graph appends to it.
func (s *Instance) histPlanes() []histPlane {
	var out []histPlane
	for li := s.lo; li < s.hi; li++ {
		g := s.m.Composing().BlockGraph(li)
		if g == nil {
			continue
		}
		for _, o := range g.Ops {
			if o.Kind != graph.OpKVAppend || len(o.Writes) != 1 || o.Writes[0].Hist == 0 || o.Writes[0].From != 0 {
				continue
			}
			h := int(o.Writes[0].Hist)
			dup := false
			for _, p := range out {
				dup = dup || p.block == li && p.hist == h
			}
			if !dup {
				out = append(out, histPlane{li, h, max(1, o.Attr.Span), g.Values[o.In[0]].Width})
			}
		}
	}
	return out
}
