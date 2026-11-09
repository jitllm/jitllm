package engine

import (
	"fmt"

	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// expertsOnCPU says a placement's word for where a block's routed experts
// run names the CPU device.
func expertsOnCPU(where string) bool { return onCPU(where) || where == "host" }

// cutBlock is block li cut at the routed experts its front end states
// (graph.Cut), for a placement that runs them on the CPU device and the rest
// of the block on a card: the card's plan of the first half and the CPU
// device's statement of the second. ok is false for a block stating none.
func (s *Instance) cutBlock(li int) (first nn.LayerPlan, rest nn.BlockSpec, ok bool, err error) {
	g := s.m.Composing().BlockGraph(li)
	if g == nil || g.Routed == (graph.Region{}) {
		return first, rest, false, nil
	}
	pre, post, err := graph.Cut(g, g.Routed)
	if err != nil {
		return first, rest, false, fmt.Errorf("engine: block %d's routed experts: %w", li, err)
	}
	w := pre.Values[pre.Out[0]].Width
	first = *s.planFor(li)
	first.Graph, first.Cut = pre, w
	first.NExpert, first.NExpertUsed, first.MoEBias = 0, 0, false
	if first.Spec != nil {
		sp := s.specOf(li, pre)
		first.Spec = &sp
	}
	rest = s.specOf(li, post)
	rest.Cut = w
	return first, rest, true, nil
}

// offerCut offers block li to the card cut at its routed experts: the first
// half's plan and the block's weights but its banks and router, which are
// the CPU device's. Once the card takes it the block is placed there, and
// then the CPU device composes the second half -- after, since placing the
// block on the card lets go of what the CPU device held of it; took is false
// where the block states no routed experts or the card declines.
func (s *Instance) offerCut(adm nn.LayerDevice, li int) (took bool, err error) {
	first, rest, ok, err := s.cutBlock(li)
	if err != nil || !ok {
		return false, err
	}
	w := s.halfWeights(li)
	if took, err = s.prepOffered(adm, li, &first, &w); err != nil || !took {
		return false, err
	}
	s.markOnDev(li)
	if rest.Why != nn.DeclineServes || !s.places[routeLanes].PrepLayer(li, &nn.LayerPlan{Spec: &rest}, nil) {
		return false, fmt.Errorf("engine: the CPU device does not compose block %d's routed experts: %v", li, rest.Why)
	}
	s.holdFrame(li, routeLanes, 1)
	if len(s.cutW) < len(s.m.Layers) {
		s.cutW = append(s.cutW, make([]int, len(s.m.Layers)-len(s.cutW))...)
	}
	s.cutW[li] = first.Cut
	return true, nil
}

// cutAt says block li runs cut: its first half on its card, its second on
// the CPU device.
func (s *Instance) cutAt(li int) bool { return li < len(s.cutW) && s.cutW[li] > 0 }

// anyCut says a block of the instance runs cut: no one device steps its
// rows whole, and a step of them takes the routed arm (stepRouted).
func (s *Instance) anyCut() bool {
	for _, w := range s.cutW {
		if w > 0 {
			return true
		}
	}
	return false
}

// clearCut makes block li whole again where it leaves its card: the CPU
// device's second half and its hold on the block's frame let go.
func (s *Instance) clearCut(li int) {
	if !s.cutAt(li) {
		return
	}
	s.places[routeLanes].ReleaseLayers(li, li+1)
	s.holdFrame(li, routeLanes, -1)
	s.cutW[li] = 0
}

// runEnd is the end of a run of blocks [li, hi) on one route: the block
// after the first that runs cut, whose second half is another device's.
func (s *Instance) runEnd(li, hi int) int {
	for k := li; k < hi; k++ {
		if s.cutAt(k) {
			return k + 1
		}
	}
	return hi
}

// cutFor hands route r the cut its call reads or writes where block li, the
// call's last block or its first, runs cut: n rows of its crossing values,
// in a buffer grown to the widest call and kept.
func (s *Instance) cutFor(r *route, li, n int) {
	if !s.cutAt(li) || r.cut == nil {
		return
	}
	if w := n * s.cutW[li]; len(s.cutRow) < w {
		s.cutRow = grow32(s.cutRow, w)
	}
	r.cut.SetCut(s.cutRow[:n*s.cutW[li]])
}

// cutRest runs block li's second half on the CPU device over the n rows the
// card's first half left in the cut, writing the residual x; the caller has
// opened the CPU device's run.
func (s *Instance) cutRest(li, pos, n int, x, cs, csSWA []float32) error {
	r := &s.routes[routeLanes]
	s.cutFor(r, li, n)
	if r.run.Layers(li, li+1, pos, n, x, cs, csSWA, nil) {
		return nil
	}
	if err := s.runRefusal(r, li); err != nil {
		return err
	}
	return fmt.Errorf("engine: block %d's routed experts failed on the CPU device", li)
}
