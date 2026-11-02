package tier

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/graph"
)

// The shadow arm (Config.ComposedShadow) runs the composer's form of a block
// beside the tier's own kernels on the same input at every step, and a
// reduction the composer forms writes how far apart the two outputs are into
// the block's own slot.
//
// The tier's kernels serve. At a block the composer would serve, the input row
// is copied twice, once for the comparison and once for the form, which runs
// on its copy in place as a served block does. The tier's kernels then run the
// block on the stream, and the comparison reads their output beside the
// form's. Both are launches of the submission, so a recorded step carries
// them.
//
// The history is the tier's: the form appends its key and value to the page
// the step writes before the tier's kernels write the same row, so the row
// that stays is theirs and the logits are what the tier computes without the
// arm (TestComposedShadowHoldsEveryBlock).

// shadowing says the shadow arm runs: it compares the forms of the blocks the
// composer is asked for, so it needs them asked for.
func (g *devTier) shadowing() bool {
	return g.ComposedShadow
}

// The comparison's inputs and its output, in the order of graph.Drift: the
// block's input, the tier's output, the form's, and the state, which the
// output replaces in place.
const (
	driftState = 3
	driftNext  = 0
)

// servedShadow is the comparison of block l's two outputs for rows of n
// floats, formed through the composer like any other part. It is kept under
// the block's graph, so it goes with the block (dropServed). The state is
// bound as its input and its output, so a form that read it at or after the
// kernel writing the new one is refused.
func (g *devTier) servedShadow(l *layer, n int) (*servedPart, error) {
	sk := servedKey{kind: partShadow, graph: l.graph, rows: 1}
	sv, err := g.servedFor(sk, func() (composedPart, error) {
		return composedPart{kind: partShadow, graph: (&graph.Drift{NEmbd: n}).Build()}, nil
	})
	if err == nil && !sv.f.inPlace(driftState, driftNext) {
		err = fmt.Errorf("the composed shadow reads its state after writing the new one")
	}
	return sv, err
}

// shadowStats fills s with every placed block's worst drift and the steps
// compared, read from the device. Callers hold g.mu.
func (g *devTier) shadowStats(s *Stats) {
	n := 0
	for li, l := range g.layers {
		if l != nil && l.shadow != nil {
			n = max(n, li+1)
		}
	}
	if n == 0 {
		return
	}
	s.ShadowWorst, s.ShadowSteps = make([]float32, n), make([]int, n)
	var raw [8]byte
	for li, l := range g.layers {
		if l == nil || l.shadow == nil {
			continue
		}
		if err := l.shadow.Read(raw[:]); err != nil {
			s.LastErr = fmt.Sprintf("reading block %d's shadow state: %v", li, err)
			continue
		}
		s.ShadowWorst[li] = math.Float32frombits(binary.LittleEndian.Uint32(raw[:4]))
		s.ShadowSteps[li] = int(math.Float32frombits(binary.LittleEndian.Uint32(raw[4:])))
	}
}
