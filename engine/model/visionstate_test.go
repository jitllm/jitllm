package model

import "github.com/jitllm/jitllm/engine/nn"

// testState is an instance over the tower's vision segment for a gate that
// encodes on its own: on share's JIT when one is given (a text instance's, as
// Instance.Vision does), on one of its own otherwise, which the instance closes. A
// session reaches the segment through Instance.Vision; this is the same instance
// without a text instance around it.
func (t *Tower) testState(share ...*nn.JIT) *Instance {
	if len(share) > 0 {
		return t.newState(share[0], false)
	}
	return t.newState(t.model.newJIT(), true)
}
