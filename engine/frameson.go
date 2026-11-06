package engine

import (
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/nn"
)

// FramesOn is the frames device dev reads this instance's blocks into: the
// CPU device's for nn.CPU, a card's own for its name (nn.FramesHolder), nil
// for a device that holds none of the model.
func (s *Instance) FramesOn(dev string) *jlm.Frames {
	if onCPU(dev) {
		return s.fr
	}
	if fh, ok := s.ld.(nn.FramesHolder); ok {
		return fh.FramesOf(dev, s.m.Container)
	}
	return nil
}
