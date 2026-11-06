package engine

import (
	"unsafe"

	"github.com/jitllm/jitllm/engine/device/cpudev"
	"github.com/jitllm/jitllm/nn"
)

// An embedding model's readout on the CPU device (cpudev.Pool): the pooled,
// unit-length row (poolGraph) from what the composer forms for it, at the
// segment's rows, composed once at its placement (cpuPlacePool) and counted
// served or declined by name (CPUPoolSteps).

// cpuPlacePool composes the model's readout for the CPU at the segment's
// rows, its weights bound, and makes the CPU device the instance's readout.
func (s *Instance) cpuPlacePool() {
	g := s.m.FrontGraphs().Pool
	if g == nil {
		return
	}
	part := s.cpuPart(g, nn.DeclineNoGraph, s.maxSeq)
	var weights []unsafe.Pointer
	if f := part.F; f != nil {
		ws, why := s.ws.bind(f, -1, nil)
		if why != nn.DeclineServes {
			part.F, part.Why = nil, why
		}
		weights = ws
	}
	s.readout = cpudev.NewPool(s.cpu.x.Device(), s.cpu.x.Steps(), part, weights, s.maxSeq, s.c.NEmbd, s.poolWidth(), len(g.Out) > 1)
}

// poolWidth is the readout's width: the last dense head's rows, or the
// model's width where it has none.
func (s *Instance) poolWidth() int {
	if g := s.m.FrontGraphs().Pool; g != nil {
		return g.Values[g.Out[0]].Width
	}
	return s.c.NEmbd
}
