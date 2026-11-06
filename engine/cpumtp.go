package engine

import (
	"github.com/jitllm/jitllm/engine/device/cpudev"
	"github.com/jitllm/jitllm/nn"
)

// cpuPlaceMTP composes the entry of the prediction block a draft instance
// starts at, if it starts at one (mtpEntryGraph), for the CPU device: a form
// of one row and one of a chunk's.
func (s *Instance) cpuPlaceMTP() {
	fg := s.m.FrontGraphs()
	i := s.lo - s.c.NLayer
	if s.cpu.x.Device() == nil || i < 0 || i >= len(fg.MtpEntry) {
		return
	}
	g := fg.MtpEntry[i]
	var p [2]cpudev.Part
	p[0] = s.cpuPart(g, nn.DeclineNoGraph, 1)
	p[1] = s.cpuPart(g, nn.DeclineNoGraph, cpuRowsCap)
	s.entry = cpudev.NewEntry(s.cpu.x.Device(), s.cpu.x.Steps(), p[0], p[1], blockWeights{ws: &s.ws, li: s.lo, l: s.layer(s.lo)},
		s.c.ResidW())
}

// mtpEntry writes `rows` rows of z, the rows the prediction block reads,
// from the tokens toks and the rows of x (the trunk's hidden rows, every
// stream) on the device the instance places the entry on, or says why not.
func (s *Instance) mtpEntry(z, x []float32, toks []int32, rows int) nn.Decline {
	if s.entry == nil {
		return nn.DeclineCaller
	}
	return s.entry.Entry(z, x, toks, rows)
}
