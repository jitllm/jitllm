package engine

import (
	"unsafe"

	"github.com/jitllm/jitllm/engine/device/cpudev"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// blockWeights is a part's weights as block li reads them (l its copy, nil
// for the model's dense tensors): where the device binds a part's form.
type blockWeights struct {
	ws *weightSource
	li int
	l  *model.Layer
}

func (b blockWeights) Bind(f *cpudev.Form) ([]unsafe.Pointer, nn.Decline) {
	return b.ws.bind(f, b.li, b.l)
}

// cpuPlaceStreams composes the streams' expansion and collapse for the CPU
// device at the instance's placement, a decode's row at one form and a
// chunk's rows at cpuRowsCap (frontstreams.go): the collapse a prediction
// block's own head mix where the instance runs one.
func (s *Instance) cpuPlaceStreams() {
	if !s.c.StatesStreams() || s.cpu.x.Device() == nil {
		return
	}
	fg := s.m.FrontGraphs()
	var p [4]cpudev.Part
	p[0] = s.cpuPart(fg.Expand, nn.DeclineNoGraph, 1)
	p[1] = s.cpuPart(fg.Expand, nn.DeclineNoGraph, cpuRowsCap)
	collapse := fg.Collapse
	cw := blockWeights{ws: &s.ws, li: -1}
	if i := s.lo - s.c.NLayer; i >= 0 && i < len(fg.MtpCollapse) {
		collapse, cw.li, cw.l = fg.MtpCollapse[i], s.lo, s.layer(s.lo)
	}
	p[2] = s.cpuPart(collapse, nn.DeclineNoGraph, 1)
	p[3] = s.cpuPart(collapse, nn.DeclineNoGraph, cpuRowsCap)
	st := cpudev.NewStreams(s.cpu.x.Device(), s.cpu.x.Steps(), p[0], p[1], p[2], p[3], blockWeights{ws: &s.ws, li: -1}, cw,
		s.c.Streams(), s.c.NEmbd)
	st.Fault = s.cpu.fault == cpuFaultStreams
	s.streams = st
}
