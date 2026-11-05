package engine

import (
	"unsafe"

	"github.com/jitllm/jitllm/engine/nn"
)

// An embedding model's readout on the CPU device: the pooled, unit-length
// row (poolGraph) from what the composer forms for it, at the segment's
// rows, composed once at its placement (cpuPlace) and counted served or
// declined by name (CPUPoolSteps). Its live rows are the positions table's
// first word (graph.OpPool); the rows it reads are a staged copy of the
// segment's, every row the form reaches held.

// cpuPoolState is a instance's composed readout and its census.
type cpuPoolState struct {
	form cpuHeadForm
	// pos is the positions table the form reads its live count from, in,
	// out its staged rows and its pooled rows, raw the pooled rows before
	// the dense heads where the model has them.
	pos          []uint32
	in, out, raw []float32
}

// cpuPlacePool composes the model's readout for the CPU at the segment's
// rows, with the buffers its call binds.
func (s *Instance) cpuPlacePool() {
	g := s.m.FrontGraphs().Pool
	if g == nil {
		return
	}
	p := &s.cpu.pool
	rows := s.maxSeq
	s.cpuComposeHead(&p.form, g, nn.DeclineNoGraph, rows)
	if p.form.f != nil {
		p.pos = make([]uint32, rows+1)
		p.in, p.out = make([]float32, rows*s.c.NEmbd), make([]float32, rows*g.Values[g.Out[0]].Width)
		if len(g.Out) > 1 {
			p.raw = make([]float32, rows*s.c.NEmbd)
		}
	}
}

// cpuPoolWidth is the readout's width: the last dense head's rows, or the
// model's width where it has none.
func (s *Instance) cpuPoolWidth() int {
	if g := s.m.FrontGraphs().Pool; g != nil {
		return g.Values[g.Out[0]].Width
	}
	return s.c.NEmbd
}

// cpuPool reads the pooled, unit-length row of x's n rows into out on the
// CPU the composer's way -- with raw, the row before the dense heads --
// reporting whether it served it; a decline is counted by name.
func (s *Instance) cpuPool(x []float32, n int, out []float32, raw bool) bool {
	p := &s.cpu.pool
	f, why := p.form.formOf()
	if why == nn.DeclineServes {
		why = s.cpuWeights(f, -1, nil)
	}
	if why == nn.DeclineServes && n > len(p.pos)-1 {
		why = nn.DeclineSessions
	}
	if why != nn.DeclineServes {
		s.cpu.steps.Decline(nn.PartPool, why)
		return false
	}
	d := s.c.NEmbd
	copy(p.in, x[:n*d])
	w := len(out)
	p.pos[0] = uint32(n)
	b := &s.cpu.bound
	*b = cpuBound{own: s.cpu.dev.Own(f), n: n}
	b.io[0], b.io[1] = unsafe.Pointer(&p.in[0]), unsafe.Pointer(&p.out[0])
	if p.raw != nil {
		b.io[2] = unsafe.Pointer(&p.raw[0])
	}
	s.cpu.call[0] = uint32(n)
	s.cpuRunForm(b, f)
	if raw && p.raw != nil {
		w = s.c.NEmbd
		copy(out[:w], p.raw[:w])
		clear(out[w:])
	} else {
		copy(out, p.out[:w])
	}
	s.cpu.steps.Serve(nn.PartPool)
	return true
}
