package cpudev

import (
	"unsafe"

	"github.com/jitllm/jitllm/nn"
)

// Part is a graph the CPU device runs whole, apart from the blocks -- the
// input rows, the head and its samplers, the readout, the streams'
// expansion and collapse, the rotary tables -- composed once at the
// instance's placement, or why it is not served.
type Part struct {
	Composed bool
	F        *Form
	Why      nn.Decline
	// Err is the composer's, where it formed none.
	Err error
}

// Form is the part's form, or why it is not served.
func (p *Part) Form() (*Form, nn.Decline) {
	if !p.Composed {
		return nil, nn.DeclineUnplaced
	}
	return p.F, p.Why
}

// Pool is an embedding model's readout on the CPU device: the pooled,
// unit-length row of a call's rows (the front end's pool graph), composed
// once at the instance's placement, its weights -- the model's dense
// tensors, which no page holds -- bound then, and the buffers its call
// binds: the positions table its live count is read from, the staged rows,
// the pooled rows and, where the model has dense heads, the rows before
// them.
type Pool struct {
	part    Part
	dev     *Device
	steps   *nn.Steps
	weights []unsafe.Pointer
	pos     []uint32
	call    [4]uint32
	zero    []uint32
	in, out []float32
	raw     []float32
	d       int
	bind    Call
}

// NewPool is the readout part of rows rows of width d on dev, counted in
// steps, its form's weights at weights; raw keeps the rows before the dense
// heads too.
func NewPool(dev *Device, steps *nn.Steps, part Part, weights []unsafe.Pointer, rows, d, width int, raw bool) *Pool {
	p := &Pool{part: part, dev: dev, steps: steps, weights: weights, d: d}
	if part.F == nil {
		return p
	}
	p.pos = make([]uint32, rows+1)
	p.zero = make([]uint32, 4)
	p.in, p.out = make([]float32, rows*d), make([]float32, rows*width)
	if raw {
		p.raw = make([]float32, rows*d)
	}
	return p
}

// Pool reads the pooled, unit-length row of x's n rows into out -- with raw,
// the row before the dense heads -- reporting whether it served it; a
// decline is counted by name (nn.PoolDevice). req is the request words a
// marker head's readout reads, nil for a pool's.
func (p *Pool) Pool(x []float32, n int, req []uint32, out []float32, raw bool) bool {
	f, why := p.part.Form()
	if why == nn.DeclineServes && n > len(p.pos)-1 {
		why = nn.DeclineSessions
	}
	if why != nn.DeclineServes {
		p.steps.Decline(nn.PartPool, why)
		return false
	}
	d := p.d
	copy(p.in, x[:n*d])
	p.pos[0] = uint32(n)
	p.call[0] = uint32(n)
	c := &p.bind
	*c = Call{Own: p.dev.Own(f), N: n, Weights: p.weights}
	c.IO[0], c.IO[1] = unsafe.Pointer(&p.in[0]), unsafe.Pointer(&p.out[0])
	if p.raw != nil {
		c.IO[2] = unsafe.Pointer(&p.raw[0])
	}
	c.At[SrcCall], c.At[SrcPoolPos], c.At[SrcZero] = unsafe.Pointer(&p.call[0]), unsafe.Pointer(&p.pos[0]),
		unsafe.Pointer(&p.zero[0])
	c.At[SrcRequest] = ptr(req)
	for i := range f.Codes {
		p.dev.Call(c, &f.Codes[i])
	}
	if raw && p.raw != nil {
		copy(out[:d], p.raw[:d])
		clear(out[d:])
	} else {
		copy(out, p.out[:len(out)])
	}
	p.steps.Serve(nn.PartPool)
	return true
}
