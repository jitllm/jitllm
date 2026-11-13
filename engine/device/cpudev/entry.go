package cpudev

import (
	"unsafe"

	"github.com/jitllm/jitllm/nn"
)

// Entry is a prediction block's entry on the CPU device: the rows the block
// reads from the tokens and the trunk's hidden rows (every stream), a form
// of one row and one of a chunk's, composed once at the draft instance's
// placement. A call is counted served or declined by name, a row at a time
// (nn.PartMTP).
type Entry struct {
	one, many Part
	w         Weights
	dev       *Device
	steps     *nn.Steps
	width     int
	// tok and pos are a call's tokens table and its positions table (its
	// live rows); pad holds a call's input and its output, each a form's
	// unit of rows, where the call is shorter than the unit.
	tok, pos []uint32
	pad      []float32
	call     [4]uint32
	zero     []uint32
	recRows  [3]uint32
	bind     Call
	// places is the calls' IO, held from one to the next.
	places places
}

// NewEntry is the entry part of rows width wide on dev, counted in steps:
// its forms of one row and of a chunk's, their weights bound by w.
func NewEntry(dev *Device, steps *nn.Steps, one, many Part, w Weights, width int) *Entry {
	e := &Entry{one: one, many: many, w: w, dev: dev, steps: steps, width: width}
	u := 1
	for _, p := range []*Part{&e.one, &e.many} {
		if p.F != nil {
			u = max(u, p.F.Unit, p.F.Rows)
		}
	}
	e.tok, e.pos = make([]uint32, u), make([]uint32, u+1)
	e.pad = make([]float32, 2*u*width)
	e.zero = make([]uint32, 4)
	return e
}

// Entry writes `rows` rows of z, the rows the prediction block reads, from
// the tokens toks and the rows of x, or says why not.
func (e *Entry) Entry(z, x []float32, toks []int32, rows int) nn.Decline {
	p := &e.one
	if rows > 1 {
		p = &e.many
	}
	f, why := p.Form()
	var weights []unsafe.Pointer
	if why == nn.DeclineServes {
		weights, why = e.w.Bind(f)
	}
	if why != nn.DeclineServes {
		e.steps.Declined[nn.PartMTP][why] += rows
		return why
	}
	w := e.width
	c := &e.bind
	for r0 := 0; r0 < rows; {
		k := min(f.Rows, rows-r0)
		u := max(f.Unit, 1)
		padded := k < u
		if !padded {
			k -= k % u
		}
		*c = Call{Own: e.dev.Own(f), N: k, Weights: weights, IO: e.places.of(f.Places)}
		e.pos[0] = uint32(k)
		for i := range k {
			e.tok[i] = uint32(toks[r0+i])
		}
		if padded {
			copy(e.pad[:k*w], x[r0*w:(r0+k)*w])
			c.IO[0], c.IO[1] = unsafe.Pointer(&e.pad[0]), unsafe.Pointer(&e.pad[u*w])
		} else {
			c.IO[0], c.IO[1] = unsafe.Pointer(&x[r0*w]), unsafe.Pointer(&z[r0*w])
		}
		e.call[0] = uint32(k)
		e.recRows = [3]uint32{1, 0, 0}
		c.At[SrcCall], c.At[SrcZero] = unsafe.Pointer(&e.call[0]), unsafe.Pointer(&e.zero[0])
		c.At[SrcRecRows] = unsafe.Pointer(&e.recRows[0])
		c.At[SrcTokens], c.At[SrcPos] = unsafe.Pointer(&e.tok[0]), unsafe.Pointer(&e.pos[0])
		for i := range f.Codes {
			e.dev.Call(c, &f.Codes[i])
		}
		if padded {
			copy(z[r0*w:(r0+k)*w], e.pad[u*w:(u+k)*w])
		}
		r0 += k
	}
	e.steps.Served[nn.PartMTP] += rows
	return nn.DeclineServes
}
