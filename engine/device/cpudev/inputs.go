package cpudev

import (
	"unsafe"

	"github.com/jitllm/jitllm/nn"
)

// Inputs is a model's input rows on the CPU device: the token rows' graph,
// writing a row and, where the model has them, its per-layer inputs, and an
// embedding span's given rows' graph, writing the row where a position
// table moves it and then the per-layer inputs -- each a decode's row at
// one form and a chunk's rows at another, composed once at the instance's
// placement. A call is counted served or declined by name, a row at a time
// (nn.PartEmbed).
type Inputs struct {
	one, many, givenOne, givenMany Part
	given                          bool
	// writesX says the given rows' graph writes the row; otherwise the row
	// is the caller's, copied.
	writesX bool
	w       Weights
	dev     *Device
	steps   *nn.Steps
	d, pw   int
	// tok and pos are a call's tokens table and its positions table (its
	// live rows, then a row's position).
	tok, pos []uint32
	// pad holds a call's outputs where it is shorter than its form's unit,
	// gin a given rows' call's input rows.
	pad, gin []float32
	call     [4]uint32
	zero     []uint32
	recRows  [3]uint32
	bind     Call
	// places is the calls' IO, held from one to the next.
	places places
}

// NewInputs is the input rows' part of rows of width d and per-layer inputs
// of width pw on dev, counted in steps: the token rows' forms of one row and
// of a chunk's, and where given the given rows' (writesX where that graph
// writes the row), their weights bound by w.
func NewInputs(dev *Device, steps *nn.Steps, one, many, givenOne, givenMany Part, given, writesX bool, w Weights, d, pw int) *Inputs {
	in := &Inputs{one: one, many: many, givenOne: givenOne, givenMany: givenMany, given: given, writesX: writesX,
		w: w, dev: dev, steps: steps, d: d, pw: pw}
	u := 1
	for _, p := range []*Part{&in.one, &in.many, &in.givenOne, &in.givenMany} {
		if p.F != nil {
			u = max(u, p.F.Unit, p.F.Rows)
		}
	}
	in.tok, in.pos = make([]uint32, u), make([]uint32, u+1)
	in.pad = make([]float32, u*(d+pw))
	if given {
		in.gin = make([]float32, u*d)
	}
	in.zero = make([]uint32, 4)
	return in
}

// Embed writes the input rows of toks, at positions pos, into the rows of x
// and, where the model has them, their per-layer inputs into the rows of
// ple, or says why not.
func (in *Inputs) Embed(x, ple []float32, toks []int32, pos []int) nn.Decline {
	return in.run(&in.one, &in.many, nil, true, x, ple, toks, pos)
}

// Given writes rows, an embedding span's given rows, into the rows of x
// with their position rows where the model has a position table, and their
// per-layer inputs into the rows of ple from the tokens toks at positions
// pos; a model that adds nothing to a given row has its rows copied.
func (in *Inputs) Given(x, ple, rows []float32, toks []int32, pos []int) nn.Decline {
	if !in.given {
		copy(x, rows)
		in.steps.Served[nn.PartEmbed] += len(toks)
		return nn.DeclineServes
	}
	return in.run(&in.givenOne, &in.givenMany, rows, in.writesX, x, ple, toks, pos)
}

// run runs one of the input rows' forms of one row (one) or of a chunk's
// (many) over toks at positions pos, reading src's rows where it is set (the
// given rows' graph) and writing x and ple, a call of at most the form's
// rows at a time.
func (in *Inputs) run(one, many *Part, src []float32, writesX bool, x, ple []float32, toks []int32, pos []int) nn.Decline {
	rows := len(toks)
	p := one
	if rows > 1 {
		p = many
	}
	f, why := p.Form()
	var weights []unsafe.Pointer
	if why == nn.DeclineServes {
		weights, why = in.w.Bind(f)
	}
	if why != nn.DeclineServes {
		in.steps.Declined[nn.PartEmbed][why] += rows
		return why
	}
	d, pw := in.d, in.pw
	if src != nil && !writesX {
		copy(x, src)
	}
	c := &in.bind
	for r0 := 0; r0 < rows; {
		k := min(f.Rows, rows-r0)
		u := max(f.Unit, 1)
		padded := k < u
		if !padded {
			k -= k % u
		}
		*c = Call{Own: in.dev.Own(f), N: k, Weights: weights, IO: in.places.of(f.Places)}
		in.pos[0] = uint32(k)
		for i := range k {
			in.tok[i] = uint32(toks[r0+i])
			in.pos[1+i] = uint32(pos[r0+i])
		}
		o := 0
		if src != nil {
			copy(in.gin[:k*d], src[r0*d:(r0+k)*d])
			c.IO[0], o = unsafe.Pointer(&in.gin[0]), 1
		}
		if padded {
			if writesX {
				c.IO[o], o = unsafe.Pointer(&in.pad[0]), o+1
			}
			if pw > 0 {
				c.IO[o] = unsafe.Pointer(&in.pad[u*d])
			}
		} else {
			if writesX {
				c.IO[o], o = unsafe.Pointer(&x[r0*d]), o+1
			}
			if pw > 0 {
				c.IO[o] = unsafe.Pointer(&ple[r0*pw])
			}
		}
		in.call[0] = uint32(k)
		in.recRows = [3]uint32{1, 0, 0}
		c.At[SrcCall], c.At[SrcZero] = unsafe.Pointer(&in.call[0]), unsafe.Pointer(&in.zero[0])
		c.At[SrcRecRows] = unsafe.Pointer(&in.recRows[0])
		c.At[SrcTokens], c.At[SrcPos] = unsafe.Pointer(&in.tok[0]), unsafe.Pointer(&in.pos[0])
		for i := range f.Codes {
			in.dev.Call(c, &f.Codes[i])
		}
		if padded {
			if writesX {
				copy(x[r0*d:(r0+k)*d], in.pad[:k*d])
			}
			if pw > 0 {
				copy(ple[r0*pw:(r0+k)*pw], in.pad[u*d:u*d+k*pw])
			}
		}
		r0 += k
	}
	in.steps.Served[nn.PartEmbed] += rows
	return nn.DeclineServes
}
