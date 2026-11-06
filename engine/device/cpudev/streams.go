package cpudev

import (
	"unsafe"

	"github.com/jitllm/jitllm/nn"
)

// Weights is where a part's form reads its weights, resolved for the block
// the part reads them in, or why the device cannot read one.
type Weights interface {
	Bind(f *Form) ([]unsafe.Pointer, nn.Decline)
}

// Streams is a model's residual streams on the CPU device: the expansion of
// stream 0 into the others and the collapse of every stream into the row the
// head reads, a decode's row at one form and a chunk's rows at another,
// each composed once at the instance's placement and each stream bound at
// its offset in the stream-major residual. A call is counted served or
// declined by name, a row at a time (nn.PartStreams).
type Streams struct {
	expand, expandRows, collapse, collapseRows Part
	ew, cw                                     Weights
	dev                                        *Device
	steps                                      *nn.Steps
	d, ns                                      int
	// pad holds a call's rows of each stream where its form's units reach
	// past them, out the collapse's rows before they go over stream 0.
	pad, out []float32
	in, outs []int
	call     [4]uint32
	zero     []uint32
	recRows  [3]uint32
	bind     Call
	// Fault swaps two of the expansion's outputs and repeats one of the
	// collapse's inputs: a gate's violation.
	Fault bool
}

// NewStreams is the streams part of ns streams of width d on dev, counted
// in steps: the expansion's and the collapse's forms of one row and of a
// chunk's, their weights bound by ew and cw.
func NewStreams(dev *Device, steps *nn.Steps, expand, expandRows, collapse, collapseRows Part, ew, cw Weights, ns, d int) *Streams {
	st := &Streams{expand: expand, expandRows: expandRows, collapse: collapse, collapseRows: collapseRows,
		ew: ew, cw: cw, dev: dev, steps: steps, d: d, ns: ns}
	u := 1
	for _, p := range []*Part{&st.expand, &st.expandRows, &st.collapse, &st.collapseRows} {
		if p.F != nil {
			u = max(u, p.F.Unit, p.F.Rows)
		}
	}
	st.pad, st.out = make([]float32, ns*u*d), make([]float32, u*d)
	st.in, st.outs = make([]int, 0, ns), make([]int, 0, ns)
	st.zero = make([]uint32, 4)
	return st
}

// Parts is the streams' four forms: the expansion's of one row and of a
// chunk's, then the collapse's.
func (st *Streams) Parts() [4]*Part {
	return [4]*Part{&st.expand, &st.expandRows, &st.collapse, &st.collapseRows}
}

// Expand writes streams 1 and up of `rows` rows of x from stream 0, or says
// why not.
func (st *Streams) Expand(x []float32, rows int) nn.Decline {
	st.in, st.outs = append(st.in[:0], 0), st.outs[:0]
	for k := 1; k < st.ns; k++ {
		st.outs = append(st.outs, k)
	}
	if st.Fault && len(st.outs) > 2 {
		st.outs[0], st.outs[1] = st.outs[1], st.outs[0]
	}
	return st.run(&st.expand, &st.expandRows, st.ew, x, rows, false)
}

// Collapse writes the row the head reads into stream 0 of `rows` rows of x,
// or says why not.
func (st *Streams) Collapse(x []float32, rows int) nn.Decline {
	st.in, st.outs = st.in[:0], append(st.outs[:0], 0)
	for k := range st.ns {
		st.in = append(st.in, k)
	}
	if st.Fault && len(st.in) > 3 {
		st.in[2] = st.in[1]
	}
	return st.run(&st.collapse, &st.collapseRows, st.cw, x, rows, true)
}

// run runs a form of one row (one) or of a chunk's (many) over `rows` rows of
// the stream-major x, reading the streams st.in and writing st.outs, a call
// of at most the form's rows at a time; apart writes the outputs in a buffer
// of their own first, for a graph writing a stream it reads.
func (st *Streams) run(one, many *Part, w Weights, x []float32, rows int, apart bool) nn.Decline {
	p := one
	if rows > 1 {
		p = many
	}
	f, why := p.Form()
	var weights []unsafe.Pointer
	if why == nn.DeclineServes {
		weights, why = w.Bind(f)
	}
	if why != nn.DeclineServes {
		st.steps.Declined[nn.PartStreams][why] += rows
		return why
	}
	d := st.d
	at := func(k, r int) unsafe.Pointer { return unsafe.Pointer(&x[(k*rows+r)*d]) }
	c := &st.bind
	for r0 := 0; r0 < rows; {
		k := min(f.Rows, rows-r0)
		u := max(f.Unit, 1)
		padded := k < u
		if !padded {
			k -= k % u
		}
		*c = Call{Own: st.dev.Own(f), N: k, Weights: weights}
		for i, sk := range st.in {
			if padded {
				p := st.pad[i*u*d : (i*u+k)*d]
				copy(p, x[(sk*rows+r0)*d:(sk*rows+r0+k)*d])
				c.IO[i] = unsafe.Pointer(&p[0])
				continue
			}
			c.IO[i] = at(sk, r0)
		}
		for j, sk := range st.outs {
			switch {
			case apart:
				c.IO[len(st.in)+j] = unsafe.Pointer(&st.out[0])
			case padded:
				c.IO[len(st.in)+j] = unsafe.Pointer(&st.pad[(len(st.in)+j)*u*d])
			default:
				c.IO[len(st.in)+j] = at(sk, r0)
			}
		}
		st.call[0] = uint32(k)
		st.recRows = [3]uint32{1, 0, 0}
		c.At[SrcCall], c.At[SrcZero] = unsafe.Pointer(&st.call[0]), unsafe.Pointer(&st.zero[0])
		c.At[SrcRecRows] = unsafe.Pointer(&st.recRows[0])
		for i := range f.Codes {
			st.dev.Call(c, &f.Codes[i])
		}
		for j, sk := range st.outs {
			switch {
			case apart:
				copy(x[(sk*rows+r0)*d:(sk*rows+r0+k)*d], st.out[:k*d])
			case padded:
				o := (len(st.in) + j) * u * d
				copy(x[(sk*rows+r0)*d:(sk*rows+r0+k)*d], st.pad[o:o+k*d])
			}
		}
		r0 += k
	}
	st.steps.Served[nn.PartStreams] += rows
	return nn.DeclineServes
}
