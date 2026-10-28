package model

import (
	"unsafe"

	"github.com/jitllm/jitllm/jit"
)

// The input rows' streams and the head's row on the CPU device, for a model
// of predicted streams (frontstreams.go): where the composed CPU path is on,
// the host's expansion of stream 0 into the others and its collapse of every
// stream into the row the head reads run what the composer forms for their
// graphs, a decode's row at one form and a chunk's rows at cpuRowsCap, each
// stream bound at its offset in the stream-major residual. Each is composed
// once at the State's placement and counted served or declined by name
// (CPUStreamSteps); main's code runs one it declines.

// cpuStreamsState is a State's composed expansion and collapse and their
// census.
type cpuStreamsState struct {
	expand, expandRows, collapse, collapseRows cpuHeadForm
	served                                     int
	declined                                   [jit.NumDeclines]int
	// pad holds a call's rows of each stream where its form's units reach
	// past them, out the collapse's rows before they go over stream 0.
	pad, out []float32
	in, outs []int
}

// CPUStreamSteps is how many rows the composed expansion and collapse
// served on the CPU, and how many they declined, by reason. With the
// composed CPU off nothing is counted.
func (s *State) CPUStreamSteps() (served int, declined [jit.NumDeclines]int) {
	return s.cpu.streams.served, s.cpu.streams.declined
}

// cpuPlaceStreams composes the expansion's and the collapse's graphs for
// the CPU at the State's placement.
func (s *State) cpuPlaceStreams() {
	st := &s.cpu.streams
	if !s.c.statesStreams() && !s.c.mixesStreamsSinkhorn() {
		return
	}
	fg := s.m.frontGraphs()
	s.cpuComposeHead(&st.expand, fg.expand, jit.DeclineNoGraph, 1)
	s.cpuComposeHead(&st.expandRows, fg.expand, jit.DeclineNoGraph, cpuRowsCap)
	s.cpuComposeHead(&st.collapse, fg.collapse, jit.DeclineNoGraph, 1)
	s.cpuComposeHead(&st.collapseRows, fg.collapse, jit.DeclineNoGraph, cpuRowsCap)
	d, ns := s.c.NEmbd, s.c.Streams()
	u := 1
	for _, hf := range []*cpuHeadForm{&st.expand, &st.expandRows, &st.collapse, &st.collapseRows} {
		if hf.f != nil {
			u = max(u, hf.f.unit, hf.f.rows)
		}
	}
	st.pad, st.out = make([]float32, ns*u*d), make([]float32, u*d)
	st.in, st.outs = make([]int, 0, ns), make([]int, 0, ns)
}

// cpuExpand writes streams 1 and up of `rows` rows of x from stream 0
// through the composed expansion, or says why not.
func (s *State) cpuExpand(x []float32, rows int) jit.Decline {
	st := &s.cpu.streams
	st.in, st.outs = append(st.in[:0], 0), st.outs[:0]
	for k := 1; k < s.c.Streams(); k++ {
		st.outs = append(st.outs, k)
	}
	if s.cpu.fault == cpuFaultStreams && len(st.outs) > 2 {
		st.outs[0], st.outs[1] = st.outs[1], st.outs[0]
	}
	return s.cpuStreamsRun(&st.expand, &st.expandRows, x, rows, false)
}

// cpuCollapse writes the row the head reads into stream 0 of `rows` rows
// of x through the composed collapse, or says why not.
func (s *State) cpuCollapse(x []float32, rows int) jit.Decline {
	st := &s.cpu.streams
	st.in, st.outs = st.in[:0], append(st.outs[:0], 0)
	for k := range s.c.Streams() {
		st.in = append(st.in, k)
	}
	if s.cpu.fault == cpuFaultStreams && len(st.in) > 3 {
		st.in[2] = st.in[1]
	}
	return s.cpuStreamsRun(&st.collapse, &st.collapseRows, x, rows, true)
}

// cpuStreamsRun runs a form of one row (one) or of a chunk's (many) over
// `rows` rows of the stream-major x, reading the streams st.in and writing
// st.outs, a call of at most the form's rows at a time; apart writes the
// outputs in a buffer of their own first, for a graph writing a stream it
// reads.
func (s *State) cpuStreamsRun(one, many *cpuHeadForm, x []float32, rows int, apart bool) jit.Decline {
	if !s.cpuHeadOn() {
		return jit.DeclineCaller
	}
	st := &s.cpu.streams
	hf := one
	if rows > 1 {
		hf = many
	}
	f, why := hf.formOf()
	if why == jit.DeclineServes {
		why = s.cpuWeights(f, -1, nil)
	}
	if why != jit.DeclineServes {
		st.declined[why] += rows
		return why
	}
	d := s.c.NEmbd
	at := func(k, r int) unsafe.Pointer { return unsafe.Pointer(&x[(k*rows+r)*d]) }
	for r0 := 0; r0 < rows; {
		k := min(f.rows, rows-r0)
		u := max(f.unit, 1)
		b := &s.cpu.bound
		*b = cpuBound{own: s.cpuOwn(f), n: k}
		padded := k < u
		if !padded {
			k -= k % u
			b.n = k
		}
		for i, sk := range st.in {
			if padded {
				p := st.pad[i*u*d : (i*u+k)*d]
				copy(p, x[(sk*rows+r0)*d:(sk*rows+r0+k)*d])
				b.io[i] = unsafe.Pointer(&p[0])
				continue
			}
			b.io[i] = at(sk, r0)
		}
		for j, sk := range st.outs {
			switch {
			case apart:
				b.io[len(st.in)+j] = unsafe.Pointer(&st.out[0])
			case padded:
				b.io[len(st.in)+j] = unsafe.Pointer(&st.pad[(len(st.in)+j)*u*d])
			default:
				b.io[len(st.in)+j] = at(sk, r0)
			}
		}
		s.cpu.call[0] = uint32(k)
		for i := range f.codes {
			s.cpuCall(b, &f.codes[i])
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
	st.served += rows
	return jit.DeclineServes
}
