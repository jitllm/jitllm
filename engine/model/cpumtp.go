package model

import (
	"unsafe"

	"github.com/jitllm/jitllm/jit"
)

// cpuMTPState is a draft instance's prediction block entry composed for the CPU
// (mtpEntryGraph): a form of one row and one of a chunk's.
type cpuMTPState struct {
	one, many cpuHeadForm
	// pad holds a call's two inputs and its output, each a form's unit of
	// rows, where the call is shorter than the unit.
	pad      []float32
	served   int
	declined [jit.NumDeclines]int
}

// CPUMTPSteps is how many rows the composed prediction block entry served on
// the CPU, and how many it declined, by reason.
func (s *Instance) CPUMTPSteps() (served int, declined [jit.NumDeclines]int) {
	return s.cpu.mtp.served, s.cpu.mtp.declined
}

// cpuPlaceMTP composes the entry of the prediction block a draft instance
// starts at, if it starts at one.
func (s *Instance) cpuPlaceMTP() {
	st := &s.cpu.mtp
	fg := s.m.frontGraphs()
	i := s.lo - s.c.NLayer
	if i < 0 || i >= len(fg.mtpEntry) {
		return
	}
	g := fg.mtpEntry[i]
	s.cpuComposeHead(&st.one, g, jit.DeclineNoGraph, 1)
	s.cpuComposeHead(&st.many, g, jit.DeclineNoGraph, cpuRowsCap)
	u := 1
	for _, hf := range []*cpuHeadForm{&st.one, &st.many} {
		if hf.f != nil {
			u = max(u, hf.f.unit, hf.f.rows)
		}
	}
	st.pad = make([]float32, 3*u*s.c.NEmbd)
}

// cpuMTPEntry writes `rows` rows of z, the rows the prediction block reads,
// from the rows of e (each token's embedding) and x (the trunk's hidden
// rows) through the composed entry, or says why not.
func (s *Instance) cpuMTPEntry(z, e, x []float32, rows int) jit.Decline {
	if !s.cpuHeadOn() {
		return jit.DeclineCaller
	}
	st := &s.cpu.mtp
	hf := &st.one
	if rows > 1 {
		hf = &st.many
	}
	f, why := hf.formOf()
	if why == jit.DeclineServes {
		why = s.cpuWeights(f, s.lo, &s.m.layers[s.lo])
	}
	if why != jit.DeclineServes {
		st.declined[why] += rows
		return why
	}
	d := s.c.NEmbd
	for r0 := 0; r0 < rows; {
		k := min(f.rows, rows-r0)
		u := max(f.unit, 1)
		b := &s.cpu.bound
		*b = cpuBound{li: s.lo, l: &s.m.layers[s.lo], own: s.cpuOwn(f), n: k}
		padded := k < u
		if !padded {
			k -= k % u
			b.n = k
		}
		for i, v := range [2][]float32{e, x} {
			if padded {
				p := st.pad[i*u*d : (i*u+k)*d]
				copy(p, v[r0*d:(r0+k)*d])
				b.io[i] = unsafe.Pointer(&p[0])
				continue
			}
			b.io[i] = unsafe.Pointer(&v[r0*d])
		}
		if padded {
			b.io[2] = unsafe.Pointer(&st.pad[2*u*d])
		} else {
			b.io[2] = unsafe.Pointer(&z[r0*d])
		}
		s.cpu.call[0] = uint32(k)
		for i := range f.codes {
			s.cpuCall(b, &f.codes[i])
		}
		if padded {
			copy(z[r0*d:(r0+k)*d], st.pad[2*u*d:(2*u+k)*d])
		}
		r0 += k
	}
	st.served += rows
	return jit.DeclineServes
}
