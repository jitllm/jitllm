package engine

import (
	"unsafe"

	"github.com/jitllm/jitllm/composer"
)

// cpuMTPState is a draft instance's prediction block entry composed for the CPU
// (mtpEntryGraph): a form of one row and one of a chunk's.
type cpuMTPState struct {
	one, many cpuHeadForm
	// tok and pos are a call's tokens table and its positions table (its
	// live rows); pad holds a call's input and its output, each a form's
	// unit of rows, where the call is shorter than the unit.
	tok, pos []uint32
	pad      []float32
	served   int
	declined [composer.NumDeclines]int
}

// CPUMTPSteps is how many rows the composed prediction block entry served on
// the CPU, and how many it declined, by reason.
func (s *Instance) CPUMTPSteps() (served int, declined [composer.NumDeclines]int) {
	return s.cpu.mtp.served, s.cpu.mtp.declined
}

// cpuPlaceMTP composes the entry of the prediction block a draft instance
// starts at, if it starts at one.
func (s *Instance) cpuPlaceMTP() {
	st := &s.cpu.mtp
	fg := s.m.FrontGraphs()
	i := s.lo - s.c.NLayer
	if i < 0 || i >= len(fg.MtpEntry) {
		return
	}
	g := fg.MtpEntry[i]
	s.cpuComposeHead(&st.one, g, composer.DeclineNoGraph, 1)
	s.cpuComposeHead(&st.many, g, composer.DeclineNoGraph, cpuRowsCap)
	u := 1
	for _, hf := range []*cpuHeadForm{&st.one, &st.many} {
		if hf.f != nil {
			u = max(u, hf.f.Unit, hf.f.Rows)
		}
	}
	st.tok, st.pos = make([]uint32, u), make([]uint32, u+1)
	st.pad = make([]float32, 2*u*s.c.ResidW())
}

// cpuMTPEntry writes `rows` rows of z, the rows the prediction block
// reads, from the tokens toks and the rows of x (the trunk's hidden rows,
// every stream) through the composed entry, or says why not.
func (s *Instance) cpuMTPEntry(z, x []float32, toks []int32, rows int) composer.Decline {
	if !s.cpuHeadOn() {
		return composer.DeclineCaller
	}
	st := &s.cpu.mtp
	hf := &st.one
	if rows > 1 {
		hf = &st.many
	}
	f, why := hf.formOf()
	if why == composer.DeclineServes {
		why = s.cpuWeights(f, s.lo, &s.m.Layers[s.lo])
	}
	if why != composer.DeclineServes {
		st.declined[why] += rows
		return why
	}
	w := s.c.ResidW()
	for r0 := 0; r0 < rows; {
		k := min(f.Rows, rows-r0)
		u := max(f.Unit, 1)
		b := &s.cpu.bound
		*b = cpuBound{li: s.lo, l: &s.m.Layers[s.lo], own: s.cpu.dev.Own(f), n: k, tok: st.tok, pos: st.pos}
		padded := k < u
		if !padded {
			k -= k % u
			b.n = k
		}
		st.pos[0] = uint32(k)
		for i := range k {
			st.tok[i] = uint32(toks[r0+i])
		}
		if padded {
			copy(st.pad[:k*w], x[r0*w:(r0+k)*w])
			b.io[0], b.io[1] = unsafe.Pointer(&st.pad[0]), unsafe.Pointer(&st.pad[u*w])
		} else {
			b.io[0], b.io[1] = unsafe.Pointer(&x[r0*w]), unsafe.Pointer(&z[r0*w])
		}
		s.cpu.call[0] = uint32(k)
		s.cpuRunForm(b, f)
		if padded {
			copy(z[r0*w:(r0+k)*w], st.pad[u*w:(u+k)*w])
		}
		r0 += k
	}
	st.served += rows
	return composer.DeclineServes
}
