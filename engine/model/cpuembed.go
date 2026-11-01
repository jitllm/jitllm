package model

import (
	"unsafe"

	"github.com/jitllm/jitllm/jit"
)

// cpuInputRows is the block a call of the input rows' graph binds: the
// model's own tensors, its token table the embedding's and not the head's.
const cpuInputRows = -2

// cpuEmbedState is the input rows' graph (embedGraph) composed for the CPU:
// a form of one row and one of a chunk's. Its outputs are the row and, where
// the model has them, the row's per-layer inputs.
type cpuEmbedState struct {
	one, many cpuHeadForm
	// tok and pos are a call's tokens table and its positions table (its live
	// rows, then a row's position); tok1 and pos1 a decode step's one row.
	tok, pos []uint32
	tok1     [1]int32
	pos1     [1]int
	// zpos is a call's rows' positions where the caller holds none (positions).
	zpos []int
	// pad holds a call's outputs where it is shorter than its form's unit.
	pad      []float32
	served   int
	declined [jit.NumDeclines]int
}

// CPUEmbedSteps is how many input rows the composed CPU served, and how many
// it declined, by reason.
func (s *State) CPUEmbedSteps() (served int, declined [jit.NumDeclines]int) {
	return s.cpu.embed.served, s.cpu.embed.declined
}

// cpuPlaceEmbed composes the input rows' graph for the CPU at the State's
// placement. A graph writing the row over every residual stream as well is
// declined by name.
func (s *State) cpuPlaceEmbed() {
	st := &s.cpu.embed
	g, why := s.m.frontGraphs().embed, jit.DeclineNoEmbed
	if g != nil && s.c.Streams() > 1 {
		g, why = nil, jit.DeclineEmbedStreams
	}
	s.cpuComposeHead(&st.one, g, why, 1)
	s.cpuComposeHead(&st.many, g, why, cpuRowsCap)
	u := 1
	for _, hf := range []*cpuHeadForm{&st.one, &st.many} {
		if hf.f != nil {
			u = max(u, hf.f.unit, hf.f.rows)
		}
	}
	st.tok, st.pos = make([]uint32, u), make([]uint32, u+1)
	st.pad = make([]float32, u*(s.c.NEmbd+s.c.pleWidth()))
}

// positions is a call's n rows' positions, the caller's to write.
func (st *cpuEmbedState) positions(n int) []int {
	if len(st.zpos) < n {
		st.zpos = make([]int, n)
	}
	return st.zpos[:n]
}

// cpuEmbed writes the input rows of toks, at positions pos, into the rows
// of x and, where the model has them, their per-layer inputs into the rows
// of ple, through the composed graph, or says why not.
func (s *State) cpuEmbed(x, ple []float32, toks []int32, pos []int) jit.Decline {
	if !s.cpuHeadOn() {
		return jit.DeclineCaller
	}
	st := &s.cpu.embed
	rows := len(toks)
	hf := &st.one
	if rows > 1 {
		hf = &st.many
	}
	f, why := hf.formOf()
	if why == jit.DeclineServes {
		why = s.cpuWeights(f, cpuInputRows, nil)
	}
	if why != jit.DeclineServes {
		st.declined[why] += rows
		return why
	}
	d, pw := s.c.NEmbd, s.c.pleWidth()
	for r0 := 0; r0 < rows; {
		k := min(f.rows, rows-r0)
		u := max(f.unit, 1)
		b := &s.cpu.bound
		*b = cpuBound{li: cpuInputRows, own: s.cpuOwn(f), n: k, tok: st.tok, pos: st.pos}
		padded := k < u
		if !padded {
			k -= k % u
			b.n = k
		}
		st.pos[0] = uint32(k)
		for i := range k {
			st.tok[i] = uint32(toks[r0+i])
			st.pos[1+i] = uint32(pos[r0+i])
		}
		if padded {
			b.io[0] = unsafe.Pointer(&st.pad[0])
			if pw > 0 {
				b.io[1] = unsafe.Pointer(&st.pad[u*d])
			}
		} else {
			b.io[0] = unsafe.Pointer(&x[r0*d])
			if pw > 0 {
				b.io[1] = unsafe.Pointer(&ple[r0*pw])
			}
		}
		s.cpu.call[0] = uint32(k)
		for i := range f.codes {
			s.cpuCall(b, &f.codes[i])
		}
		if padded {
			copy(x[r0*d:(r0+k)*d], st.pad[:k*d])
			if pw > 0 {
				copy(ple[r0*pw:(r0+k)*pw], st.pad[u*d:u*d+k*pw])
			}
		}
		r0 += k
	}
	st.served += rows
	return jit.DeclineServes
}
