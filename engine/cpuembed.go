package engine

import (
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/engine/nn"
)

// cpuInputRows is the block a call of the input rows' graph binds: the
// model's own tensors, its token table the embedding's and not the head's.
const cpuInputRows = -2

// cpuEmbedState is the input rows' graphs composed for the CPU: the token
// rows' (embedGraph without the streams past the first) and an embedding
// span's given rows' (givenGraph), each a form of one row and one of a
// chunk's. The token rows' outputs are the row and, where the model has
// them, the row's per-layer inputs; the given rows' the row where a
// position table moves it, then the per-layer inputs.
type cpuEmbedState struct {
	one, many           cpuHeadForm
	givenOne, givenMany cpuHeadForm
	// tok and pos are a call's tokens table and its positions table (its live
	// rows, then a row's position); tok1 and pos1 a decode step's one row.
	tok, pos []uint32
	tok1     [1]int32
	pos1     [1]int
	// zpos is a call's rows' positions where the caller holds none
	// (positions), toks a call's rows' tokens (fillRows).
	zpos []int
	toks []int32
	// pad holds a call's outputs where it is shorter than its form's unit,
	// gin a given rows' call's input rows.
	pad, gin []float32
}

// cpuPlaceEmbed composes the input rows' graphs for the CPU at the
// instance's placement.
func (s *Instance) cpuPlaceEmbed() {
	st := &s.cpu.embed
	fg := s.m.FrontGraphs()
	s.cpuComposeHead(&st.one, fg.CpuEmbed, nn.DeclineNoEmbed, 1)
	s.cpuComposeHead(&st.many, fg.CpuEmbed, nn.DeclineNoEmbed, cpuRowsCap)
	if fg.Given != nil {
		s.cpuComposeHead(&st.givenOne, fg.Given, nn.DeclineNoEmbed, 1)
		s.cpuComposeHead(&st.givenMany, fg.Given, nn.DeclineNoEmbed, cpuRowsCap)
	}
	u := 1
	for _, hf := range []*cpuHeadForm{&st.one, &st.many, &st.givenOne, &st.givenMany} {
		if hf.f != nil {
			u = max(u, hf.f.Unit, hf.f.Rows)
		}
	}
	st.tok, st.pos = make([]uint32, u), make([]uint32, u+1)
	st.pad = make([]float32, u*(s.c.NEmbd+s.c.PleWidth()))
	if fg.Given != nil {
		st.gin = make([]float32, u*s.c.NEmbd)
	}
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
func (s *Instance) cpuEmbed(x, ple []float32, toks []int32, pos []int) nn.Decline {
	return s.cpuInputCall(&s.cpu.embed.one, &s.cpu.embed.many, nil, x, ple, toks, pos)
}

// cpuGiven writes rows, an embedding span's given rows, into the rows of x
// with their position rows where the model has a position table, and their
// per-layer inputs into the rows of ple from the tokens toks at positions
// pos, through the composed given rows; a model that adds nothing to a
// given row has its rows copied.
func (s *Instance) cpuGiven(x, ple, rows []float32, toks []int32, pos []int) nn.Decline {
	st := &s.cpu.embed
	if s.m.FrontGraphs().Given == nil {
		copy(x, rows)
		s.cpu.steps.Served[nn.PartEmbed] += len(toks)
		return nn.DeclineServes
	}
	return s.cpuInputCall(&st.givenOne, &st.givenMany, rows, x, ple, toks, pos)
}

// cpuInputCall runs one of the input rows' forms of one row (one) or of a
// chunk's (many) over toks at positions pos, reading in's rows where it is
// set (the given rows' graph) and writing x and ple, a call of at most the
// form's rows at a time.
func (s *Instance) cpuInputCall(one, many *cpuHeadForm, in, x, ple []float32, toks []int32, pos []int) nn.Decline {
	if !s.cpuHeadOn() {
		return nn.DeclineCaller
	}
	st := &s.cpu.embed
	rows := len(toks)
	hf := one
	if rows > 1 {
		hf = many
	}
	f, why := hf.formOf()
	if why == nn.DeclineServes {
		why = s.cpuWeights(f, cpuInputRows, nil)
	}
	if why != nn.DeclineServes {
		s.cpu.steps.Declined[nn.PartEmbed][why] += rows
		return why
	}
	d, pw := s.c.NEmbd, s.c.PleWidth()
	// A given rows' graph writes the row only where a position table moves
	// it; otherwise the row is the caller's, copied.
	writesX := in == nil || len(s.m.FrontGraphs().Given.Out) > 1 || pw == 0
	if in != nil && !writesX {
		copy(x, in)
	}
	for r0 := 0; r0 < rows; {
		k := min(f.Rows, rows-r0)
		u := max(f.Unit, 1)
		b := &s.cpu.bound
		*b = cpuBound{li: cpuInputRows, own: s.cpu.dev.Own(f), n: k, tok: st.tok, pos: st.pos}
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
		o := 0
		if in != nil {
			copy(st.gin[:k*d], in[r0*d:(r0+k)*d])
			b.io[0], o = unsafe.Pointer(&st.gin[0]), 1
		}
		if padded {
			if writesX {
				b.io[o], o = unsafe.Pointer(&st.pad[0]), o+1
			}
			if pw > 0 {
				b.io[o] = unsafe.Pointer(&st.pad[u*d])
			}
		} else {
			if writesX {
				b.io[o], o = unsafe.Pointer(&x[r0*d]), o+1
			}
			if pw > 0 {
				b.io[o] = unsafe.Pointer(&ple[r0*pw])
			}
		}
		s.cpu.call[0] = uint32(k)
		s.cpuRunForm(b, f)
		if padded {
			if writesX {
				copy(x[r0*d:(r0+k)*d], st.pad[:k*d])
			}
			if pw > 0 {
				copy(ple[r0*pw:(r0+k)*pw], st.pad[u*d:u*d+k*pw])
			}
		}
		r0 += k
	}
	s.cpu.steps.Served[nn.PartEmbed] += rows
	return nn.DeclineServes
}

// embedDeclined is the error of input rows no composed form serves.
func (s *Instance) embedDeclined(why nn.Decline) error {
	return fmt.Errorf("model: the input rows have no composed form on this instance (%s): %v %v",
		nn.DeclineName(int(why)), s.cpu.formErr, s.m.FrontGraphs().CpuInErr)
}

// rowSrc is a prompt's rows: row i is the vocabulary row of tok[i], or,
// within a given run, the caller's row, whose per-layer inputs read tok[i]
// (token 0's where it is -1, a row naming no token).
type rowSrc struct {
	tok   []int32
	given []givenRun
}

// givenRun is the caller's rows from row at on, n_embd each, and where its
// span carries them its tap rows: deep is the span's [taps][n][n_embd], the
// run's first row the span's row n0.
type givenRun struct {
	at    int
	rows  []float32
	deep  []float32
	n, n0 int
}

// tokenRows is the source of ids' vocabulary rows.
func tokenRows(ids []int32) rowSrc { return rowSrc{tok: ids} }

func (r rowSrc) len() int { return len(r.tok) }

// from is r's rows from row n on.
func (r rowSrc) from(n int, d int) rowSrc {
	out := rowSrc{tok: r.tok[min(n, len(r.tok)):]}
	for _, g := range r.given {
		end := g.at + len(g.rows)/d
		if end <= n {
			continue
		}
		lo := max(g.at, n)
		out.given = append(out.given, givenRun{at: lo - n, rows: g.rows[(lo-g.at)*d:], deep: g.deep, n: g.n, n0: g.n0 + lo - g.at})
	}
	return out
}

// givenAt is the caller's row i, or nil where row i is a token's.
func (r rowSrc) givenAt(i, d int) []float32 {
	for _, g := range r.given {
		if i >= g.at && i < g.at+len(g.rows)/d {
			return g.rows[(i-g.at)*d : (i-g.at+1)*d]
		}
	}
	return nil
}

// fillRows writes rows base..base+n-1 of src into x (n_embd each) and,
// where the model has them, their per-layer inputs into ple, at positions
// pos: a run of token rows through the composed input rows (cpuEmbed), a
// given run through the composed given rows (cpuGiven).
func (s *Instance) fillRows(src rowSrc, base, n int, x, ple []float32, pos []int) error {
	d, pw := s.c.NEmbd, s.c.PleWidth()
	st := &s.cpu.embed
	if len(st.toks) < n {
		st.toks = make([]int32, n)
	}
	toks := st.toks[:n]
	for i := range toks {
		toks[i] = max(src.tok[base+i], 0)
	}
	pleOf := func(a, b int) []float32 {
		if pw == 0 {
			return nil
		}
		return ple[a*pw : b*pw]
	}
	tokens := func(a, b int) error {
		if a >= b {
			return nil
		}
		if why := s.cpuEmbed(x[a*d:b*d], pleOf(a, b), toks[a:b], pos[a:b]); why != nn.DeclineServes {
			return s.embedDeclined(why)
		}
		return nil
	}
	at := 0
	for _, g := range src.given {
		lo, hi := max(g.at, base)-base, min(g.at+len(g.rows)/d, base+n)-base
		if hi <= lo {
			continue
		}
		if err := tokens(at, lo); err != nil {
			return err
		}
		rows := g.rows[(base+lo-g.at)*d : (base+hi-g.at)*d]
		if why := s.cpuGiven(x[lo*d:hi*d], pleOf(lo, hi), rows, toks[lo:hi], pos[lo:hi]); why != nn.DeclineServes {
			return s.embedDeclined(why)
		}
		at = hi
	}
	return tokens(at, n)
}
