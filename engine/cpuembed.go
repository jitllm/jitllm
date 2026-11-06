package engine

import (
	"fmt"

	"github.com/jitllm/jitllm/nn"
)

// cpuInputRows is the block a call of the input rows' graph binds: the
// model's own tensors, its token table the embedding's and not the head's.
const cpuInputRows = -2

// inputScratch is the caller's side of a call's input rows: a decode step's
// one token and position (tok1, pos1), a call's rows' positions where the
// caller holds none (positions) and its rows' tokens (fillRows).
type inputScratch struct {
	tok1 [1]int32
	pos1 [1]int
	zpos []int
	toks []int32
}

// positions is a call's n rows' positions, the caller's to write.
func (st *inputScratch) positions(n int) []int {
	if len(st.zpos) < n {
		st.zpos = make([]int, n)
	}
	return st.zpos[:n]
}

// embedRows writes the input rows of toks, at positions pos, into the rows
// of x and their per-layer inputs into the rows of ple on the device the
// instance places them on (nn.InputsDevice.Embed), or says why not.
func (s *Instance) embedRows(x, ple []float32, toks []int32, pos []int) nn.Decline {
	if s.inputs == nil {
		return nn.DeclineCaller
	}
	return s.inputs.Embed(x, ple, toks, pos)
}

// givenRows writes an embedding span's given rows into the rows of x
// (nn.InputsDevice.Given), or says why not.
func (s *Instance) givenRows(x, ple, rows []float32, toks []int32, pos []int) nn.Decline {
	if s.inputs == nil {
		return nn.DeclineCaller
	}
	return s.inputs.Given(x, ple, rows, toks, pos)
}

// embedDeclined is the error of input rows no composed form serves.
func (s *Instance) embedDeclined(why nn.Decline) error {
	return fmt.Errorf("model: the input rows have no composed form on this instance (%s): %v %v",
		nn.DeclineName(int(why)), s.cpu.x.Composer().Err(), s.m.FrontGraphs().CpuInErr)
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
	st := &s.inScr
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
		if why := s.embedRows(x[a*d:b*d], pleOf(a, b), toks[a:b], pos[a:b]); why != nn.DeclineServes {
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
		if why := s.givenRows(x[lo*d:hi*d], pleOf(lo, hi), rows, toks[lo:hi], pos[lo:hi]); why != nn.DeclineServes {
			return s.embedDeclined(why)
		}
		at = hi
	}
	return tokens(at, n)
}
