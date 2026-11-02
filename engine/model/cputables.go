package model

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit"
)

// cpuTablesBlock is the block a call of the rotary tables' graphs binds: the
// planes it reads are its rotary's (cpuTablePlanes), no block's.
const cpuTablesBlock = -3

// cpuTableState is the rotary tables' graphs (frontGraphs().tables) composed
// for the CPU: a form a rotary of one row and one of a chunk's, and a call's
// rows as the tables read them (jit.TableRopeRows): its live count, then a
// row's cache position and where its sequence's marks start.
type cpuTableState struct {
	one, many []cpuHeadForm
	rows      []uint32
	// zero is the marks a model of no marks binds, one empty block.
	zero [1]uint32
}

// cpuPlaceTables composes the rotary tables' graphs for the CPU at the
// instance's placement.
func (s *Instance) cpuPlaceTables() {
	st := &s.cpu.tables
	tabs := s.m.frontGraphs().tables
	st.one, st.many = make([]cpuHeadForm, len(tabs)), make([]cpuHeadForm, len(tabs))
	u := 1
	for i, g := range tabs {
		s.cpuComposeTable(&st.one[i], g, 1)
		s.cpuComposeTable(&st.many[i], g, cpuRowsCap)
		for _, hf := range []*cpuHeadForm{&st.one[i], &st.many[i]} {
			if hf.f != nil {
				u = max(u, hf.f.unit, hf.f.rows)
			}
		}
	}
	st.rows = make([]uint32, 1+2*u)
}

// cpuComposeTable is hf composed from rotary table graph g at rows rows, the
// most marks a sequence holds the instance's.
func (s *Instance) cpuComposeTable(hf *cpuHeadForm, g *graph.Graph, rows int) {
	*hf = cpuHeadForm{composed: true}
	switch {
	case !jit.HostCPULowers():
		hf.why = jit.DeclineISA
	case g == nil:
		hf.why = jit.DeclineNoGraph
	default:
		req := jit.Request{Graph: g, Step: jit.Step{Rows: rows, Page: 1, Keys: 1, Seqs: max(s.nseq, 1), Marks: s.m.ropeMarksCap(s.maxSeq),
			ActWindow: s.jit.ActWindow()}, Device: s.composedCPU(), Apart: true}
		f, err := s.m.cpuFormOf(req, g, rows, true, false, false)
		if err != nil {
			hf.why = s.cpuFormDecline(err)
			return
		}
		hf.f = f
	}
}

// ropeMarksCap is the most rotary marks a sequence of maxSeq positions holds:
// a picture's grid and the text after it, a picture at least a row; none on
// a model whose rotary turns over one axis, every row at its cache position.
func (m *Model) ropeMarksCap(maxSeq int) int {
	if len(m.rope.Runs) == 0 {
		return 0
	}
	return 2*maxSeq + 1
}

// cpuTables writes the rotary tables of rows rows -- row i at cache position
// pos[i] of slot seq[i] (slot 0 where seq is nil) of the session whose marks
// are marks -- into cs and, where the model trains a second rotary, csSWA,
// through the composed tables: each row's positions on the axes from its
// sequence's marks (graph.OpRopePos), then its angles.
func (s *Instance) cpuTables(marks *ropeMarks, seq, pos []int, cs, csSWA []float32) error {
	st := &s.cpu.tables
	outs := [2][]float32{cs, csSWA}
	widths := [2]int{s.c.RopeW(), s.c.NRotSWA}
	rows := len(pos)
	for i := range st.one {
		if i >= len(outs) || outs[i] == nil {
			continue
		}
		hf := &st.one[i]
		if rows > 1 {
			hf = &st.many[i]
		}
		f, why := hf.formOf()
		if why != jit.DeclineServes {
			return fmt.Errorf("model: the rotary table %d on the CPU: %s", i, why)
		}
		for r0 := 0; r0 < rows; {
			k := min(f.rows, rows-r0)
			st.rows[0] = uint32(k)
			for j := range f.rows {
				r := r0 + min(j, k-1)
				q := 0
				if seq != nil {
					q = seq[r]
				}
				st.rows[1+2*j] = uint32(pos[r])
				if marks != nil && s.m.ropeMarksCap(s.maxSeq) > 0 {
					st.rows[2+2*j] = uint32(marks.offset(q))
				}
			}
			b := &s.cpu.bound
			*b = cpuBound{li: cpuTablesBlock, own: s.cpuOwn(f), n: k, ropeRows: st.rows, ropeMarks: st.zero[:]}
			if marks != nil && len(marks.buf) > 0 {
				b.ropeMarks = marks.buf
			}
			b.cs, b.csSWA = outs[0], outs[1]
			if i == 0 {
				b.cs = outs[0][r0*widths[0]:]
			} else {
				b.csSWA = outs[1][r0*widths[1]:]
			}
			s.cpu.call[0] = uint32(k)
			for c := range f.codes {
				s.cpuCall(b, &f.codes[c])
			}
			r0 += k
		}
	}
	return nil
}

// cpuTablePlanes is rotary table i's planes as its composed form reads them
// (jit.TablePlanes): the model's rotary's, folded over its axes where it has
// several, or its sliding blocks'.
func (m *Model) cpuTablePlanes(i int) []float32 {
	c := &m.cpuForms
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.tabPlanes[i] == nil {
		switch i {
		case 0:
			if len(m.rope.Runs) > 0 {
				c.tabPlanes[i] = ropeAxesPlanes(&m.rope, m.Cfg.NRot, m.Cfg.RopeXD)
			} else {
				c.tabPlanes[i] = ropeTabPlanes(&m.rope, m.Cfg.NRot)
			}
		case 1:
			c.tabPlanes[i] = ropeTabPlanes(m.ropeSWA, m.Cfg.NRotSWA)
		}
	}
	return c.tabPlanes[i]
}

// cpuOneTable makes a decode row's rotary tables in s.cs and s.csSWA, which
// the CPU's blocks read, once a token: at the row's cache position, from
// its sequence's marks.
func (s *Instance) cpuOneTable() error {
	if s.cpu.oneFresh {
		return nil
	}
	s.cpu.onePos[0] = s.pos
	if err := s.cpuTables(&s.rmarks, nil, s.cpu.onePos[:], s.cs, s.csSWA); err != nil {
		return err
	}
	s.cpu.oneFresh = true
	return nil
}

// cpuRowTables makes a step's rows' rotary tables in s.bcs and s.bcsSWA,
// which the CPU's blocks read, once a step: row i of slot seq[i] at cache
// position pos[i], each run of one session's rows from its marks.
func (s *Instance) cpuRowTables(seq, pos []int) error {
	if s.cpu.tabsFresh || len(pos) == 0 {
		return nil
	}
	n := len(pos)
	w, ws := s.c.RopeW(), s.c.NRotSWA
	if cap(s.cpu.tabSeq) < n {
		s.cpu.tabSeq = make([]int, n)
	}
	slots := s.cpu.tabSeq[:n]
	for i := 0; i < n; {
		o, sl := s.rowOwner(i, seq)
		slots[i] = sl
		j := i + 1
		for ; j < n; j++ {
			oj, sj := s.rowOwner(j, seq)
			if oj != o {
				break
			}
			slots[j] = sj
		}
		var swa []float32
		if s.bcsSWA != nil {
			swa = s.bcsSWA[i*ws:]
		}
		if err := s.cpuTables(&o.rmarks, slots[i:j], pos[i:j], s.bcs[i*w:], swa); err != nil {
			return err
		}
		i = j
	}
	s.cpu.tabsFresh = true
	return nil
}
