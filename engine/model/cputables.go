package model

import (
	"fmt"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/graph"
)

// cpuTablesBlock is the block a call of the rotary tables' graphs binds: the
// planes it reads are its rotary's (cpuTablePlanes), no block's.
const cpuTablesBlock = -3

// cpuTableState is the rotary tables' graphs (frontGraphs().tables) composed
// for the CPU: a form a rotary of one row and one of a chunk's, and a call's
// rows as the tables read them (composer.TableRopeRows): its live count, then a
// row's cache position and where its sequence's marks start.
type cpuTableState struct {
	one, many []cpuHeadForm
	rows      []uint32
	// zero is the marks a model of no marks binds, one empty block.
	zero [1]uint32
	// req and pos are a grid table's request and rows (a tower's picture):
	// the grid's width, and the live rows and their positions.
	req, pos []uint32
}

// cpuPlaceTables composes the rotary tables' graphs for the CPU at the
// instance's placement.
func (s *Instance) cpuPlaceTables() {
	st := &s.cpu.tables
	tabs := s.tableGraphs()
	st.one, st.many = make([]cpuHeadForm, len(tabs)), make([]cpuHeadForm, len(tabs))
	widths := [2]int{s.c.RopeW(), s.c.NRotSWA}
	u := 1
	for i, g := range tabs {
		// The tables are the model's rotaries; an instance whose config has
		// none (a tower's) composes none.
		if i < len(widths) && widths[i] == 0 {
			continue
		}
		s.cpuComposeTable(&st.one[i], g, 1)
		// A tower's grid table turns a picture's rows in one call.
		many := cpuRowsCap
		if s.towered() {
			many = s.maxSeq
		}
		s.cpuComposeTable(&st.many[i], g, many)
		for _, hf := range []*cpuHeadForm{&st.one[i], &st.many[i]} {
			if hf.f != nil {
				u = max(u, hf.f.Unit, hf.f.Rows)
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
	case !composer.LowersCPU(uint64(s.m.cpuISA())):
		hf.why = composer.DeclineISA
	case g == nil:
		hf.why = composer.DeclineNoGraph
	default:
		req := composer.Request{Graph: g, Step: composer.Step{Rows: rows, Page: 1, Keys: 1, Seqs: max(s.nseq, 1),
			ActWindow: s.m.actWindow}, Device: s.composedCPU(), Apart: true}
		req.Step.Marks = s.marksCap()
		f, err := s.m.cpuFormOf(req, g, rows, true, false, false)
		if err != nil {
			hf.why, hf.err = s.cpuFormDecline(err), err
			return
		}
		hf.f = f
	}
}

// towered says the instance's blocks are the model's tower's.
func (s *Instance) towered() bool { return s.m.tower != nil && s.lo >= s.m.towerBase }

// tableGraphs is the rotary tables' graphs the instance's blocks read: its
// tower's own (towerRotary), none for a tower that turns nothing, the
// model's for any other instance.
func (s *Instance) tableGraphs() []*graph.Graph {
	if !s.towered() {
		return s.m.frontGraphs().tables
	}
	if t := s.m.tower.table; t != nil {
		return []*graph.Graph{t}
	}
	return nil
}

// rotaries is every rotary the instance's blocks turn by: its tower's own,
// none for a tower that turns nothing, the model's for any other instance
// (composing.Rotaries).
func (s *Instance) rotaries() []graph.Rotary {
	if !s.towered() {
		return composing{s.m.container, s.m}.Rotaries()
	}
	if t := s.m.container.Vision().Tower; len(t.Block.Rope.Runs) > 0 {
		return []graph.Rotary{towerRotaryOf(t)}
	}
	return nil
}

// tablePlanes is rotary table i's planes as the instance's composed tables
// read them: its tower's, or the model's (cpuTablePlanes).
func (s *Instance) tablePlanes(i int) []float32 {
	if !s.towered() {
		return s.m.cpuTablePlanes(i)
	}
	if i == 0 {
		return s.m.tower.planes
	}
	return nil
}

// marksCap is the most rotary marks a sequence of the instance holds: none
// on a tower's, whose table reads its rows' cells (towerRotary), the model's
// on any other (ropeMarksCap).
func (s *Instance) marksCap() int {
	if s.towered() {
		return 0
	}
	return s.m.ropeMarksCap(s.maxSeq)
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
		if i >= len(outs) || len(outs[i]) == 0 || widths[i] == 0 {
			continue
		}
		if len(outs[i]) < rows*widths[i] {
			return fmt.Errorf("model: the rotary table %d on the CPU: %d values for %d rows of %d", i, len(outs[i]), rows, widths[i])
		}
		hf := &st.one[i]
		if rows > 1 {
			hf = &st.many[i]
		}
		f, why := hf.formOf()
		if why != composer.DeclineServes {
			return fmt.Errorf("model: the rotary table %d on the CPU: %s: %w", i, why, hf.err)
		}
		if st.pos != nil {
			// A grid table reads the picture's rows and request whole.
			if f.Rows < rows {
				return fmt.Errorf("model: the grid table on the CPU turns %d rows, a picture's %d", f.Rows, rows)
			}
			b := &s.cpu.bound
			*b = cpuBound{li: cpuTablesBlock, own: s.cpu.dev.Own(f), n: rows, req: st.req, pos: st.pos, ropeMarks: st.zero[:]}
			b.cs, b.csSWA = outs[0], outs[1]
			s.cpu.call[0] = uint32(rows)
			s.cpuRunForm(b, f)
			continue
		}
		for r0 := 0; r0 < rows; {
			k := min(f.Rows, rows-r0)
			st.rows[0] = uint32(k)
			for j := range f.Rows {
				r := r0 + min(j, k-1)
				q := 0
				if seq != nil {
					q = seq[r]
				}
				st.rows[1+2*j] = uint32(pos[r])
				if marks != nil && s.marksCap() > 0 {
					st.rows[2+2*j] = uint32(marks.offset(q))
				}
			}
			b := &s.cpu.bound
			*b = cpuBound{li: cpuTablesBlock, own: s.cpu.dev.Own(f), n: k, ropeRows: st.rows, ropeMarks: st.zero[:]}
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
			s.cpuRunForm(b, f)
			r0 += k
		}
	}
	return nil
}

// cpuTablePlanes is rotary table i's planes as its composed form reads them
// (composer.TablePlanes): the model's rotary's, folded over its axes where it
// has several, or its sliding blocks'; nil where the model has no such
// rotary.
func (m *Model) cpuTablePlanes(i int) []float32 {
	switch {
	case i == 0 && len(m.rope.Runs) > 0:
		return ropeAxesPlanes(&m.rope, m.Cfg.NRot, m.Cfg.RopeXD)
	case i == 0:
		return ropeTabPlanes(&m.rope, m.Cfg.NRot)
	}
	return ropeTabPlanes(m.ropeSWA, m.Cfg.NRotSWA)
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
