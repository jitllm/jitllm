package engine

import (
	"fmt"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/graph"
)

// cpuTablesBlock is the block a call of the rotary tables' graphs binds: the
// planes it reads are its rotary's (cpuTablePlanes), no block's.
const cpuTablesBlock = -3

// towered says the instance's blocks are the model's tower's.
func (s *Instance) towered() bool { return s.m.Tower != nil && s.lo >= s.m.TowerBase }

// tableGraphs is the rotary tables' graphs the instance's blocks read: its
// tower's own (towerRotary), none for a tower that turns nothing, the
// model's for any other instance.
func (s *Instance) tableGraphs() []*graph.Graph {
	if !s.towered() {
		return s.m.FrontGraphs().Tables
	}
	if t := s.m.Tower.Table; t != nil {
		return []*graph.Graph{t}
	}
	return nil
}

// axesPlanes is the planes of the instance's rotary over several position
// axes, for a device (nn.LayerPlan.RopeTabAxes): the model's, none on a
// tower's, whose rotary is a grid table (gridPlanes).
func (s *Instance) axesPlanes() []float32 {
	if s.towered() {
		return nil
	}
	return model.RopeAxesPlanes(&s.m.Rope, s.c.NRot, s.c.RopeXD)
}

// gridPlanes is a tower's grid table's planes for a device
// (nn.LayerPlan.GridPlanes), each pair's axis after them; nil on any other
// instance and on a tower that turns nothing.
func (s *Instance) gridPlanes() []float32 {
	if !s.towered() {
		return nil
	}
	return s.m.Tower.Planes
}

// rotaries is every rotary the instance's blocks turn by: its tower's own,
// none for a tower that turns nothing, the model's for any other instance
// (composing.Rotaries).
func (s *Instance) rotaries() []graph.Rotary {
	if !s.towered() {
		return s.m.Composing().Rotaries()
	}
	if t := s.m.Container.Vision().Tower; len(t.Block.Rope.Runs) > 0 {
		return []graph.Rotary{model.TowerRotaryOf(t)}
	}
	return nil
}

// tablePlanes is rotary table i's planes as the instance's composed tables
// read them: its tower's, or the model's (cpuTablePlanes).
func (s *Instance) tablePlanes(i int) []float32 {
	if !s.towered() {
		return s.m.CpuTablePlanes(i)
	}
	if i == 0 {
		return s.m.Tower.Planes
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
	return s.m.RopeMarksCap(s.maxSeq)
}

// ropeTables writes the rotary tables of rows rows -- row i at cache
// position pos[i] of slot seq[i] (slot 0 where seq is nil) of the session
// whose marks are marks -- into cs and, where the model trains a second
// rotary, csSWA, on the device the instance places them on
// (nn.TablesDevice).
func (s *Instance) ropeTables(marks *model.RopeMarks, seq, pos []int, cs, csSWA []float32) error {
	if s.tables == nil {
		return fmt.Errorf("model: the rotary tables have no device on this instance")
	}
	n := len(pos)
	if cap(s.tabDesc) < 2*n {
		s.tabDesc = make([]uint32, 2*n)
	}
	desc := s.tabDesc[:2*n]
	marked := marks != nil && s.marksCap() > 0
	for r := range n {
		q := 0
		if seq != nil {
			q = seq[r]
		}
		desc[2*r], desc[2*r+1] = uint32(pos[r]), 0
		if marked {
			desc[2*r+1] = uint32(marks.Offset(q))
		}
	}
	var buf []uint32
	if marks != nil {
		buf = marks.Buf
	}
	return s.tables.Tables(desc, buf, cs, csSWA)
}

// cpuOneTable makes a decode row's rotary tables in s.cs and s.csSWA, which
// the CPU's blocks read, once a token: at the row's cache position, from
// its sequence's marks.
func (s *Instance) cpuOneTable() error {
	if s.cpu.oneFresh {
		return nil
	}
	s.cpu.onePos[0] = s.pos
	if err := s.ropeTables(&s.rmarks, nil, s.cpu.onePos[:], s.cs, s.csSWA); err != nil {
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
	w, ws := s.c.RopeW(), s.c.LocalNRot()
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
		if err := s.ropeTables(&o.rmarks, slots[i:j], pos[i:j], s.bcs[i*w:], swa); err != nil {
			return err
		}
		i = j
	}
	s.cpu.tabsFresh = true
	return nil
}
