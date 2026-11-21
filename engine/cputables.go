package engine

import (
	"fmt"

	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"

	"github.com/jitllm/jitllm/composer"
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
// axes, its first table's (nn.LayerPlan.RopeTabs): the model's, none on a
// tower's, whose rotary is a grid table (gridPlanes), nor on a rotary of
// one position.
func (s *Instance) axesPlanes() []float32 {
	if s.towered() {
		return nil
	}
	return model.RopeAxesPlanes(&s.m.Rope)
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
		return s.m.CpuTablePlanes(i, s.maxSeq)
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
// whose marks are marks -- into each of cs's tables, on the device the
// instance places them on (nn.TablesDevice).
func (s *Instance) ropeTables(marks *model.RopeMarks, seq, pos []int, cs nn.RopeTables) error {
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
	return s.tables.Tables(desc, buf, cs)
}

// oneTable makes a decode row's rotary tables in s.cs, which
// the CPU's blocks read, once a token: at the row's cache position, from
// its sequence's marks.
func (s *Instance) oneTable() error {
	if s.oneFresh {
		return nil
	}
	s.onePos[0] = s.pos
	if err := s.ropeTables(&s.rmarks, nil, s.onePos[:], s.cs); err != nil {
		return err
	}
	s.oneFresh = true
	return nil
}

// rowTables makes a step's rows' rotary tables in s.bcs,
// which the CPU's blocks read, once a step: row i of slot seq[i] at cache
// position pos[i], each run of one session's rows from its marks.
func (s *Instance) rowTables(seq, pos []int) error {
	if s.rowsFresh || len(pos) == 0 {
		return nil
	}
	n := len(pos)
	if cap(s.rowSlots) < n {
		s.rowSlots = make([]int, n)
	}
	slots := s.rowSlots[:n]
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
		var at nn.RopeTables
		for t, b := range s.bcs {
			if b != nil {
				at[t] = b[i*s.tabW[t]:]
			}
		}
		if err := s.ropeTables(&o.rmarks, slots[i:j], pos[i:j], at); err != nil {
			return err
		}
		i = j
	}
	s.rowsFresh = true
	return nil
}

// blockRotary is which of the instance's rotary tables block graph g's
// rotary turns by (composer.RotaryIndex), 0 for a block that turns none.
func (s *Instance) blockRotary(g *graph.Graph) int {
	if g == nil {
		return 0
	}
	for _, o := range g.Ops {
		if o.Kind == graph.OpRope {
			if i, ok := composer.RotaryIndex(s.rotaries(), o.Attr); ok {
				return i
			}
		}
	}
	return 0
}
