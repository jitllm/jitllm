package model

import (
	"fmt"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// The tower's 2-D rotary table as the composer forms it (graph.RopeTable's
// Grid): each row's two coordinates -- the ones its family turns it by, a
// patch's grid row and column, Llama 4's from one, Gemma 4's column first --
// stated as one cell a0*w+a1 of a grid w wide, w one more than the largest
// second coordinate; the planes the tower rotary's, each pair's axis after
// them, so the table turns a pair by the cell's row (axis 0) or its column.
// Computed by the device that runs the tower's blocks, at the encode's rows;
// what Go computes is the cells, the picture's geometry.

// towerTableGraph is the tower rotary's table: HeadDim floats a row, its
// pairs the tower rotary's.
func (c TowerConfig) towerTableGraph() *graph.Graph {
	return composer.TableGraph(graph.Rotary{NRot: c.HeadDim, Grid: true}, 0)
}

// towerGridPlanes is the tower rotary's planes with each pair's axis after
// them (decomposeRopeTable's grid planes).
func (c TowerConfig) towerGridPlanes() []float32 {
	r := c.towerRope()
	P := c.HeadDim / 2
	p := r.TabPlanes(P)
	out := append(p[:len(p):len(p)], make([]float32, P)...)
	at := len(p)
	for _, run := range r.Runs {
		for range run.Pairs {
			if at < len(out) {
				out[at] = float32(run.Axis)
			}
			at++
		}
	}
	return out
}

// towerCoords is row i's two coordinates as the tower's rotary turns it.
func (s *Instance) towerCoords(i, gh, gw int) (a0, a1 int) {
	c := s.vis.t.Cfg
	switch c.Kind {
	case jlm.ProjLlama4:
		var pos [2]int
		llama4Coords(i, gh*gw, gw, pos[:], c.viol)
		return pos[0], pos[1]
	case jlm.ProjGemma4V:
		// (column, row): the column turns the first half.
		p := s.patchAt(i)
		if c.viol == violRowFirst {
			return p / gw, p % gw
		}
		return p % gw, p / gw
	}
	p := s.patchAt(i)
	return p / gw, p % gw
}

// towerCells writes the n rows' cells into dst -- the live count, then a
// row's cell -- and returns the grid's width.
func (s *Instance) towerCells(n int, dst []uint32) int {
	gh, gw := s.patchGrid()
	w := 1
	for i := range n {
		_, a1 := s.towerCoords(i, gh, gw)
		w = max(w, a1+1)
	}
	dst[0] = uint32(n)
	for i := range n {
		a0, a1 := s.towerCoords(i, gh, gw)
		dst[1+i] = uint32(a0*w + a1)
	}
	return w
}

// towerGridCells hands the device the encode's n rows' cells
// (nn.GridCellsDevice); a device that takes none reads cs, the CPU's table.
func (s *Instance) towerGridCells(n int, cs []float32) error {
	v := s.vis
	if cap(v.gridCells) < 1+n {
		v.gridCells = make([]uint32, 1+n)
	}
	w := s.towerCells(n, v.gridCells[:1+n])
	if gd, ok := s.ld.(nn.GridCellsDevice); ok && gd.SetGridCells(v.gridCells[:1+n], w) {
		return nil
	}
	return s.towerCPUTable(n, cs)
}

// towerCPUTable writes the encode's n rows' table into cs on the CPU, once
// an encode.
func (s *Instance) towerCPUTable(n int, cs []float32) error {
	v := s.vis
	if v.tabFresh {
		return nil
	}
	if err := s.cpuTowerTableRows(n, cs[:n*v.t.Cfg.HeadDim]); err != nil {
		return err
	}
	v.tabFresh = true
	return nil
}

// cpuTowerTable is the tower table's form on the CPU, the planes it reads
// and a call's words: its cells and its request (the grid's width).
type cpuTowerTable struct {
	hf     cpuHeadForm
	planes []float32
	pos    []uint32
	req    []uint32
}

// cpuPlaceTowerTable composes the tower's table for the CPU at the encode's
// rows, at the vision instance's placement.
func (s *Instance) cpuPlaceTowerTable() {
	st := &s.cpu.towerTab
	c := s.vis.t.Cfg
	if !c.Rope {
		return
	}
	rows := c.MaxSeq()
	s.cpuComposeTable(&st.hf, c.towerTableGraph(), rows)
	st.planes = c.towerGridPlanes()
	st.pos = make([]uint32, 1+rows)
	st.req = make([]uint32, composer.RequestRows+rows)
}

// cpuTowerTableRows writes the tower's table of the encode's n rows into cs
// through the composed table, or says why not.
func (s *Instance) cpuTowerTableRows(n int, cs []float32) error {
	st := &s.cpu.towerTab
	f, why := st.hf.formOf()
	if why != composer.DeclineServes {
		return fmt.Errorf("model: the tower's rotary table on the CPU: %s", why)
	}
	if n > f.rows || n+1 > len(st.pos) {
		return fmt.Errorf("model: the tower's rotary table on the CPU: %d rows past the %d it was formed for", n, f.rows)
	}
	st.req[composer.RequestGrid] = uint32(s.towerCells(n, st.pos))
	b := &s.cpu.bound
	*b = cpuBound{li: cpuTablesBlock, own: s.cpuOwn(f), n: n, cs: cs, pos: st.pos, req: st.req, planes: st.planes}
	s.cpu.call[0] = uint32(n)
	for i := range f.codes {
		s.cpuCall(b, &f.codes[i])
	}
	return nil
}
