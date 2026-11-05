package gpu

import (
	"encoding/binary"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// A non-causal call's rotary table on the card (nn.GridCellsDevice): the
// plan's grid table (graph.RopeTable over cells, LayerPlan.GridPlanes),
// formed at the tower set's rows when its blocks are placed and run first in
// the call's submission, writing cs as the caller's table would, from the
// rows' cells and the grid's width the session hands it (SetGridCells).

// SetGridCells records the cells and width of the session's non-causal calls
// that follow, on every device (nn.GridCellsDevice).
func (g *GPU) SetGridCells(cells []uint32, width int) bool {
	return g.setGridCells(0, cells, width)
}

// setGridCells is SetGridCells for session sid.
func (g *GPU) setGridCells(sid uint64, cells []uint32, width int) bool {
	g.mu.Lock()
	ds := g.devs
	g.mu.Unlock()
	for _, d := range ds {
		d.mu.Lock()
		ss := d.sessOf(sid)
		ss.gridCells = append(ss.gridCells[:0], u32b(cells)...)
		ss.gridReq = binary.LittleEndian.AppendUint32(ss.gridReq[:0], uint32(width))
		d.mu.Unlock()
	}
	return true
}

// SetGridCells forwards to the cards for this session.
func (s *gpuSession) SetGridCells(cells []uint32, width int) bool {
	defer s.leave(s.step())
	return s.g.setGridCells(s.sid, cells, width)
}

// gridServes says set bs runs a call's grid table itself: it holds one, and
// the session gave its cells.
func (g *devTier) gridServes(bs *blockScratch) bool {
	return bs.gridGraph != nil && bs.cs != nil && len(g.gridCells) > 0
}

// servedGridTable is bs's grid table at R rows, writing bs.cs from bs's
// cells, request and planes.
func (g *devTier) servedGridTable(bs *blockScratch, R int) (*servedPart, error) {
	sk := servedKey{kind: nn.PartTable, graph: bs.gridGraph, rows: R}
	sv, err := g.servedFor(sk, func() (composedPart, error) {
		return composedPart{kind: nn.PartTable, part: composer.PartTable, graph: bs.gridGraph, formats: map[jlm.Role]jlm.Type{},
			tables: map[composer.TableKey]backend.Buf{{Table: composer.TableAngles}: bs.cs, {Table: composer.TablePlanes}: bs.gridPlanes,
				tableKey(composer.TablePos): bs.gridPos, tableKey(composer.TableRequest): bs.gridReq}}, nil
	})
	if err != nil {
		return nil, err
	}
	t := sv.sl.tables
	t[composer.TableKey{Table: composer.TableAngles}], t[composer.TableKey{Table: composer.TablePlanes}] = bs.cs, bs.gridPlanes
	t[tableKey(composer.TablePos)], t[tableKey(composer.TableRequest)] = bs.gridPos, bs.gridReq
	return sv, nil
}
