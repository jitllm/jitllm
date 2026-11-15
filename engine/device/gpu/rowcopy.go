package gpu

import (
	"github.com/jitllm/jitllm/composer"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// servedRowCopy is a copy of rows of w floats, formed through the composer
// like any other part (graph.Copy).
func (g *devTier) servedRowCopy(w, rows int) (*servedPart, error) {
	gr := g.copyGraphs[w]
	if gr == nil {
		gr = (&graph.Copy{Width: w}).Build()
		if g.copyGraphs == nil {
			g.copyGraphs = map[int]*graph.Graph{}
		}
		g.copyGraphs[w] = gr
	}
	return g.servedFor(servedKey{kind: nn.PartCopy, graph: gr, rows: rows}, func() (composedPart, error) {
		return composedPart{kind: nn.PartCopy, graph: gr}, nil
	})
}

// servedRowTakeOf is the out rows of rows rows of w floats request req names
// moved into rows of their own (graph.TakeRow), formed through the composer
// like any other part: a head chunk's rows (headChunk).
func (g *devTier) servedRowTakeOf(req backend.Buf, w, rows, out int) (*servedPart, error) {
	key := [2]int{w, out}
	gr := g.takesGraphs[key]
	if gr == nil {
		gr = (&graph.TakeRow{Width: w, Rows: out}).Build()
		if g.takesGraphs == nil {
			g.takesGraphs = map[[2]int]*graph.Graph{}
		}
		g.takesGraphs[key] = gr
	}
	sv, err := g.servedFor(servedKey{kind: nn.PartCopy, graph: gr, rows: rows}, func() (composedPart, error) {
		return composedPart{kind: nn.PartCopy, graph: gr, tables: map[composer.TableKey]backend.Buf{tableKey(composer.TableRequest): req}}, nil
	})
	if err != nil {
		return nil, err
	}
	sv.sl.tables[tableKey(composer.TableRequest)] = req
	return sv, nil
}

// servedRowTake is the row of rows rows of w floats hb's request names
// moved into a row of its own (graph.TakeRow), formed through the composer
// like any other part.
func (g *devTier) servedRowTake(hb *blockScratch, w, rows int) (*servedPart, error) {
	gr := g.takeGraphs[w]
	if gr == nil {
		gr = (&graph.TakeRow{Width: w}).Build()
		if g.takeGraphs == nil {
			g.takeGraphs = map[int]*graph.Graph{}
		}
		g.takeGraphs[w] = gr
	}
	sv, err := g.servedFor(servedKey{kind: nn.PartCopy, graph: gr, rows: rows}, func() (composedPart, error) {
		return composedPart{kind: nn.PartCopy, graph: gr, tables: map[composer.TableKey]backend.Buf{tableKey(composer.TableRequest): hb.foldReq}}, nil
	})
	if err != nil {
		return nil, err
	}
	sv.sl.tables[tableKey(composer.TableRequest)] = hb.foldReq
	return sv, nil
}
