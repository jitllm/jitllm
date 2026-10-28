package tier

import (
	"errors"
	"fmt"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// The head of a model of several residual streams on the card: a call of
// the head alone hands it every stream, stream-major in x, and the
// composer's form of the front end's collapse (nn.Head.Collapse) writes the
// row the head's graph reads, in a buffer of its own, before the head runs
// over it. Its tensors are held for the model as the picture rows' are
// (rowsWeight, rowsVector), at no block's slot.

// prepCollapse holds h's collapse where the composer serves the head:
// its tensors, and the row it writes. h.Collapses is the answer. Callers
// hold mu.
func (g *devTier) prepCollapse(bs *blockScratch, h *nn.Head) {
	bs.collapse, h.Collapses = nil, false
	if g.Composed&ComposedHead == 0 || h.Collapse == nil || h.Graph == nil || bs.p.Streams() < 2 {
		return
	}
	if _, err := g.collapseParts(h.Collapse); err != nil {
		g.LastErr = "the streams' collapse: " + err.Error()
		return
	}
	rs := &g.rows
	if rs.coll == nil {
		b, err := g.dev.Alloc(batchWidth * bs.p.NEmbd * 4)
		if err != nil {
			g.LastErr = "the streams' collapse: " + err.Error()
			return
		}
		rs.coll = b
	}
	bs.collapse, h.Collapses = h.Collapse, true
}

// collapseParts is the formats, weights and vectors the collapse p binds,
// held on the card.
func (g *devTier) collapseParts(p *nn.RowsPart) (composedPart, error) {
	cp := composedPart{kind: partCollapse, graph: p.Graph, formats: map[jlm.Role]jlm.Type{},
		weights: map[jlm.Role]*resident{}, vectors: map[jlm.Role]backend.Buf{}, apart: true}
	for _, o := range p.Graph.Ops {
		for _, b := range o.Weight {
			if w, ok := p.Weights[b.Role]; ok {
				r, err := g.rowsWeight(w)
				if err != nil {
					return composedPart{}, fmt.Errorf("%v: %w", b.Role, err)
				}
				t, _ := jlmType(r.t)
				cp.formats[b.Role], cp.weights[b.Role] = t, r
				continue
			}
			v, ok := p.Vectors[b.Role]
			if !ok || len(v) == 0 {
				return composedPart{}, fmt.Errorf("it reads %v, which the model hands no tensor of", b.Role)
			}
			buf, err := g.rowsVector(v)
			if err != nil {
				return composedPart{}, fmt.Errorf("%v: %w", b.Role, err)
			}
			cp.formats[b.Role], cp.vectors[b.Role] = jlm.TypeF32, buf
		}
	}
	return cp, nil
}

// servedCollapse is the collapse p at a step of rows.
func (g *devTier) servedCollapse(p *nn.RowsPart, rows int) (*servedPart, error) {
	if p == nil {
		return nil, errors.New("the streams' collapse is not held")
	}
	return g.servedFor(servedKey{kind: partCollapse, graph: p.Graph, rows: rows}, func() (composedPart, error) {
		return g.collapseParts(p)
	})
}

// runCollapse launches the collapse p of bs's R rows of every stream, then
// copies its rows over the first stream's, where every head reads its
// input: the composer's and the tier's own alike.
func (g *devTier) runCollapse(lc *launcher, p *nn.RowsPart, bs *blockScratch, R int) error {
	sv, err := g.servedCollapse(p, R)
	if err != nil {
		return err
	}
	cp, err := g.servedRowCopy(bs.p.NEmbd, R)
	if err != nil {
		return err
	}
	_, out, err := bs.servedStreams(R, bs.p.Streams())
	if err != nil {
		return err
	}
	rs := &g.rows
	rs.collOut[0] = rs.coll
	if err := sv.run(g, lc, sv.inst, out, rs.collOut[:]); err != nil {
		return err
	}
	rs.collIn[0] = out[0]
	return cp.run(g, lc, cp.inst, rs.collOut[:], rs.collIn[:])
}

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
	return g.servedFor(servedKey{kind: partCopy, graph: gr, rows: rows}, func() (composedPart, error) {
		return composedPart{kind: partCopy, graph: gr}, nil
	})
}
