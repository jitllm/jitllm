package gpu

import (
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
)

// A non-causal block -- a vision tower's -- served whole by the composer:
// its graph (nn.LayerPlan.Graph) formed at the call's rows and launched in
// place of the tier's kernels for it, as a prompt chunk's block is. Its
// rows are one run attending both ways, each row's descriptor naming the
// call's rows; its keys and values go to the device's one transient pair
// (transientKV), read as the one page of a pool of the block's geometry,
// written and read inside the block's own launch as the tier's kernels do.

// towerServes is why non-causal block li is not one the composer serves at
// bs's rows, serves where it is: the scratch holds the rows' descriptors
// and, where the block rotates, the rows' rotary table; the device holds its
// transient pair; the block attends over the whole run or the windows the
// call staged (SetKeyRuns, SetRowWindows) -- a symmetric window its graph
// states is the form's own, computed from each row's place in the run
// (composer's history.start/end); and the composer forms it at the
// scratch's rows.
func (g *devTier) towerServes(l *layer, li int, p *nn.LayerPlan, bs *blockScratch) partDecline {
	switch {
	case bs.pics == nil || bs.calls == nil:
		return partFeeds
	case bs.cs == nil && rotates(l.graph):
		return partTableHeld
	case g.tkv == nil || g.tkv.kc == nil || g.tkv.vc == nil:
		return partNoPool
	case l.windowed && len(g.winRuns) == 0, g.windowCuts(l):
		return partWindow
	}
	if _, _, _, err := g.towerStep(bs, l, li, p, bs.rows); err != nil {
		return composer.DeclineNotFormed
	}
	return partServes
}

// prewarmTower forms non-causal block li's part at its set's rows and holds
// its buffers, at the block's placement: under a budget the pager would
// otherwise give the room to the blocks it turns over, and the part meet none
// at its first call. It reports why the part does not fit, "" where it fits
// or the set does not hold the call's feeds yet. Callers hold g.mu.
func (g *devTier) prewarmTower(li int) (roomless string) {
	if g.NoPromptPrewarm {
		return ""
	}
	defer g.useGeom(g.enterGeomOf(li))
	bs, l := g.bs, g.layers[li]
	if l.conv != nil {
		// Formed at its placement (prepConv).
		return ""
	}
	// A windowed block forms here too: the windows a call stages
	// (SetKeyRuns, SetRowWindows) are descriptors its form reads, not a
	// shape it is formed for.
	if bs == nil || bs.pics == nil || bs.calls == nil || bs.cs == nil && rotates(l.graph) ||
		g.tkv == nil || g.tkv.kc == nil {
		return ""
	}
	_, inst, _, err := g.towerStep(bs, l, li, &bs.p, bs.rows)
	if err != nil {
		g.LastErr = "the composed tower block: " + err.Error()
		if errors.As(err, new(partRoomError)) {
			return fmt.Sprintf("tower block %d: its composed part does not fit beside the blocks placed (%s)", li, err)
		}
		return ""
	}
	if inst == nil {
		// The scratch's instance the budget did not hold (promptInst named it).
		return fmt.Sprintf("tower block %d: its composed part does not fit beside the blocks placed (%s)", li, g.LastErr)
	}
	return ""
}

// windowCuts reports a symmetric window block l's graph states that can cut
// a run of the set. The form centres such a window on each row's write
// position, which a non-causal set's rows do not carry as the window needs:
// on the card embeddinggemma's sliding layers read the answer of no window
// at all (TestEmbeddingsOnEveryDevice). Where half the window covers the
// set's largest run the window cuts nothing and the block is served.
// Callers hold g.mu.
func (g *devTier) windowCuts(l *layer) bool {
	return l.win != 0 && (g.visMax == 0 || l.win/2 < g.visMax)
}

// rotates says graph gr turns rows by a rotary table.
func rotates(gr *graph.Graph) bool {
	for _, o := range gr.Ops {
		if o.Kind == graph.OpRope {
			return true
		}
	}
	return false
}

// towerStep is non-causal block li's served part at a call of R rows and
// the scratch's instance it launches in (servedStep).
func (g *devTier) towerStep(bs *blockScratch, l *layer, li int, p *nn.LayerPlan, R int) (sv *servedPart, inst *partInst, cp *servedPart, err error) {
	cpl := g.capped(p)
	// The pair holds MaxSeq+1 positions of the block's row (kvCacheBytes):
	// the pool's one page.
	page := cpl.MaxSeq + 1
	pl := &l.picPool
	pl.geom, pl.k, pl.v, pl.tab = g.kvGeomOf(p), g.tkv.kc, g.tkv.vc, bs.zero
	sv, err = g.servedBlock(l, li, bs.cs, pl, bs.pics, bs.picsW, bs.roff, bs.calls, nil, cpl.MaxSeq, page, R, 0, nil)
	if err != nil {
		return nil, nil, nil, err
	}
	if !sv.f.inPlace(0, 0) {
		if cp, err = g.servedCopy(bs, R); err != nil {
			return nil, nil, nil, err
		}
	}
	return sv, g.promptInst(bs, sv.f), cp, nil
}

// picRow is the descriptor of row r of a call over rows [pos, pos+n) of a
// picture attending positions [lo, hi): its key and value written at its
// position. A row-chunked block's first phase (nn.RowChunker) reads only
// what its rows wrote -- its output is the caller's scratch -- and its
// second reads every key the first phases wrote.
func picRow(d []byte, at, lo, hi, phase int) {
	if phase == 1 {
		hi = at + 1
	}
	if rowsFaulted("piccausal") {
		hi = min(hi, at+1)
	}
	binary.LittleEndian.PutUint32(d[4*composer.RowTab:], 0)
	binary.LittleEndian.PutUint32(d[4*composer.RowStart:], uint32(min(lo, at)))
	binary.LittleEndian.PutUint32(d[4*composer.RowEnd:], uint32(hi))
	binary.LittleEndian.PutUint32(d[4*composer.RowWrite:], uint32(at))
}

// picDesc is the descriptors of a call's n rows from position pos: each row
// attends the picture's rows, the call's where it is not a phase of a
// row-chunked block (keys rows where it is).
func (bs *blockScratch) picDesc(pos, n, phase, keys int) []byte {
	b, w := bs.picStage, 4*composer.RowWords
	end := pos + n
	if phase == 2 {
		end = keys
	}
	for r := 0; r < len(b)/w; r++ {
		picRow(b[r*w:], pos+r, 0, end, phase)
	}
	return b
}

// picWinDesc is picDesc with each row attending its window
// (graph.Attrs.RowWindows): its own run where the windows are a row each,
// else the run holding it; the picture where no run holds it.
func (bs *blockScratch) picWinDesc(pos, n, phase, keys int, runs []nn.KeyRun, perRow bool) []byte {
	b, w := bs.picStageW, 4*composer.RowWords
	end := pos + n
	if phase == 2 {
		end = keys
	}
	for r := 0; r < len(b)/w; r++ {
		at := pos + r
		lo, hi := 0, end
		switch {
		case perRow && at < len(runs):
			lo, hi = runs[at].Lo, runs[at].Hi
		case !perRow:
			for _, k := range runs {
				if at >= k.Lo && at < k.Hi {
					lo, hi = k.Lo, k.Hi
					break
				}
			}
		}
		picRow(b[r*w:], at, lo, hi, phase)
	}
	return b
}
