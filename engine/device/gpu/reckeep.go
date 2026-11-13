package gpu

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/graph"
)

// A speculation's rows rollback on the card (nn.RecRowKeeper). A step whose
// calls keep their rows runs each linear block's keep form -- the block's
// graph with graph.Attrs.Keeps, formed when the speculation prepares
// (RecKeepPrep), never at a step -- which stores the block's state and
// window after each of the step's rows to the session's keep slots, a slot
// a row. Returning to a row copies its slots over the session's seat, both
// halves, on the card. The plain forms and every other step are as they
// were; a keeping step is its session's own, issued unrecorded.

// recKeep is a session's kept rows on one device: each linear block's keep
// graph and its state (s) and window (c) after each row of a step that kept
// them, slots of each; the rows its steps keep (armed), those the last one
// kept (ran), and whether that one ran as groups (split), whose rows are not
// its slots. Only its session writes it, and its steps read it outside g.mu.
type recKeep struct {
	graphs            map[*layer]*graph.Graph
	s, c              map[*layer]backend.Buf
	slots, armed, ran int
	split             bool
	bytes             uint64
	// list is a restore's copies, kept for the next.
	list []backend.CopyRange
}

// keepGraph is l's graph with each op that writes a recurrent state keeping
// every row's (graph.Attrs.Keeps), one for the blocks that share a graph.
// Callers hold g.mu.
func (g *devTier) keepGraph(l *layer) *graph.Graph {
	if kg := g.keepGraphs[l.graph]; kg != nil {
		return kg
	}
	kg := *l.graph
	kg.Ops = slices.Clone(l.graph.Ops)
	for i := range kg.Ops {
		for _, w := range kg.Ops[i].Writes {
			if w.State == graph.StateRecState || w.State == graph.StateRecConv {
				kg.Ops[i].Attr.Keeps = true
			}
		}
	}
	if g.keepGraphs == nil {
		g.keepGraphs = map[*graph.Graph]*graph.Graph{}
	}
	g.keepGraphs[l.graph] = &kg
	return &kg
}

// keepArmed says the step of rag keeps its rows: the session's steps keep
// them, its rows are one sequence's and its slots hold them.
func (g *devTier) keepArmed(rag *ragStep) bool {
	k := g.kept
	return k != nil && k.armed > 0 && rag != nil && rag.oneSequence() && len(rag.pos) <= k.slots
}

// keepStep is keepArmed for block l, prepared to keep, recording the
// step's rows as the ones kept.
func (g *devTier) keepStep(rag *ragStep, l *layer) bool {
	if !g.keepArmed(rag) || g.kept.graphs[l] == nil {
		return false
	}
	g.kept.ran = len(rag.pos)
	return true
}

// linearBlocks is the linear blocks this device holds. Callers hold g.mu.
func (g *devTier) linearBlocks() []int {
	var lis []int
	for li, l := range g.layers {
		if l != nil && l.ok && l.linear {
			lis = append(lis, li)
		}
	}
	return lis
}

// recKeepPrep readies session sid's steps to keep up to rows rows here: each
// linear block's keep form at the width a verification runs at (the reserved
// prompt width, rowsComposed), formed on lane0 and given to every lane, and
// the slots.
func (g *devTier) recKeepPrep(sid uint64, rows int) error {
	v := g.as(sid)
	if v == nil {
		return fmt.Errorf("tier: session %d is not on this device", sid)
	}
	defer v.done()
	g = v
	g.mu.Lock()
	defer g.mu.Unlock()
	lis := g.linearBlocks()
	if len(lis) == 0 || rows < 1 {
		return nil
	}
	v0 := &devTier{devShared: g.devShared, devSess: g.devSess, lane: g.lane0, Stats: g.Stats}
	bs := v0.bbs[v0.promptW]
	if bs == nil {
		return fmt.Errorf("tier: a step keeping its rows runs at the reserved prompt width, which this device holds no scratch for (%s)", g.LastErr)
	}
	if g.kept == nil {
		g.kept = &recKeep{graphs: map[*layer]*graph.Graph{}, s: map[*layer]backend.Buf{}, c: map[*layer]backend.Buf{}}
	}
	// The slots first: a keep form binds them as it forms.
	if err := g.growKept(rows); err != nil {
		return err
	}
	was := g.prewarming
	g.prewarming, g.keeping = true, true
	defer func() { g.prewarming, g.keeping = was, false }()
	for _, li := range lis {
		kg := g.keepGraph(g.layers[li])
		g.kept.graphs[g.layers[li]] = kg
		if err := v0.keepForm(bs, li); err != nil {
			return err
		}
		for _, ln := range g.lanes {
			if err := g.spreadParts(g.lane0, ln, kg); err != nil {
				return fmt.Errorf("tier: block %d's keep form on a lane: %w", li, err)
			}
		}
	}
	return nil
}

// keepForm forms block li's keep form in bs's width. Callers hold g.mu,
// keeping and prewarming.
func (g *devTier) keepForm(bs *blockScratch, li int) error {
	defer g.useGeom(g.enterGeomOf(li))
	if _, _, _, err := g.servedStep(bs, g.layers[li], li, &bs.p, bs.rows, true); err != nil {
		return fmt.Errorf("tier: block %d's keep form: %w", li, err)
	}
	return nil
}

// growKept gives the session's keep slots, made by recKeepPrep, room for
// steps of rows rows on every linear block here, charged to the device.
// Callers hold g.mu.
func (g *devTier) growKept(rows int) error {
	k := g.kept
	if rows <= k.slots {
		return nil
	}
	lis := g.linearBlocks()
	sB, cB := max(rows*g.recS, 1)*4, max(rows*g.recC, 1)*4
	need := uint64(len(lis)) * uint64(sB+cB)
	g.freeKeptBufs(k)
	if !g.stateFits(need, nil) {
		return fmt.Errorf("tier: %d kept rows of %d linear blocks want %.1f MiB, which this device's budget does not hold",
			rows, len(lis), float64(need)/(1<<20))
	}
	for _, li := range lis {
		l := g.layers[li]
		s, err := g.dev.Alloc(sB)
		if err != nil {
			g.freeKeptBufs(k)
			return fmt.Errorf("tier: block %d's kept states: %w", li, err)
		}
		c, err := g.dev.Alloc(cB)
		if err != nil {
			s.Free()
			g.freeKeptBufs(k)
			return fmt.Errorf("tier: block %d's kept windows: %w", li, err)
		}
		k.s[l], k.c[l] = s, c
	}
	k.slots, k.bytes = rows, need
	g.charge(need)
	return nil
}

// freeKeptBufs frees k's slots and refunds them. Callers hold g.mu.
func (g *devTier) freeKeptBufs(k *recKeep) {
	for l, b := range k.s {
		b.Free()
		delete(k.s, l)
	}
	for l, b := range k.c {
		b.Free()
		delete(k.c, l)
	}
	g.refund(k.bytes)
	k.slots, k.bytes, k.ran = 0, 0, 0
}

// freeKept frees session ds's kept rows. Callers hold g.mu.
func (g *devTier) freeKept(ds *devSess) {
	if ds.kept != nil {
		g.freeKeptBufs(ds.kept)
		ds.kept = nil
	}
}

// recKeepRows has session sid's next steps keep their first rows+1 rows,
// growing its slots where they hold fewer; zero keeps none.
func (g *devTier) recKeepRows(sid uint64, rows int) {
	v := g.as(sid)
	if v == nil {
		return
	}
	defer v.done()
	g = v
	g.mu.Lock()
	defer g.mu.Unlock()
	k := g.kept
	if k == nil {
		return // prepared nothing here: RecRestoreRow says so
	}
	k.armed, k.ran, k.split = rows, 0, false
	if rows > 0 {
		if err := g.growKept(rows + 1); err != nil {
			g.LastErr = err.Error()
			k.armed = 0
		}
	}
}

// recRestoreRow returns session sid's every linear block here to its state
// after row of the last step that kept them: the row's slots copied over the
// session's seat, both halves, as one list on the card.
func (g *devTier) recRestoreRow(sid uint64, row int) bool {
	v := g.as(sid)
	if v == nil {
		return false
	}
	defer v.done()
	g = v
	g.mu.Lock()
	defer g.mu.Unlock()
	st := g.recSeats[sid]
	k := g.kept
	if k == nil {
		return !g.keepsLinear(sid)
	}
	list := k.list[:0]
	defer func() { k.list = list[:0] }()
	for li, l := range g.layers {
		if l == nil || !l.linear || l.recOf(sid) == nil || l.pool == nil {
			continue
		}
		if k.split || row < 0 || row >= k.ran || k.s[l] == nil {
			g.LastErr = fmt.Sprintf("recurrent restore: block %d kept no row %d", li, row)
			return false
		}
		for h := range 2 {
			if g.recC > 0 {
				list = append(list, backend.CopyRange{Dst: l.pool.c[h], DstOff: st.base * g.recC * 4, Src: k.c[l], SrcOff: row * g.recC * 4, N: g.recC * 4})
			}
			if g.recS > 0 && l.pool.s[h] != nil {
				list = append(list, backend.CopyRange{Dst: l.pool.s[h], DstOff: st.base * g.recS * 4, Src: k.s[l], SrcOff: row * g.recS * 4, N: g.recS * 4})
			}
		}
	}
	if len(list) == 0 {
		return true
	}
	if err := backend.CopyAll(g.dev, list); err != nil {
		g.LastErr = "recurrent restore: " + err.Error()
		return false
	}
	return true
}

// keepsLinear says session sid holds a linear block's state here. Callers
// hold g.mu.
func (g *devTier) keepsLinear(sid uint64) bool {
	for _, l := range g.layers {
		if l != nil && l.linear && l.recOf(sid) != nil && l.pool != nil {
			return true
		}
	}
	return false
}

// RecKeepPrep is recKeepPrep for the zero session: a caller that never attached.
func (g *devTier) RecKeepPrep(rows int) error { return g.recKeepPrep(0, rows) }

// RecKeepRows is recKeepRows for the zero session.
func (g *devTier) RecKeepRows(rows int) { g.recKeepRows(0, rows) }

// RecRestoreRow is recRestoreRow for the zero session.
func (g *devTier) RecRestoreRow(row int) bool { return g.recRestoreRow(0, row) }

// RecKeepPrep is recKeepPrep for the zero session: a caller that never attached.
func (g *GPU) RecKeepPrep(rows int) error { return g.recKeepPrep(0, rows) }

// RecKeepRows is recKeepRows for the zero session.
func (g *GPU) RecKeepRows(rows int) { g.recKeepRows(0, rows) }

// RecRestoreRow is recRestoreRow for the zero session.
func (g *GPU) RecRestoreRow(row int) bool { return g.recRestoreRow(0, row) }

// recKeepPrep prepares the session on every device.
func (g *GPU) recKeepPrep(sid uint64, rows int) error {
	for _, d := range g.devs {
		if err := d.recKeepPrep(sid, rows); err != nil {
			return err
		}
	}
	return nil
}

// recKeepRows arms the session on every device.
func (g *GPU) recKeepRows(sid uint64, rows int) {
	for _, d := range g.devs {
		d.recKeepRows(sid, rows)
	}
}

// recRestoreRow restores the session on every device; false when any kept
// no such row.
func (g *GPU) recRestoreRow(sid uint64, row int) bool {
	ok := true
	for _, d := range g.devs {
		ok = d.recRestoreRow(sid, row) && ok
	}
	return ok
}
