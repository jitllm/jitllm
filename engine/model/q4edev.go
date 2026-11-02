package model

import (
	"slices"

	"github.com/jitllm/jitllm/engine/nn"
)

// Qwen3.8-Flash-Next on a device: the plan and weights a block carries
// (nn.Q4EPlan, nn.HCWeights, nn.NGramWeights), and the n-gram rows the host
// hands over before a call that runs the n-gram block.

// q4ePlan fills block li's Qwen3.8-Flash-Next fields of plan.
func (s *Instance) q4ePlan(li int, plan *nn.LayerPlan) {
	c := s.m.Cfg
	if !c.Q4E() {
		return
	}
	q := &nn.Q4EPlan{HCMult: c.HCMult, HCLowRank: c.HCLowRank, Fault: c.q4eFault}
	if c.QSAAt(li) {
		q.QSARatio = c.QSARatio
		if c.NRot > 0 {
			q.QSARope = s.m.rope.TabPlanes(c.NRot / 2)
		}
		plan.IdxHeads, plan.IdxHeadDim, plan.IdxTopK = c.IdxHeads, c.IdxHeadDim, c.IdxTopK
	} else {
		plan.IdxHeads, plan.IdxHeadDim, plan.IdxTopK = 0, 0, 0
	}
	if c.NGramBlock != 0 {
		q.NGramWidth, q.NGramConv, q.NGramDil = c.NGramWidth(), c.NGramConv, c.NGram
		q.NGramHere = c.NGramAt(li)
	}
	if c.LayerKind(li).Recurrent() {
		plan.Recurrent.SigmoidGate = c.q4eSigmoidGate()
		plan.Recurrent.TailLen = c.recTailLen()
	}
	plan.Q4E = q
}

// q4eWeights adds block li's Qwen3.8-Flash-Next tensors to w.
func (s *Instance) q4eWeights(li int, w *nn.LayerWeights) {
	l := &s.m.layers[li]
	d := l.q4e
	if d == nil {
		return
	}
	hc := func(h *q4eHC) nn.HCWeights {
		return nn.HCWeights{Norm: h.norm, Down: wt(h.down), Up: wt(h.up), Inject: wt(h.inj)}
	}
	w.HCAttn, w.HCFFN = hc(&d.attn), hc(&d.ffn)
	if s.m.Cfg.QSAAt(li) {
		w.IdxQ, w.IdxK = wt(l.idxQ), wt(l.idxK)
		w.IdxQNorm, w.IdxKNorm = l.idxQNorm, l.idxKNorm
	}
	if s.m.Cfg.NGramAt(li) {
		w.NGram = nn.NGramWeights{Key: wt(d.ngKey), Value: wt(d.ngVal), NormKey: d.ngNormK,
			NormQuery: d.ngNormQ, NormC: d.ngNormC, Conv: d.ngCv}
	}
}

// ngramToDevice hashes and gathers the n-gram rows of n rows -- row r the
// next position of sequence slot(r) of instance owner(r) with token tok(r) --
// and hands them to the device that runs the n-gram block of [lo, hi), when
// one does. order is the rows in each sequence's position order (nil for
// 0..n-1). The host keeps the predecessors whichever side runs the block.
func (s *Instance) ngramToDevice(lo, hi, n int, order []int, owner func(r int) (*Session, int),
	tok func(r int) int32) error {
	c := s.m.Cfg
	li := c.NGramBlock - 1
	if c.NGramBlock == 0 || li < lo || li >= hi {
		return nil
	}
	nd := s.ldNGram
	if nd == nil {
		return nil
	}
	s.growQ4E(n)
	// The predecessors as they stand, for ngramUndo: hashing moves them on,
	// and a device call that fails leaves the host to run these rows again.
	g, np := s.q4.ng, c.NGram-1
	g.undo = g.undo[:0]
	for r := 0; r < n; r++ {
		o, sl := owner(r)
		prev, _ := s.ngramStateOf(o, li, sl)
		i := len(g.undo)
		g.undo = append(g.undo, ngramUndo{o: o, slot: sl})
		copy(g.undoPrev[i*np:(i+1)*np], prev)
	}
	if err := s.ngramRows(li, n, order, owner, tok); err != nil {
		s.ngramUndoRows()
		return err
	}
	nd.SetNGramRows(g.emb[:n*c.NGramWidth()])
	return nil
}

// ngramForCPU hashes and gathers the n-gram rows of n rows into the n-gram
// scratch, the side input the CPU's n-gram block li reads (cpuSideRows),
// moving each sequence's predecessors on; nothing where li is not the
// n-gram block.
func (s *Instance) ngramForCPU(li, n int, order []int, owner func(r int) (*Session, int), tok func(r int) int32) error {
	c := s.m.Cfg
	if !c.NGramAt(li) {
		return nil
	}
	s.growQ4E(n)
	return s.ngramRows(li, n, order, owner, tok)
}

// ngramUndo is one sequence's predecessors before ngramToDevice moved them.
type ngramUndo struct {
	o    *Session
	slot int
}

// ngramUndoRows puts back the predecessors ngramToDevice moved on, for a
// device call that failed and whose rows the host runs instead. A sequence
// with several rows in the call is put back to the first row's.
func (s *Instance) ngramUndoRows() {
	c := s.m.Cfg
	if s.q4 == nil || s.q4.ng == nil || c.NGramBlock == 0 {
		return
	}
	g, np, li := s.q4.ng, c.NGram-1, c.NGramBlock-1
	for i := len(g.undo) - 1; i >= 0; i-- {
		u := g.undo[i]
		prev, _ := s.ngramStateOf(u.o, li, u.slot)
		copy(prev, g.undoPrev[i*np:(i+1)*np])
	}
	g.undo = g.undo[:0]
}

// selfSlot0 is a one-row call's owner: this session's sequence 0.
func (s *Instance) selfSlot0(int) (*Session, int) { return s.Session, 0 }

// tok1At is a one-row call's token, which tokenIDs was handed too.
func (s *Instance) tok1At(int) int32 { return s.tok1[0] }

// setCall names the rows of a call over one sequence for ngramToDevice:
// sequence slot of instance o, row i's token ids[i] (-1 past ids).
func (g *q4eScratch) setCall(o *Instance, slot int, ids []int32) {
	if g == nil {
		return
	}
	g.callState, g.callSlot, g.callIDs = o, slot, ids
}

// callOwner and callTok are setCall's rows.
func (g *q4eScratch) callOwner(int) (*Session, int) { return g.callState.Session, g.callSlot }

func (g *q4eScratch) callTok(r int) int32 {
	if r < len(g.callIDs) {
		return g.callIDs[r]
	}
	return -1
}

// setRows names a batch's rows for ngramToDevice: row i is sequence seq[i]
// (of the instance rowOwner says) with token ids[i].
func (g *q4eScratch) setRows(o *Instance, seq []int, ids []int32) {
	g.callState, g.callSeq, g.callIDs = o, seq, ids
}

func (g *q4eScratch) rowsOwner(i int) (*Session, int) { return g.callState.rowOwner(i, g.callSeq) }

// setStep names a device step's rows for ngramToDevice: row i is the
// sequence whose slot base slot[i] is (a multiple of maxSeq), or the instance a
// multi-session step names, with token ids[i].
func (g *q4eScratch) setStep(o *Instance, pos, slot []int, ids []int32) {
	g.callState, g.callPos, g.callSlots, g.callIDs = o, pos, slot, ids
}

func (g *q4eScratch) stepOwner(i int) (*Session, int) {
	s := g.callState
	if len(s.rowOwn) > 0 {
		return s.rowOwn[i], 0
	}
	return s.Session, g.callSlots[i] / s.maxSeq
}

// causalOrder is the rows of a step in position order, stable: each
// sequence's rows in the order it wrote them.
func (s *Instance) causalOrder(pos []int) []int {
	o := slices.Grow(s.rowCausal[:0], len(pos))[:len(pos)]
	s.rowCausal = o
	for i := range o {
		o[i] = i
	}
	slices.SortStableFunc(o, func(a, b int) int { return pos[a] - pos[b] })
	return o
}
