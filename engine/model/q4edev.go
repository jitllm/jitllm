package model

import (
	"slices"

	"github.com/jitllm/jitllm/engine/nn"
)

// Qwen3.8-Flash-Next on a device: the plan and weights a block carries
// (nn.Q4EPlan, nn.HCWeights, nn.NGramWeights), and the n-gram rows the host
// hands over before a call that runs the n-gram block.

// q4ePlan fills block li's Qwen3.8-Flash-Next fields of plan.
func (s *State) q4ePlan(li int, plan *nn.LayerPlan) {
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
	if c.NGramBlock != 0 && c.q4eFault != q4eFaultNoNGram {
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
func (s *State) q4eWeights(li int, w *nn.LayerWeights) {
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
// next position of sequence slot(r) of State owner(r) with token tok(r) --
// and hands them to the device that runs the n-gram block of [lo, hi), when
// one does. order is the rows in each sequence's position order (nil for
// 0..n-1). The host keeps the predecessors whichever side runs the block.
func (s *State) ngramToDevice(lo, hi, n int, order []int, owner func(r int) (*State, int),
	tok func(r int) int32) error {
	c := s.m.Cfg
	s.qsaMarksToDevice(lo, hi, n, owner)
	li := c.NGramBlock - 1
	if c.NGramBlock == 0 || li < lo || li >= hi || c.q4eFault == q4eFaultNoNGram {
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
		prev, _ := o.ngramState(li, sl)
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

// qsaMarksToDevice hands the device each of n rows' rotary marks -- where a
// picture moved its sequence's positions, which a QSA block's pooled keys
// turn by (qsa.go's ropeOf) -- when [lo, hi) holds a QSA block: per row its
// count, then its (position, delta) pairs.
func (s *State) qsaMarksToDevice(lo, hi, n int, owner func(r int) (*State, int)) {
	c := s.m.Cfg
	if s.ldQSA == nil || c.QSARatio == 0 {
		return
	}
	any := false
	for li := lo; li < hi && !any; li++ {
		any = c.QSAAt(li)
	}
	if !any {
		return
	}
	g := s.q4
	m := g.marks[:0]
	for r := 0; r < n; r++ {
		o, sl := owner(r)
		var ms []ropeMark
		if sl < len(o.rmarks) {
			ms = o.rmarks[sl]
		}
		m = append(m, int32(len(ms)))
		for _, k := range ms {
			m = append(m, int32(k.at), int32(k.delta))
		}
	}
	g.marks = m
	s.ldQSA.SetRopeMarks(m)
}

// ngramUndo is one sequence's predecessors before ngramToDevice moved them.
type ngramUndo struct {
	o    *State
	slot int
}

// ngramUndoRows puts back the predecessors ngramToDevice moved on, for a
// device call that failed and whose rows the host runs instead. A sequence
// with several rows in the call is put back to the first row's.
func (s *State) ngramUndoRows() {
	c := s.m.Cfg
	if s.q4 == nil || s.q4.ng == nil || c.NGramBlock == 0 {
		return
	}
	g, np, li := s.q4.ng, c.NGram-1, c.NGramBlock-1
	for i := len(g.undo) - 1; i >= 0; i-- {
		u := g.undo[i]
		prev, _ := u.o.ngramState(li, u.slot)
		copy(prev, g.undoPrev[i*np:(i+1)*np])
	}
	g.undo = g.undo[:0]
}

// selfSlot0 is a one-row call's owner: this State's sequence 0.
func (s *State) selfSlot0(int) (*State, int) { return s, 0 }

// tok1At is a one-row call's token, which tokenIDs was handed too.
func (s *State) tok1At(int) int32 { return s.tok1[0] }

// setCall names the rows of a call over one sequence for ngramToDevice:
// sequence slot of State o, row i's token ids[i] (-1 past ids).
func (g *q4eScratch) setCall(o *State, slot int, ids []int32) {
	if g == nil {
		return
	}
	g.callState, g.callSlot, g.callIDs = o, slot, ids
}

// callOwner and callTok are setCall's rows.
func (g *q4eScratch) callOwner(int) (*State, int) { return g.callState, g.callSlot }

func (g *q4eScratch) callTok(r int) int32 {
	if r < len(g.callIDs) {
		return g.callIDs[r]
	}
	return -1
}

// setRows names a batch's rows for ngramToDevice: row i is sequence seq[i]
// (of the State rowOwner says) with token ids[i].
func (g *q4eScratch) setRows(o *State, seq []int, ids []int32) {
	g.callState, g.callSeq, g.callIDs = o, seq, ids
}

func (g *q4eScratch) rowsOwner(i int) (*State, int) { return g.callState.rowOwner(i, g.callSeq) }

// setStep names a device step's rows for ngramToDevice: row i is the
// sequence whose slot base slot[i] is (a multiple of maxSeq), or the State a
// multi-session step names, with token ids[i].
func (g *q4eScratch) setStep(o *State, pos, slot []int, ids []int32) {
	g.callState, g.callPos, g.callSlots, g.callIDs = o, pos, slot, ids
}

func (g *q4eScratch) stepOwner(i int) (*State, int) {
	s := g.callState
	if len(s.rowOwn) > 0 {
		return s.rowOwn[i], 0
	}
	return s, g.callSlots[i] / s.maxSeq
}

// causalOrder is the rows of a step in position order, stable: each
// sequence's rows in the order it wrote them.
func (s *State) causalOrder(pos []int) []int {
	o := slices.Grow(s.rowCausal[:0], len(pos))[:len(pos)]
	s.rowCausal = o
	for i := range o {
		o[i] = i
	}
	slices.SortStableFunc(o, func(a, b int) int { return pos[a] - pos[b] })
	return o
}
