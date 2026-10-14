package model

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/engine/nn"
)

// ForwardBatchGreedy runs one token for each of the batch's sequences on the
// device and returns each row's greedy token. nn.RowsDevice puts row i's
// history at slots i*maxSeq.. of one device cache and attends each row over
// its own.
//
// Every block and the head must be on one device that implements it. A prompt
// goes through here a token per step as well; the rows are independent in
// position, so rows with different prompt lengths simply sit at different
// positions.
func (s *State) ForwardBatchGreedy(tokens []int32) ([]int32, error) {
	defer s.m.enterPager()()
	if err := s.batchStepOK(tokens); err != nil {
		return nil, err
	}
	// No head bias: the device's argmax reads the projection's output, and a
	// head bias is added on the host (finishLogits).
	if s.m.outB != nil {
		return nil, fmt.Errorf("model: ForwardBatchGreedy cannot apply a head bias on the device")
	}
	pos, slot := s.rowsAtBpos()
	if err := s.rowsStep(tokens, pos, slot, nil, 0); err != nil {
		return nil, err
	}
	s.advanceRows()
	return append([]int32(nil), s.head.Tokens[:len(tokens)]...), nil
}

// forwardRowsDevice is forwardRows when every block and the head are on one
// device: the ragged step with the first nlogit rows' logits read back, so a
// caller can sample. The logits land in s.blogits, as the host path leaves
// them. The rows' history and positions are the device's to keep apart: each
// row's slot names its sequence and its position its place in it.
func (s *State) forwardRowsDevice(tokens []int32, seq, pos, took []int, nlogit int) ([]float32, error) {
	nv := s.c.NVocab
	if len(s.blogits) < nlogit*nv {
		s.blogits = make([]float32, nlogit*nv)
	}
	if err := s.rowsStep(tokens, pos, s.flatSlots(seq, pos), s.blogits[:nlogit*nv], nlogit); err != nil {
		return nil, err
	}
	s.advanceSeqs(took)
	s.finishDevLogits(s.blogits[:nlogit*nv])
	return s.blogits[:nlogit*nv], nil
}

// rowsTake checks a step's rows against the sequences' positions and returns
// how many rows each slot takes: every row's slot in range and position inside
// the context, and each slot's positions exactly its next ones, bpos.. up to
// the rows it takes, each once. A gap would leave a position no row wrote and
// a later one attends; a repeat would write one position twice.
func (s *State) rowsTake(seq, pos []int) ([]int, error) {
	// The State's own scratch, kept from step to step: took is the
	// caller's until the next step.
	s.rtTook = slices.Grow(s.rtTook[:0], s.nseq)[:s.nseq]
	s.rtStart = slices.Grow(s.rtStart[:0], s.nseq)[:s.nseq]
	s.rtSeen = slices.Grow(s.rtSeen[:0], len(seq))[:len(seq)]
	took, start, seen := s.rtTook, s.rtStart, s.rtSeen
	clear(took)
	clear(start)
	clear(seen)
	for _, q := range seq {
		if q < 0 || q >= s.nseq {
			return nil, errSlot{q, s.nseq}
		}
		took[q]++
	}
	// Each slot's run in seen, one flag per position it takes.
	for q := 1; q < s.nseq; q++ {
		start[q] = start[q-1] + took[q-1]
	}
	for r, q := range seq {
		if pos[r] >= s.maxSeq {
			return nil, errFull{s.maxSeq, s.reqSeq}
		}
		off := pos[r] - s.bpos[q]
		if off < 0 || off >= took[q] || seen[start[q]+off] {
			return nil, fmt.Errorf("model: row %d writes position %d of slot %d, whose next %d position(s) start at %d",
				r, pos[r], q, took[q], s.bpos[q])
		}
		seen[start[q]+off] = true
	}
	return took, nil
}

// relocateRows is relocateFor for a step in which slot q takes took[q] rows:
// what has to fit is each stepping sequence's next took[q] positions.
func (s *State) relocateRows(took []int) {
	if s.ld == nil {
		return
	}
	end := 0
	for q, k := range took {
		if k > 0 {
			end = max(end, s.bpos[q]+k)
		}
	}
	sd, ok := s.ld.(nn.SeqKVDevice)
	if !ok || !sd.PerSequenceKV() || s.nseq == 1 {
		s.relocateFor(end)
		return
	}
	var bases, ends []int
	for q, k := range took {
		if k > 0 {
			bases, ends = append(bases, q*s.maxSeq), append(ends, s.bpos[q]+k)
		}
	}
	s.relocateUntil(end, func() bool { return sd.ReserveKVSeqs(bases, ends) })
}

// flatSlots is each row's place in the device's flat history, sequence q's
// position p at q*maxSeq+p (nn.RowsDevice).
func (s *State) flatSlots(seq, pos []int) []int {
	slot := make([]int, len(seq))
	for r, q := range seq {
		slot[r] = q*s.maxSeq + pos[r]
	}
	return slot
}

// advanceSeqs moves slot q on by the took[q] positions its rows wrote, and
// s.pos to the furthest.
func (s *State) advanceSeqs(took []int) {
	s.pos = 0
	for q, k := range took {
		s.bpos[q] += k
		s.pos = max(s.pos, s.bpos[q])
	}
}

// prefillSeqDevice runs a prompt into one row of a device batch as rows of a
// single sequence: positions p..p+k-1 at that row's own slots, which is a
// causal prefill (nn.RowsDevice). Only the prompt's last token needs logits,
// so it runs as its own one-row step and the chunk before it reads back none.
// A recurrent block steps the rows in position order through the slot's one
// state, as a prefill would (nn.RowsDevice).
func (s *State) prefillSeqDevice(slot int, tokens []int32) ([]float32, error) {
	if len(tokens) == 0 {
		return nil, errEmptyPrompt{}
	}
	if s.bpos[slot]+len(tokens) > s.maxSeq {
		return nil, errFull{s.maxSeq, s.reqSeq}
	}
	nv := s.c.NVocab
	if len(s.blogits) < s.nseq*nv {
		s.blogits = make([]float32, s.nseq*nv)
	}
	for done := 0; done < len(tokens); {
		k := min(len(tokens)-1-done, nn.MaxDevicePrefillChunk)
		var lg []float32
		if k == 0 {
			k = 1
			lg = s.blogits[slot*nv : (slot+1)*nv]
		}
		pos, sl := make([]int, k), make([]int, k)
		for j := range pos {
			pos[j] = s.bpos[slot] + j
			sl[j] = slot*s.maxSeq + pos[j]
		}
		if err := s.rowsStep(tokens[done:done+k], pos, sl, lg, len(lg)/nv); err != nil {
			return nil, err
		}
		s.bpos[slot] += k
		s.pos = max(s.pos, s.bpos[slot])
		done += k
	}
	out := s.blogits[slot*nv : (slot+1)*nv]
	s.finishDevLogits(out)
	return out, nil
}

// batchStepOK is the checks every device batch step makes before it runs.
func (s *State) batchStepOK(tokens []int32) error {
	c := s.c
	if !s.batched {
		return errBatch{}
	}
	if err := s.retireErr; err != nil {
		s.retireErr = nil
		return err
	}
	if len(tokens) != s.nseq {
		return errBatchWidth{len(tokens), s.nseq}
	}
	for i, id := range tokens {
		if int(id) < 0 || int(id) >= c.NVocab {
			return errToken{id, c.NVocab}
		}
		if s.bpos[i] >= s.maxSeq {
			return errFull{s.maxSeq, s.reqSeq}
		}
	}
	return nil
}

// onRowsDevice is the device a batch step can run on: every block and the head
// on one device that runs ragged rows. A final softcap rides the head there
// as it does in decode (the tier caps each row's logits before its argmax).
func (s *State) onRowsDevice() (nn.RowsDevice, bool) {
	return s.ldRows, s.rowsRefusal() == nil
}

// rowsRefusal is why onRowsDevice is false, nil where it is true.
func (s *State) rowsRefusal() error {
	switch {
	case s.ldRows == nil:
		return errStepNoRows
	case !s.headWithLastBlock():
		return errStepHeadElsewhere
	case s.devCount() != s.hi-s.lo:
		return errStepSplit
	}
	return nil
}

// headWithLastBlock is the head on the device that runs the last block, the
// only place a rows step takes it (tier.GPU.LayersRows): a placement that put
// the projection on another card runs it on the host instead.
func (s *State) headWithLastBlock() bool {
	last := s.hi - 1
	if s.head == nil || !s.devAt(last) {
		return false
	}
	d := s.ldNamed
	return d == nil || d.HeadWith(last) // without it, one device
}

// rowsRun runs device blocks [lo, hi) of a split placement for the rows at
// (pos[i], slot[i]), whose residual is x; with head it also takes the
// projection and reads the first nlogit rows' logits into logits. The host runs
// the blocks around it through its own batched path.
func (s *State) rowsRun(lo, hi int, pos, slot []int, x, cs, csSWA []float32, head bool, logits []float32, nlogit int) error {
	rd := s.ldRows
	if rd == nil {
		return errBatchDevice{}
	}
	var h *nn.Head
	if head && nlogit > 0 {
		h = s.head
		keep := h.Logits
		h.RowLogits, h.Logits, h.LogitRows = true, logits, nlogit
		defer func() { h.RowLogits, h.Logits, h.LogitRows = false, keep, 0 }()
	}
	s.layerInputs(s.bple[:len(pos)*s.c.pleWidth()])
	if !rd.LayersRows(lo, hi, pos, slot, s.maxSeq, x, cs, csSWA, h) {
		why := "no reason given"
		if e, ok := s.ld.(nn.ErrReporter); ok {
			why = e.Err()
		}
		return fmt.Errorf("model: the device refused blocks [%d,%d) for %d rows: %s", lo, hi, len(pos), why)
	}
	return nil
}

// rowsAtBpos is every row at its own next position.
func (s *State) rowsAtBpos() (pos, slot []int) {
	pos, slot = make([]int, s.nseq), make([]int, s.nseq)
	for i := range pos {
		pos[i], slot[i] = s.bpos[i], i*s.maxSeq+s.bpos[i]
	}
	return pos, slot
}

func (s *State) advanceRows() {
	s.pos = 0
	for i := range s.bpos {
		s.bpos[i]++
		s.pos = max(s.pos, s.bpos[i])
	}
}

// rowsStep runs tokens[i] at (pos[i], slot[i]) through every block and the
// head on the device. logits, when non-nil, receives the first nlogit rows' raw
// logits; a step with logits and nlogit 0 wants none, and runs no head.
func (s *State) rowsStep(tokens []int32, pos, slot []int, logits []float32, nlogit int) error {
	m, c := s.m, s.c
	rd, ok := s.onRowsDevice()
	if !ok {
		return fmt.Errorf("model: a device batch step needs every block and the head on "+
			"one device that runs ragged rows: %w", s.rowsRefusal())
	}
	n, nrot := len(tokens), c.RopeW()
	s.growBatch(n)
	for i := 0; i < n; i++ {
		row := s.bx[i*c.NEmbd : (i+1)*c.NEmbd]
		if err := m.embedRow(row, int(tokens[i])); err != nil {
			return err
		}
		if c.EmbdScale != 1 {
			s.scale(row, float32(c.EmbdScale))
		}
		if err := m.addPos(row, s.prow, pos[i]); err != nil {
			return err
		}
		if c.PLEDim != 0 {
			w := c.pleWidth()
			s.growBPLE(n)
			if err := s.pleInputs(s.bple[i*w:(i+1)*w], tokens[i], row); err != nil {
				return err
			}
		}
		s.ropeRow(i, s.ropeOf(slot[i]/s.maxSeq, pos[i]))
	}
	s.layerInputs(s.bple[:n*c.pleWidth()])
	s.tokenIDs(tokens)
	if c.streamHead() {
		return s.altRowsStep(rd, pos, slot, tokens, n, logits, nlogit)
	}
	// The head's Logits is the single-sequence destination (s.logits); a step
	// borrows it for the rows and hands it back.
	h := s.head
	keep := h.Logits
	switch {
	case logits != nil && nlogit == 0:
		h = nil
	case logits != nil:
		h.RowLogits, h.Logits, h.LogitRows = true, logits, nlogit
	}
	defer func() { s.head.RowLogits, s.head.Logits, s.head.LogitRows = false, keep, 0 }()
	if !rd.LayersRows(s.lo, s.hi, pos, slot, s.maxSeq, s.bx[:n*c.NEmbd], s.bcs[:n*nrot], s.bswaTable(n, c.NRotSWA), h) {
		why := "no reason given"
		if e, ok := s.ld.(nn.ErrReporter); ok {
			why = e.Err()
		}
		return fmt.Errorf("model: the device refused a %d-row step: %s", n, why)
	}
	return nil
}
