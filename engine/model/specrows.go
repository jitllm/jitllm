package model

import (
	"errors"
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
)

// The State half of speculative decoding (spec.go): one call that runs rows of
// one sequence and hands back, besides the KV it writes, the logits and the
// normed hidden state of the rows a speculation step reads, and on a host
// hybrid the recurrent state after each row; and the rewind that undoes the
// rows a verification rejected.

// rowsOut is what a speculation step wants back from prefillSrc (State.rows).
//
// Rows [from, n) of the call want their logits and hidden state: every row of
// a verification (from 0), the last row alone of a draft step or of the
// prompt's catch-up. Both slices are row-major over those rows, NVocab and
// NEmbd wide.
type rowsOut struct {
	from           int
	logits, hidden []float32
	// noHead runs the rows for their KV and recurrent state alone: a replay
	// whose outputs the step already has.
	noHead bool
	// prompt runs the rows as prompt chunks even where a device takes ragged
	// rows: the prediction block's catch-up over a whole prompt, once a
	// session, whose length a ragged step would compile and record for.
	prompt bool
	// argmax asks a one-row step for its greedy token alone where the head
	// is on a device (nn.Head.ArgmaxOnly, ForwardGreedy's path): token is
	// then that token, the logits are not read back, and -1 says they were.
	// A greedy round that drafted nothing reads what plain decode reads --
	// the vocabulary's logits are a megabyte a token across the bus.
	//
	// On a step of several rows that runs as rows on a device it asks the
	// same of every wanted row: tokens holds each one's greedy token (the
	// device's per-row argmax, nn.Head.Tokens) and the logits are not read
	// back -- four megabytes a verification on a 248k vocabulary. tokens is
	// nil where the logits came back instead.
	argmax bool
	token  int32
	tokens []int32
	// snap[r], when there are that many, receives every host linear block's
	// recurrent state after row r of the call: a rejection after row r takes
	// it back (recSnap). Device blocks keep their own (nn.RecRewinder).
	snap []recSnap
	// noCap is a gate's break (TestSpecOverAFinalSoftcap): the host's head
	// over the step's rows skips the final softcap. false in every step a
	// session makes.
	noCap bool
}

// recSnap is one copy of every linear block's recurrent state for one row,
// indexed by block; nil for an attention block.
type recSnap struct{ conv, state [][]float32 }

// hiddenOf points a head's hidden readback at the step's destination for the
// duration of one call, and returns what puts it back. Callable on a nil
// rowsOut, which is every call but a speculation step's.
func (r *rowsOut) hiddenOf(h *nn.Head) func() {
	if r == nil || h == nil {
		return func() {}
	}
	h.Hidden = r.hidden
	return func() { h.Hidden = nil }
}

// foldHead readies the head a chunk folds into its submission for a
// speculation step. The rows it projects are the chunk's last (a draft step's
// one row) or every row of the chunk (a verification), and only when the rows
// the step wants are exactly those: anything else is projected on the host
// after the loop (finishRows), and false says so.
func (r *rowsOut) foldHead(h *nn.Head, base, n int) (func(), bool) {
	end := base + n
	if r.noHead {
		return nil, false
	}
	switch {
	case r.from == end-1:
		h.Hidden = r.hidden
		return func() { h.Hidden = nil }, true
	case r.from == base:
		keep := h.Logits
		h.RowLogits, h.Logits, h.LogitRows, h.Hidden = true, r.logits, 0, r.hidden
		return func() { h.RowLogits, h.Logits, h.LogitRows, h.Hidden = false, keep, 0, nil }, true
	}
	return nil, false
}

// stepRows runs src -- rows of slot 0 at the session's next positions -- and
// fills out (see rowsOut). It is prefill with every output a speculation step
// reads: one pass over the weights for all the rows, the KV written for each.
// The caller holds the pager (enterPager), as every inner path assumes.
func (s *State) stepRows(src []func(dst []float32) error, out *rowsOut) error {
	c := s.c
	out.token, out.tokens = -1, nil
	switch {
	case len(src) == 0:
		return errEmptyPrompt{}
	case out.noHead && len(src) < 2:
		return fmt.Errorf("model: a replay of %d row(s) is decode, which runs the head", len(src))
	case out.from < 0 || out.from >= len(src):
		return fmt.Errorf("model: a speculation step of %d rows wanting rows from %d", len(src), out.from)
	case !out.noHead && (len(out.logits) < (len(src)-out.from)*c.NVocab || len(out.hidden) < (len(src)-out.from)*c.NEmbd):
		return fmt.Errorf("model: a speculation step's destinations hold fewer than its %d rows", len(src)-out.from)
	}
	// A final softcap needs nothing here: every head a step's rows reach caps
	// them -- the device's per-row head and its folded one as decode's does
	// (nn.Head.Softcap), the host's in finishRows -- which
	// TestSpecOverAFinalSoftcap holds to plain decode on every tier.
	// On a device that runs ragged rows, a step of several rows goes as rows
	// of this one sequence: decode's matvec over every row, recorded like
	// decode, where a prompt chunk's batched kernels cost several decode
	// tokens at a verification's handful of rows -- and on a hybrid far more:
	// a verification carrying replayed rows ahead of the ones it wants cost
	// tens of milliseconds as a chunk of five to seven rows, against a few
	// as rows. Whatever rows it wants (stepRowsDevice).
	if len(src) >= 2 && !out.prompt && s.rowsVerifiable() {
		err := s.stepRowsDevice(src, out)
		// A refusal (a kernel the ragged path lacks, as a float expert bank)
		// left positions and every recurrent state where they were, so a
		// model with no linear block takes the prompt chunk, and its per-row
		// retry, instead. A hybrid's refusal stands: its step is rolled back
		// as one pass (prefillSrc).
		var ref errRowsRefused
		if !errors.As(err, &ref) || s.recurrent() {
			return err
		}
	}
	s.rows = out
	defer func() { s.rows = nil }()
	h := s.head
	if out.argmax && len(src) == 1 && h != nil && s.m.outB == nil { // as ForwardGreedy
		h.ArgmaxOnly, h.Token = true, -1
		defer func() { h.ArgmaxOnly = false }()
	}
	lg, err := s.prefillSrc(0, src, nil, nil)
	if err != nil {
		return err
	}
	// One row ran as decode (forward), whose logits come back as ever and
	// whose hidden state the head's readback or the host norm filled.
	switch {
	case len(src) != 1:
	case h != nil && h.ArgmaxOnly && h.Token >= 0:
		out.token = h.Token
	default:
		copy(out.logits, lg)
	}
	return nil
}

// errRowsRefused is the device's refusal of a speculation step of n rows,
// with its reason.
type errRowsRefused struct {
	n   int
	why string
}

func (e errRowsRefused) Error() string {
	return fmt.Sprintf("model: the device refused a %d-row speculation step: %s", e.n, e.why)
}

// rowsVerifiable reports a session whose speculation steps can run as ragged
// rows: every block it runs and the head on one device that runs rows, no
// block the ragged path refuses (latent attention).
func (s *State) rowsVerifiable() bool {
	_, ok := s.onRowsDevice()
	return ok && !s.c.MLA() && s.nseq == 1
}

// stepRowsDevice is stepRows as one ragged step on the device. The rows are
// ordered so the ones the step wants lead (nn.Head.LogitRows): every row of a
// verification in position order, or the last row first for a step that
// wants it alone -- a draft, whose block is attention and takes its rows in
// any order. Any other tail -- a verification behind replayed rows, or a
// hybrid's last row, whose linear blocks chain the rows in position order
// (tier.ragStep.chain) -- runs every row through the head in position order
// into s.rowsLog and s.rowsHid, and the tail is copied out: a few rows of
// head beside a prompt chunk.
func (s *State) stepRowsDevice(src []func(dst []float32) error, out *rowsOut) error {
	if s.c.PLEDim != 0 {
		// A row here is an embedding without its token, and Gemma 4's
		// per-layer inputs are built from the token.
		return fmt.Errorf("model: a speculation step on the device of a model with per-layer embeddings")
	}
	m, c := s.m, s.c
	n, nrot := len(src), c.RopeW()
	p0 := s.bpos[0]
	if p0+n > s.maxSeq {
		return errFull{s.maxSeq, s.reqSeq}
	}
	order := make([]int, n)
	for j := range order {
		order[j] = j
	}
	want := n
	tail := out.from > 0 && !out.noHead
	if out.from == n-1 && !s.recurrent() {
		order[0] = n - 1
		for j := 1; j < n; j++ {
			order[j] = j - 1
		}
		want, tail = 1, false
	}
	logits, hidden := out.logits, out.hidden
	// Greedy rows read back their tokens alone, where nothing the host
	// applies to the logits could move an argmax: a head bias could, and a
	// logit scale (a positive divisor) cannot.
	toks := out.argmax && !out.noHead && s.m.outB == nil
	if tail {
		s.rowsHid = grow32(s.rowsHid, n*c.NEmbd)
		hidden = s.rowsHid
		if !toks {
			s.rowsLog = grow32(s.rowsLog, n*c.NVocab)
			logits = s.rowsLog
		}
	}
	s.growDeviceBatch(n)
	pos, slot := make([]int, n), make([]int, n)
	for j, i := range order {
		row := s.bx[j*c.NEmbd : (j+1)*c.NEmbd]
		if err := src[i](row); err != nil {
			return err
		}
		pos[j] = p0 + i
		slot[j] = pos[j] // sequence 0's slots are its positions (nn.RowsDevice)
		if err := m.addPos(row, s.prow, pos[j]); err != nil {
			return err
		}
		s.ropeRow(j, s.ropeOf(0, pos[j]))
	}
	s.relocateFor(p0 + n)
	rd, _ := s.onRowsDevice()
	// A step with no head (a replay's commit, a piece of the prediction
	// block's catch-up) runs the blocks alone.
	var h *nn.Head
	if !out.noHead {
		h = s.head
		keep := h.Logits
		h.RowLogits, h.Logits, h.LogitRows, h.Hidden = !toks, nil, want, hidden[:want*c.NEmbd]
		if !toks {
			h.Logits = logits[:want*c.NVocab]
		}
		defer func() { h.RowLogits, h.Logits, h.LogitRows, h.Hidden = false, keep, 0, nil }()
	}
	if !rd.LayersRows(s.lo, s.hi, pos, slot, s.maxSeq, s.bx[:n*c.NEmbd], s.bcs[:n*nrot], s.bswaTable(n, c.NRotSWA), h) {
		why := "no reason given"
		if e, ok := s.ld.(nn.ErrReporter); ok {
			why = e.Err()
		}
		return errRowsRefused{n, why}
	}
	s.advance(0, n)
	from := 0
	if tail {
		from = out.from
	}
	switch {
	case h != nil && toks:
		s.rowsTok = grow32i(s.rowsTok, want-from)
		copy(s.rowsTok, h.Tokens[from:want])
		out.tokens = s.rowsTok[:want-from]
	case h != nil:
		s.finishDevLogits(logits[:want*c.NVocab])
		if tail {
			copy(out.logits[:(n-from)*c.NVocab], logits[from*c.NVocab:n*c.NVocab])
		}
	}
	if h != nil && tail {
		copy(out.hidden[:(n-from)*c.NEmbd], hidden[from*c.NEmbd:n*c.NEmbd])
	}
	return s.kvCheck()
}

// finishRows is prefillSrc's tail for a speculation step: the wanted rows'
// logits and hidden state, from the device's head where the chunk folded it
// (headDone) and from the host's otherwise. The wanted rows are in the last
// chunk, whose residual rows are in s.bx from lastBase.
func (s *State) finishRows(headDone bool, lastBase, total int) error {
	m, c, r := s.m, s.c, s.rows
	if r.noHead {
		return s.kvCheck()
	}
	cnt := total - r.from
	if headDone {
		if cnt == 1 {
			copy(r.logits, s.logits[:c.NVocab])
		}
		s.finishDevLogits(r.logits[:cnt*c.NVocab])
		return s.kvCheck()
	}
	off := r.from - lastBase
	if off < 0 {
		return fmt.Errorf("model: a speculation step wants row %d, before its last chunk at %d", r.from, lastBase)
	}
	// A host head over the wanted rows: the same norm and matmul the batched
	// decode step runs (batch.go), so under the exact GEMM a row's logits are
	// the decode path's bit for bit.
	s.growBatch(cnt)
	s.batchNormB(s.bh, s.bx[off*c.NEmbd:], s.outNorm, m.outNormB, c.NEmbd, cnt)
	if err := s.mm(r.logits, *s.outW, s.bh, cnt); err != nil {
		return err
	}
	copy(r.hidden, s.bh[:cnt*c.NEmbd])
	if r.noCap {
		s.headBias(r.logits[:cnt*c.NVocab])
		s.scaleLogits(r.logits[:cnt*c.NVocab])
		return s.kvCheck()
	}
	s.finishLogits(r.logits[:cnt*c.NVocab])
	return s.kvCheck()
}

// snapRow copies every host linear block's recurrent state for the slot after
// row `row` of a speculation step into its snapshot, when the step asked for
// one that far. Called once per (block, row), after the block steps the row.
func (r *rowsOut) snapRow(s *State, li, slot, row int) {
	if r == nil || row >= len(r.snap) || s.rconv == nil || s.rconv[li] == nil {
		return
	}
	g := s.dg
	nc, ns := g.convStateLen(), g.stateStride()
	sn := &r.snap[row]
	copy(sn.conv[li], s.rconv[li][slot*nc:(slot+1)*nc])
	copy(sn.state[li], s.rstate[li][slot*ns:(slot+1)*ns])
}

// newRecSnap allocates one snapshot of every host linear block's state for
// one row, or an empty one on a model with none.
func (s *State) newRecSnap() recSnap {
	if s.rconv == nil {
		return recSnap{}
	}
	g := s.dg
	sn := recSnap{conv: make([][]float32, len(s.rconv)), state: make([][]float32, len(s.rstate))}
	for li := range s.rconv {
		if s.rconv[li] != nil {
			sn.conv[li] = make([]float32, g.convStateLen())
			sn.state[li] = make([]float32, g.stateStride())
		}
	}
	return sn
}

// saveRec copies slot 0's recurrent state of every host linear block into
// sn, and loadRec copies it back. A device block's state is the device's
// (nn.RecRewinder): the host's copy of it is not the live one, and copying it
// was a measurable share of a speculation round.
func (s *State) saveRec(sn *recSnap) { s.copyRec(sn, true) }
func (s *State) loadRec(sn *recSnap) { s.copyRec(sn, false) }

func (s *State) copyRec(sn *recSnap, save bool) {
	g := s.dg
	nc, ns := g.convStateLen(), g.stateStride()
	for li := range s.rconv {
		if s.rconv[li] == nil || !s.c.LayerKind(li).Recurrent() || s.devAt(li) {
			continue
		}
		if save {
			copy(sn.conv[li], s.rconv[li][:nc])
			copy(sn.state[li], s.rstate[li][:ns])
		} else {
			copy(s.rconv[li][:nc], sn.conv[li])
			copy(s.rstate[li][:ns], sn.state[li])
		}
	}
}

// rewind takes slot 0 back to position p, forgetting what was written past
// it. Attention reads only up to a row's own position, so the keys and values
// past p are never addressed again and the next rows overwrite them; the
// page store's seal mark and the noted token ids fall with it. A recurrent
// state has no position to rewind: the caller restores it (loadRec,
// nn.RecRewinder) before or after, which is why this is not public.
func (s *State) rewind(p int) {
	if p > s.bpos[0] {
		panic(fmt.Sprintf("model: rewind forward from %d to %d", s.bpos[0], p))
	}
	// A windowed layer keeps nn.KVWindowSlack positions behind its window and
	// releases the rest: a rewind further back would read released pages.
	for li := s.kv.lo; li < len(s.kv.layers); li++ {
		if pg := &s.kv.layers[li]; pg.win > 0 {
			if j := pg.keyStart(p) / pg.p; j < pg.gone && !pg.resident(j) {
				panic(fmt.Sprintf("model: rewind from %d to %d reaches block %d's page %d, released behind "+
					"its window", s.bpos[0], p, li, j))
			}
		}
	}
	s.bpos[0], s.pos = p, p
	s.rewindRope(0, p)
	s.kv.seal(p)
	s.kv.note(p)
}
