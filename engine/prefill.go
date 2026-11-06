package engine

import (
	"fmt"
	"time"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/nn"
)

// MaxPrefillChunk is the most rows a step on the CPU device carries.
const MaxPrefillChunk = nn.MaxHostPrefillChunk

// chunkWidth is how many rows one prompt chunk runs: the host's chunk, or the
// device's where the session places blocks.
func (s *Instance) chunkWidth() int {
	if s.devCount() > 0 {
		return s.chunkOn(routeView)
	}
	return s.chunkOn(routeLanes)
}

// Prefill runs a whole prompt through the model in batches and returns the
// logits for its LAST token.
//
// Decode reads every weight once per token and is memory-bound; prefill reads
// them once per chunk, so arithmetic intensity rises with the batch. The graph
// is identical to Forward's: every matvec becomes a matmul over the chunk, and
// attention stays per token because token i attends only up to p0+i.
func (s *Instance) Prefill(tokens []int32) ([]float32, error) {
	defer s.fr.Enter()()
	if err := s.retireErr; err != nil {
		s.retireErr = nil
		return nil, err
	}
	return s.prefillInto(0, tokens)
}

// PrefillMixed prefills a prompt made of token runs and precomputed embedding
// runs. It is the prefill half of the embedding seam; see
// ForwardEmbd.
func (s *Instance) PrefillMixed(spans ...model.Span) ([]float32, error) {
	return s.prefillMixedAt(0, spans...)
}

// prefillMixedAt is PrefillMixed into one sequence slot.
func (s *Instance) prefillMixedAt(slot int, spans ...model.Span) ([]float32, error) {
	defer s.fr.Enter()()
	return s.prefillSpans(slot, spans...)
}

// prefillSpans is prefillMixedAt under the pager the caller holds.
func (s *Instance) prefillSpans(slot int, spans ...model.Span) ([]float32, error) {
	spans, err := s.encodePictures(spans, 0)
	if err != nil {
		return nil, err
	}
	src, err := s.spanSrc(spans)
	if err != nil {
		return nil, err
	}
	if err := s.bidirRuns(slot, spans); err != nil {
		return nil, err
	}
	// The device forgets the runs with the prompt: a later call at those
	// positions (a decode, another session's step) is causal.
	defer func() {
		if len(s.bidir) > 0 {
			if kd, ok := s.ld.(nn.KeyRunDevice); ok {
				kd.SetKeyRuns(nil, nil)
			}
		}
		s.bidir = s.bidir[:0]
	}()
	// Every row's place on the rotary axes, from the spans: an image's rows
	// at their grid coordinates and the text after it where the reference
	// resumes it.
	p0, total := s.seqPos(slot), src.len()
	next, err := s.spanRope(slot, p0, spans)
	if err != nil {
		return nil, err
	}
	// The rows after the prompt -- decode, a step -- turn where it left off,
	// which an image has moved away from their cache positions.
	s.markRope(slot, p0+total, next-(p0+total))
	// Every row's id for the page keys: a named picture's rows are named as
	// tokens are (spanKey), so the pages from here on can be sealed and found.
	if slot == 0 && s.kv.ns != "" {
		s.kv.note(p0, s.spanKey(p0, spans)...)
	}
	defer s.tokensOf(spans)()
	return s.prefillSrc(slot, src)
}

// bidirRuns records the bidirectional runs of a mixed prompt, at the positions
// they will occupy. Single sequence only: a batch slot's rows go through the
// device as a causal prefill (nn.RowsDevice), which has no run to widen.
func (s *Instance) bidirRuns(slot int, spans []model.Span) error {
	c := s.c
	p := s.seqPos(slot)
	if s.bidirAll {
		// The whole prompt is one run (a non-causal decoder's, its pictures'
		// rows among the text's).
		n := 0
		for _, sp := range spans {
			n += sp.N(c.NEmbd)
		}
		if n > nn.MaxDevicePrefillChunk {
			return fmt.Errorf("model: PrefillMixed: a bidirectional prompt of %d rows is longer "+
				"than one %d-row chunk", n, nn.MaxDevicePrefillChunk)
		}
		s.bidir = append(s.bidir, nn.KeyRun{Lo: p, Hi: p + n})
		return nil
	}
	for _, sp := range spans {
		n := sp.N(c.NEmbd)
		if sp.Bidir && n > 0 {
			if n > nn.MaxDevicePrefillChunk {
				return fmt.Errorf("model: PrefillMixed: a bidirectional span of %d rows is longer "+
					"than one %d-row chunk", n, nn.MaxDevicePrefillChunk)
			}
			s.bidir = append(s.bidir, nn.KeyRun{Lo: p, Hi: p + n})
		}
		p += n
	}
	return nil
}

// tokensOf sets spanTok to every row's token of a mixed prompt -- a token
// span's ids, an embedding span's Tokens when it names them, -1 for a row
// with none -- for a model that reads the token past the embedding (per-layer
// embeddings, ple.go), and returns its reset. A no-op elsewhere.
func (s *Instance) tokensOf(spans []model.Span) func() {
	if s.c.PLEDim == 0 {
		return func() {}
	}
	s.spanTok = s.spanTok[:0]
	for _, sp := range spans {
		if sp.Rows() && sp.Tokens == nil {
			for range sp.N(s.c.NEmbd) {
				s.spanTok = append(s.spanTok, -1)
			}
			continue
		}
		s.spanTok = append(s.spanTok, sp.Tokens...)
	}
	return func() { s.spanTok = nil }
}

// spanSrc is one flat row source over a mixed prompt's spans, so the chunk
// loop indexes it the same way it indexes tokens. An embedding span's Tokens
// name its rows' ids and are not looked up; a picture carrying tokens is a
// caller bug and is refused rather than resolved by precedence.
func (s *Instance) spanSrc(spans []model.Span) (rowSrc, error) {
	c := s.c
	total := 0
	for i, sp := range spans {
		if sp.Embd != nil && len(sp.Embd)%c.NEmbd != 0 {
			return rowSrc{}, fmt.Errorf("model: PrefillMixed: span %d is %d floats, not a multiple of n_embd %d",
				i, len(sp.Embd), c.NEmbd)
		}
		if sp.Embd != nil && sp.Tokens != nil {
			if len(sp.Tokens) != sp.N(c.NEmbd) {
				return rowSrc{}, fmt.Errorf("model: PrefillMixed: span %d names %d token ids for %d rows",
					i, len(sp.Tokens), sp.N(c.NEmbd))
			}
			for _, id := range sp.Tokens {
				if int(id) < 0 || int(id) >= c.NVocab {
					return rowSrc{}, errToken{id, c.NVocab}
				}
			}
		}
		total += sp.N(c.NEmbd)
	}
	if total == 0 {
		return rowSrc{}, errEmptyPrompt{}
	}
	src := rowSrc{tok: make([]int32, 0, total)}
	for _, sp := range spans {
		if sp.Embd != nil {
			src.given = append(src.given, givenRun{at: len(src.tok), rows: sp.Embd, deep: sp.Deep, n: len(sp.Embd) / c.NEmbd})
			if sp.Tokens != nil {
				src.tok = append(src.tok, sp.Tokens...)
				continue
			}
			for range sp.N(c.NEmbd) {
				src.tok = append(src.tok, -1)
			}
			continue
		}
		for _, id := range sp.Tokens {
			if int(id) < 0 || int(id) >= c.NVocab {
				return rowSrc{}, errToken{id, c.NVocab}
			}
		}
		src.tok = append(src.tok, sp.Tokens...)
	}
	return src, nil
}

func (s *Instance) prefillInto(slot int, tokens []int32) ([]float32, error) {
	c := s.c
	for _, id := range tokens {
		if int(id) < 0 || int(id) >= c.NVocab {
			return nil, errToken{id, c.NVocab}
		}
	}
	if slot == 0 {
		s.kv.note(s.seqPos(0), tokens...)
	}
	return s.prefillSrc(slot, tokenRows(tokens))
}

// ErrPrefillInterrupted is a prefill stopped between chunks because its
// interrupt (SetPrefillInterrupt) asked. The sequence holds the chunks that
// ran and nothing after them; the caller resets it before reusing the Instance.
const ErrPrefillInterrupted modelError = "model: prefill interrupted"

// SetPrefillInterrupt has every chunked prefill ask f before each chunk and
// stop with ErrPrefillInterrupted when it answers true: a server stops a long
// prompt whose client has gone without waiting out the whole prefill. nil
// removes it. f runs on the prefill's goroutine, between chunks, never
// inside one.
func (s *Session) SetPrefillInterrupt(f func() bool) { s.interrupt = f }

// promptCalls tells the device whether the calls that follow are a prompt's
// (nn.PromptDevice).
func (s *Instance) promptCalls(on bool) {
	if s.ldPrompt != nil {
		s.ldPrompt.SetPrompt(on)
	}
}

// prefillSrc is the chunked prefill over an abstract row source. src[i] fills
// one n_embd-wide row of the residual stream, and is either a vocabulary
// lookup or a caller-supplied embedding -- the two entry points differ in
// nothing else, which is the whole point of the seam. ids is the prompt when
// every row is a vocabulary lookup (nil otherwise), which lets a device holding
// the tied embedding gather the rows itself (nn.EmbedDevice). rp, when not
// nil, is every row's multi-axis rotary position (spanRope); nil rows advance
// like text (ropeOf).
func (s *Instance) prefillSrc(slot int, src rowSrc) ([]float32, error) {
	if err := s.marksHeld(); err != nil {
		return nil, err
	}
	m, c := s.m, s.c
	total := src.len()
	if total == 0 {
		return nil, errEmptyPrompt{}
	}
	// Every row's token for what reads it past the embedding (per-layer
	// embeddings, DeepSeek V4's hash blocks): -1 where a row has none.
	rowIDs := src.tok
	s.relocateFresh(total)
	// One token is decode; the batched path would allocate megabytes to beat
	// it by nothing. Single-sequence only, because Forward writes row 0; a
	// batch slot takes the chunked path at any length.
	if (total == 1 || total <= s.forwardRowsFor() && len(src.given) == 0) && s.emb == nil {
		var lg []float32
		var err error
		s.forwardPrompts++
		for i := range total {
			s.tok = rowIDs[i]
			if lg, err = s.forward(func() error {
				g := src.givenAt(i, c.NEmbd)
				if c.DeepTaps > 0 {
					s.fillDeep(src, i, 1, s.ple)
				}
				if g == nil {
					// A token's row, as a decode step embeds it.
					return s.embed(rowIDs[i])
				}
				st := &s.inScr
				st.tok1[0], st.pos1[0] = max(rowIDs[i], 0), s.pos
				if why := s.givenRows(s.x, s.ple, g, st.tok1[:], st.pos1[:]); why != nn.DeclineServes {
					return s.embedDeclined(why)
				}
				s.trace(-1, "embd", s.x)
				return nil
			}); err != nil {
				return nil, err
			}
			s.promptRows++
		}
		// The decode rows after it are tokens: their tap rows are zeros.
		if c.DeepTaps > 0 {
			clear(s.ple)
		}
		return lg, nil
	}
	if s.seqPos(slot)+total > s.maxSeq {
		return nil, errFull{s.maxSeq, s.reqSeq}
	}
	s.recPrefill = true
	defer func() { s.recPrefill = false }()
	// Every device call from here on is the prompt's, its last row's head
	// too (nn.PromptDevice).
	s.promptCalls(true)
	defer s.promptCalls(false)
	// A bidirectional embedding run is one chunk: no row's attention exists
	// until every row's k and v are in the cache. A mean-pooled one is one
	// chunk too: its readout pools every row in one call.
	bidir := s.emb != nil && s.emb.bidir
	whole := s.emb != nil && (s.emb.bidir || s.emb.pool == jlm.PoolMean)
	chunkCap := s.chunkWidth()
	// A speculation step that wants more than its last row's outputs runs as
	// one chunk: the rows it wants are read out of the last chunk's residual,
	// and a verification is a handful of rows.
	spanAll := s.rows != nil && s.rows.from < total-1
	if whole || spanAll {
		chunkCap = total
	}
	chunkRows := min(chunkCap, total)
	sizeFor := func(rows int) {
		if s.allOnDevice() {
			s.growDeviceBatch(rows)
		} else {
			s.growBatch(rows)
		}
		if c.MoE() {
			if w := rows * c.NExpertUsed; len(s.bsel) < w {
				s.bsel = make([]int32, w)
				s.bw = make([]float32, w)
			}
		}
	}
	sizeFor(chunkRows)

	// The prefill pool, where the JIT has one (nn.Config.PrefillCores). A
	// mixture stays on the decode pool: its expert GEMMs are a few routed
	// tokens each and a wider pool measured slower (nn/prefillcores.go).
	// Chunk by chunk it is placed beside any session decoding (serving.go).
	var place *prefillPlace
	if c.NExpert == 0 {
		place = &prefillPlace{s: s}
		defer place.end()
	}
	// headDone says the last chunk's submission ran the output projection on
	// its last row (the folded head), so s.logits are filled and the tail call
	// below is not needed.
	headDone := false
	// lastBase is where the last chunk starts: a speculation step's wanted
	// rows are in it (finishRows).
	lastBase := 0
	// The prompt's host blocks may stream through the device for it
	// (prefillstream.go): measured on its first two chunks, undone after the
	// last.
	chunks := (total + nn.MaxDevicePrefillChunk - 1) / nn.MaxDevicePrefillChunk
	// Not with a bidirectional run either: a streamed block runs on the device
	// without being told of it.
	ps := s.newPrefillStreamer(chunks, bidir || len(s.bidir) > 0)
	defer func() { ps.finish(s.seqPos(slot)) }()
	for base := 0; base < total; {
		if s.interrupt != nil && s.interrupt() {
			return nil, ErrPrefillInterrupted
		}
		ps.before(s.seqPos(slot))
		if place != nil {
			place.at()
		}
		width := s.chunkOn(routeLanes)
		if whole || spanAll {
			width = total
		}
		// Snapped to the page boundary when a hybrid is being cached: a
		// recurrent summary exists for a boundary only at the instant the
		// position is exactly there, so a chunk must not step over it. A
		// speculation step's rows are its own and may be rolled back, so it
		// seals no summary mid-step (the boundary is sealed when the position
		// next passes it).
		if s.view.Recurrent() && s.kv.ns != "" && !spanAll {
			if pb := s.recurrentBoundary(); pb > 0 {
				if next := ((s.seqPos(slot) / pb) + 1) * pb; next > s.seqPos(slot) {
					width = min(width, next-s.seqPos(slot))
				}
			}
		}
		// The device gets the full width, not the host tuner's. A chunk's
		// kernels are built for that width, and what it costs follows the
		// rows it carries, not the width: measured, d1-3B every block placed,
		// 64 rows 13.3 ms and 512 57.4 ms on the V100 (CUDA), 116 ms and
		// 834 ms on the M4 (Metal).
		if s.devCount() > 0 && !whole && !spanAll {
			width = s.chunkOn(routeView)
			// One submission's width when a run is in the prompt: a pipelined
			// chunk is cut into pieces, and a piece must not cut a run.
			if len(s.bidir) > 0 {
				width = min(width, nn.MaxDevicePrefillChunk)
			}
		}
		n := min(width, total-base)
		// A bidirectional run is never cut (bidir.go): a chunk that starts
		// in one ends where it does, past the chunk's width if it must.
		if len(s.bidir) > 0 && !s.bidirCutFault {
			n = min(s.bidirChunk(s.seqPos(slot), n), total-base)
		}
		// A block an earlier chunk moved (relocateFor) can widen the
		// device's chunk past what the prompt was sized for -- a second card
		// makes it a pipeline (chunkOn) -- and so can a run.
		if n > chunkRows {
			chunkRows = n
			sizeFor(n)
		}
		chunkBidir := bidir || s.bidirIn(s.seqPos(slot), n)
		lastBase = base
		p0 := s.seqPos(slot)
		s.relocateFor(p0 + n)
		chunkStart := time.Now()

		// A closure, as forward.go's fill is: a device call may write part of
		// the residual before it fails, so a restart begins from the
		// embeddings again.
		//
		refill := func() error {
			// On the card when the block run starts there and the device
			// gathers the step's input rows (nn.Head.GatherRow), or its tier
			// gathers the tied embedding: far faster than the host (EmbedRows).
			// Its input rows are the front end's, a position table's included.
			// A model of per-layer embeddings gathers on the card only with
			// every block there: a CPU block reads the per-layer inputs the
			// CPU's input rows write.
			if len(src.given) == 0 && s.head != nil && s.devAt(s.lo) && (c.PLEDim == 0 || s.allOnDevice()) &&
				(s.head.GatherRow || s.c.TiedEmbd && c.PLEDim == 0 && m.PosEmbd.E == nil) {
				// The card reads the per-layer inputs it writes back into the
				// chunk's own, handed before it runs.
				if c.PLEDim != 0 {
					s.growBPLE(n)
					s.layerInputs(s.bple[:n*c.PleWidth()])
				}
				if s.ldEmbed != nil && s.ldEmbed.EmbedRows(rowIDs[base:base+n], p0, s.bx[:n*c.NEmbd]) {
					// A token chunk's tap rows are zeros.
					s.tokenSide(n)
					return nil
				}
			}
			// The chunk's rows through the composed input rows, from this
			// goroutine: a token's row, its scale, its position row and its
			// per-layer inputs; a given row and its own.
			var bple []float32
			if c.PLEDim != 0 {
				s.growBPLE(n)
				bple = s.bple[:n*c.PleWidth()]
			}
			at := s.inScr.positions(n)
			for i := range at {
				at[i] = p0 + i
			}
			if c.DeepTaps > 0 {
				s.growBPLE(n)
				s.fillDeep(src, base, n, s.bple[:n*c.SideWidth()])
			}
			return s.fillRows(src, base, n, s.bx[:n*c.NEmbd], bple, at)
		}
		// The streams past the first, from each row's embedding.
		if c.StreamHead() {
			fill := refill
			refill = func() error {
				if err := fill(); err != nil {
					return err
				}
				return s.expandStreams(s.bx[:n*c.ResidW()], n, base)
			}
		}
		if err := refill(); err != nil {
			return nil, err
		}

		// The chunk's rotary tables are made for a device that reads them
		// (openPrompt); a device that makes its own is handed none.
		s.rowsFresh = false
		// The chunk's rows' tokens, -1 for a supplied embedding's, for a block
		// that reads them (graph.OpToken).
		s.chunkTok = grow32i(s.chunkTok, n)
		for i := range n {
			s.chunkTok[i] = -1
			if rowIDs != nil {
				s.chunkTok[i] = rowIDs[base+i]
			}
		}
		// Device blocks go as runs, not a prefix (see forward.go), each run as
		// one batch so the weights are read once for n tokens.
		// A run of blocks on one device is one call, whichever device: the
		// chunk's n rows go to it and come back once.
		for li := s.lo; li < s.hi; li++ {
			r := s.routeAt(li)
			hi := li + 1
			// A tower's run ends where a tap keeps the rows a block leaves.
			for hi < s.hi && s.routeAt(hi) == r && !s.tapsAfter(hi-1) {
				hi++
			}
			// The last chunk carries the head when the device runs every
			// block, so the projection runs on the last row inside the same
			// submission. A device that cannot refuses before submitting and
			// the tail call below runs instead.
			// A batch row's history on the device lives at its own slots,
			// which the per-sequence call cannot address: the run goes as
			// rows of this one sequence, a causal prefill (nn.RowsDevice),
			// which a linear block steps in position order through the
			// slot's one state.
			var fh *nn.Head
			// A tap reads every chunk's residual, which a folded head keeps
			// on the device.
			if li == s.lo && hi == s.hi && base+n == total && s.emb == nil && s.head != nil && !s.noHeadFold && s.rowTap == nil &&
				!c.StreamHead() && r.run == nn.LayerRunner(s.ld) {
				fh = s.head
				// A speculation step folds the head only over the rows it
				// wants; otherwise the host projects them after the loop.
				if s.rows != nil {
					if undo, ok := s.rows.foldHead(fh, base, n); ok {
						defer undo()
					} else {
						fh = nil
					}
				}
			}
			// A chunk of a few rows over the whole model is one ragged step
			// of them (tailrows.go): decode's matvec over the live rows,
			// where the chunk would run its batched kernels at the device's
			// grain. The head is the tail call after the loop. A decline is
			// the prompt's error (refusal); a fault falls through to the
			// chunk, whose call takes the fault's path.
			if !s.opt.noTailRows && n >= 2 && n <= tailRowsMax && li == s.lo && hi == s.hi && !chunkBidir && s.rows == nil && r.run == nn.LayerRunner(s.ld) &&
				!s.recurrent() && s.rowsRefusal() == nil {
				pos := s.tailPos[:0]
				for j := 0; j < n; j++ {
					pos = append(pos, p0+j)
				}
				s.tailPos = pos
				// One sequence: slot is the position (nn.RowsDevice).
				s.layerInputs(s.bple[:n*c.SideWidth()])
				if s.rowsOn(r, li, hi, pos, pos, s.bx[:n*c.ResidW()], s.bcs[:n*c.RopeW()],
					s.bswaTable(n, c.LocalNRot()), false, nil, 0) == nil {
					s.tailRowChunks++
					li = hi - 1
					continue
				}
				if err := s.refusal(li); err != nil {
					return nil, err
				}
				// The step may have written part of the residual.
				if err := refill(); err != nil {
					return nil, err
				}
			}
			if err := s.openPrompt(r, slot, base, p0, n, chunkRows, bidir); err != nil {
				return nil, err
			}
			if s.callTellOn(r, p0, n) && r.run.Layers(li, hi, p0, n, s.bx[:n*c.ResidW()], s.bcs[:n*c.RopeW()],
				s.bswaTable(n, c.LocalNRot()), fh) {
				headDone = fh != nil
				li = hi - 1
				s.keepTaps(li, n)
				continue
			}
			if err := s.runRefusal(r, li); err != nil {
				return nil, err
			}
			// A fault: a hybrid's speculation step is rolled back as one
			// pass, so its fault is the step's error; any other chunk's
			// demotes and restarts on the CPU device.
			if s.rows != nil && s.recurrent() {
				why := "no reason given"
				if e, ok := s.ld.(nn.ErrReporter); ok && e.Err() != "" {
					why = e.Err()
				}
				return nil, fmt.Errorf("model: the device refused a %d-row speculation step on blocks [%d,%d): %s",
					n, li, hi, why)
			}
			if !s.fallsBack() {
				return nil, errDevice{li, s.devWhy()}
			}
			// A hybrid cannot restart: re-running the linear blocks before li
			// would apply their gated delta rule twice. See errRecurRetry.
			if s.recurrent() {
				return nil, errRecurRetry{li, s.devWhy()}
			}
			if !s.demoteAll() {
				return nil, errDevice{li, s.devWhy()}
			}
			if err := refill(); err != nil {
				return nil, err
			}
			li = s.lo - 1
			continue
		}
		if s.emb != nil {
			if err := s.emb.collect(s, base, n); err != nil {
				return nil, err
			}
		}
		if s.rowTap != nil {
			s.rowTap(p0, s.bx[:n*c.ResidW()])
		}
		s.recChunkPoint(slot, p0+n)
		s.advance(slot, n)
		// The last chunk's last row is the only one the caller can use, so it
		// becomes the decode path's one-row residual, every stream of it, and
		// the session continues from there.
		//
		// It is written for a batch row too: nothing decodes from it there, but
		// the output projection below reads it.
		for k := range c.Streams() {
			copy(c.Stream(s.resid(), k, 1, 0), c.Stream(s.bx, k, n, n-1))
		}
		// A head over several streams reads their collapse: collapsed in
		// place, the chunk's rows' stream 0 is their head input, and the last
		// row's is s.x's. A device head that collapses them itself is handed
		// the last row's as they are (devCollapses).
		if c.StreamHead() && base+n == total && !s.devCollapses() {
			if s.rows != nil {
				s.rows.resBase = base
			}
			if err := s.collapseStreams(s.bx[:n*c.ResidW()], n); err != nil {
				return nil, err
			}
			copy(s.x, s.bx[(n-1)*c.NEmbd:n*c.NEmbd])
		}
		ps.after(n, time.Since(chunkStart).Seconds(), s.seqPos(slot))
		s.promptRows += n
		base += n
	}

	if s.rows != nil {
		return nil, s.finishRows(headDone, lastBase, total)
	}
	// Only the final token needs logits; the earlier positions exist to fill
	// the KV cache. The head runs on the device where the device holds it, as
	// forward.go's partial-seam tail does (an empty block range with a head).
	// An embedding run stops at the residual.
	if s.emb != nil {
		return nil, s.kvCheck()
	}
	if headDone {
		if err := s.devLogits(); err != nil {
			return nil, err
		}
		if err := s.kvCheck(); err != nil {
			return nil, err
		}
		return s.logits, nil
	}
	devCollapse := s.devCollapses()
	if s.head != nil && s.devCount() > 0 {
		if s.ld.Layers(s.gpuLayers, s.gpuLayers, s.seqPos(slot)-1, 1, s.resid(), s.cs[:c.RopeW()], s.swaTable(c.LocalNRot()), s.head) {
			if err := s.devLogits(); err != nil {
				return nil, err
			}
			// A page the store could not return fails the request. See kvCache.faultIn.
			if err := s.kvCheck(); err != nil {
				return nil, err
			}
			return s.logits, nil
		}
		if err := s.refusal(s.gpuLayers); err != nil {
			return nil, err
		}
		if !s.fallsBack() {
			return nil, errDevice{s.gpuLayers, s.devWhy()}
		}
	}
	if devCollapse {
		if err := s.collapseStreams(s.resid(), 1); err != nil {
			return nil, err
		}
	}
	why := nn.DeclineServes
	if len(s.HostPick) > 0 {
		why = s.pickLabels(s.x)
	} else {
		why = s.headRow(s.x, s.logits, nil)
	}
	if why != nn.DeclineServes {
		return nil, cpuHeadRefused(why)
	}
	// A page the store could not return fails the request. See kvCache.faultIn.
	if err := s.kvCheck(); err != nil {
		return nil, err
	}
	return s.logits, nil
}

// devCollapses says a prompt's head over several streams goes to a device
// that collapses them itself (nn.Head.Collapses), handed the last row's
// every stream as forward.go's tail hands a token's: the host collapsing
// them first would have the device collapse stream 0's collapse beside
// streams the last row never wrote.
func (s *Instance) devCollapses() bool {
	c := s.m.Cfg
	return c.StreamHead() && s.head != nil && s.head.Collapses && s.rows == nil && s.devCount() > 0 && s.lo < c.NLayer
}

// chunkOn is how many prompt rows route r's device takes in one chunk: its
// own limit (nn.ChunkLimiter), a pipeline's chunks where every block crosses
// its cards, else a device's chunk.
func (s *Instance) chunkOn(r int) int {
	if l := s.chunks[r]; l != nil && l.ChunkRows() > 0 {
		return l.ChunkRows()
	}
	if p, ok := s.places[r].(nn.Pipeliner); ok && s.allOnDevice() && p.PipelineDepth() > 1 {
		return nn.MaxDevicePrefillChunk * pipelineChunks
	}
	return nn.MaxDevicePrefillChunk
}

// pipelineChunks is how many device chunks a pipelined prompt chunk holds:
// llama.cpp's 2048-token batch over 512-token micro-batches. With d cards a
// chunk takes (pieces + d - 1) piece-times instead of d x pieces.
// Deeper pipelines (more cards) gain from more pieces.
const pipelineChunks = 4

func (s *Instance) growDeviceBatch(chunk int) {
	c := s.c
	// Every buffer the rows read, the second rotary's table too: a set
	// whose residual rows suffice may hold a shorter table, or none.
	if len(s.bx) >= chunk*c.ResidW() && len(s.bcs) >= chunk*c.RopeW() &&
		(s.m.RopeSWA == nil || len(s.bcsSWA) >= chunk*c.LocalNRot()) {
		return
	}
	b := s.takeSpare()
	if b != nil {
		s.spareHost = b.host // growBatch's, if a host block wants it
	}
	if b != nil && len(b.bx) >= chunk*c.ResidW() && len(b.bcs) >= chunk*c.RopeW() &&
		(s.m.RopeSWA == nil || len(b.bcsSWA) >= chunk*c.LocalNRot()) {
		s.bx, s.bcs, s.bcsSWA = b.bx, b.bcs, b.bcsSWA
		return
	}
	s.bx = make([]float32, chunk*c.ResidW())
	s.bcs = make([]float32, chunk*c.RopeW())
	if s.m.RopeSWA != nil {
		s.bcsSWA = make([]float32, chunk*c.LocalNRot())
	}
}

// promptBuf is a session's prompt buffers -- a chunk's residual rows and its
// rotary tables -- kept by the model when the instance closes. A request is a
// fresh instance, and allocating these per request cost a 512-row prompt about
// a fifth of its time (a fresh 4 MiB heap span faults in page by page;
// gpu-kernels.md, "The prompt lost 22%").
//
// One spare serves serial requests; concurrent instances on one model would
// need a free list.
type promptBuf struct {
	bx, bcs, bcsSWA []float32
	host            hostBatch
}

// takeSpare is the model's spare prompt buffers, or nil, emptying the slot.
func (s *Instance) takeSpare() *promptBuf {
	s.eng.host.mu.Lock()
	defer s.eng.host.mu.Unlock()
	b := s.eng.host.spare
	s.eng.host.spare = nil
	return b
}

// keepSpare gives a closing session's prompt buffers to its spares, keeping the
// larger of them and any it already holds.
func (s *Instance) keepSpare() {
	if s.bx == nil {
		return
	}
	s.eng.host.mu.Lock()
	defer s.eng.host.mu.Unlock()
	if s.eng.host.spare == nil || len(s.eng.host.spare.bx) < len(s.bx) {
		s.eng.host.spare = &promptBuf{s.bx, s.bcs, s.bcsSWA, s.takeHostBatch()}
	}
	s.bx, s.bcs, s.bcsSWA = nil, nil, nil
}

// allOnDevice reports whether every block of this instance runs on a device.
func (s *Instance) allOnDevice() bool {
	for li := s.lo; li < s.hi; li++ {
		if !s.devAt(li) {
			return false
		}
	}
	return true
}

func (s *Instance) growBatch(chunk int) {
	c := s.c
	qDim := c.MaxQDim()
	s.growDeviceBatch(chunk)
	if len(s.bh) >= chunk*max(c.NEmbd, qDim) {
		return
	}
	// A previous instance's set where it is big enough (hostBatch); scratch
	// allocates what is not.
	h := s.spareHost
	s.spareHost.bh = nil
	s.bh = scratch(h.bh, chunk*max(c.NEmbd, qDim), 0)
	// One attention scores row per token, for the same kernels decode uses:
	// sized where it is used, by the keys the chunk reaches (rowsStride).
	s.battf = h.battf
	if c.AttnOutGate {
		s.qgate = scratch(h.qgate, chunk*2*qDim, 0)
		s.ogate = scratch(h.ogate, chunk*qDim, 0)
	}
	s.bqf = scratch(h.bqf, chunk*qDim, 0)
	s.bxbf = scratch(h.bxbf, chunk*qDim, 0)
}

type errEmptyPrompt struct{}

func (errEmptyPrompt) Error() string { return "model: Prefill called with no tokens" }

// bswaTable and bswaTableAt are swaTable for the batched path: the prefill
// tables are per token, so the device gets the same n-row slice of the local
// one that it gets of the global one. nil where the architecture has one base.
func (s *Instance) bswaTable(n, nrot int) []float32 {
	if s.bcsSWA == nil {
		return nil
	}
	return s.bcsSWA[:n*nrot]
}

// forwardRowsFor is the longest prompt run through the decode path: the
// caller's WithPrefillForwardRows, else one.
func (s *Instance) forwardRowsFor() int {
	if s.opt.forwardRowsSet {
		return s.opt.forwardRows
	}
	return 1
}
