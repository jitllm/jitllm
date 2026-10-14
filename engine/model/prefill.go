package model

import (
	"errors"
	"fmt"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"time"

	"github.com/jitllm/jitllm/engine/nn"
)

// MaxPrefillChunk sizes the batched scratch. The width used is tuned per
// machine (nn.JIT.PrefillChunk), so buffers fit the widest candidate.
const MaxPrefillChunk = nn.MaxHostPrefillChunk

// chunkWidth is how many rows one prompt chunk runs: the host's chunk, or the
// device's where the session places blocks.
func (s *State) chunkWidth() int {
	if s.devCount() > 0 {
		return s.deviceChunk()
	}
	return MaxPrefillChunk
}

// Prefill runs a whole prompt through the model in batches and returns the
// logits for its LAST token.
//
// Decode reads every weight once per token and is memory-bound; prefill reads
// them once per chunk, so arithmetic intensity rises with the batch. The graph
// is identical to Forward's: every matvec becomes a matmul over the chunk, and
// attention stays per token because token i attends only up to p0+i.
func (s *State) Prefill(tokens []int32) ([]float32, error) {
	defer s.m.enterPager()()
	if s.batched {
		return nil, errBatch{}
	}
	if err := s.retireErr; err != nil {
		s.retireErr = nil
		return nil, err
	}
	return s.prefillInto(0, tokens)
}

// PrefillSeq runs a prompt into ROW slot of a batch session, leaving the other
// rows untouched.
//
// It is what admission needs: without it a new sequence's prompt would step the
// whole batch once per token.
//
// The row must be at position 0, i.e. a retired slot; Retire is the only thing
// that resets a row's positions.
func (s *State) PrefillSeq(slot int, tokens []int32) ([]float32, error) {
	defer s.m.enterPager()()
	if !s.batched {
		return nil, errBatch{}
	}
	if slot < 0 || slot >= s.nseq {
		return nil, errSlot{slot, s.nseq}
	}
	// A device that pages each row's history takes this prompt's pages first,
	// and moves blocks off when they do not fit -- before the path is chosen,
	// since a block that left makes it a split placement.
	if sd, ok := s.ld.(nn.SeqKVDevice); ok && s.devCount() > 0 && sd.PerSequenceKV() {
		end := s.bpos[slot] + len(tokens)
		s.relocateUntil(end, func() bool { return sd.ReserveKVSeqs([]int{slot * s.maxSeq}, []int{end}) })
	}
	// A model wholly on one device that runs ragged rows prefills there, the
	// prompt laid out as rows of this slot. A split placement prefills on the
	// host and hands each device run over as rows at this slot's positions.
	if s.devCount() > 0 {
		if _, ok := s.onRowsDevice(); ok {
			return s.prefillSeqDevice(slot, tokens)
		}
		if _, ok := s.ld.(nn.RowsDevice); !ok {
			return nil, errBatchDevice{}
		}
	}
	return s.prefillInto(slot, tokens)
}

// Span is one run of a mixed prompt: token ids, a picture, or precomputed
// embeddings -- one of them. Embd is n*NEmbd floats laid out row-major, and is
// the residual stream rather than a vocabulary row -- EmbdScale is not applied
// to it, for the reason ForwardEmbd's comment gives. A Picture is rows the
// State's vision segment encodes when the span is prefilled (picture.go), and
// its Grid and Key are the span's.
//
// An Embd span may carry Tokens too, one per row: the ids its rows embed,
// which are not looked up but read where a model reads the token past the
// embedding -- DeepSeek V4's hash-routed blocks, which refuse a row without one
// as its references cannot run it, and per-layer embeddings, which take token
// 0's for such a row (ple.go). A picture's rows have no ids.
type Span struct {
	Tokens  []int32
	Embd    []float32
	Picture *Picture
	// Bidir says the span's rows attend to each other in both directions
	// rather than causally (Gemma 3's image tokens, which its reference masks
	// so). The rows before and after it stay causal.
	Bidir bool
	// Grid is an image span's shape, which is where its rows turn on a
	// multi-axis rotary (mrope.go). It is required there and ignored by a
	// model whose rotary is a plain index.
	Grid ImageGrid
	// Key names the picture an embedding span's rows came from (ImageKey), so
	// the prefix cache can name the pages that hold them and every page after.
	// nil leaves them unnamed, and nothing from them on is cached.
	Key *ImageKey
	// Deep is an embedding span's deepstack rows (Qwen3-VL), [taps][rows][NEmbd]:
	// tap k adds into the span's rows after the text model's block k (deep.go).
	// nil for every other span; a Picture span's are its tower's.
	Deep []float32
}

// rows reports whether the span is rows rather than tokens: a picture or an
// embedding run.
func (sp Span) rows() bool { return sp.Embd != nil || sp.Picture != nil }

func (sp Span) n(nEmbd int) int {
	if sp.Embd != nil {
		return len(sp.Embd) / nEmbd
	}
	if sp.Picture != nil {
		return sp.Picture.Rows
	}
	return len(sp.Tokens)
}

// named is the spans with each picture's grid and name on its span, so the
// rotary and the page keys read them before the picture is encoded.
func named(spans []Span) []Span {
	var out []Span
	for i, sp := range spans {
		if sp.Picture == nil || (!sp.Grid.zero() && sp.Key != nil) {
			continue
		}
		if out == nil {
			out = append([]Span(nil), spans...)
		}
		if sp.Grid.zero() {
			out[i].Grid = sp.Picture.Grid
		}
		if sp.Key == nil {
			out[i].Key = &sp.Picture.Key
		}
	}
	if out == nil {
		return spans
	}
	return out
}

// PrefillMixed prefills a prompt made of token runs, pictures and precomputed
// embedding runs. It is the prefill half of the embedding seam; see
// ForwardEmbd.
func (s *State) PrefillMixed(spans ...Span) ([]float32, error) {
	return s.PrefillSeqMixed(0, spans...)
}

// PrefillSeqMixed is PrefillMixed into one batch slot.
func (s *State) PrefillSeqMixed(slot int, spans ...Span) ([]float32, error) {
	defer s.m.enterPager()()
	spans, err := s.pictureRows(named(spans), 0)
	if err != nil {
		return nil, err
	}
	src, err := s.spanSrc(spans)
	if err != nil {
		return nil, err
	}
	undoDeep, err := s.deepRuns(spans, 0)
	if err != nil {
		return nil, err
	}
	defer undoDeep()
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
	p0, total := s.seqPos(slot), len(src)
	rp, next, err := s.spanRope(slot, p0, spans)
	if err != nil {
		return nil, err
	}
	// The rows after the prompt -- decode, a step -- turn where it left off,
	// which an image has moved away from their cache positions.
	s.markRope(slot, p0+total, next-(p0+total))
	// Every row's id for the page keys: a named picture's rows are named as
	// tokens are (spanKey), so the pages from here on can be sealed and found.
	if slot == 0 && !s.batched && s.kv.ns != "" {
		s.kv.note(p0, s.spanKey(p0, spans, rp)...)
	}
	defer s.tokensOf(spans)()
	return s.prefillSrc(slot, src, nil, rp)
}

// bidirRuns records the bidirectional runs of a mixed prompt, at the positions
// they will occupy. Single sequence only: a batch slot's rows go through the
// device as a causal prefill (nn.RowsDevice), which has no run to widen.
func (s *State) bidirRuns(slot int, spans []Span) error {
	c := s.c
	p := s.seqPos(slot)
	for _, sp := range spans {
		n := sp.n(c.NEmbd)
		if sp.Bidir && n > 0 {
			if s.batched {
				return fmt.Errorf("model: PrefillMixed: a bidirectional span in a batch slot is not implemented")
			}
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
// embeddings, ple.go; DeepSeek V4's hash blocks, ds4.go), and returns its
// reset. A no-op elsewhere.
func (s *State) tokensOf(spans []Span) func() {
	if s.c.PLEDim == 0 && s.c.NHashLayers == 0 {
		return func() {}
	}
	s.spanTok = s.spanTok[:0]
	for _, sp := range spans {
		if sp.rows() && sp.Tokens == nil {
			for range sp.n(s.c.NEmbd) {
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
func (s *State) spanSrc(spans []Span) ([]func(dst []float32) error, error) {
	c := s.c
	total := 0
	for i, sp := range spans {
		if sp.Picture != nil && sp.Tokens != nil {
			return nil, fmt.Errorf("model: PrefillMixed: span %d carries both tokens and a picture", i)
		}
		if sp.Embd != nil && len(sp.Embd)%c.NEmbd != 0 {
			return nil, fmt.Errorf("model: PrefillMixed: span %d is %d floats, not a multiple of n_embd %d",
				i, len(sp.Embd), c.NEmbd)
		}
		if sp.Embd != nil && sp.Tokens != nil {
			if len(sp.Tokens) != sp.n(c.NEmbd) {
				return nil, fmt.Errorf("model: PrefillMixed: span %d names %d token ids for %d rows",
					i, len(sp.Tokens), sp.n(c.NEmbd))
			}
			for _, id := range sp.Tokens {
				if int(id) < 0 || int(id) >= c.NVocab {
					return nil, errToken{id, c.NVocab}
				}
			}
		}
		total += sp.n(c.NEmbd)
	}
	if total == 0 {
		return nil, errEmptyPrompt{}
	}
	// Resolve position -> (span, row) once. A prompt is thousands of rows at
	// most and this runs once per prefill, not per chunk.
	src := make([]func(dst []float32) error, 0, total)
	for i, sp := range spans {
		sp := sp
		// A picture no row of the run reads (the prefix cache restored it
		// whole, pictureRows): rows that fail if anything does read them.
		if sp.Picture != nil && sp.Embd == nil {
			for range sp.n(c.NEmbd) {
				src = append(src, func([]float32) error {
					return fmt.Errorf("model: span %d's picture was read before it was encoded", i)
				})
			}
			continue
		}
		if sp.Embd != nil {
			for i := 0; i < sp.n(c.NEmbd); i++ {
				i := i
				src = append(src, func(dst []float32) error {
					copy(dst, sp.Embd[i*c.NEmbd:(i+1)*c.NEmbd])
					return nil
				})
			}
			continue
		}
		for _, id := range sp.Tokens {
			id := id
			if int(id) < 0 || int(id) >= c.NVocab {
				return nil, errToken{id, c.NVocab}
			}
			src = append(src, func(dst []float32) error { return s.embedInto(dst, id) })
		}
	}
	return src, nil
}

// embedInto is embed() writing somewhere other than s.x.
func (s *State) embedInto(dst []float32, token int32) error {
	m, c := s.m, s.c
	if err := m.embedRow(dst, int(token)); err != nil {
		return err
	}
	if c.EmbdScale != 1 {
		s.scale(dst, float32(c.EmbdScale))
	}
	return nil
}

func (s *State) prefillInto(slot int, tokens []int32) ([]float32, error) {
	c := s.c
	for _, id := range tokens {
		if int(id) < 0 || int(id) >= c.NVocab {
			return nil, errToken{id, c.NVocab}
		}
	}
	if slot == 0 {
		s.kv.note(s.seqPos(0), tokens...)
	}
	return s.prefillSrc(slot, nil, tokens, nil)
}

// ErrPrefillInterrupted is a prefill stopped between chunks because its
// interrupt (SetPrefillInterrupt) asked. The sequence holds the chunks that
// ran and nothing after them; the caller resets it before reusing the State.
var ErrPrefillInterrupted = errors.New("model: prefill interrupted")

// SetPrefillInterrupt has every chunked prefill ask f before each chunk and
// stop with ErrPrefillInterrupted when it answers true: a server stops a long
// prompt whose client has gone without waiting out the whole prefill. nil
// removes it. f runs on the prefill's goroutine, between chunks, never
// inside one.
func (s *State) SetPrefillInterrupt(f func() bool) { s.interrupt = f }

// prefillSrc is the chunked prefill over an abstract row source. src[i] fills
// one n_embd-wide row of the residual stream, and is either a vocabulary
// lookup or a caller-supplied embedding -- the two entry points differ in
// nothing else, which is the whole point of the seam. ids is the prompt when
// every row is a vocabulary lookup (nil otherwise), which lets a device holding
// the tied embedding gather the rows itself (nn.EmbedDevice). rp, when not
// nil, is every row's multi-axis rotary position (spanRope); nil rows advance
// like text (ropeOf).
func (s *State) prefillSrc(slot int, src []func(dst []float32) error, ids []int32, rp [][4]int) ([]float32, error) {
	m, c := s.m, s.c
	// src nil: every row is a vocabulary lookup of ids, with no function a
	// row (prefillInto), so a warm prompt makes no closure per token.
	total := len(src)
	if src == nil {
		total = len(ids)
	}
	if total == 0 {
		return nil, errEmptyPrompt{}
	}
	// Every row's token for what reads it past the embedding (per-layer
	// embeddings, DeepSeek V4's hash blocks): the prompt, or a mixed prompt's
	// rows (tokensOf), -1 where a row has none. nil when neither has any.
	rowIDs := ids
	if rowIDs == nil {
		rowIDs = s.spanTok
	}
	s.relocateFresh(total)
	// One token is decode; the batched path would allocate megabytes to beat
	// it by nothing. Single-sequence only, because Forward writes row 0; a
	// batch slot takes the chunked path at any length.
	if !s.batched && total == 1 && s.emb == nil && len(s.deep) == 0 {
		var lg []float32
		var err error
		s.tok = -1
		if rowIDs != nil {
			s.tok = rowIDs[0]
		}
		for i := range total {
			if lg, err = s.forward(func() error {
				if err := s.fillRow(src, ids, i, s.x); err != nil {
					return err
				}
				s.m.trace(-1, "embd", s.x)
				if c.PLEDim != 0 {
					tok := int32(0) // a row with no id takes token 0's; see ple.go
					if rowIDs != nil {
						tok = max(rowIDs[0], 0)
					}
					return s.pleInputs(s.ple, tok, s.x)
				}
				return nil
			}); err != nil {
				return nil, err
			}
			s.promptRows++
		}
		return lg, nil
	}
	if s.seqPos(slot)+total > s.maxSeq {
		return nil, errFull{s.maxSeq, s.reqSeq}
	}
	// A bidirectional embedding run is one chunk: no row's attention exists
	// until every row's k and v are in the cache.
	bidir := s.emb != nil && s.emb.bidir
	chunkCap := s.chunkWidth()
	// A speculation step that wants more than its last row's outputs runs as
	// one chunk: the rows it wants are read out of the last chunk's residual,
	// and a verification is a handful of rows.
	spanAll := s.rows != nil && s.rows.from < total-1
	if bidir || spanAll {
		chunkCap = total
	}
	chunkRows := min(chunkCap, total)
	if s.allOnDevice() {
		s.growDeviceBatch(chunkRows)
	} else {
		s.growBatch(chunkRows)
	}
	if c.MoE() {
		if w := min(chunkCap, total) * c.NExpertUsed; len(s.bsel) < w {
			s.bsel = make([]int32, w)
			s.bw = make([]float32, w)
		}
	}

	// The prefill pool, where the JIT has one (nn.Config.PrefillCores). A
	// mixture stays on the decode pool: its expert GEMMs are a few routed
	// tokens each and a wider pool measured slower (engine/nn/prefillcores.go).
	if c.NExpert == 0 {
		s.jit.BeginPrefill()
		defer s.jit.EndPrefill()
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
		// Asked per chunk, not once: while the tuner is cycling, successive
		// chunks deliberately use different widths.
		width := s.jit.PrefillChunk()
		if bidir || spanAll {
			width = total
		}
		// Snapped to the page boundary when a hybrid is being cached: a
		// recurrent summary exists for a boundary only at the instant the
		// position is exactly there, so a chunk must not step over it. A
		// speculation step's rows are its own and may be rolled back, so it
		// seals no summary mid-step (the boundary is sealed when the position
		// next passes it).
		if s.rconv != nil && s.kv.ns != "" && !spanAll {
			if pb := s.recurrentBoundary(); pb > 0 {
				if next := ((s.seqPos(slot) / pb) + 1) * pb; next > s.seqPos(slot) {
					width = min(width, next-s.seqPos(slot))
				}
			}
		}
		// The device runs one fixed grid (kernels bake their element count), so
		// a narrower chunk runs the same work with surplus rows zeroed; it
		// always gets the full width, not the host tuner's.
		if s.devCount() > 0 && !bidir && !spanAll {
			width = s.deviceChunk()
			// One submission's width when a run is in the prompt: a pipelined
			// chunk is cut into pieces, and a piece must not cut a run.
			if len(s.bidir) > 0 {
				width = min(width, nn.MaxDevicePrefillChunk)
			}
		}
		n := min(width, total-base)
		// A bidirectional run is never cut (bidir.go).
		if len(s.bidir) > 0 && !s.bidirCutFault {
			n = min(s.bidirChunk(s.seqPos(slot), n), total-base)
		}
		chunkBidir := s.bidirIn(s.seqPos(slot), n)
		lastBase = base
		p0 := s.seqPos(slot)
		s.relocateFor(p0 + n)
		chunkStart := time.Now()

		// A closure, as forward.go's fill is: a device call may write part of
		// the residual before it fails, so a restart begins from the
		// embeddings again.
		//
		// Over the pool: in the packed layout one row's words are a
		// vocabulary's width apart (see Model.embedRow), so each row is many
		// cache misses, and rows are independent.
		refill := func() error {
			// On the card when it holds the tied embedding and the block run
			// starts there: the device gathers the rows far faster (EmbedRows).
			if ids != nil && s.head != nil && s.devAt(s.lo) && s.c.TiedEmbd && c.PLEDim == 0 {
				if ed, ok := s.ld.(nn.EmbedDevice); ok && ed.EmbedRows(ids[base:base+n], s.bx[:n*c.NEmbd]) {
					return nil
				}
			}
			return s.embedRun(src, ids, base, n)
		}
		// Gemma 4's per-layer inputs are built from each row's token and its
		// embedding, inside refill for the same reason.
		if c.PLEDim != 0 {
			fill := refill
			refill = func() error {
				if err := fill(); err != nil {
					return err
				}
				w := c.pleWidth()
				s.growBPLE(n)
				for i := 0; i < n; i++ {
					tok := int32(0) // a row with no id takes token 0's; see ple.go
					if rowIDs != nil {
						tok = max(rowIDs[base+i], 0)
					}
					if err := s.pleInputs(s.bple[i*w:(i+1)*w], tok, s.bx[i*c.NEmbd:(i+1)*c.NEmbd]); err != nil {
						return err
					}
				}
				return nil
			}
		}
		// AltUp's other streams, from each row's embedding (altup.go), and
		// DeepSeek V4's copies of it (ds4.go).
		if c.AltUp != 0 {
			fill := refill
			refill = func() error {
				if err := fill(); err != nil {
					return err
				}
				return s.altExpand(s.bx[:n*c.ResidW()], n)
			}
		}
		if c.DSV4() {
			fill := refill
			refill = func() error {
				if err := fill(); err != nil {
					return err
				}
				s.ds4Expand(s.bx[:n*c.ResidW()], n)
				return nil
			}
		}
		if c.ResAttn() {
			fill := refill
			refill = func() error {
				if err := fill(); err != nil {
					return err
				}
				s.k3Expand(s.bx[:n*c.ResidW()], n)
				return nil
			}
		}
		if c.Q4E() {
			fill := refill
			refill = func() error {
				if err := fill(); err != nil {
					return err
				}
				s.q4eExpand(s.bx[:n*c.ResidW()], n)
				return nil
			}
		}
		// A learned position table (starcoder) is added inside refill, so every
		// restart gets its positions back; serially, since one scratch row
		// serves the loop.
		if m.posEmbd.e != nil {
			fill := refill
			refill = func() error {
				if err := fill(); err != nil {
					return err
				}
				for i := 0; i < n; i++ {
					if err := m.addPos(s.bx[i*c.NEmbd:(i+1)*c.NEmbd], s.prow, p0+i); err != nil {
						return err
					}
				}
				return nil
			}
		}
		if err := refill(); err != nil {
			return nil, err
		}

		// One rotary table per row, per chunk, shared by the device and every
		// host layer, at the row's rotary position (mrope.go).
		for i := 0; i < n; i++ {
			if rp != nil {
				s.ropeTableRow(i, rp[base+i])
				continue
			}
			s.ropeRow(i, s.ropeOf(slot, p0+i))
		}
		// Device blocks go as runs, not a prefix (see forward.go), each run as
		// one batch so the weights are read once for n tokens.
		for li := s.lo; li < s.hi; li++ {
			if s.devAt(li) {
				hi := li + 1
				// A deepstack row adds in after each of the first blocks, so
				// a chunk holding one runs them one submission each.
				for hi < s.hi && s.devAt(hi) && !s.deepIn(hi-1, base, n) {
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
				// The chunk's token ids, which DeepSeek V4's hash blocks route
				// by (-1 or nil for a supplied embedding with none, which they
				// refuse).
				var cids []int32
				if rowIDs != nil {
					cids = rowIDs[base : base+n]
				}
				s.tokenIDs(cids)
				// Qwen3.8-Flash-Next's n-gram rows, when the run holds its
				// n-gram block; every failure below puts the predecessors back.
				s.q4.setCall(s, slot, cids)
				if err := s.ngramToDevice(li, hi, n, nil, s.q4.callOwner, s.q4.callTok); err != nil {
					return nil, err
				}
				if s.batched {
					pos, sl := make([]int, n), make([]int, n)
					for j := range pos {
						pos[j] = p0 + j
						sl[j] = slot*s.maxSeq + pos[j]
					}
					if err := s.rowsRun(li, hi, pos, sl, s.bx[:n*c.ResidW()], s.bcs[:n*c.RopeW()], s.bswaTable(n, c.NRotSWA),
						false, nil, 0); err != nil {
						return nil, err
					}
					s.addDeep(hi-1, base, n)
					li = hi - 1
					continue
				}
				var fh *nn.Head
				if li == s.lo && hi == s.hi && base+n == total && s.emb == nil && s.head != nil && !s.noHeadFold &&
					!c.streamHead() {
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
				// grain. The head is the tail call after the loop. A refusal
				// falls through to the chunk.
				if !s.m.opt.noTailRows && n >= 2 && n <= tailRowsMax && li == s.lo && hi == s.hi && !chunkBidir && s.rows == nil &&
					!s.recurrent() && s.rowsRefusal() == nil {
					pos := s.tailPos[:0]
					for j := 0; j < n; j++ {
						pos = append(pos, p0+j)
					}
					s.tailPos = pos
					// One sequence: slot is the position (nn.RowsDevice).
					if s.rowsRun(li, hi, pos, pos, s.bx[:n*c.ResidW()], s.bcs[:n*c.RopeW()],
						s.bswaTable(n, c.NRotSWA), false, nil, 0) == nil {
						s.tailRowChunks++
						s.addDeep(hi-1, base, n)
						li = hi - 1
						continue
					}
					// The step may have written part of the residual.
					if err := refill(); err != nil {
						return nil, err
					}
					s.ngramUndoRows()
					if err := s.ngramToDevice(li, hi, n, nil, s.q4.callOwner, s.q4.callTok); err != nil {
						return nil, err
					}
				}
				s.layerInputs(s.bple[:n*c.pleWidth()])
				if s.bidirTell(p0, n) && s.ld.Layers(li, hi, p0, n, s.bx[:n*c.ResidW()], s.bcs[:n*c.RopeW()],
					s.bswaTable(n, c.NRotSWA), fh) {
					headDone = fh != nil
					s.addDeep(hi-1, base, n)
					li = hi - 1
					continue
				}
				s.ngramUndoRows()
				// A hybrid's speculation step is rolled back as one pass, and a
				// recurrent block run row by row could not be: the device's
				// refusal is the step's error. Any other model's step takes the
				// retry below like a prompt chunk, and its head on the host
				// (finishRows).
				if s.rows != nil && s.recurrent() {
					why := "no reason given"
					if e, ok := s.ld.(nn.ErrReporter); ok && e.Err() != "" {
						why = e.Err()
					}
					return nil, fmt.Errorf("model: the device refused a %d-row speculation step on blocks [%d,%d): %s",
						n, li, hi, why)
				}
				// The per-row retry is only for a run at block 0: refill()
				// restarts from the embeddings, which is sound only while no
				// earlier block has been applied. At li > 0 we demote and
				// restart on the host. The retry is per token on the device,
				// since the device still owns those blocks' KV cache.
				// Not for a chunk holding a bidirectional run: one row at a time
				// is causal by construction. A row of several streams (AltUp's,
				// DeepSeek V4's) is gathered out of the chunk's stream-major
				// layout and put back (rowStreams).
				if li == s.lo && !chunkBidir {
					s.devRowChunks++
					// A hybrid's batch that advanced a linear block before it
					// failed cannot go round again from the embeddings: the
					// retry would apply that block's rule twice.
					if s.recurrent() && s.recStepped() != 0 {
						return nil, errRecurRetry{li, s.devWhy()}
					}
					// From the embeddings: the failed batch may have written
					// some rows already.
					if err := refill(); err != nil {
						return nil, err
					}
					s.ngramUndoRows()
					done := true
					for i := 0; i < n; i++ {
						row := s.bx[i*c.NEmbd : (i+1)*c.NEmbd]
						if c.Streams() > 1 {
							row = s.rowStreams(s.bx[:n*c.ResidW()], i, n, false)
						}
						s.layerInputs(s.bple[i*c.pleWidth() : (i+1)*c.pleWidth()])
						if cids != nil {
							s.tokenIDs(cids[i : i+1])
						}
						s.q4.setCall(s, slot, cids[min(i, len(cids)):min(i+1, len(cids))])
						if err := s.ngramToDevice(li, hi, 1, nil, s.q4.callOwner, s.q4.callTok); err != nil {
							return nil, err
						}
						if s.ld.Layers(li, hi, p0+i, 1, row,
							s.bcs[i*c.RopeW():(i+1)*c.RopeW()],
							s.bswaTableAt(i, c.NRotSWA), nil) {
							if c.Streams() > 1 {
								s.rowStreams(s.bx[:n*c.ResidW()], i, n, true)
							}
							continue
						}
						s.ngramUndoRows()
						if !s.devFallback {
							return nil, errDevice{0, s.devWhy()}
						}
						// The host restart re-runs rows 0..i from the
						// embeddings, which a hybrid's advanced rows forbid.
						if s.recurrent() && (i > 0 || s.recStepped() != 0) {
							return nil, errRecurRetry{li, s.devWhy()}
						}
						if !s.demoteAll() {
							return nil, errDevice{0, s.devWhy()}
						}
						// And again before the host restart: rows 0..i-1 have
						// already been through the device's blocks.
						if err := refill(); err != nil {
							return nil, err
						}
						done = false
						break
					}
					if done {
						s.addDeep(hi-1, base, n)
						li = hi - 1
						continue
					}
					li = s.lo - 1
					continue
				}
				if !s.devFallback {
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
			// DeepSeek V4's hash blocks route by the chunk's tokens (-1 for a
			// supplied embedding), which hostRows does not see.
			if c.DSV4() {
				s.growDS4(chunkRows)
				for i := 0; i < n; i++ {
					s.ds.rid[i] = -1
					if rowIDs != nil {
						s.ds.rid[i] = rowIDs[base+i]
					}
				}
			}
			// Qwen3.8-Flash-Next's n-gram block hashes the chunk's tokens
			// (-1 for a supplied embedding).
			if c.NGramAt(li) {
				s.q4eRowIDs(n, func(i int) int32 {
					if rowIDs != nil {
						return rowIDs[base+i]
					}
					return -1
				})
			}
			// A host block needs the host's rows, which a device-only prompt
			// did not allocate (growDeviceBatch); a demotion lands here.
			if err := s.hostRows(li, slot, base, p0, n, chunkRows, bidir); err != nil {
				return nil, err
			}
			s.addDeep(li, base, n)
		}
		if s.emb != nil {
			s.emb.collect(s, base, n)
		}
		if s.rowTap != nil {
			s.rowTap(p0, s.bx[:n*c.NEmbd])
		}
		s.advance(slot, n)
		// The last chunk's last row is the only one the caller can use, so it
		// becomes the decode path's residual stream and the session continues
		// from there.
		//
		// It is written for a batch row too: nothing decodes from it there, but
		// the output projection below reads s.x.
		copy(s.x, s.bx[(n-1)*c.NEmbd:n*c.NEmbd])
		// AltUp's head reads every stream of the row: collapsed in place, the
		// chunk's rows' stream 0 is their head input, and the last row's is
		// s.x's (altup.go).
		if c.AltUp != 0 && base+n == total {
			if err := s.altCollapse(s.bx[:n*c.ResidW()], n); err != nil {
				return nil, err
			}
			copy(s.x, s.bx[(n-1)*c.NEmbd:n*c.NEmbd])
		}
		if c.DSV4() && base+n == total {
			if err := s.ds4Collapse(s.bx[:n*c.ResidW()], n); err != nil {
				return nil, err
			}
			copy(s.x, s.bx[(n-1)*c.NEmbd:n*c.NEmbd])
		}
		if c.ResAttn() && base+n == total {
			if err := s.k3Collapse(s.bx[:n*c.ResidW()], n); err != nil {
				return nil, err
			}
			copy(s.x, s.bx[(n-1)*c.NEmbd:n*c.NEmbd])
		}
		if c.Q4E() && base+n == len(src) {
			if err := s.q4eCollapse(s.bx[:n*c.ResidW()], n); err != nil {
				return nil, err
			}
			copy(s.x, s.bx[(n-1)*c.NEmbd:n*c.NEmbd])
		}
		// The batch width is tuned on the whole chunk's time, not the matmul
		// rate inside it. See nn.JIT.ObserveChunk.
		s.jit.ObserveChunk(width, n, time.Since(chunkStart))
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
		// The device ran the norm, the projection and gemma2's final softcap
		// (nn.Head.Softcap) on the last row; only the scale is left.
		s.finishDevLogits(s.logits)
		if err := s.kvCheck(); err != nil {
			return nil, err
		}
		return s.logits, nil
	}
	if s.head != nil && s.devCount() > 0 {
		s.ropeFill(s.cs, s.ropeOf(slot, s.seqPos(slot)-1))
		if m.ropeSWA != nil {
			s.jit.RopeTable(*m.ropeSWA, s.csSWA, s.ropeOf(slot, s.seqPos(slot)-1))
		}
		if s.ld.Layers(s.gpuLayers, s.gpuLayers, s.seqPos(slot)-1, 1, s.resid(), s.cs[:c.RopeW()], s.swaTable(c.NRotSWA), s.head) {
			// The device ran the norm, the projection and gemma2's final
			// softcap (nn.Head.Softcap); only the scale is left.
			s.finishDevLogits(s.logits)
			// A page the store could not return fails the request. See kvFault.
			if err := s.kvCheck(); err != nil {
				return nil, err
			}
			return s.logits, nil
		}
		if !s.devFallback {
			return nil, errDevice{s.gpuLayers, s.devWhy()}
		}
	}
	s.norm(s.h, s.x, s.outNorm, m.outNormB)
	s.jit.NewInput()
	if s.pickOnHost() {
		if err := s.kvCheck(); err != nil {
			return nil, err
		}
		return s.logits, nil
	}
	if err := s.mv(s.logits, *s.outW, s.h); err != nil {
		return nil, err
	}
	s.finishLogits(s.logits)
	// A page the store could not return fails the request. See kvFault.
	if err := s.kvCheck(); err != nil {
		return nil, err
	}
	return s.logits, nil
}

// mm is the batched matvec seam, mirroring mv. A false from MatMul is a
// supported answer -- there is no GEMM for this quantization yet, or the batch
// is too small to pay -- and the loop below produces identical tokens.
func (s *State) mm(out []float32, w tensor, x []float32, ntok int) error {
	// The batched path is timed here; the fallback is timed by each s.mv, so a
	// defer over the whole call would count it twice.
	t0 := s.tick()
	// The GGUF GEMM only for a GGUF-layout weight, tested first: nn.MatMul
	// walks GGUF blocks and would read out of range on a packed payload.
	if w.packed == nil && s.jit.MatMul(out, w.typ, w.data, x, w.rows, w.k, ntok) {
		s.tock(opMatVec, t0)
		return nil
	}
	// The packed batch, one pool region for the whole chunk. It declines for a
	// shape it cannot serve, and the row loop below runs instead.
	if w.packed != nil && s.jit.MatMulPacked(out, w.typ, w.packed, x, w.rows, w.k, ntok) {
		s.tock(opMatVec, t0)
		return nil
	}
	for i := 0; i < ntok; i++ {
		s.jit.NewInput()
		if err := s.mv(out[i*w.rows:(i+1)*w.rows], w, x[i*w.k:(i+1)*w.k]); err != nil {
			return err
		}
	}
	return nil
}

func (s *State) batchNorm(dst, src, weight []float32, dim, ntok int) {
	s.batchNormB(dst, src, weight, nil, dim, ntok)
}

// batchNormB is batchNorm with the norm's bias, and it is State.norm per row:
// a LayerNorm on a classic block (C6), an RMSNorm everywhere else.
func (s *State) batchNormB(dst, src, weight, bias []float32, dim, ntok int) {
	s.normRows(dst, src, weight, bias, dim, ntok, s.c.LayerNorm, s.c.RMSEps, 1)
	s.jit.NewInput()
}

// normRows is State.norm over ntok rows of width dim on the pool, chunk rows
// at a time: a LayerNorm with its bias where ln is set, an RMSNorm otherwise,
// a copy where there is no weight.
func (s *State) normRows(dst, src, weight, bias []float32, dim, ntok int, ln bool, eps float64, chunk int) {
	j := &s.rg.rows
	j.op, j.dst, j.src, j.w, j.b, j.dim, j.ln, j.eps = rowNorm, dst, src, weight, bias, dim, ln, eps
	s.rowRun(ntok, chunk)
}

// axpyRows adds alpha*b to each of n rows of dst, b one row of width dim or,
// where idx is set, the table whose row idx[r] goes to row r.
func (s *State) axpyRows(dst, b []float32, idx []int32, dim, n, chunk int, alpha float32) {
	j := &s.rg.rows
	j.op, j.dst, j.b, j.idx, j.dim, j.alpha = rowAxpy, dst, b, idx, dim, alpha
	s.rowRun(n, chunk)
}

// growDeviceBatch sizes what a chunk needs when every block runs on a device:
// the residual rows it uploads and downloads and the rotary rows beside them.
//
// Nothing else: growBatch's full host set is tens of MB at the device's chunk
// width, and allocating it cost a fresh State measurable time per prompt.
// deviceChunk is the prompt chunk a device takes: MaxDevicePrefillChunk, or
// pipelineChunks of them when every block is on a device that crosses
// several cards, so the cards can work on different pieces at once
// (nn.Pipeliner). Only then: host blocks between device runs take the chunk
// whole.
func (s *State) deviceChunk() int {
	if p, ok := s.ld.(nn.Pipeliner); ok && s.allOnDevice() && p.PipelineDepth() > 1 {
		return nn.MaxDevicePrefillChunk * pipelineChunks
	}
	return nn.MaxDevicePrefillChunk
}

// pipelineChunks is how many device chunks a pipelined prompt chunk holds:
// llama.cpp's 2048-token batch over 512-token micro-batches. With d cards a
// chunk takes (pieces + d - 1) piece-times instead of d x pieces.
// Deeper pipelines (more cards) gain from more pieces.
const pipelineChunks = 4

func (s *State) growDeviceBatch(chunk int) {
	c := s.c
	if len(s.bx) >= chunk*c.ResidW() {
		return
	}
	b := s.m.takeSpare()
	if b != nil {
		s.spareHost = b.host // growBatch's, if a host block wants it
	}
	if b != nil && len(b.bx) >= chunk*c.ResidW() && len(b.bcs) >= chunk*c.RopeW() &&
		(s.m.ropeSWA == nil || len(b.bcsSWA) >= chunk*c.NRotSWA) {
		s.bx, s.bcs, s.bcsSWA = b.bx, b.bcs, b.bcsSWA
		return
	}
	s.bx = make([]float32, chunk*c.ResidW())
	s.bcs = make([]float32, chunk*c.RopeW())
	if s.m.ropeSWA != nil {
		s.bcsSWA = make([]float32, chunk*c.NRotSWA)
	}
}

// promptBuf is a State's prompt buffers -- a chunk's residual rows and its
// rotary tables -- kept by the model when the State closes. A request is a
// fresh State, and allocating these per request cost a 512-row prompt about
// a fifth of its time (a fresh 4 MiB heap span faults in page by page;
// gpu-kernels.md, "The prompt lost 22%").
//
// One spare serves serial requests; concurrent States on one model would
// need a free list.
type promptBuf struct {
	bx, bcs, bcsSWA []float32
	host            hostBatch
}

// takeSpare is the model's spare prompt buffers, or nil, emptying the slot.
func (m *Model) takeSpare() *promptBuf {
	m.spareMu.Lock()
	defer m.spareMu.Unlock()
	b := m.spare
	m.spare = nil
	return b
}

// keepSpare gives a closing State's prompt buffers to the model, keeping the
// larger of them and any it already holds.
func (s *State) keepSpare() {
	if s.bx == nil {
		return
	}
	m := s.m
	m.spareMu.Lock()
	defer m.spareMu.Unlock()
	if m.spare == nil || len(m.spare.bx) < len(s.bx) {
		m.spare = &promptBuf{s.bx, s.bcs, s.bcsSWA, s.takeHostBatch()}
	}
	s.bx, s.bcs, s.bcsSWA = nil, nil, nil
}

// allOnDevice reports whether every block of this State runs on a device.
func (s *State) allOnDevice() bool {
	for li := s.lo; li < s.hi; li++ {
		if !s.devAt(li) {
			return false
		}
	}
	return true
}

func (s *State) growBatch(chunk int) {
	c := s.c
	qDim := c.MaxQDim()
	s.growDeviceBatch(chunk)
	if len(s.bh) >= chunk*max(c.NEmbd, qDim) {
		return
	}
	// A previous State's set where it is big enough (hostBatch); scratch
	// allocates what is not.
	h := s.spareHost
	s.spareHost.bh = nil
	s.bh = scratch(h.bh, chunk*max(c.NEmbd, qDim), 0)
	s.bq = scratch(h.bq, chunk*qDim, 0)
	s.bxb = scratch(h.bxb, chunk*qDim, 0)
	s.bk = scratch(h.bk, chunk*c.MaxKVDim(), 0)
	s.bv = scratch(h.bv, chunk*c.MaxKVDim(), 0)
	// Padded, so the ragged tail of a width that is not a whole number of
	// vectors has somewhere legal to go and the generated activation serves
	// every width (see State.actmul).
	// An ungated vision MLP has no gate: every tower's but Qwen2.5-VL's and
	// Pixtral's.
	if s.vis == nil || s.vis.t.Cfg.Gated {
		s.bgate = scratch(h.bgate, chunk*c.MaxNFFN(), ffnPad(chunk*c.MaxNFFN()))
	}
	s.bup = scratch(h.bup, chunk*c.MaxNFFN(), ffnPad(chunk*c.MaxNFFN()))
	if c.Parallel || c.SSDAttn() {
		s.bhf = scratch(h.bhf, chunk*c.NEmbd, 0)
	}
	// One attention scores row per token, for the same kernels decode uses:
	// sized where it is used, by the keys the chunk reaches (rowsStride).
	s.battf = h.battf
	if c.AttnOutGate {
		s.qgate = scratch(h.qgate, chunk*2*qDim, 0)
		s.ogate = scratch(h.ogate, chunk*qDim, 0)
	}
	// A vision segment's rows attend through visAttend, which reads the
	// queries in place; the staging is the text attention's.
	if s.vis == nil {
		s.bqf = scratch(h.bqf, chunk*qDim, 0)
		s.bxbf = scratch(h.bxbf, chunk*qDim, 0)
	}
	s.growMLABatch(chunk)
	s.growMSARows(chunk)
}

type errEmptyPrompt struct{}

func (errEmptyPrompt) Error() string { return "model: Prefill called with no tokens" }

// ropeRow writes rotary position p's rows into row j of the batch tables:
// the global one, and the local one where the architecture trains two bases.
// p is the row's ROTARY position (ropeOf), not its cache position.
// ropeFill writes the plain rotary table at position p into cs: NRot floats,
// or under XD-RoPE the same table twice (a text row's halves turn alike).
func (s *State) ropeFill(cs []float32, p int) {
	n := s.c.NRot
	s.jit.RopeTable(s.m.rope, cs[:n], p)
	if s.c.RopeXD {
		copy(cs[n:2*n], cs[:n])
	}
}

func (s *State) ropeRow(j, p int) {
	nrot, nrotSWA := s.c.RopeW(), s.c.NRotSWA
	s.ropeFill(s.bcs[j*nrot:(j+1)*nrot], p)
	if s.m.ropeSWA != nil {
		s.jit.RopeTable(*s.m.ropeSWA, s.bcsSWA[j*nrotSWA:(j+1)*nrotSWA], p)
	}
}

// bswaTable and bswaTableAt are swaTable for the batched path: the prefill
// tables are per token, so the device gets the same n-row slice of the local
// one that it gets of the global one. nil where the architecture has one base.
func (s *State) bswaTable(n, nrot int) []float32 {
	if s.bcsSWA == nil {
		return nil
	}
	return s.bcsSWA[:n*nrot]
}

func (s *State) bswaTableAt(i, nrot int) []float32 {
	if s.bcsSWA == nil {
		return nil
	}
	return s.bcsSWA[i*nrot : i*nrot+nrot]
}

// hostRows runs block li on the host for n rows of one sequence -- slot's,
// positions p0.. -- reading and writing the residual in s.bx: the block body
// prefill runs for a chunk, and the one a vision segment runs for an image
// (vision.go), whose rows attend over their key runs instead of causally.
// base is the chunk's first row in the prompt and chunkRows the scratch the
// chunk is sized for.
func (s *State) hostRows(li, slot, base, p0, n, chunkRows int, bidir bool) error {
	m, c := s.m, s.c
	// c.AttnScale, not 1/sqrt(HeadDim): they differ under YaRN's magnitude
	// correction (DeepSeek-V2-Lite). The widths are per layer (HeadDimAt).
	scale := c.AttnScale
	act := c.Act
	s.growBatch(chunkRows)
	// Same fault as decode: a block is a page and visiting it brings
	// it in. One nil compare when the budget holds the model.
	if err := m.pageIn(li); err != nil {
		return err
	}
	l := &m.layers[li]
	// DeepSeek V4's block is its own (ds4.go): the chunk's rows are
	// successive positions of this slot, their tokens set by the caller.
	if c.DSV4() {
		s.growDS4(chunkRows)
		for i := 0; i < n; i++ {
			s.ds.rslot[i], s.ds.rpos[i] = slot, p0+i
		}
		return s.ds4Block(li, l, s.bx[:n*c.ResidW()], n)
	}
	if c.AltUp != 0 {
		s.growAltUpRows(chunkRows)
		if err := s.altPredict(l, s.bx[:n*c.ResidW()], s.baltPred, n); err != nil {
			return err
		}
	}

	tn := s.tick()
	// Kimi-K3's attention reads its mix over the bank (k3.go).
	attnIn := s.bx
	if c.Q4E() {
		// The n-gram embedding, then the attention's hyper-connection mix
		// (q4e.go).
		s.growQ4E(chunkRows)
		x := s.bx[:n*c.ResidW()]
		if c.NGramAt(li) {
			if err := s.ngramApply(li, l, x, n, nil, func(int) (*State, int) { return s, slot },
				func(i int) int32 { return s.q4.rid[i] }); err != nil {
				return err
			}
		}
		if err := s.hcMix(x, n, &l.q4e.attn); err != nil {
			return err
		}
		attnIn = s.q4.mix
	}
	if c.ResAttn() {
		var err error
		if attnIn, err = s.k3AttnIn(li, l, s.bx[:n*c.ResidW()], n); err != nil {
			return err
		}
	}
	s.batchNormB(s.bh, attnIn, l.attnNorm, l.attnNormB, c.NEmbd, n)
	if c.AltUp != 0 {
		for r := 0; r < n; r++ {
			if err := s.laurel(l, s.bh[r*c.NEmbd:(r+1)*c.NEmbd], s.baltLaur[r*c.NEmbd:(r+1)*c.NEmbd]); err != nil {
				return err
			}
		}
	}
	// A parallel block's FFN input, taken before attention adds into
	// the residual; see forward.go.
	if c.Parallel {
		if l.ffnNorm != nil {
			s.batchNormB(s.bhf, s.bx, l.ffnNorm, l.ffnNormB, c.NEmbd, n)
		} else {
			copy(s.bhf[:n*c.NEmbd], s.bh[:n*c.NEmbd])
		}
	}
	s.tock(opRMSNorm, tn)
	// The linear layer's recurrence is a per-row loop: each row's state
	// depends on the row before it, and a batched recurrence would be a
	// different algorithm (a chunked scan). Its projections are not
	// sequential and run as GEMMs over the chunk (linrows.go).
	// A block that attends as well (Falcon-H1) leaves its mixer's rows in bhf
	// and runs attention from the same bh.
	kind := c.LayerKind(li)
	if kind.Recurrent() {
		// The projections as GEMMs over the chunk around the row loop
		// (linrows.go); the rows overwrite bh, so the output projection
		// waits for the loop to finish.
		outRows := s.bh
		if kind.Attends() {
			outRows = s.bhf
		}
		batched := s.linearBatched()
		if batched {
			if err := s.linearPre(l, s.bh, n); err != nil {
				return err
			}
		}
		ta := s.tick()
		for r := 0; r < n; r++ {
			h := s.bh[r*c.NEmbd : (r+1)*c.NEmbd]
			out := outRows[r*c.NEmbd : (r+1)*c.NEmbd]
			s.lin.r = r
			s.jit.NewInput()
			if err := s.linearAttn(li, l, slot, h, out); err != nil {
				s.lin.on = false
				return err
			}
			s.rows.snapRow(s, li, slot, base+r)
		}
		s.tock(opAttn, ta)
		if batched {
			if err := s.linearPost(li, l, outRows, n); err != nil {
				return err
			}
		}
	}
	if !kind.Attends() {
	} else if c.MLA() {
		// One call replaces q, k, v and the attnPrep loop: it leaves
		// every row's absorbed query in s.bmlaAbs and every row's
		// latent in the cache.
		if err := s.mlaProjectRows(li, l, n, s.bcs, s.bcsSWA,
			func(i int) (int, int) { return slot, p0 + i }); err != nil {
			return err
		}
	} else {
		kvDim, qDim := c.KVDimAt(li), c.QDimAt(li)
		if _, err := s.projectQ(l, s.bq, s.ogate, s.clampIn(l, clampQ, s.bh[:n*c.NEmbd]), n, true); err != nil {
			return err
		}
		s.clampOut(l, clampQ, s.bq[:n*qDim])
		// A KV-sharing block: q alone, then its source's history.
		if c.KVShared(li) {
			for i := 0; i < n; i++ {
				s.attnPrep(l, s.bq[i*qDim:(i+1)*qDim], nil, nil,
					s.ropeTable(li, s.bcs, s.bcsSWA, i), li, slot, p0+i)
			}
			goto attend
		}
		if err := s.mm(s.bk, l.wk, s.clampIn(l, clampK, s.bh[:n*c.NEmbd]), n); err != nil {
			return err
		}
		s.addBiasRows(s.bk, l.bk, n)
		s.clampOut(l, clampK, s.bk[:n*kvDim])
		if l.vFromK {
			// v is k's projection (Gemma 4's global layers), copied
			// before attnPrep norms and rotates k.
			copy(s.bv[:n*kvDim], s.bk[:n*kvDim])
		} else if err := s.mm(s.bv, l.wv, s.clampIn(l, clampV, s.bh[:n*c.NEmbd]), n); err != nil {
			return err
		}
		s.addBiasRows(s.bv, l.bv, n)
		if !l.vFromK {
			s.clampOut(l, clampV, s.bv[:n*kvDim])
		}
		// MiniMax Sparse Attention's indexer key and query, every row's
		// (msa.go).
		msa := c.MSAAt(li)
		if msa {
			if err := s.msaProjectRows(l, n, s.bcs, s.bcsSWA, li); err != nil {
				return err
			}
		}
		qsa := c.QSAAt(li)
		if qsa {
			if err := s.qsaProjectRows(l, n, s.bcs, s.bcsSWA, li); err != nil {
				return err
			}
		}

		// RoPE and the cache write must both finish for the WHOLE chunk
		// before any attention runs: token i attends to earlier tokens of
		// this same chunk, whose K and V are only valid once written.
		if s.vis != nil {
			// A picture's rows, on the pool: the one page the block was lent
			// holds every row, so no row grows or faults it (visPrepRows).
			s.visPrep(li, n)
			goto attend
		}
		for i := 0; i < n; i++ {
			pos := p0 + i
			k, v := s.bk[i*kvDim:(i+1)*kvDim], s.bv[i*kvDim:(i+1)*kvDim]
			if msa {
				k, v = s.msaRowOf(li, i, k, v)
			}
			if qsa {
				k, v = s.qsaRowOf(li, i, k, v)
			}
			// This row's rotary table, built once above and exactly NRot
			// wide, shared by all heads of this token.
			s.attnPrep(l, s.bq[i*qDim:(i+1)*qDim], k, v,
				s.ropeTable(li, s.bcs, s.bcsSWA, i), li, slot, pos)
		}
	}
attend:
	if kind.Attends() {

		qDim, mla := c.QDimAt(li), c.MLA()
		if s.vis != nil {
			// A vision segment's rows attend over their key runs, the whole
			// image or a window, from the page this block wrote (vision.go).
			ta := s.tick()
			s.visAttend(li, n)
			s.tock(opAttn, ta)
		} else {
			// Attention, parallel over tokens. Each token owns its own scores
			// buffer, so heads can share it without any coordination, and the
			// causal bound p0+i is what keeps this equal to the decode path.
			ta := s.tick()
			bq, bxb, bqf, bxbf := s.bq, s.bxb, s.bqf, s.bxbf
			// A row reaches keys below p0+n, or to the end of the
			// bidirectional run it is in.
			reach := p0 + n
			for i := 0; i < n; i++ {
				reach = max(reach, s.bidirEnd(p0+i))
			}
			astride := s.rowsStride(reach)
			s.battf = scratch(s.battf, n*astride, 0)
			attf, fast := s.battf, s.attnAt(li)
			hd, gqa := c.HeadDimAt(li), c.GQAAt(li)
			// MLA reads a wider query than it writes an output, and every
			// head reads the same cached row: qw is the whole row, ow its
			// latent prefix, and gqa becomes NHead so hh/gqa is 0. The
			// absorbed query is read directly, without the bqf staging.
			qw, ow := hd, hd
			qsrc, odst := bqf, bxbf
			gqaN := gqa
			if mla {
				qw, ow = c.KVLoraRank+c.NRot, c.KVLoraRank
				qsrc, odst, gqaN = s.bmlaAbs, s.bmlaAcc, c.NHead
			}
			// Residency before the fan-out; see kvEnsureWindow. [0, p0+n) is
			// a superset of every row's window, which is always safe.
			kvli := c.KVSource(li)
			s.kvEnsureWindow(kvli, 0, p0+n)
			masks, mstride := s.idxMasks, 0
			if c.Indexer() {
				s.idxRows(kvli, n, func(int) int { return slot }, func(i int) int { return p0 + i + 1 }, masks)
			}
			if c.MSAAt(li) {
				// The masks are laid out at the State's own stride (msa.go).
				masks, mstride = s.msaMasks, s.attStride
				s.msaRows(kvli, n, func(int) int { return slot }, func(i int) int { return p0 + i + 1 }, masks)
			}
			if c.QSAAt(li) {
				masks, mstride = s.qsa().masks, 0
				s.qsaRows(kvli, l, n, func(int) (*State, int) { return s, slot }, func(i int) int { return p0 + i + 1 }, masks)
			}
			s.rg.pattn = prefAttnRegion{l: l, li: li, kvli: kvli, slot: slot, p0: p0, n: n,
				qDim: qDim, qw: qw, ow: ow, hd: hd, gqa: gqaN, astride: astride, mstride: mstride,
				mla: mla, bidir: bidir, scale: scale, bq: bq, bxb: bxb, qsrc: qsrc, odst: odst,
				attf: attf, masks: masks, fast: fast, fn: s.rg.pattn.fn}
			if s.rg.pattn.fn == nil {
				s.rg.pattn.fn = s.prefAttnRows
			}
			s.jit.Parallel(n, 1, s.rg.pattn.fn)
			s.rg.pattn = prefAttnRegion{fn: s.rg.pattn.fn}

			s.tock(opAttn, ta)
		}

		// The output gate goes before the projection; see forward.go.
		if mla {
			// W_v un-absorbs out of the latent and wo follows, both
			// inside mlaOutRows, with Kimi-K3's output gate between them.
			if err := s.mlaOutRows(l, n); err != nil {
				return err
			}
		} else {
			asrc := s.bxb
			if c.AttnOutGate {
				s.applyOutGate(s.ogate, s.bxb, n)
				s.jit.NewInput()
				asrc = s.ogate
			}
			// BitNet's norm over the attention output, before wo.
			if l.attnSubNorm != nil {
				tn := s.tick()
				s.batchNorm(asrc, asrc, l.attnSubNorm, qDim, n)
				s.jit.NewInput()
				s.tock(opRMSNorm, tn)
			}
			if err := s.mm(s.bh, l.wo, s.clampIn(l, clampO, asrc[:n*qDim]), n); err != nil {
				return err
			}
			s.clampOut(l, clampO, s.bh[:n*c.NEmbd])
		}
		// Before the residual, for every architecture.
		s.addBiasRows(s.bh, l.bo, n)
		if kind.Recurrent() {
			s.addInto(s.bh[:n*c.NEmbd], s.bhf[:n*c.NEmbd])
		}
	}
	// gemma2/gemma3 normalise the attention output before the residual,
	// as forward.go does; TestPrefillMatchesForward catches a miss.
	if l.postAttnNorm != nil {
		tn := s.tick()
		s.batchNorm(s.bh, s.bh, l.postAttnNorm, c.NEmbd, n)
		s.tock(opRMSNorm, tn)
	}
	tr := s.tick()
	if c.Q4E() {
		s.hcPlace(s.bx[:n*c.ResidW()], n, s.bh)
	} else if c.ResAttn() {
		s.k3Resid(li, s.bx, s.bh, n)
	} else {
		s.addInto(s.bx[:n*c.NEmbd], s.bh[:n*c.NEmbd])
	}
	if c.AltUp != 0 {
		s.laurelJoin(s.bx[:n*c.NEmbd], s.baltLaur[:n*c.NEmbd])
	}
	s.tock(opResid, tr)
	// A block with no FFN ends at the first residual; see forward.go.
	if l.noFFN {
		return nil
	}

	tn = s.tick()
	ffnIn := s.bh
	if c.Parallel {
		ffnIn = s.bhf
		s.jit.NewInput()
	} else if c.ResAttn() {
		in, err := s.k3FFNIn(li, l, s.bx[:n*c.ResidW()], n)
		if err != nil {
			return err
		}
		s.batchNormB(s.bh, in, l.ffnNorm, l.ffnNormB, c.NEmbd, n)
	} else if c.Q4E() {
		if err := s.hcMix(s.bx[:n*c.ResidW()], n, &l.q4e.ffn); err != nil {
			return err
		}
		copy(s.bh[:n*c.NEmbd], s.q4.mix[:n*c.NEmbd])
	} else {
		s.batchNormB(s.bh, s.bx, l.ffnNorm, l.ffnNormB, c.NEmbd, n)
	}
	s.tock(opRMSNorm, tn)
	// A mixture visits the bank expert-major over the chunk; see
	// moeBatch. Gemma 4's block runs row by row; see denseMoERows.
	if l.router.data != nil && c.DenseMoE {
		if err := s.denseMoERows(li, l, ffnIn, s.bx, n); err != nil {
			return err
		}
	} else if l.router.data != nil && c.ExpertLatent != 0 {
		if err := s.k3MoEBatch(li, l, ffnIn, s.bx, n); err != nil {
			return err
		}
	} else if l.router.data != nil && c.Q4E() {
		out := s.q4.out[:n*c.NEmbd]
		clear(out)
		if err := s.moeBatch(li, l, ffnIn, out, n); err != nil {
			return err
		}
		s.hcPlace(s.bx[:n*c.ResidW()], n, out)
	} else if l.router.data != nil {
		if err := s.moeBatch(li, l, ffnIn, s.bx, n); err != nil {
			return err
		}
	} else {
		if l.gate.e == nil {
			// The ungated FFN (C6); see forward.go.
			if err := s.mm(s.bup, l.up, ffnIn, n); err != nil {
				return err
			}
			s.addBiasRows(s.bup, l.upB, n)
			tac := s.tick()
			s.ungatedAct(l, s.bup[:n*c.NFFN])
			s.jit.NewInput()
			s.tock(opAct, tac)
			if err := s.mm(s.bh, l.down, s.bup, n); err != nil {
				return err
			}
			s.addBiasRows(s.bh, l.downB, n)
		} else {
			if err := s.mm(s.bgate, l.gate, s.clampIn(l, clampGate, ffnIn[:n*c.NEmbd]), n); err != nil {
				return err
			}
			s.clampOut(l, clampGate, s.bgate[:n*c.NFFNAt(li)])
			if err := s.mm(s.bup, l.up, s.clampIn(l, clampUp, ffnIn[:n*c.NEmbd]), n); err != nil {
				return err
			}
			s.clampOut(l, clampUp, s.bup[:n*c.NFFNAt(li)])
			// A gated MLP with biases (Qwen2.5-VL's vision blocks); nil and
			// skipped on every text block.
			s.addBiasRows(s.bgate, l.gateB, n)
			s.addBiasRows(s.bup, l.upB, n)
			gate, up := s.bgate[:n*c.NFFNAt(li)], s.bup[:n*c.NFFNAt(li)]
			tac := s.tick()
			s.gaussTopK(li, gate, c.NFFNAt(li))
			s.actmulAll(gate, up, act)
			s.tock(opAct, tac)
			// BitNet's norm over the product, before down.
			if l.ffnSubNorm != nil {
				tn := s.tick()
				s.batchNorm(gate, gate, l.ffnSubNorm, c.NFFNAt(li), n)
				s.jit.NewInput()
				s.tock(opRMSNorm, tn)
			}
			if err := s.mm(s.bh, l.down, s.clampIn(l, clampDown, gate), n); err != nil {
				return err
			}
			s.clampOut(l, clampDown, s.bh[:n*c.NEmbd])
			s.addBiasRows(s.bh, l.downB, n)
		}
		if l.postFFNNorm != nil {
			tn := s.tick()
			s.batchNorm(s.bh, s.bh, l.postFFNNorm, c.NEmbd, n)
			s.tock(opRMSNorm, tn)
		}
		tr = s.tick()
		s.addInto(s.bx[:n*c.NEmbd], s.bh[:n*c.NEmbd])
		s.tock(opResid, tr)
	}
	if c.AltUp != 0 {
		if err := s.altCorrect(li, l, s.bx[:n*c.ResidW()], s.baltPred, s.bple, n); err != nil {
			return err
		}
	} else if err := s.pleRows(li, l, s.bx, n); err != nil {
		return err
	}
	s.layerOutScale(l, s.bx[:n*c.NEmbd])
	return nil
}

// pickOnHost is the label pick on the host (State.HostPick): the projection
// at the picked rows alone, each a one-row view of the packed head through
// the generated matvec, where the whole vocabulary was projected for a
// handful of ids -- on Lev, 248320 rows of Q8_0 read for nine. It reads the
// normed row in s.h and leaves s.logits as it was, so a caller that picked
// reads HostPickVals. False, with nothing done, where the pick does not
// apply: no pick asked, a head the generated matvec does not read packed, or
// a model that adjusts its logits after the head (a bias, a final softcap or a
// logit scale), which the picked values would have to take too.
func (s *State) pickOnHost() bool {
	s.HostPicked = false
	w := s.outW
	if len(s.HostPick) == 0 || len(s.HostPickVals) < len(s.HostPick) || w.packed == nil || s.m.outB != nil ||
		s.c.FinalSoftcap != 0 || s.c.LogitScale != 1 {
		return false
	}
	// A narrow format packs DSlots rows' scales in a word, so a view starts
	// on a word: the slot's rows are projected and the one asked kept.
	sl := 1
	if q, ok := kernels.QuantOf(w.typ); ok {
		sl = max(kernels.DSlots(q), 1)
	}
	tmp := &s.pickTmp
	if sl > len(tmp) {
		return false
	}
	for j, id := range s.HostPick {
		lo := int(id) - int(id)%sl
		if id < 0 || lo+sl > w.rows {
			return false
		}
		v := &s.pickView
		*v = *w.packed
		v.Stride, v.Row = max(v.Stride, w.rows), v.Row+lo
		if !s.jit.MatVecPacked(tmp[:sl], w.typ, v, s.h, sl, w.k) {
			return false
		}
		s.HostPickVals[j] = tmp[int(id)-lo]
	}
	s.HostPicked = true
	return true
}
