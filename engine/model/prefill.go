package model

import (
	"errors"
	"fmt"
	"time"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/nn"
)

// MaxPrefillChunk sizes the batched scratch. The width used is tuned per
// machine (nn.JIT.PrefillChunk), so buffers fit the widest candidate.
const MaxPrefillChunk = nn.MaxHostPrefillChunk

// chunkWidth is how many rows one prompt chunk runs: the host's chunk, or the
// device's where the session places blocks.
func (s *Instance) chunkWidth() int {
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
func (s *Instance) Prefill(tokens []int32) ([]float32, error) {
	defer s.m.enterPager()()
	if err := s.retireErr; err != nil {
		s.retireErr = nil
		return nil, err
	}
	return s.prefillInto(0, tokens)
}

// prefillBidir is Prefill over a prompt whose rows all attend to each other
// (a non-causal decoder's, vllm-d3's): PrefillMixed's one bidirectional run,
// through the token path, so a warm prompt allocates nothing (the mixed
// path's per-row sources do).
func (s *Instance) prefillBidir(tokens []int32) ([]float32, error) {
	defer s.m.enterPager()()
	if err := s.retireErr; err != nil {
		s.retireErr = nil
		return nil, err
	}
	n := len(tokens)
	if n > nn.MaxDevicePrefillChunk {
		return nil, fmt.Errorf("model: a bidirectional prompt of %d rows is longer than one %d-row chunk",
			n, nn.MaxDevicePrefillChunk)
	}
	p := s.seqPos(0)
	s.bidir = append(s.bidir[:0], nn.KeyRun{Lo: p, Hi: p + n})
	defer func() {
		if kd, ok := s.ld.(nn.KeyRunDevice); ok {
			kd.SetKeyRuns(nil, nil)
		}
		s.bidir = s.bidir[:0]
	}()
	return s.prefillInto(0, tokens)
}

// Span is one run of a mixed prompt: token ids, a picture, or precomputed
// embeddings -- one of them. Embd is n*NEmbd floats laid out row-major, and is
// the residual stream rather than a vocabulary row -- EmbdScale is not applied
// to it, for the reason ForwardEmbd's comment gives. A Picture is rows the
// instance's vision segment encodes when the span is prefilled (picture.go), and
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
func (s *Instance) PrefillMixed(spans ...Span) ([]float32, error) {
	return s.prefillMixedAt(0, spans...)
}

// prefillMixedAt is PrefillMixed into one sequence slot.
func (s *Instance) prefillMixedAt(slot int, spans ...Span) ([]float32, error) {
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
	return s.prefillSrc(slot, src, nil)
}

// bidirRuns records the bidirectional runs of a mixed prompt, at the positions
// they will occupy. Single sequence only: a batch slot's rows go through the
// device as a causal prefill (nn.RowsDevice), which has no run to widen.
func (s *Instance) bidirRuns(slot int, spans []Span) error {
	c := s.c
	p := s.seqPos(slot)
	if s.bidirAll {
		// The whole prompt is one run (a non-causal decoder's, its pictures'
		// rows among the text's).
		n := 0
		for _, sp := range spans {
			n += sp.n(c.NEmbd)
		}
		if n > nn.MaxDevicePrefillChunk {
			return fmt.Errorf("model: PrefillMixed: a bidirectional prompt of %d rows is longer "+
				"than one %d-row chunk", n, nn.MaxDevicePrefillChunk)
		}
		s.bidir = append(s.bidir, nn.KeyRun{Lo: p, Hi: p + n})
		return nil
	}
	for _, sp := range spans {
		n := sp.n(c.NEmbd)
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
// embeddings, ple.go; DeepSeek V4's hash blocks, ds4.go), and returns its
// reset. A no-op elsewhere.
func (s *Instance) tokensOf(spans []Span) func() {
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
func (s *Instance) spanSrc(spans []Span) ([]func(dst []float32) error, error) {
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

// embedInto is embed() writing somewhere other than s.x. Prompt rows are
// filled from the pool's workers (embedRun), so it reads no instance scratch.
func (s *Instance) embedInto(dst []float32, token int32) error {
	m, c := s.m, s.c
	if err := m.embedRow(dst, int(token)); err != nil {
		return err
	}
	if c.EmbdScale != 1 {
		s.scale(dst, float32(c.EmbdScale))
	}
	return nil
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
	return s.prefillSrc(slot, nil, tokens)
}

// ErrPrefillInterrupted is a prefill stopped between chunks because its
// interrupt (SetPrefillInterrupt) asked. The sequence holds the chunks that
// ran and nothing after them; the caller resets it before reusing the Instance.
var ErrPrefillInterrupted = errors.New("model: prefill interrupted")

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
func (s *Instance) prefillSrc(slot int, src []func(dst []float32) error, ids []int32) ([]float32, error) {
	if err := s.marksHeld(); err != nil {
		return nil, err
	}
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
	if (total == 1 || total <= s.forwardRowsFor() && src == nil) && s.emb == nil && len(s.deep) == 0 {
		var lg []float32
		var err error
		s.forwardPrompts++
		for i := range total {
			s.tok = -1
			if rowIDs != nil {
				s.tok = rowIDs[i]
			}
			if lg, err = s.forward(func() error {
				if src == nil {
					// A token's row, as a decode step embeds it.
					return s.embed(ids[i])
				}
				if err := s.fillRow(src, ids, i, s.x); err != nil {
					return err
				}
				s.m.trace(-1, "embd", s.x)
				if c.PLEDim != 0 {
					tok := int32(0) // a row with no id takes token 0's; see ple.go
					if rowIDs != nil {
						tok = max(rowIDs[i], 0)
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
	s.recPrefill = true
	defer func() { s.recPrefill = false }()
	// Every device call from here on is the prompt's, its last row's head
	// too (nn.PromptDevice).
	s.promptCalls(true)
	defer s.promptCalls(false)
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
	// tokens each and a wider pool measured slower (engine/nn/prefillcores.go).
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
		// The device gets the full width, not the host tuner's. A chunk's
		// kernels are built for that width, and what it costs follows the
		// rows it carries, not the width: measured, d1-3B every block placed,
		// 64 rows 13.3 ms and 512 57.4 ms on the V100 (CUDA), 116 ms and
		// 834 ms on the M4 (Metal).
		if s.devCount() > 0 && !bidir && !spanAll {
			width = s.deviceChunk()
			// One submission's width when a run is in the prompt: a pipelined
			// chunk is cut into pieces, and a piece must not cut a run.
			if len(s.bidir) > 0 {
				width = min(width, nn.MaxDevicePrefillChunk)
			}
		}
		n := min(width, total-base)
		// A block an earlier chunk moved (relocateFor) can widen the
		// device's chunk past what the prompt was sized for: a second card
		// makes it a pipeline (deviceChunk).
		if n > chunkRows {
			chunkRows = n
			sizeFor(n)
		}
		// A bidirectional run is never cut (bidir.go).
		if len(s.bidir) > 0 && !s.bidirCutFault {
			n = min(s.bidirChunk(s.seqPos(slot), n), total-base)
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
		// Over the pool: in the packed layout one row's words are a
		// vocabulary's width apart (see Model.embedRow), so each row is many
		// cache misses, and rows are independent.
		gathered, composedRows := false, false
		refill := func() error {
			// On the card when the block run starts there and the device
			// gathers the step's input rows (nn.Head.GatherRow), or its tier
			// gathers the tied embedding: far faster than the host (EmbedRows).
			// Its input rows are the front end's, a position table's included.
			gathered, composedRows = false, false
			if ids != nil && s.head != nil && s.devAt(s.lo) &&
				(s.head.GatherRow || s.c.TiedEmbd && c.PLEDim == 0 && m.posEmbd.e == nil) {
				if s.ldEmbed != nil && s.ldEmbed.EmbedRows(ids[base:base+n], p0, s.bx[:n*c.NEmbd]) {
					gathered = true
					return nil
				}
			}
			// A chunk's token rows through the composed input rows, from this
			// goroutine: the row, its scale, its position row and its
			// per-layer inputs, which the wrappers below then leave be.
			if src == nil {
				var bple []float32
				if c.PLEDim != 0 {
					s.growBPLE(n)
					bple = s.bple[:n*c.pleWidth()]
				}
				at := s.cpu.embed.positions(n)
				for i := range at {
					at[i] = p0 + i
				}
				if s.cpuEmbed(s.bx[:n*c.NEmbd], bple, ids[base:base+n], at) == composer.DeclineServes {
					composedRows = true
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
				if err := fill(); err != nil || composedRows {
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
		if c.PredStreams != 0 {
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
				s.q4eExpandAt(s.bx[:n*c.ResidW()], n, base)
				return nil
			}
		}
		// A learned position table (starcoder) is added inside refill, so every
		// restart gets its positions back; serially, since one scratch row
		// serves the loop.
		if m.posEmbd.e != nil {
			fill := refill
			refill = func() error {
				if err := fill(); err != nil || gathered || composedRows {
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

		// The chunk's rotary tables are the CPU's to make where a block runs
		// there (cpuRowTables); a device makes its own.
		s.cpu.tabsFresh = false
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
				var fh *nn.Head
				// A tap reads every chunk's residual, which a folded head keeps
				// on the device.
				if li == s.lo && hi == s.hi && base+n == total && s.emb == nil && s.head != nil && !s.noHeadFold && s.rowTap == nil &&
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
				// grain. The head is the tail call after the loop. A decline is
				// the prompt's error (refusal); a fault falls through to the
				// chunk, whose call takes the fault's path.
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
					if err := s.refusal(li); err != nil {
						return nil, err
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
				if s.callTell(p0, n) && s.ld.Layers(li, hi, p0, n, s.bx[:n*c.ResidW()], s.bcs[:n*c.RopeW()],
					s.bswaTable(n, c.NRotSWA), fh) {
					headDone = fh != nil
					s.addDeep(hi-1, base, n)
					li = hi - 1
					continue
				}
				if err := s.refusal(li); err != nil {
					return nil, err
				}
				s.ngramUndoRows()
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
			// A block routing by token reads the chunk's (-1 for a supplied
			// embedding).
			if c.NHashLayers > 0 {
				s.cpuTokens(n, func(i int) int32 {
					if rowIDs != nil {
						return rowIDs[base+i]
					}
					return -1
				})
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
			s.rowTap(p0, s.bx[:n*c.ResidW()])
		}
		s.recChunkPoint(slot, p0+n)
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
		if c.PredStreams != 0 && base+n == total {
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
		if c.Q4E() && base+n == total {
			if s.rows != nil {
				s.rows.resBase = base
			}
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
		if err := s.refusal(s.gpuLayers); err != nil {
			return nil, err
		}
		if !s.fallsBack() {
			return nil, errDevice{s.gpuLayers, s.devWhy()}
		}
	}
	why := composer.DeclineServes
	if len(s.HostPick) > 0 {
		why = s.cpuPick(s.x)
	} else {
		why = s.cpuHead(s.x, s.logits, nil)
	}
	if why != composer.DeclineServes {
		return nil, cpuHeadRefused(why)
	}
	// A page the store could not return fails the request. See kvFault.
	if err := s.kvCheck(); err != nil {
		return nil, err
	}
	return s.logits, nil
}

// mm is the batched matvec seam, mirroring mv. A false from MatMul is a
// supported answer -- there is no GEMM for this quantization yet, or the batch
// is too small to pay -- and the loop below produces identical tokens.
func (s *Instance) mm(out []float32, w tensor, x []float32, ntok int) error {
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

// batchNormB is batchNorm with the norm's bias, and it is Instance.norm per row:
// a LayerNorm on a classic block (C6), an RMSNorm everywhere else.
func (s *Instance) batchNormB(dst, src, weight, bias []float32, dim, ntok int) {
	s.normRows(dst, src, weight, bias, dim, ntok, s.c.LayerNorm, s.c.RMSEps, 1)
	s.jit.NewInput()
}

// normRows is Instance.norm over ntok rows of width dim on the pool, chunk rows
// at a time: a LayerNorm with its bias where ln is set, an RMSNorm otherwise,
// a copy where there is no weight.
func (s *Instance) normRows(dst, src, weight, bias []float32, dim, ntok int, ln bool, eps float64, chunk int) {
	j := &s.rg.rows
	j.op, j.dst, j.src, j.w, j.b, j.dim, j.ln, j.eps = rowNorm, dst, src, weight, bias, dim, ln, eps
	s.rowRun(ntok, chunk)
}

// axpyRows adds alpha*b to each of n rows of dst, b one row of width dim or,
// where idx is set, the table whose row idx[r] goes to row r.
func (s *Instance) axpyRows(dst, b []float32, idx []int32, dim, n, chunk int, alpha float32) {
	j := &s.rg.rows
	j.op, j.dst, j.b, j.idx, j.dim, j.alpha = rowAxpy, dst, b, idx, dim, alpha
	s.rowRun(n, chunk)
}

// growDeviceBatch sizes what a chunk needs when every block runs on a device:
// the residual rows it uploads and downloads and the rotary rows beside them.
//
// Nothing else: growBatch's full host set is tens of MB at the device's chunk
// width, and allocating it cost a fresh instance measurable time per prompt.
// deviceChunk is the prompt chunk a device takes: MaxDevicePrefillChunk, or
// pipelineChunks of them when every block is on a device that crosses
// several cards, so the cards can work on different pieces at once
// (nn.Pipeliner). Only then: host blocks between device runs take the chunk
// whole.
func (s *Instance) deviceChunk() int {
	if l, ok := s.ld.(nn.ChunkLimiter); ok && l.ChunkRows() > 0 {
		return l.ChunkRows()
	}
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

func (s *Instance) growDeviceBatch(chunk int) {
	c := s.c
	// Every buffer the rows read, the second rotary's table too: a set
	// whose residual rows suffice may hold a shorter table, or none.
	if len(s.bx) >= chunk*c.ResidW() && len(s.bcs) >= chunk*c.RopeW() &&
		(s.m.ropeSWA == nil || len(s.bcsSWA) >= chunk*c.NRotSWA) {
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
func (m *Model) takeSpare() *promptBuf {
	m.spareMu.Lock()
	defer m.spareMu.Unlock()
	b := m.spare
	m.spare = nil
	return b
}

// keepSpare gives a closing session's prompt buffers to the model, keeping the
// larger of them and any it already holds.
func (s *Instance) keepSpare() {
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
	s.bq = scratch(h.bq, chunk*qDim, 0)
	s.bxb = scratch(h.bxb, chunk*qDim, 0)
	s.bk = scratch(h.bk, chunk*c.MaxKVDim(), 0)
	s.bv = scratch(h.bv, chunk*c.MaxKVDim(), 0)
	// Padded, so the ragged tail of a width that is not a whole number of
	// vectors has somewhere legal to go and the generated activation serves
	// every width (see Instance.actmul).
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

// bswaTable and bswaTableAt are swaTable for the batched path: the prefill
// tables are per token, so the device gets the same n-row slice of the local
// one that it gets of the global one. nil where the architecture has one base.
func (s *Instance) bswaTable(n, nrot int) []float32 {
	if s.bcsSWA == nil {
		return nil
	}
	return s.bcsSWA[:n*nrot]
}

// hostRows runs block li on the host for n rows of one sequence -- slot's,
// positions p0.. -- reading and writing the residual in s.bx: the block body
// prefill runs for a chunk, and the one a vision segment runs for an image
// (vision.go), whose rows attend over their key runs instead of causally.
// base is the chunk's first row in the prompt and chunkRows the scratch the
// chunk is sized for.
func (s *Instance) hostRows(li, slot, base, p0, n, chunkRows int, bidir bool) error {
	m, c := s.m, s.c
	s.growBatch(chunkRows)
	// Same fault as decode: a block is a page and visiting it brings
	// it in. One nil compare when the budget holds the model.
	if err := m.pageIn(li); err != nil {
		return err
	}
	l := &m.layers[li]
	why := composer.DeclineBidir
	if s.vis != nil {
		var err error
		if why, err = s.cpuTowerRows(li, l, n); err != nil {
			return err
		}
	} else if !bidir || !c.LayerKind(li).Recurrent() {
		// A non-causal decoder's chunk (an embedding run's) is one run
		// of rows attending both ways; a symmetric window its graph
		// states clips it per row (graph.Attrs.Bidir).
		if bidir {
			s.bidir = append(s.bidir[:0], nn.KeyRun{Lo: p0, Hi: p0 + n})
		}
		seq, pos := s.cpuRowsOf(slot, p0, n)
		var err error
		if c.NGramAt(li) {
			s.q4.setCall(s, slot, s.q4.rid[:n])
			if err := s.ngramForCPU(li, n, nil, s.q4.callOwner, s.q4.callTok); err != nil {
				return err
			}
		}
		if c.LayerKind(li).Recurrent() {
			why, err = s.cpuRecRows(li, l, s.bx[:n*c.ResidW()], s.bcs, seq, pos, nil, base, p0)
		} else {
			why, err = s.cpuRows(li, l, s.bx[:n*c.ResidW()], s.bcs, s.bcsSWA, seq, pos)
		}
		if bidir {
			s.bidir = s.bidir[:0]
		}
		if err != nil {
			return err
		}
	}
	if why == composer.DeclineServes {
		s.cpu.served++
		return nil
	}
	return s.cpuRefused(li, why)
}

// forwardRowsFor is the longest prompt run through the decode path: the
// caller's WithPrefillForwardRows, else hostRowsByRow once a device has run
// this model's experts on the host, else one.
func (s *Instance) forwardRowsFor() int {
	switch {
	case s.m.opt.forwardRowsSet:
		return s.m.opt.forwardRows
	case s.m.hostExperts.Load():
		return hostRowsByRow
	}
	return 1
}
