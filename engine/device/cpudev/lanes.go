package cpudev

import (
	"fmt"
	"slices"
	"unsafe"

	"github.com/jitllm/jitllm/nn"
)

// Lanes is a session's run of blocks on the CPU device (Layers), as a card's
// view runs a session's blocks there: each block read in through its
// source, its history readied and committed through the session's History,
// its recurrent state the session's View's, the run one fused pool region.
// The token and the per-layer inputs are handed before the call (SetToken,
// SetLayerInputs), as a card's per-layer inputs are.
type Lanes struct {
	e      *Exec
	b      *Bindings
	src    Source
	h      History
	v      *View
	seqs   int
	stride int
	side   []float32
	// cut is the cut rows the next call's block cut in two reads (SetCut).
	cut  []float32
	tok  int32
	toks []int32
	// req is the request words the next calls' blocks read (SetRequest).
	req []uint32
	// seqBuf, zeros, order, last and oruns are a rows call's scratch: the
	// rows' sequences, a session's rows' (every one its session's 0), a
	// linear block's rows in causal order, each sequence's last position,
	// and a step across sessions' runs of one session's rows.
	seqBuf, zeros, order, last []int
	oruns                      []ownRun
	own                        []*SessionLanes
	posBuf                     []int
	// slot, base and bidir are the prompt chunk the next call runs
	// (SetPromptRows); prompt is the session's say over a linear block's
	// summary as a prompt's rows run (attach).
	slot, base int
	bidir      bool
	chunkRun   [1]nn.KeyRun
	prompt     Prompt
	// keep is the attached session's copies of its recurrent state
	// (recKeep).
	keep *recKeep
	// ChunkFault starts each prompt chunk after the first from a cleared
	// summary.
	ChunkFault bool
	// width and streams are a row's width and its residual streams, convLen
	// a sequence's recurrent window.
	width, streams, convLen int
	// runs is the prompt's bidirectional runs, windowed that a row reads
	// inside its run only, wins each position's own window (KeyRuns).
	runs     []nn.KeyRun
	windowed bool
	wins     []nn.KeyRun
	// spans and in are a call of rows' spans and record.
	spans []uint32
	in    Rows
	// RecFault binds every linear block to the recurrent state of the first:
	// a gate's violation.
	RecFault bool
	// trace is told each block the run finished.
	trace func(li int)
	why   nn.Decline
	at    int
	err   error
}

// NewLanes runs e's blocks for a session, their weights bound by b and read
// in through src, a row's residual streams stride apart, their history seqs
// sequences wide; trace, where set, is told each block a run finishes.
func NewLanes(e *Exec, b *Bindings, src Source, stride, seqs, streams, convLen int, trace func(li int)) *Lanes {
	return &Lanes{e: e, b: b, src: src, stride: stride, seqs: seqs, width: stride, streams: streams, convLen: convLen,
		trace: trace}
}

// KeyRuns is the prompt's bidirectional runs the next calls of rows read,
// windowed that a row reads inside its run only, and each position's own
// window where the rows' windows overlap (nil for none), as a card is
// handed them (nn.KeyRunDevice, nn.RowWindowDevice).
func (l *Lanes) KeyRuns(runs []nn.KeyRun, windowed bool, wins []nn.KeyRun) {
	l.runs, l.windowed, l.wins = runs, windowed, wins
}

// RowsCall is a call of a block's rows: the rows of X, the stream-major
// residual Stride apart, their per-layer inputs Side (SideW a row), their
// rotary tables CS and CSSWA (NRot and NSWA a row), positions Pos of
// sequences Seq and tokens Tok.
type RowsCall struct {
	X        []float32
	Stride   int
	Side     []float32
	SideW    int
	CS       []float32
	NRot     int
	CSSWA    []float32
	NSWA     int
	Seq, Pos []int
	Tok      []int32
}

// RowsForm is block li's rows' form, its weights read in t bound where its
// call reads them, or why not: the block not composed here, no rows' form
// composed for it, a weight the device cannot read.
func (l *Lanes) RowsForm(li int, t Tensors) (*Form, nn.Decline) {
	cb, why := l.e.Block(li)
	if why != nn.DeclineServes {
		return nil, why
	}
	if cb.Rows == nil {
		return nil, cb.RowsWhy
	}
	if _, why := l.b.Bind(cb.Rows, li, t); why != nn.DeclineServes {
		return nil, why
	}
	return cb.Rows, nn.DeclineServes
}

// ReadyRows makes block li's history ready for the rows at pos its form f
// runs, or says why f cannot run them: a history past f's key bound, a page
// outside the arena.
func (l *Lanes) ReadyRows(li int, f *Form, pos []int) nn.Decline {
	if cb, _ := l.e.Block(li); cb != nil && cb.Cut > 0 {
		return nn.DeclineServes
	}
	return l.h.PrepareRows(li, pos, f.Keys)
}

// Rows runs block li's form f of rows over c, its weights read in t: each
// row's history read from where it starts to where its bidirectional run or
// its window ends, the history committed once the rows ran.
func (l *Lanes) Rows(li int, f *Form, t Tensors, c *RowsCall) error {
	n := len(c.Pos)
	if f.Toks && c.Tok == nil {
		return fmt.Errorf("model: block %d reads its rows' tokens, which the step gives none of", li)
	}
	cb, _ := l.e.Block(li)
	cut := cb != nil && cb.Cut > 0
	var hb nn.HistoryBinding
	if !cut {
		var err error
		if hb, err = l.h.Rows(li, l.seqs); err != nil {
			return err
		}
	}
	var sideOff int
	var bidir, whole bool
	if cb != nil {
		sideOff, bidir, whole = cb.SideOff, cb.Bidir, cb.Whole
	}
	runs := bidir && runsIn(l.runs, c.Pos[0], n)
	if cap(l.spans) < 4*n {
		l.spans = make([]uint32, 4*n)
	}
	spans := l.spans[:4*n]
	end := 0
	for i, p := range c.Pos {
		e, st := p+1, 0
		if !cut {
			st = l.h.KeyStart(li, p)
		}
		if runs {
			lo, hi := runOf(l.runs, p)
			e = max(e, hi)
			if l.windowed {
				st = max(st, lo)
			}
		}
		if w := l.wins; w != nil && !whole {
			st, e = max(st, w[p].Lo), w[p].Hi
		}
		end = max(end, p+1)
		spans[i*4], spans[i*4+1], spans[i*4+2], spans[i*4+3] = uint32(c.Seq[i]*hb.Slots), uint32(st), uint32(e), uint32(p)
	}
	w, _ := l.b.Bind(f, li, t)
	in := &l.in
	*in = Rows{X: c.X, Stride: c.Stride, Width: l.width, Streams: l.streams, Side: c.Side, SideW: c.SideW,
		SideOff: sideOff, CS: c.CS, NRot: c.NRot, CSSWA: c.CSSWA, NSWA: c.NSWA, Pages: hb.Pages, PagePos: hb.PagePos,
		Spans: spans, Pos: c.Pos, Tok: c.Tok, Req: l.req, Seq: c.Seq, Weights: w, Keys: hb.Keys, Values: hb.Values,
		Entries: hb.Entries, EntryPages: hb.EntryPages}
	if cb != nil && cb.Cut > 0 {
		in.Cut, in.CutW, in.Graph = l.cut, cb.Cut, cb.Graph
	}
	if f.Rec {
		in.RecLi = l.recLi(li)
		in.RecState, in.RecConv = l.recAddrs(in.RecLi)
		in.Conv, in.ConvLen = l.v.Conv(in.RecLi), l.convLen
	}
	if err := l.e.rows.Rows(li, f, in, runs); err != nil {
		return err
	}
	if !cut {
		l.h.Commit(li, end-1)
	}
	return nil
}

// runsIn says one of runs holds a position of [p0, p0+n).
func runsIn(runs []nn.KeyRun, p0, n int) bool {
	for _, r := range runs {
		if r.Lo < p0+n && r.Hi > p0 {
			return true
		}
	}
	return false
}

// runOf is the run holding pos, zero for none.
func runOf(runs []nn.KeyRun, pos int) (lo, hi int) {
	for _, r := range runs {
		if pos >= r.Lo && pos < r.Hi {
			return r.Lo, r.Hi
		}
	}
	return 0, 0
}

// Attach runs the lanes over a session's history h and recurrent state v.
func (l *Lanes) Attach(h History, v *View) { l.h, l.v = h, v }

// SetTokens is the tokens of the next call's rows.
func (l *Lanes) SetTokens(toks []int32) {
	l.toks = toks
	if len(toks) > 0 {
		l.tok = toks[0]
	}
}

// SetKeyRuns is the prompt's runs the next calls' rows read both ways in,
// full or each a window its rows read inside only (nn.KeyRunDevice); it
// takes back any row windows.
func (l *Lanes) SetKeyRuns(full, windowed []nn.KeyRun) {
	l.runs, l.windowed, l.wins = full, false, nil
	if len(windowed) > 0 {
		l.runs, l.windowed = windowed, true
	}
}

// SetRowWindows is each row's own window for the next calls
// (nn.RowWindowDevice).
func (l *Lanes) SetRowWindows(wins []nn.KeyRun) {
	l.wins = wins
}

// CloseRun closes the fused pool region the last call left open.
func (l *Lanes) CloseRun() { l.e.Unfuse() }

// SetRequest is the request words the next calls' blocks read
// (nn.RequestDevice).
func (l *Lanes) SetRequest(req []uint32) { l.req = req }

// SetLayerInputs is the per-layer inputs the next call's blocks read
// (nn.LayerInputDevice).
func (l *Lanes) SetLayerInputs(in []float32) { l.side = in }

// SetCut is the cut rows the next call's block cut in two reads
// (nn.CutDevice).
func (l *Lanes) SetCut(cut []float32) { l.cut = cut }

// Layers runs blocks [lo, hi) for the row at pos in x, its rotary tables cs
// and csSWA, one row at a time and no head: false where a block is declined
// (Declined) or fails (Err).
func (l *Lanes) Layers(lo, hi, pos, n int, x, cs, csSWA []float32, head *nn.Head) bool {
	l.why, l.at, l.err = nn.DeclineServes, -1, nil
	if n > 1 && head == nil {
		return l.chunk(lo, hi, pos, n, x, cs, csSWA)
	}
	if n != 1 || head != nil {
		l.err = fmt.Errorf("cpudev: a run of %d rows with a head is the rows' call", n)
		return false
	}
	for li := lo; li < hi; li++ {
		// A block that has to be read ends the fused run before the read,
		// and the run reopens after it.
		if l.e.fused && !l.src.Held(li) {
			l.e.Unfuse()
		}
		t, err := l.src.Ensure(li)
		if err != nil {
			l.err = err
			return false
		}
		l.e.Fuse()
		st := Step{X: x, Stride: l.stride, CS: cs, CSSWA: csSWA, Side: l.side, Pos: pos, Tok: l.tok, Toks: true,
			Faults: true, Req: l.req, Tensors: t, Cut: l.cut}
		why, err := l.Row(li, &st)
		if err != nil {
			l.err = err
			return false
		}
		if why != nn.DeclineServes {
			l.why, l.at = why, li
			l.e.steps.Decline(nn.PartBlock, why)
			return false
		}
		l.e.steps.Serve(nn.PartBlock)
		// The caller traces a run's last block, whichever device ran it.
		if l.trace != nil && li < hi-1 {
			l.trace(li)
		}
	}
	return true
}

// Row runs block li's form of one row over st on the session the lanes are
// attached to: its history and recurrent state the session's.
func (l *Lanes) Row(li int, st *Step) (nn.Decline, error) {
	st.History, st.Seqs = l.h, l.seqs
	st.RecLi = l.recLi(li)
	st.RecState, st.RecConv = l.recAddrs(st.RecLi)
	return l.e.Run(li, st, l.b)
}

// Declined is the block the last call stopped at and why, DeclineServes
// where none declined.
func (l *Lanes) Declined() (int, nn.Decline) { return l.at, l.why }

// RunErr is the last call's failure, nil where it ran or was declined
// (nn.RunFailer).
func (l *Lanes) RunErr() error { return l.err }

// Err is why the last call declined (nn.ErrReporter): the block's decline,
// and the composer's first error where the block was not formed.
func (l *Lanes) Err() string {
	if l.why == nn.DeclineServes {
		return ""
	}
	if l.why == nn.DeclineNotFormed && l.e.comp.Err() != nil {
		return fmt.Sprintf("the composer does not serve block %d on the CPU: %s (the first: %v)", l.at, nn.DeclineName(int(l.why)), l.e.comp.Err())
	}
	return fmt.Sprintf("the composer does not serve block %d on the CPU: %s", l.at, nn.DeclineName(int(l.why)))
}

// ErrCause is a decline's cause (nn.CauseReporter): a call the lanes refuse
// is a decline, never a fault.
func (l *Lanes) ErrCause() nn.FailCause { return nn.FailDecline }

// recLi is the block whose recurrent state block li steps: its own, or under
// RecFault the first block's that holds one.
func (l *Lanes) recLi(li int) int {
	if !l.RecFault {
		return li
	}
	for r := range l.v.Blocks() {
		if l.v.State(r) != nil {
			return r
		}
	}
	return li
}

// recAddrs is block li's recurrent state and window in the session's view,
// nil for a block it holds none of.
func (l *Lanes) recAddrs(li int) (state, conv unsafe.Pointer) {
	if l.v == nil || li < 0 || li >= l.v.Blocks() {
		return nil, nil
	}
	if l.keep != nil {
		l.keep.stepping(l.v, li)
	}
	return unsafe.Pointer(unsafe.SliceData(l.v.State(li))), unsafe.Pointer(unsafe.SliceData(l.v.Conv(li)))
}

// SessionLanes is a session's calls on the CPU device: the instance's lanes
// run over the session's history and its view, as a card's session view
// (gpu.GPU.Attach) runs the card's kernels over the session's. Each call
// attaches the history first, so one instance's lanes serve every session
// leasing it.
type SessionLanes struct {
	l    *Lanes
	of   HistoryOf
	pr   Prompt
	keep recKeep
}

// Bind makes s the calls of the session whose history of reads and whose
// say over the prompt pr is, on l, keeping what s holds of its own (its
// rewind copies).
func (s *SessionLanes) Bind(l *Lanes, of HistoryOf, pr Prompt) { s.l, s.of, s.pr = l, of, pr }

// Layers runs blocks [lo, hi) for the row at pos (nn.LayerRunner) over the
// session's history.
func (s *SessionLanes) Layers(lo, hi, pos, n int, x, cs, csSWA []float32, head *nn.Head) bool {
	s.attach()
	return s.l.Layers(lo, hi, pos, n, x, cs, csSWA, head)
}

// SetTokens is the tokens of the next call's rows (nn.TokenReader).
func (s *SessionLanes) SetTokens(toks []int32) { s.l.SetTokens(toks) }

// SetLayerInputs is the per-layer inputs the next call's blocks read
// (nn.LayerInputDevice).
func (s *SessionLanes) SetLayerInputs(in []float32) { s.l.SetLayerInputs(in) }

// SetCut is the cut rows the next call's block cut in two reads
// (nn.CutDevice).
func (s *SessionLanes) SetCut(cut []float32) { s.l.SetCut(cut) }

// SetRequest is the request words the next calls' blocks read
// (nn.RequestDevice).
func (s *SessionLanes) SetRequest(req []uint32) { s.l.SetRequest(req) }

// ReadsTables says the blocks read the call's rotary tables (nn.TableReader).
func (s *SessionLanes) ReadsTables() bool { return true }

// RunErr is the last call's failure (nn.RunFailer).
func (s *SessionLanes) RunErr() error { return s.l.RunErr() }

// Err is why the last call declined (nn.ErrReporter).
func (s *SessionLanes) Err() string { return s.l.Err() }

// ErrCause is a decline's cause (nn.CauseReporter).
func (s *SessionLanes) ErrCause() nn.FailCause { return s.l.ErrCause() }

// CloseRun closes the fused pool region the last call left open
// (nn.RunCloser).
func (s *SessionLanes) CloseRun() { s.l.CloseRun() }

// ownRun is rows [a, b) of a step across sessions, one session's.
type ownRun struct{ a, b int }

// LayersRows runs blocks [lo, hi) over the rows of x (nn.RowsDevice): row i
// at position pos[i] of the sequence its slot names (slot[i] / seqLen), its
// rotary tables the i-th of cs and csSWA, each block faulted in first.
func (l *Lanes) LayersRows(lo, hi int, pos, slot []int, seqLen int, x, cs, csSWA []float32, head *nn.Head) bool {
	l.why, l.at, l.err = nn.DeclineServes, -1, nil
	if head != nil {
		l.err = fmt.Errorf("cpudev: rows with a head are the head's call")
		return false
	}
	l.seqBuf = slices.Grow(l.seqBuf[:0], len(slot))[:len(slot)]
	for i, f := range slot {
		l.seqBuf[i] = f / max(seqLen, 1)
	}
	return l.blocksRows(lo, hi, x, cs, csSWA, l.seqBuf, pos, nil)
}

// blocksRows runs blocks [lo, hi) over the rows: their sequences seq, or
// with own each row's session.
func (l *Lanes) blocksRows(lo, hi int, x, cs, csSWA []float32, seq, pos []int, own []*SessionLanes) bool {
	return l.blocks(lo, hi, x, cs, csSWA, seq, pos, own, -1)
}

// blocks is blocksRows for a prompt chunk's rows too: base the chunk's first
// row in the prompt, -1 for a step's rows.
func (l *Lanes) blocks(lo, hi int, x, cs, csSWA []float32, seq, pos []int, own []*SessionLanes, base int) bool {
	for li := lo; li < hi; li++ {
		t, err := l.src.Ensure(li)
		if err != nil {
			l.err = err
			return false
		}
		cb, why := l.e.Block(li)
		switch {
		case why != nn.DeclineServes:
		case cb.Rec:
			why, err = l.recRows(li, cb, t, x, cs, csSWA, seq, pos, own, base)
		case own != nil:
			why, err = l.sessionRows(li, t, x, cs, csSWA, pos, own)
		default:
			why, err = l.rows(li, t, x, cs, csSWA, seq, pos)
		}
		if err != nil {
			l.err = err
			return false
		}
		if why != nn.DeclineServes {
			l.why, l.at = why, li
			l.e.steps.Decline(nn.PartBlock, why)
			return false
		}
		l.e.steps.Serve(nn.PartBlock)
	}
	return true
}

// call is a rows call over rows [a, b) of a step of n rows.
func (l *Lanes) call(x, cs, csSWA []float32, seq, pos []int, a, b, n int) RowsCall {
	d := l.width
	nrot, nswa, pw := len(cs)/n, len(csSWA)/n, len(l.side)/n
	c := RowsCall{X: x[a*d:], Stride: n * d, SideW: pw, NRot: nrot, NSWA: nswa, Seq: seq[a:b], Pos: pos[a:b]}
	if pw != 0 {
		c.Side = l.side[a*pw:]
	}
	if nrot != 0 {
		c.CS = cs[a*nrot:]
	}
	if nswa != 0 {
		c.CSSWA = csSWA[a*nswa:]
	}
	if l.toks != nil {
		c.Tok = l.toks[a:b]
	}
	return c
}

// rows runs block li's rows' form over the rows, each row's history in its
// own sequence's pages.
func (l *Lanes) rows(li int, t Tensors, x, cs, csSWA []float32, seq, pos []int) (nn.Decline, error) {
	f, why := l.RowsForm(li, t)
	if why != nn.DeclineServes {
		return why, nil
	}
	if why := l.ReadyRows(li, f, pos); why != nn.DeclineServes {
		return why, nil
	}
	c := l.call(x, cs, csSWA, seq, pos, 0, len(pos), len(pos))
	return nn.DeclineServes, l.Rows(li, f, t, &c)
}

// sessionRows runs block li over the rows of a step across sessions, row
// i's history own[i]'s: every row's page made its own before any session's
// rows are made ready, each session's rows run on its own lanes' form, the
// runs in order of position so a session's later row reads the keys its
// earlier rows wrote.
func (l *Lanes) sessionRows(li int, t Tensors, x, cs, csSWA []float32, pos []int, own []*SessionLanes) (nn.Decline, error) {
	n := len(pos)
	for a := range n {
		own[a].of.History().Own(li, pos[a])
	}
	l.oruns = l.oruns[:0]
	for a := 0; a < n; {
		b := a + 1
		for b < n && own[b] == own[a] {
			b++
		}
		l.oruns = append(l.oruns, ownRun{a, b})
		a = b
	}
	slices.SortStableFunc(l.oruns, func(p, q ownRun) int { return pos[p.a] - pos[q.a] })
	l.zeros = slices.Grow(l.zeros[:0], n)[:n]
	clear(l.zeros)
	for _, r := range l.oruns {
		o := own[r.a]
		f, why := o.l.RowsForm(li, t)
		if why != nn.DeclineServes {
			return why, nil
		}
		o.attach()
		if why := o.l.ReadyRows(li, f, pos[r.a:r.b]); why != nn.DeclineServes {
			return why, nil
		}
	}
	for _, r := range l.oruns {
		o := own[r.a]
		f, _ := o.l.RowsForm(li, t)
		o.attach()
		c := l.call(x, cs, csSWA, l.zeros, pos, r.a, r.b, n)
		if err := o.l.Rows(li, f, t, &c); err != nil {
			return nn.DeclineServes, err
		}
	}
	return nn.DeclineServes, nil
}

// recRows runs linear block li over the rows: on its rows' form where the
// rows run on it whole (recWhole), else on its form of one row, a row at a
// time in causal order, each stepping its own sequence's recurrent state --
// with own, its session's (its sequence that session's 0).
func (l *Lanes) recRows(li int, cb *Block, t Tensors, x, cs, csSWA []float32, seq, pos []int, own []*SessionLanes, base int) (nn.Decline, error) {
	n := len(pos)
	l.order = slices.Grow(l.order[:0], n)[:n]
	for i := range l.order {
		l.order[i] = i
	}
	slices.SortStableFunc(l.order, func(a, b int) int { return pos[a] - pos[b] })
	pr := l.prompt
	if base < 0 {
		pr = nil
	}
	if own == nil && l.keep.whole() {
		if f := l.recWhole(li, t, cb, seq, pos); f != nil {
			if why := l.ReadyRows(li, f, pos); why != nn.DeclineServes {
				return why, nil
			}
			if l.ChunkFault && base > 0 {
				l.clearRec(li, seq[0])
			}
			points := pr != nil && pr.Points()
			for a := 0; a < n; {
				b := n
				for e := a + 1; points && e < n; e++ {
					if pr.Point(pos[0] + e) {
						b = e
						break
					}
				}
				c := l.call(x, cs, csSWA, seq, pos, a, b, n)
				if err := l.Rows(li, f, t, &c); err != nil {
					return nn.DeclineServes, err
				}
				if pr != nil {
					pr.Seal(li, seq[a], pos[0]+b)
				}
				a = b
			}
			l.e.steps.Whole++
			return nn.DeclineServes, nil
		}
	}
	if base >= 0 && l.bidir && cb.Bidir && n > 1 {
		// A bidirectional chunk's rows each read the whole run (Rows'
		// runs); the stepped form reads a row's past alone.
		return nn.DeclineServes, fmt.Errorf("block %d: a bidirectional chunk of %d rows runs whole on its rows' form, "+
			"one sequence in order with no rewind mark, and these rows are not one", li, n)
	}
	if n > 1 {
		l.e.steps.Stepped++
	}
	if _, why := l.b.Bind(cb.F, li, t); why != nn.DeclineServes {
		return why, nil
	}
	d, stride := l.width, len(x)/l.streams
	nrot, nswa, pw := len(cs)/n, len(csSWA)/n, len(l.side)/n
	h, v := l.h, l.v
	defer l.Attach(h, v)
	for _, r := range l.order {
		if l.toks == nil && cb.F.Toks {
			return nn.DeclineServes, fmt.Errorf("model: block %d reads its rows' tokens, which the step gives none of", li)
		}
		st := Step{X: x[r*d:], Stride: stride, Pos: pos[r], Tensors: t}
		if own != nil {
			hr := own[r].of.History()
			l.Attach(hr, hr.View())
		} else {
			st.Slot = seq[r]
		}
		if l.ChunkFault && base > 0 && r == l.order[0] {
			l.clearRec(li, st.Slot)
		}
		if l.toks != nil {
			st.Tok, st.Toks = l.toks[r], true
		}
		if pw != 0 {
			st.Side = l.side[r*pw:]
		}
		if nrot > 0 {
			st.CS = cs[r*nrot:]
		}
		if nswa > 0 {
			st.CSSWA = csSWA[r*nswa:]
		}
		if why, err := l.Row(li, &st); why != nn.DeclineServes || err != nil {
			return why, err
		}
		if l.keep != nil && base >= 0 {
			l.keep.stepped(l.v, li, base+r)
		}
		if pr != nil {
			pr.Seal(li, st.Slot, pos[r]+1)
		}
	}
	return nn.DeclineServes, nil
}

// clearRec clears linear block li's summary of sequence slot (ChunkFault).
func (l *Lanes) clearRec(li, slot int) {
	if st, cv := l.v.State(li), l.v.Conv(li); st != nil && cv != nil {
		ns, nc := len(st)/max(l.seqs, 1), l.convLen
		clear(st[slot*ns:][:ns])
		clear(cv[slot*nc:][:nc])
	}
}

// chunk runs blocks [lo, hi) over a prompt chunk's n rows of the sequence
// SetPromptRows named, at positions pos..pos+n-1; a non-causal prompt's
// chunk is one run of rows attending both ways.
func (l *Lanes) chunk(lo, hi, pos, n int, x, cs, csSWA []float32) bool {
	l.seqBuf = slices.Grow(l.seqBuf[:0], n)[:n]
	l.posBuf = slices.Grow(l.posBuf[:0], n)[:n]
	for i := range n {
		l.seqBuf[i], l.posBuf[i] = l.slot, pos+i
	}
	if l.bidir {
		runs, windowed, wins := l.runs, l.windowed, l.wins
		l.chunkRun[0] = nn.KeyRun{Lo: pos, Hi: pos + n}
		l.runs, l.windowed = l.chunkRun[:], false
		defer func() { l.runs, l.windowed, l.wins = runs, windowed, wins }()
	}
	return l.blocks(lo, hi, x, cs, csSWA, l.seqBuf, l.posBuf, nil, l.base)
}

// SetPromptRows is the prompt chunk the next chunk call runs
// (nn.PromptRows).
func (l *Lanes) SetPromptRows(slot, base int, bidir bool) {
	l.slot, l.base, l.bidir = slot, base, bidir
}

// recWhole is linear block li's rows' form where the rows run on it whole:
// several rows in causal order, each sequence's rows in order of position;
// nil where not.
func (l *Lanes) recWhole(li int, t Tensors, cb *Block, seq, pos []int) *Form {
	if len(pos) < 2 || cb.Rows == nil {
		return nil
	}
	for i, r := range l.order {
		if r != i {
			return nil
		}
	}
	l.last = slices.Grow(l.last[:0], max(l.seqs, 1))[:max(l.seqs, 1)]
	for i := range l.last {
		l.last[i] = -1
	}
	for i, q := range seq {
		if q < 0 || q >= len(l.last) || pos[i] <= l.last[q] {
			return nil
		}
		l.last[q] = pos[i]
	}
	if f, why := l.RowsForm(li, t); why == nn.DeclineServes {
		return f
	}
	return nil
}

// attach binds the session's history to its lanes.
func (s *SessionLanes) attach() {
	h := s.of.History()
	s.l.Attach(h, h.View())
	s.l.prompt = s.pr
	s.l.keep = &s.keep
}

// SetPromptRows hands the lanes the prompt chunk the next call runs
// (nn.PromptRows).
func (s *SessionLanes) SetPromptRows(slot, base int, bidir bool) {
	s.l.SetPromptRows(slot, base, bidir)
}

// LayersRows runs blocks [lo, hi) over the session's rows (nn.RowsDevice).
func (s *SessionLanes) LayersRows(lo, hi int, pos, slot []int, seqLen int, x, cs, csSWA []float32, head *nn.Head) bool {
	s.attach()
	return s.l.LayersRows(lo, hi, pos, slot, seqLen, x, cs, csSWA, head)
}

// LayersSessions runs blocks [lo, hi) over a step's rows across sessions
// (nn.SessionStepper), row i's session sess[i] -- its calls on this device,
// a SessionLanes.
func (s *SessionLanes) LayersSessions(lo, hi int, sess []nn.LayerRunner, pos []int, x, cs, csSWA []float32, head *nn.Head) bool {
	l := s.l
	l.why, l.at, l.err = nn.DeclineServes, -1, nil
	if head != nil {
		l.err = fmt.Errorf("cpudev: rows with a head are the head's call")
		return false
	}
	l.own = l.own[:0]
	for i, d := range sess {
		o, ok := d.(*SessionLanes)
		if !ok || o == nil || o.l == nil {
			l.err = fmt.Errorf("cpudev: row %d's session has no calls on the CPU device", i)
			return false
		}
		l.own = append(l.own, o)
	}
	defer s.attach()
	ok := l.blocksRows(lo, hi, x, cs, csSWA, nil, pos, l.own)
	clear(l.own)
	return ok
}

// SetKeyRuns hands the lanes the prompt's runs (nn.KeyRunDevice).
func (s *SessionLanes) SetKeyRuns(full, windowed []nn.KeyRun) {
	s.l.SetKeyRuns(full, windowed)
}

// SetRowWindows hands the lanes each row's window (nn.RowWindowDevice).
func (s *SessionLanes) SetRowWindows(wins []nn.KeyRun) { s.l.SetRowWindows(wins) }

// MigrateKV is the CPU's half of moving block li's history of pos positions
// (nn.KVMigrator): the session's history takes k and v, or gives its rows
// into them, through the device's conversions.
func (s *SessionLanes) MigrateKV(li int, k, v []float32, pos int, toDevice bool) bool {
	var conv nn.KVConvDevice
	if c := s.l.e.kvconv; c != nil {
		conv = c
	}
	return s.of.History().MigrateKV(li, k, v, pos, toDevice, conv)
}

// MigrateRec is the CPU's half of moving block li's recurrent state
// (nn.RecMigrator): out of the session's view into conv and state, or into
// it.
func (s *SessionLanes) MigrateRec(li int, conv, state []float32, toDevice bool) bool {
	v := s.of.History().View()
	c, st := v.Conv(li), v.State(li)
	if c == nil || st == nil || len(conv) != len(c) || len(state) != len(st) {
		return false
	}
	if toDevice {
		copy(c, conv)
		copy(st, state)
	} else {
		copy(conv, c)
		copy(state, st)
	}
	return true
}
