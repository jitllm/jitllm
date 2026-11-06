package cpudev

import (
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/nn"
)

// Source is where an instance's blocks' weights are read in: Held says block
// li's page is in, so reading it reads nothing; Ensure reads it in where it
// is not and is the block's copy, bound at it, that its forms read.
type Source interface {
	Held(li int) bool
	Ensure(li int) (Tensors, error)
}

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
	tok    int32
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
	hb, err := l.h.Rows(li, l.seqs)
	if err != nil {
		return err
	}
	cb, _ := l.e.Block(li)
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
		e, st := p+1, l.h.KeyStart(li, p)
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
		Spans: spans, Pos: c.Pos, Tok: c.Tok, Seq: c.Seq, Weights: w, Keys: hb.Keys, Values: hb.Values}
	if f.Rec {
		in.RecLi = l.recLi(li)
		in.RecState, in.RecConv = l.recAddrs(in.RecLi)
		in.Conv, in.ConvLen = l.v.Conv(in.RecLi), l.convLen
	}
	if err := l.e.rows.Rows(li, f, in, runs); err != nil {
		return err
	}
	l.h.Commit(li, end-1)
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

// SetTokens is the tokens of the next call's rows (nn.TokenReader).
func (l *Lanes) SetTokens(toks []int32) { l.tok = toks[0] }

// ReadsTables says the blocks read the call's rotary tables (nn.TableReader).
func (l *Lanes) ReadsTables() bool { return true }

// CloseRun closes the fused pool region the last call left open
// (nn.RunCloser).
func (l *Lanes) CloseRun() { l.e.Unfuse() }

// SetLayerInputs is the per-layer inputs the next call's blocks read
// (nn.LayerInputDevice).
func (l *Lanes) SetLayerInputs(in []float32) { l.side = in }

// Layers runs blocks [lo, hi) for the row at pos in x, its rotary tables cs
// and csSWA, one row at a time and no head: false where a block is declined
// (Declined) or fails (Err).
func (l *Lanes) Layers(lo, hi, pos, n int, x, cs, csSWA []float32, head *nn.Head) bool {
	l.why, l.at, l.err = nn.DeclineServes, -1, nil
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
			Faults: true, Tensors: t}
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
	return unsafe.Pointer(unsafe.SliceData(l.v.State(li))), unsafe.Pointer(unsafe.SliceData(l.v.Conv(li)))
}
