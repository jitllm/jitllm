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
func NewLanes(e *Exec, b *Bindings, src Source, stride, seqs int, trace func(li int)) *Lanes {
	return &Lanes{e: e, b: b, src: src, stride: stride, seqs: seqs, trace: trace}
}

// Attach runs the lanes over a session's history h and recurrent state v.
func (l *Lanes) Attach(h History, v *View) { l.h, l.v = h, v }

// SetToken is the token the next call's row is of.
func (l *Lanes) SetToken(tok int32) { l.tok = tok }

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
		st := Step{X: x, Stride: l.stride, CS: cs, CSSWA: csSWA, Side: l.side, Pos: pos, Tok: l.tok, History: l.h,
			Seqs: l.seqs, Tensors: t, RecLi: l.recLi(li)}
		st.RecState, st.RecConv = l.recAddrs(st.RecLi)
		why, err := l.e.Run(li, &st, l.b)
		if err != nil {
			l.err = err
			return false
		}
		if why != nn.DeclineServes {
			l.why, l.at = why, li
			return false
		}
		l.e.steps.Serve(nn.PartBlock)
		if l.trace != nil {
			l.trace(li)
		}
	}
	return true
}

// Declined is the block the last call stopped at and why, DeclineServes
// where none declined.
func (l *Lanes) Declined() (int, nn.Decline) { return l.at, l.why }

// Err is the last call's failure, nil where it ran or was declined.
func (l *Lanes) Err() error { return l.err }

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
