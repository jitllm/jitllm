package engine

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/device/cpudev"
	"github.com/jitllm/jitllm/nn"
)

// route is a device the instance's blocks run on and the faces a call asks
// of it, asserted once: an interface-to-interface assertion caches its answer
// in a heap object at a random later call (setLD).
type route struct {
	run    nn.LayerRunner
	inputs nn.LayerInputDevice
	toks   nn.TokenReader
	tables nn.TableReader
	fail   nn.RunFailer
	errs   nn.ErrReporter
	cause  nn.CauseReporter
	closer nn.RunCloser
	rows   nn.RowsDevice
	steps  nn.SessionStepper
	keys   nn.KeyRunDevice
}

func newRoute(d nn.LayerRunner) route {
	r := route{run: d}
	r.inputs, _ = d.(nn.LayerInputDevice)
	r.toks, _ = d.(nn.TokenReader)
	r.tables, _ = d.(nn.TableReader)
	r.fail, _ = d.(nn.RunFailer)
	r.errs, _ = d.(nn.ErrReporter)
	r.cause, _ = d.(nn.CauseReporter)
	r.closer, _ = d.(nn.RunCloser)
	r.rows, _ = d.(nn.RowsDevice)
	r.steps, _ = d.(nn.SessionStepper)
	r.keys, _ = d.(nn.KeyRunDevice)
	return r
}

// laneRoute is the route of a session's calls on the CPU device, its faces
// the lanes' own (no assertion at all).
func laneRoute(l *cpudev.SessionLanes) route {
	return route{run: l, inputs: l, toks: l, tables: l, fail: l, errs: l, cause: l, closer: l, rows: l, steps: l, keys: l}
}

// History is the session's history as it is now (cpudev.HistoryOf).
func (s *Session) History() cpudev.History { return s.kv }

// bindLanes makes sess's calls on the CPU device run on s's lanes.
func (s *Instance) bindLanes(sess *Session) {
	if s.cpu.lanes == nil {
		return
	}
	sess.lanes = s.cpu.lanes.For(sess)
	sess.routes[1] = laneRoute(&sess.lanes)
}

// routeAt is the route block li runs on.
func (s *Instance) routeAt(li int) *route { return &s.routes[s.routeOf(li)] }

// routeOf is the index of the route block li runs on in a session's routes.
func (s *Instance) routeOf(li int) int {
	if s.devAt(li) {
		return 0
	}
	return 1
}

// openRun hands the run's device what its blocks read before it is called:
// the side inputs, the token and, where it reads them, the step's rotary
// tables; and closes the region another device holds open.
func (s *Instance) openRun(r *route) error {
	s.closeRuns(r)
	if r.inputs != nil && s.c.SideWidth() != 0 {
		r.inputs.SetLayerInputs(s.ple)
	}
	if r.toks != nil {
		s.tok1[0] = s.tok
		r.toks.SetTokens(s.tok1[:])
	}
	if r.tables != nil && r.tables.ReadsTables() {
		return s.oneTable()
	}
	return nil
}

// closeRuns closes the region every device but r holds open (nn.RunCloser).
func (s *Instance) closeRuns(r *route) {
	for i := range s.routes {
		if o := &s.routes[i]; o != r && o.closer != nil {
			o.closer.CloseRun()
		}
	}
}

// runRefusal is the error a run that route r refused ends with, nil for a
// fault: the device's own failure, else what refusal makes of its cause.
func (s *Instance) runRefusal(r *route, li int) error {
	if r.fail != nil {
		if err := r.fail.RunErr(); err != nil {
			return err
		}
	}
	why := ""
	if r.errs != nil {
		why = r.errs.Err()
	}
	cause := nn.FailDecline
	if r.cause != nil {
		cause = r.cause.ErrCause()
	}
	return s.refusalOf(li, why, cause)
}

// openRows hands a rows run's device what its blocks read: the side inputs,
// the tokens, the prompt's key runs and, where it reads them, the rows'
// rotary tables; and closes the region another device holds open.
func (s *Instance) openRows(r *route, seq, pos []int, tokens []int32) error {
	s.closeRuns(r)
	n := len(pos)
	if r.inputs != nil && s.c.SideWidth() != 0 {
		r.inputs.SetLayerInputs(s.bple[:n*s.c.SideWidth()])
	}
	if r.toks != nil {
		r.toks.SetTokens(rowsOf(tokens, 0, n))
	}
	if r.keys != nil && !r.keys.SetKeyRuns(s.bidir, nil) && s.bidirIn(0, s.pos+n) {
		return fmt.Errorf("model: the device cannot honour the prompt's bidirectional runs")
	}
	if r.tables != nil && r.tables.ReadsTables() {
		return s.cpuRowTables(seq, pos)
	}
	return nil
}

// rowsOn runs blocks [lo, hi) of a step's rows on route r (nn.RowsDevice),
// row i at (pos[i], slot[i]); with head the device takes the projection too
// and reads the first nlogit rows' logits into logits.
func (s *Instance) rowsOn(r *route, lo, hi int, pos, slot []int, x, cs, csSWA []float32, head bool, logits []float32, nlogit int) error {
	if r.rows == nil {
		return errBatchDevice{}
	}
	var h *nn.Head
	if head && nlogit > 0 {
		h = s.head
		keep := h.Logits
		h.RowLogits, h.Logits, h.LogitRows = true, logits, nlogit
		defer func() { h.RowLogits, h.Logits, h.LogitRows = false, keep, 0 }()
	}
	if !r.rows.LayersRows(lo, hi, pos, slot, s.maxSeq, x, cs, csSWA, h) {
		return s.rowsRefused(r, lo, hi, len(pos), "")
	}
	return nil
}

// sessionsOn is rowsOn for a step across sessions: every row through its
// own session's view of route k's device (nn.SessionStepper), handed its
// own instance's key runs first, so its history there is its own
// session's.
func (s *Instance) sessionsOn(k int, lo, hi int, pos []int, x, cs, csSWA []float32, head bool, logits []float32, nlogit int) error {
	r := &s.routes[k]
	if r.steps == nil {
		return errBatchDevice{}
	}
	var h *nn.Head
	if head && nlogit > 0 {
		h = s.head
		keep := h.Logits
		h.RowLogits, h.Logits, h.LogitRows = true, logits, nlogit
		defer func() { h.RowLogits, h.Logits, h.LogitRows = false, keep, 0 }()
	}
	views := s.stepSess[:0]
	for a, o := range s.rowOwn {
		if o.model != s.m {
			return fmt.Errorf("model: row %d is of another model's session", a)
		}
		v := &o.routes[k]
		if kr := v.keys; kr != nil && (a == 0 || s.rowOwn[a-1] != o) {
			oi := s
			if a < len(s.rowInst) && s.rowInst[a] != nil {
				oi = s.rowInst[a]
			}
			kr.SetKeyRuns(oi.bidir, nil)
		}
		views = append(views, v.run)
	}
	defer func() { s.stepSess = clearDevs(views) }()
	if !r.steps.LayersSessions(lo, hi, views, pos, x, cs, csSWA, h) {
		return s.rowsRefused(r, lo, hi, len(pos), " across sessions")
	}
	return nil
}

// rowsRefused is the error a rows run that route r refused ends with: the
// device's own failure, else why it refused.
func (s *Instance) rowsRefused(r *route, lo, hi, n int, across string) error {
	if r.fail != nil {
		if err := r.fail.RunErr(); err != nil {
			return err
		}
	}
	why := "no reason given"
	if r.errs != nil && r.errs.Err() != "" {
		why = r.errs.Err()
	}
	return fmt.Errorf("model: the device refused blocks [%d,%d) for %d rows%s: %s", lo, hi, n, across, why)
}
