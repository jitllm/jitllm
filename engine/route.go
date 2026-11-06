package engine

import (
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
	return r
}

// laneRoute is the route of a session's calls on the CPU device, its faces
// the lanes' own (no assertion at all).
func laneRoute(l *cpudev.SessionLanes) route {
	return route{run: l, inputs: l, toks: l, tables: l, fail: l, errs: l, cause: l, closer: l}
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
func (s *Instance) routeAt(li int) *route {
	if s.devAt(li) {
		return &s.routes[0]
	}
	return &s.routes[1]
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
