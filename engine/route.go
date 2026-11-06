package engine

import (
	"fmt"

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
	wins   nn.RowWindowDevice
	resv   nn.RowsReserver
	prompt nn.PromptRows
	kvmig  nn.KVMigrator
	recmig nn.RecMigrator
	rew    nn.RecRewinder
	rowk   nn.RecRowKeeper
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
	r.wins, _ = d.(nn.RowWindowDevice)
	r.resv, _ = d.(nn.RowsReserver)
	r.prompt, _ = d.(nn.PromptRows)
	r.kvmig, _ = d.(nn.KVMigrator)
	r.recmig, _ = d.(nn.RecMigrator)
	r.rew, _ = d.(nn.RecRewinder)
	r.rowk, _ = d.(nn.RecRowKeeper)
	return r
}

// The routes of a session: the attached device's view, and the CPU's lanes.
const (
	routeView = iota
	routeLanes
)

// History is the session's history as it is now (nn.HistoryOf).
func (s *Session) History() nn.History { return s.kv }

// bindLanes makes sess's calls on the lanes' device run for s: its view there
// is made once and bound again in place when it leases another instance, so
// a lease allocates nothing.
func (s *Instance) bindLanes(sess *Session) {
	b, ok := s.places[routeLanes].(nn.SessionBinder)
	if !ok {
		return
	}
	if v, ok := b.BindSession(sess.routes[routeLanes].run, sess, sess); ok {
		sess.routes[routeLanes] = newRoute(v)
	}
}

// routeAt is the route block li runs on.
func (s *Instance) routeAt(li int) *route { return &s.routes[s.routeOf(li)] }

// slotNone is a block the instance does not place (Instance.at).
const slotNone int8 = -1

// routeOf is the index of the route block li runs on in a session's routes,
// one of the instance's range, which it places.
func (s *Instance) routeOf(li int) int { return int(s.slotAt(li)) }

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
	if r.keys != nil {
		r.keys.SetKeyRuns(s.bidir, nil)
	}
	if r.tables != nil && r.tables.ReadsTables() {
		return s.rowTables(seq, pos)
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

// openPrompt hands a prompt chunk's run's device what its blocks read: the
// side inputs, the chunk's tokens, which sequence and prompt rows it is
// (nn.PromptRows) and, where it reads them, the chunk's rotary tables from
// its host rows, grown to the chunk's size first.
func (s *Instance) openPrompt(r *route, slot, base, p0, n, chunkRows int, bidir bool) error {
	s.closeRuns(r)
	if r.inputs != nil && s.c.SideWidth() != 0 {
		r.inputs.SetLayerInputs(s.bple[:n*s.c.SideWidth()])
	}
	if r.toks != nil {
		r.toks.SetTokens(s.chunkTok[:n])
	}
	if r.prompt != nil {
		r.prompt.SetPromptRows(slot, base, bidir)
	}
	if r.tables != nil && r.tables.ReadsTables() {
		s.growRows(chunkRows)
		seq, pos := s.chunkRows(slot, p0, n)
		return s.rowTables(seq, pos)
	}
	return nil
}

// callTellOn hands route r's device the chunk's attention before it runs:
// a non-causal prompt's rows (embedTellOn), else the prompt's key runs in
// the chunk (bidirTellOn). false is a device that cannot honour them.
func (s *Instance) callTellOn(r *route, p0, n int) bool {
	if s.emb != nil && s.emb.bidir {
		return s.embedTellOn(r, n)
	}
	return s.bidirTellOn(r, p0, n)
}

// bidirTellOn is bidirTell for route r's device.
func (s *Instance) bidirTellOn(r *route, p0, n int) bool {
	if !s.bidirIn(p0, n) {
		if r.keys != nil {
			r.keys.SetKeyRuns(nil, nil)
		}
		return true
	}
	if r.keys == nil {
		return false
	}
	r.keys.SetKeyRuns(s.bidir, nil)
	return true
}

// embedTellOn is embedTell for route r's device; a device told the prompt is
// non-causal (nn.PromptRows) makes the chunk's run itself, handed the rows'
// windows only where the run states them (a tower's windowed blocks).
func (s *Instance) embedTellOn(r *route, n int) bool {
	if r.prompt != nil {
		if r.wins != nil {
			r.wins.SetRowWindows(s.emb.rowWins)
		}
		return true
	}
	if r.resv != nil && !r.resv.ReserveRows(n) {
		return false
	}
	if r.wins == nil {
		return false
	}
	r.wins.SetRowWindows(s.embedWins(n))
	return true
}

// Points says the prompt seals restore points (cpudev.Prompt).
func (s *Session) Points() bool { return s.on != nil && s.on.recPointsOn() }

// Point says pos is one of the prompt's restore points (cpudev.Prompt).
func (s *Session) Point(pos int) bool { return s.on != nil && s.on.recPoint(pos) }

// Seal stores block li's summary of slot at pos where pos is a restore point
// (cpudev.Prompt).
func (s *Session) Seal(li, slot, pos int) {
	if s.on != nil {
		s.on.recRowPoint(li, slot, pos)
	}
}

// moveRoutes is the routes a block's history moves between: from the CPU's
// to the card's where toDevice, else back.
func (s *Instance) moveRoutes(toDevice bool) (from, to *route) {
	if toDevice {
		return &s.routes[routeLanes], &s.routes[routeView]
	}
	return &s.routes[routeView], &s.routes[routeLanes]
}

// moveKV moves block li's history of pos positions: the source device gives
// it as floats into k and v, the destination takes them.
func (s *Instance) moveKV(li, pos int, k, v []float32, from, to *route) bool {
	return from.kvmig != nil && to.kvmig != nil && from.kvmig.MigrateKV(li, k, v, pos, false) && to.kvmig.MigrateKV(li, k, v, pos, true)
}
