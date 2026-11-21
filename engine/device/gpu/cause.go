package gpu

import (
	"errors"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/nn"
)

// noteCause records what kind of refusal the device's last call was. A fault
// outranks a build failure, which outranks a decline, so a call that met both
// reports the one that cannot be moved around. A session's view records it as
// that session's, so two sessions' calls at once each read their own.
func (g *devTier) noteCause(c nn.FailCause) {
	at := &g.cause
	if g != g.dv {
		at = &g.devSess.sessCause
	}
	for {
		old := at.Load()
		if uint32(c) <= old || at.CompareAndSwap(old, uint32(c)) {
			return
		}
	}
}

// fail records why the device's call refused: as the device's LastErr,
// which the last writer wins, and as the calling session's own reason
// (devSess.own), which only that session's calls write.
func (g *devTier) fail(why string) {
	g.LastErr = why
	g.failSess().setOwn(why)
}

// failSess is the session a reason recorded through g is the call's of: a
// session's view's own, and on the device's own view the session whose call
// holds the device exclusively, the zero session's when none does.
func (g *devTier) failSess() *devSess {
	if g == g.dv {
		if c := g.exclSess.Load(); c != nil {
			return c
		}
	}
	return g.devSess
}

func (ds *devSess) setOwn(why string) {
	ds.ownMu.Lock()
	ds.own = why
	ds.ownMu.Unlock()
}

func (ds *devSess) ownErr() string {
	ds.ownMu.Lock()
	defer ds.ownMu.Unlock()
	return ds.own
}

// errMark is the reasons a question asked on the way to a call found
// (markErr), put back after it (restoreErr) so the question leaves them as
// they were.
type errMark struct {
	last, own string
	ds        *devSess
}

func (g *devTier) markErr() errMark {
	ds := g.failSess()
	return errMark{last: g.LastErr, own: ds.ownErr(), ds: ds}
}

func (g *devTier) restoreErr(m errMark) {
	g.LastErr = m.last
	m.ds.setOwn(m.own)
}

// reasonOf is why session sid's call just refused here, read inside that
// call: its own reason, and the device's last where its view recorded none
// (a write the device's own view made beside other sessions' steps).
func (g *devTier) reasonOf(sid uint64) string {
	if e := g.ownErr(sid); e != "" {
		return e
	}
	return g.Err()
}

// ownErr is why session sid's last call refused on this device, "" when
// it recorded no reason here.
func (g *devTier) ownErr(sid uint64) string {
	g.mu.Lock()
	ds := g.sess[sid]
	g.mu.Unlock()
	if ds == nil {
		return ""
	}
	return ds.ownErr()
}

// noteRoom records a call refused for room (ErrKVCapacity) as one.
func (g *devTier) noteRoom(err error) {
	if errors.Is(err, ErrKVCapacity) {
		g.noteCause(nn.FailRoom)
	}
}

// ErrCause is what kind of refusal the last call on this device was
// (nn.CauseReporter), the zero session's included.
func (g *devTier) ErrCause() nn.FailCause {
	return nn.FailCause(max(g.cause.Load(), g.devSess.sessCause.Load()))
}

// compile is Compile for a kernel the call cannot do without: a failure is a
// build failure, named as one, and the call that needed it reports it. A
// kernel with an alternative the caller falls back to on the same device is
// compiled with g.dev.Compile, which records nothing.
func (g *devTier) compile(k *ir.Kernel) (backend.Kernel, error) {
	c, err := g.dev.Compile(k)
	if err != nil {
		// A width the pipeline refused is formed again narrower
		// (narrowGroups); the call notes a build failure only if that fails.
		if !errors.Is(err, backend.ErrGroupTooWide) {
			g.noteCause(nn.FailBuild)
		}
		return nil, &nn.BuildError{Layer: -1, Err: err}
	}
	return c, nil
}

// noteFormCause records a form or a lowering that did not build.
func (g *devTier) noteFormCause() {
	if g.formFaultedAsDevice() {
		g.noteCause(nn.FailFault)
	} else {
		g.noteCause(nn.FailBuild)
	}
}

// buildFailed records a form or a lowering that did not build and returns it
// named.
func (g *devTier) buildFailed(err error) error {
	g.noteFormCause()
	if !nn.IsBuildError(err) {
		err = &nn.BuildError{Layer: -1, Err: err}
	}
	if g.FormErr == nil {
		g.FormErr = err
	}
	return err
}

// ErrCause is the worst refusal kind any device met since the last call began
// (nn.CauseReporter).
func (g *GPU) ErrCause() nn.FailCause { return g.errCause(0) }

// errCause is ErrCause for session sid: each device's own cause and the one
// sid's calls met there.
func (g *GPU) errCause(sid uint64) nn.FailCause {
	c := nn.FailDecline
	for i := 0; ; i++ {
		d := g.dev(i)
		if d == nil {
			return c
		}
		c = max(c, nn.FailCause(d.cause.Load()))
		d.mu.Lock()
		if ds := d.sess[sid]; ds != nil {
			c = max(c, nn.FailCause(ds.sessCause.Load()))
		}
		d.mu.Unlock()
	}
}

// resetCause forgets every device's last refusal kind, and session sid's, at
// the head of a call, as resetRecSteps does for its count: a call that
// refuses reads its own cause, not the one an earlier call left. A rows call
// beside another (shared false) forgets only its session's: the device's own
// may be the other call's.
func (g *GPU) resetCause(sid uint64, shared bool) {
	for i := 0; ; i++ {
		d := g.dev(i)
		if d == nil {
			return
		}
		if shared {
			d.cause.Store(0)
		}
		d.mu.Lock()
		if ds := d.sess[sid]; ds != nil {
			ds.sessCause.Store(0)
		}
		d.mu.Unlock()
	}
}

func (s *gpuSession) ErrCause() nn.FailCause { return s.g.errCause(s.sid) }
