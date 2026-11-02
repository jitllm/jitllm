package gpu

import (
	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/engine/nn"
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
		g.noteCause(nn.FailBuild)
		return nil, &nn.BuildError{Layer: -1, Err: err}
	}
	return c, nil
}

// noteFormCause records a form or a lowering that did not build.
func (g *devTier) noteFormCause() {
	if formFaultedAsDevice() {
		g.noteCause(nn.FailFault)
	} else {
		g.noteCause(nn.FailBuild)
	}
}

// buildFailed records a form or a lowering that did not build and returns it
// named.
func (g *devTier) buildFailed(err error) error {
	g.noteFormCause()
	if nn.IsBuildError(err) {
		return err
	}
	return &nn.BuildError{Layer: -1, Err: err}
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
