package tier

import (
	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/ir"
)

// noteCause records what kind of refusal the device's last call was. A fault
// outranks a build failure, which outranks a decline, so a call that met both
// reports the one that cannot be moved around.
func (g *devTier) noteCause(c nn.FailCause) {
	for {
		old := g.cause.Load()
		if uint32(c) <= old || g.cause.CompareAndSwap(old, uint32(c)) {
			return
		}
	}
}

// ErrCause is what kind of refusal the last call on this device was
// (nn.CauseReporter).
func (g *devTier) ErrCause() nn.FailCause { return nn.FailCause(g.cause.Load()) }

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

// buildFailed records a form or a lowering that did not build and returns it
// named.
func (g *devTier) buildFailed(err error) error {
	if formFaultedAsDevice() {
		g.noteCause(nn.FailFault)
	} else {
		g.noteCause(nn.FailBuild)
	}
	if nn.IsBuildError(err) {
		return err
	}
	return &nn.BuildError{Layer: -1, Err: err}
}

// ErrCause is the worst refusal kind any device met since the last call began
// (nn.CauseReporter).
func (g *GPU) ErrCause() nn.FailCause {
	c := nn.FailDecline
	for i := 0; ; i++ {
		d := g.dev(i)
		if d == nil {
			return c
		}
		c = max(c, d.ErrCause())
	}
}

// resetCause forgets every device's last refusal kind at the head of a call,
// as resetRecSteps does for its count: a call that refuses reads its own
// cause, not the one an earlier call left.
func (g *GPU) resetCause() {
	for i := 0; ; i++ {
		d := g.dev(i)
		if d == nil {
			return
		}
		d.cause.Store(0)
	}
}

func (s *gpuSession) ErrCause() nn.FailCause { return s.g.ErrCause() }
