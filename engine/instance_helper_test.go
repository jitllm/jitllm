package engine

import (
	"testing"

	"github.com/jitllm/jitllm/model"
)

// testInstance is NewInstance for a gate: a refusal fails the test.
func testInstance(tb testing.TB, m *model.Model, maxSeq int, opts ...Option) *Instance {
	tb.Helper()
	s, err := NewInstance(m, maxSeq, opts...)
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() {
		if d := s.DeviceDemotions(); d > 0 && !faultsInjected {
			tb.Errorf("the device failed %d time(s) and every block moved to the CPU device: %+v", d, s.Evacuations())
		}
	})
	return s
}

// sharedCPU is a CPU device for a test's instances to share, as a server's
// do: instances on one device step their rows together (StepRuns) and see
// each other decoding. opts are the instances' (their CPU tier and pool); the
// device closes when the test ends. Where opts name a device already
// (WithCPU), it is that one.
func sharedCPU(tb testing.TB, opts ...Option) Option {
	tb.Helper()
	var lo loadOpts
	for _, o := range opts {
		o(&lo)
	}
	if lo.cpu != nil {
		return WithCPU(lo.cpu)
	}
	c, err := OpenCPU(opts...)
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(c.Close)
	return WithCPU(c)
}

// withFuseCount counts the instance's fused runs in c.
func withFuseCount(c *fuseCount) Option { return func(l *loadOpts) { l.opt.fuse = c } }

// withStepShareFault makes the instance's host steps read the first run's
// history for every row: a gate's violation.
func withStepShareFault() Option { return func(l *loadOpts) { l.opt.stepShareFault = true } }

// withStepPosFault lays a step's rows out where f says: a gate's violation.
func withStepPosFault(f func(base, j, k int) int) Option {
	return func(l *loadOpts) { l.opt.stepPosFault = f }
}

// withStaleKeyFault names a restore's pages by f's key: a gate's violation.
func withStaleKeyFault(f func(key []int32) []int32) Option {
	return func(l *loadOpts) { l.opt.staleKeyFault = f }
}
