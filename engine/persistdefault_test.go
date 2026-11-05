package engine

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testdevices"
)

// openGPU is gpu.OpenWith for a model gate. With JITLLM_TEST_PERSIST set the
// tier runs the persistent decode path (gpu.WithPersist), and with
// JITLLM_TEST_PERSIST=deps orders those kernels with dependency counters
// (gpu.WithPersistDeps), either before the gate's own options, which still
// decide; the tier then fails t when it ends if it ran a block's launches down
// the path and launched no persistent kernel (none ordered by dependency
// counters, for deps): the gate ran the per-op path.
func openGPU(t testing.TB, opts ...gpu.Option) (*gpu.GPU, error) {
	mode := os.Getenv("JITLLM_TEST_PERSIST")
	switch mode {
	case "":
	case "deps":
		opts = append([]gpu.Option{gpu.WithPersistDeps(true, 0)}, opts...)
	default:
		opts = append([]gpu.Option{gpu.WithPersist(true, 0)}, opts...)
	}
	g, err := gpu.OpenWith(append([]gpu.Option{gpu.WithAdmit(testdevices.Admit())}, opts...)...)
	if err != nil || g == nil || mode == "" {
		return g, err
	}
	t.Cleanup(func() {
		runs, launched, deps := g.PersistLaunched()
		switch {
		case runs > 0 && launched == 0:
			t.Errorf("JITLLM_TEST_PERSIST is set and %d runs of a block's launches launched no persistent kernel: the gate ran the per-op path", runs)
		case mode == "deps" && launched > 0 && deps == 0:
			t.Errorf("JITLLM_TEST_PERSIST=deps is set and none of %d persistent kernels was ordered by dependency counters", launched)
		}
	})
	return g, err
}
