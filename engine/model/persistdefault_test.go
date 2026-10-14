//go:build jitllmtest

package model

import (
	"fmt"
	"os"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// JITLLM_TEST_PERSIST=1 in a jitllmtest build runs every device gate in the
// binary through the persistent decode path (tier.PersistByDefaultForTest),
// and fails the binary when no persistent kernel launched: run one gate per
// binary invocation and the check is that gate's. JITLLM_TEST_PERSIST=deps
// orders those kernels with dependency counters (tier.PersistDepsByDefaultForTest)
// and fails the binary when none of them launched.
func init() {
	if os.Getenv("JITLLM_TEST_PERSIST") != "" {
		tier.PersistByDefaultForTest(true)
	}
	if os.Getenv("JITLLM_TEST_PERSIST") == "deps" {
		tier.PersistDepsByDefaultForTest(true)
	}
}

func persistRan(code int) int {
	if os.Getenv("JITLLM_TEST_PERSIST") == "" || code != 0 {
		return code
	}
	if n := tier.PersistLaunchesForTest(); n == 0 {
		fmt.Fprintln(os.Stderr, "FAIL: JITLLM_TEST_PERSIST is set and no persistent kernel launched: the gate ran the per-op path")
		return 1
	} else {
		fmt.Fprintf(os.Stderr, "persistent path: %d persistent kernels launched\n", n)
	}
	if os.Getenv("JITLLM_TEST_PERSIST") == "deps" {
		if n := tier.PersistDepLaunchesForTest(); n == 0 {
			fmt.Fprintln(os.Stderr, "FAIL: JITLLM_TEST_PERSIST=deps is set and no dependency-counter kernel launched")
			return 1
		} else {
			fmt.Fprintf(os.Stderr, "dependency counters: %d kernels launched\n", n)
		}
	}
	return code
}
