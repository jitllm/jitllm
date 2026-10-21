//go:build !amd64

package probe

import (
	"fmt"
	"runtime"

	"github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/jit"
)

// Host measures the host CPU; only amd64 has its probe yet.
func Host() (cost.Device, error) {
	return cost.Device{Device: jit.Device{Kind: jit.DeviceCPU}}, fmt.Errorf("probe: no host probe for %s", runtime.GOARCH)
}
