//go:build !amd64 && !arm64

package probe

import (
	"fmt"
	"runtime"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/cost"
)

// Host measures the host CPU; only amd64 has its probe yet.
func Host() (cost.Device, error) {
	return cost.Device{Device: composer.Device{Kind: composer.DeviceCPU}}, fmt.Errorf("probe: no host probe for %s", runtime.GOARCH)
}
