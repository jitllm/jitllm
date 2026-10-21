package backend_test

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testlock"
)

// gpuLock serialises GPU tests across processes (internal/testlock):
// `go test ./...` runs packages in parallel, and packages that open a device
// contend for VRAM and the driver, failing in a different package each run.
func gpuLock(t *testing.T) { testlock.GPU(t) }
