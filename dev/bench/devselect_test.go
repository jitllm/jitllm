package bench

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
)

// TestMain puts JITLLM_TEST_DEVICES in force: no test here opens a device
// outside it, and one that names such a device fails the binary
// (internal/testdevices).
func TestMain(m *testing.M) {
	testdevices.Install(backend.Restrict)
	os.Exit(testdevices.Fail(m.Run()))
}
