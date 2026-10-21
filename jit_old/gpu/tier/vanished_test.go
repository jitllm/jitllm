package tier

import (
	"fmt"
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/vulkan"
)

// TestMain fails the binary when any Vulkan enumeration in it shrank. A
// driver that stops loading partway through a test binary turns every later
// Vulkan test into a skip on "no device", which reads as green (RULE 10).
func TestMain(m *testing.M) {
	testdevices.Install(backend.Restrict)
	code := m.Run()
	if first, now := vulkan.Vanished(); first > 0 {
		fmt.Fprintf(os.Stderr, "FAIL: the Vulkan loader enumerated %d device(s) first in this process and "+
			"%d later -- every Vulkan test after that skipped or ran elsewhere\n", first, now)
		code = 1
	}
	os.Exit(testdevices.Fail(code))
}
