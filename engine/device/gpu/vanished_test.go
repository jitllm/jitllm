package gpu

import (
	"fmt"
	"os"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/internal/testdevices"
)

// TestMain holds a Vulkan driver for the binary's life, which keeps the ICDs
// loaded while the tests open and close their own (vulkan.Driver), and fails
// the binary when the devices it first enumerated have shrunk by the end. A
// driver that stops loading partway through a test binary turns every later
// Vulkan test into a skip on "no device", which reads as green (RULE 10).
func TestMain(m *testing.M) {
	keep, _ := backend.NewVulkanDriver(backend.Opts{Admit: testdevices.Admit()})
	code := m.Run()
	if first, now := keep.Vanished(); first > 0 {
		fmt.Fprintf(os.Stderr, "FAIL: the Vulkan loader enumerated %d device(s) when this binary started and "+
			"%d at its end -- every Vulkan test after that skipped or ran elsewhere\n", first, now)
		code = 1
	}
	keep.Close()
	os.Exit(code)
}
