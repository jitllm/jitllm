package bench

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/internal/testdevices"
)

// TestMain holds a Vulkan driver for the binary's life where JITLLM_TEST_DEVICES
// admits one (internal/testdevices); every device open here carries the
// selection itself.
func TestMain(m *testing.M) {
	// A Vulkan driver held for the binary's life keeps the ICDs loaded while
	// the tests open and close their own (vulkan.Driver).
	keep, _ := backend.NewVulkanDriver(backend.Opts{Admit: testdevices.Admit()})
	code := m.Run()
	keep.Close()
	os.Exit(code)
}
