package main

import (
	"flag"
	"os"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/backend/vulkan"
	"github.com/jitllm/jitllm/internal/testdevices"
)

// TestMain puts JITLLM_TEST_DEVICES in force: no test here opens a device
// outside it, and one that names such a device fails the binary
// (internal/testdevices).
func TestMain(m *testing.M) {
	flag.Bool("update", false, "rewrite the demo's golden text from what it renders now")
	testdevices.Install(backend.Restrict)
	// A Vulkan driver held for the binary's life keeps the ICDs loaded while
	// the tests open and close their own (vulkan.Driver).
	keep, _ := vulkan.NewDriver()
	code := m.Run()
	keep.Close()
	os.Exit(code)
}
