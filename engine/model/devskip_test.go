package model

import (
	"errors"
	"fmt"
	"os"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/backend/vulkan"
	"github.com/jitllm/jitllm/internal/testdevices"
)

// noDevice is what a device arm does when its device would not open: skip,
// naming the device and the error -- unless an earlier open in this process
// saw the device, in which case the arm fails. A driver that stops loading
// partway through a test binary turns every later arm on it into "no device
// here", which reads as green (RULE 10); vulkan.ErrDevicesVanished is how the
// loader's silence becomes an error.
func noDevice(t *testing.T, what string, err error) {
	t.Helper()
	if errors.Is(err, vulkan.ErrDevicesVanished) {
		t.Fatalf("%s was there earlier in this process and is gone: %v", what, err)
	}
	t.Skipf("no %s here: %v", what, err)
}

// TestMain fails the binary when any Vulkan enumeration in it shrank, which
// catches an arm that skipped on a vanished device through a path noDevice
// does not cover.
func TestMain(m *testing.M) {
	testdevices.Install(backend.Restrict)
	os.Exit(vanishedFails(persistRan(m.Run())))
}

// vanishedFails is code, or a failure naming the Vulkan devices that went
// missing during the run.
func vanishedFails(code int) int {
	if first, now := vulkan.Vanished(); first > 0 {
		fmt.Fprintf(os.Stderr, "FAIL: the Vulkan loader enumerated %d device(s) first in this process and "+
			"%d later -- every Vulkan arm after that skipped or ran elsewhere\n", first, now)
		return 1
	}
	return code
}
