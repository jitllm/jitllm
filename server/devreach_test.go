package server

import (
	"fmt"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
)

// reachable says whether a -devices name can be opened on this host, and why
// not. It asks the backend, not the tier, so a host with no card is told apart
// from a card the gate then failed on.
func reachable(dev string) (bool, string) {
	api, _, _ := strings.Cut(dev, ":")
	switch api {
	case "cuda":
		n, err := backend.CUDACount()
		if err != nil || n == 0 {
			return false, fmt.Sprintf("no CUDA device (%d, %v)", n, err)
		}
		return true, ""
	case "vulkan":
		infos, err := backend.VulkanDevices()
		if err != nil || len(infos) == 0 {
			return false, fmt.Sprintf("no Vulkan device (%v)", err)
		}
		return true, ""
	}
	devs := backend.Open()
	for _, d := range devs {
		d.Close()
	}
	if len(devs) == 0 {
		return false, "no GPU backend opened a device"
	}
	return true, ""
}

// requireDevice skips a device gate, by name, on a host where dev is not
// reachable. Where it is, the gate runs and a failure is a failure.
func requireDevice(t testing.TB, dev string) {
	t.Helper()
	requireDeviceWith(t, dev, reachable)
}

func requireDeviceWith(t testing.TB, dev string, probe func(string) (bool, string)) {
	t.Helper()
	if ok, why := probe(dev); !ok {
		t.Skipf("NO DEVICE: %s is not reachable on this host: %s -- this gate proved nothing", dev, why)
	}
}

// TestRequireDeviceSkipsOnlyWithoutADevice: an absent device skips with its
// reason; a present one does not skip, so a gate after it can fail.
func TestRequireDeviceSkipsOnlyWithoutADevice(t *testing.T) {
	absent := func(string) (bool, string) { return false, "none here" }
	present := func(string) (bool, string) { return true, "" }
	t.Run("absent", func(t *testing.T) {
		r := &skipRecorder{TB: t}
		requireDeviceWith(r, "cuda:0", absent)
		if !strings.Contains(r.skipped, "none here") {
			t.Fatalf("an absent device did not skip with its reason: %q", r.skipped)
		}
	})
	t.Run("present", func(t *testing.T) {
		requireDeviceWith(t, "cuda:0", present)
		if t.Skipped() {
			t.Fatal("a present device skipped")
		}
	})
}

// skipRecorder records a Skipf instead of ending the test.
type skipRecorder struct {
	testing.TB
	skipped string
}

func (r *skipRecorder) Helper() {}

func (r *skipRecorder) Skipf(format string, args ...any) { r.skipped = fmt.Sprintf(format, args...) }
