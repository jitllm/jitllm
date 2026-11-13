package server

import (
	"fmt"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/internal/testdevices"
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
		infos, err := backend.VulkanDevicesWith(backend.Opts{Admit: testdevices.Admit()})
		if err != nil || len(infos) == 0 {
			return false, fmt.Sprintf("no Vulkan device (%v)", err)
		}
		return true, ""
	}
	devs := backend.OpenWith(backend.Opts{Admit: testdevices.Admit()})
	for _, d := range devs {
		d.Close()
	}
	if len(devs) == 0 {
		return false, "no GPU backend opened a device"
	}
	return true, ""
}

// requireDevice skips a device gate, by name, on a host where dev is not
// reachable. Where it is, the gate runs and a failure is a failure; where the
// run listed it (JITLLM_TEST_DEVICES) and it does not open, the card or its
// driver is broken, not absent, and the gate fails.
func requireDevice(t testing.TB, dev string) {
	t.Helper()
	requireDeviceWith(t, dev, reachable, listed(dev))
}

func requireDeviceWith(t testing.TB, dev string, probe func(string) (bool, string), listed bool) {
	t.Helper()
	ok, why := probe(dev)
	switch {
	case ok:
	case listed:
		t.Fatalf("NO DEVICE: %s is listed (%s=%s) and does not open: %s", dev, testdevices.Var,
			strings.Join(testdevices.Selection(), ","), why)
	default:
		t.Skipf("NO DEVICE: %s is not reachable on this host: %s -- this gate proved nothing", dev, why)
	}
}

// listed reports whether the run's device selection names a device dev may
// open: one of its API, or any for gpu, auto and all.
func listed(dev string) bool {
	api, _, _ := strings.Cut(dev, ":")
	for _, s := range testdevices.Selection() {
		if sa, _, _ := strings.Cut(s, ":"); sa == api || api == "gpu" || api == "auto" || api == "all" {
			return true
		}
	}
	return false
}

// TestRequireDeviceSkipsOnlyWithoutADevice: an absent device skips with its
// reason; a present one does not skip, so a gate after it can fail; a listed
// one that does not open fails rather than skips.
func TestRequireDeviceSkipsOnlyWithoutADevice(t *testing.T) {
	absent := func(string) (bool, string) { return false, "none here" }
	present := func(string) (bool, string) { return true, "" }
	t.Run("absent", func(t *testing.T) {
		r := &skipRecorder{TB: t}
		requireDeviceWith(r, "cuda:0", absent, false)
		if !strings.Contains(r.skipped, "none here") || r.failed != "" {
			t.Fatalf("an absent device did not skip with its reason: skipped %q, failed %q", r.skipped, r.failed)
		}
	})
	t.Run("listed", func(t *testing.T) {
		r := &skipRecorder{TB: t}
		requireDeviceWith(r, "cuda:0", absent, true)
		if !strings.Contains(r.failed, "none here") || r.skipped != "" {
			t.Fatalf("a listed device that does not open did not fail with its reason: failed %q, skipped %q", r.failed, r.skipped)
		}
	})
	t.Run("present", func(t *testing.T) {
		requireDeviceWith(t, "cuda:0", present, true)
		if t.Skipped() {
			t.Fatal("a present device skipped")
		}
	})
}

// skipRecorder records a Skipf or a Fatalf instead of ending the test.
type skipRecorder struct {
	testing.TB
	skipped, failed string
}

func (r *skipRecorder) Helper() {}

func (r *skipRecorder) Skipf(format string, args ...any) { r.skipped = fmt.Sprintf(format, args...) }

func (r *skipRecorder) Fatalf(format string, args ...any) { r.failed = fmt.Sprintf(format, args...) }
