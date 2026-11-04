package server

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
)

// TestAnAMDCardUnderHIPAndVulkanCountsAsHIP: one AMD card listed as vulkan:0
// and hip:0 (the PCI location both report) is one device, and the entry that
// counts is hip:0; pbBackend names it.
func TestAnAMDCardUnderHIPAndVulkanCountsAsHIP(t *testing.T) {
	amd := "uuid:00000000030000000000000000000000"
	devs := []DeviceInfo{
		{ID: "vulkan:0", Backend: "vulkan", Kind: KindDiscrete, PhysicalID: amd, Available: true},
		{ID: "hip:0", Backend: "hip", Kind: KindDiscrete, PhysicalID: amd, Available: true},
	}
	if n := markSameDevice(devs); n != 1 {
		t.Fatalf("linked %d entries, want 1", n)
	}
	if devs[0].SameDeviceAs != "hip:0" || devs[1].SameDeviceAs != "" {
		t.Errorf("vulkan:0 -> %q, hip:0 -> %q; want vulkan:0 to point at hip:0",
			devs[0].SameDeviceAs, devs[1].SameDeviceAs)
	}
	if got := pbBackend("hip").String(); got != "BACKEND_HIP" {
		t.Errorf("pbBackend(hip) = %s", got)
	}
}

// TestNoROCmListsNoHIPDevice: a ROCm directory with nothing in it lists no
// hip:N entry and is not an error.
func TestNoROCmListsNoHIPDevice(t *testing.T) {
	devs, err := probeDevices(t.TempDir(), testdevices.Admit())
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range devs {
		if d.Backend == "hip" {
			t.Errorf("%s listed with no ROCm", d.ID)
		}
	}
}
