package tier

import (
	"fmt"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/backend/vulkan"
	"github.com/jitllm/jitllm/internal/testdevices"
)

// amdCard is the identity one AMD card reports under both HIP and Vulkan: the
// PCI location as RADV's deviceUUID (hip.Device.VulkanUUID).
const amdCard = "00000000030000000000000000000000"

// amdHost stands one AMD card behind the constructors: Vulkan lists it, and
// HIP lists it nHIP times over (0 is a host without ROCm, where the real
// backend.OpenHIPWith under a directory holding no ROCm answers instead).
func amdHost(t *testing.T, nHIP int) *[]*fakeDev {
	t.Helper()
	var opened []*fakeDev
	saved := []any{cudaCount, openCUDA, vulkanDevices, openVulkan, openMetal, hipCount, openHIP}
	t.Cleanup(func() {
		cudaCount = saved[0].(func() (int, error))
		openCUDA = saved[1].(func(int, backend.Opts) (backend.Device, error))
		vulkanDevices = saved[2].(func() ([]vulkan.Info, error))
		openVulkan = saved[3].(func(string, backend.Opts) (backend.Device, error))
		openMetal = saved[4].(func(backend.Opts) (backend.Device, error))
		hipCount = saved[5].(func(backend.HIPConfig) int)
		openHIP = saved[6].(func(int, backend.Opts) (backend.Device, error))
	})
	cudaCount = func() (int, error) { return 0, nil }
	openCUDA = func(ord int, _ backend.Opts) (backend.Device, error) {
		return nil, fmt.Errorf("fake: no CUDA device %d", ord)
	}
	openMetal = func(backend.Opts) (backend.Device, error) { return nil, fmt.Errorf("fake: no Metal") }
	vulkanDevices = func() ([]vulkan.Info, error) {
		return []vulkan.Info{{Index: 0, Name: "AMD Radeon RX 7900 XTX (RADV NAVI31)", Compute: true}}, nil
	}
	openVulkan = func(sel string, _ backend.Opts) (backend.Device, error) {
		if sel != "" && sel != "0" {
			return nil, fmt.Errorf("fake: no Vulkan device %q", sel)
		}
		d := &fakeDev{name: "AMD Radeon RX 7900 XTX (RADV NAVI31)", api: "spirv", uuid: amdCard,
			free: boxCardFree, total: boxCardTotal}
		opened = append(opened, d)
		return ordFake{d, 0}, nil
	}
	if nHIP == 0 {
		// The real constructors, pointed at a directory with no ROCm in it.
		hipCount, openHIP = backend.HIPCount, backend.OpenHIPWith
		return &opened
	}
	hipCount = func(backend.HIPConfig) int { return nHIP }
	openHIP = func(ord int, _ backend.Opts) (backend.Device, error) {
		if ord < 0 || ord >= nHIP {
			return nil, fmt.Errorf("hip:%d requested; %d HIP devices", ord, nHIP)
		}
		uuid := amdCard
		if ord > 0 {
			uuid = fmt.Sprintf("%032x", ord)
		}
		d := &fakeDev{name: fmt.Sprintf("#%d AMD Radeon RX 7900 XTX (gfx1100)", ord), api: "amdgcn",
			uuid: uuid, free: boxCardFree, total: boxCardTotal}
		opened = append(opened, d)
		return ordFake{d, ord}, nil
	}
	return &opened
}

func resolve(t *testing.T, spec, rocm string) ([]backend.Device, []string, error) {
	t.Helper()
	es, err := ParseDevices(spec)
	if err != nil {
		t.Fatal(err)
	}
	o := openOpts{spec: spec, kb: knobs{tune: TuneOff}}
	o.hip.Path = rocm
	devs, _, names, err := openEntries(es, o)
	return devs, names, err
}

// TestHIPIsTheDefaultForAnAMDCardWithROCm: with ROCm present the card is
// hip:0, and the default set keeps it over the same card's Vulkan entry.
func TestHIPIsTheDefaultForAnAMDCardWithROCm(t *testing.T) {
	for _, spec := range []string{"all", "gpu"} {
		amdHost(t, 1)
		devs, _, err := resolve(t, spec, "")
		if err != nil {
			t.Fatal(err)
		}
		if len(devs) != 1 || devs[0].API() != "amdgcn" {
			t.Errorf("-devices %s on one AMD card with ROCm: %v, want the one hip device", spec, labels(devs))
		}
	}
	amdHost(t, 2)
	devs, names, err := resolve(t, "hip:1", "")
	if err != nil || len(devs) != 1 || names[0] != "hip:1" {
		t.Errorf("hip:1: %v %q %v", labels(devs), names, err)
	}
	if _, _, err := resolve(t, "rocm:0,vulkan:0", ""); err != nil {
		t.Errorf("hip and vulkan named together: %v", err)
	}
}

// TestNoROCmIsVulkanAndNotAnError: without ROCm the card is its Vulkan entry
// and opening the default set is not an error.
func TestNoROCmIsVulkanAndNotAnError(t *testing.T) {
	amdHost(t, 0)
	devs, _, err := resolve(t, "all", t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if len(devs) != 1 || devs[0].API() != "spirv" {
		t.Errorf("-devices all with no ROCm: %v, want the card through Vulkan", labels(devs))
	}
}

// TestAnExplicitHIPRequestFailsNamingTheCause: hip:N named with no ROCm, or
// past the device count, is an error naming which -- never the card through
// Vulkan, never the host.
func TestAnExplicitHIPRequestFailsNamingTheCause(t *testing.T) {
	// The no-ROCm half reaches the real backend.OpenHIPWith, which a device
	// selection refuses before ROCm is looked for: the error would name the
	// selection, not the cause, and the open is counted as a violation.
	t.Run("no-rocm", func(t *testing.T) {
		testdevices.SkipSelected(t, "it names hip:0 through the real backend on a box that has none")
		amdHost(t, 0)
		dir := t.TempDir()
		for _, spec := range []string{"hip:0", "hip"} {
			devs, _, err := resolve(t, spec, dir)
			if err == nil || !strings.Contains(err.Error(), "hip:0 requested but ROCm was not found (searched "+dir+")") {
				t.Errorf("-devices %s with no ROCm: %v, %v", spec, labels(devs), err)
			}
		}
	})
	amdHost(t, 2)
	if devs, _, err := resolve(t, "hip:3", ""); err == nil || !strings.Contains(err.Error(), "hip:3 requested; 2 HIP devices") {
		t.Errorf("hip:3 on two devices: %v, %v", labels(devs), err)
	}
}
