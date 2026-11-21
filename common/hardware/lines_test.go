package hardware

import (
	"strings"
	"testing"
)

func TestDeviceNoteSaysSharedMemoryIsSubtractedAndNotAdded(t *testing.T) {
	shared := DeviceNote(GPU{Unified: true})
	if !strings.Contains(shared, "shares system memory") {
		t.Errorf("DeviceNote(unified) = %q, want it to say the memory is shared", shared)
	}

	discrete := DeviceNote(GPU{Unified: false})
	if strings.Contains(discrete, "shares") {
		t.Errorf("DeviceNote(discrete) = %q, want no shared-memory claim", discrete)
	}
}

func TestDeviceLineReportsAnUnknownMemoryAsUnknownRatherThanZero(t *testing.T) {
	got := DeviceLine(GPU{API: "msl", Name: "a chip", MemNote: "not reported by this backend"})
	if !strings.Contains(got, "not reported") {
		t.Errorf("DeviceLine = %q, want the backend's own refusal", got)
	}
	if strings.Contains(got, "0 B of 0 B") {
		t.Errorf("DeviceLine = %q, want no zero-byte card", got)
	}
}

func TestVulkanLineCarriesTheIndexThatNamesTheDevice(t *testing.T) {
	got := VulkanLine(VulkanDev{Index: 1, Name: "Iris Xe", Type: "integrated GPU",
		Compute: true, Unified: true, Subgroups: [2]int{8, 32}, Promises32: false})
	if !strings.HasPrefix(got, "vulkan:1") {
		t.Errorf("VulkanLine = %q, want it to start with the spec that selects it", got)
	}
	if !strings.Contains(got, "no 32-lane guarantee") {
		t.Errorf("VulkanLine = %q, want the subgroup caveat", got)
	}
}

// The clipboard report carries the composer's form of the CPU and the
// instruction sets it reads.
func TestReportTextPrintsTheComposersForm(t *testing.T) {
	r := Report{
		Probed: true, GOOS: "linux", GOARCH: "amd64", GoVersion: "go1.27.1",
		CPU: "a cpu", Logical: 8, PCores: []int{0, 2}, SMT: []int{0, 1, 2, 3}, Decode: []int{0, 2},
		Arch: "amd64", ISA: "sse2, avx2", Form: "x86, wide passes",
		GPUs: []GPU{{API: "ptx", Name: "a card", Slots: 30720, Free: 1 << 30, Total: 4 << 30}},
	}
	got := ReportText(r)
	for _, want := range []string{"composer  x86, wide passes", "isa sse2, avx2", "a card", "linux/amd64"} {
		if !strings.Contains(got, want) {
			t.Errorf("ReportText missing %q in:\n%s", want, got)
		}
	}
}

func TestReportTextSaysSoBeforeAProbeRatherThanPrintingZeroes(t *testing.T) {
	got := ReportText(Report{})
	if !strings.Contains(got, "not been probed") {
		t.Errorf("ReportText(unprobed) = %q, want it to say there is no probe yet", got)
	}
}
