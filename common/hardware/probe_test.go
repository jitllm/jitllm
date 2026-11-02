package hardware_test

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/common/hardware"
)

// The spec picker is only worth having if it agrees with the engine, so the
// validator is gpu.ParseDevices itself. These gates cover the summary it
// builds and what it refuses.

func TestValidateSpecAcceptsTheGrammarAndSaysWhatItAsksFor(t *testing.T) {
	cases := []struct {
		spec string
		want []string
	}{
		{"auto", []string{"auto"}},
		{"cpu", []string{"cpu", "no device"}},
		{"cuda:0=3G", []string{"ptx:0", "3.00 GiB"}},
		{"cuda:0,vulkan:1", []string{"ptx:0", "spirv:1"}},
	}
	for _, c := range cases {
		got, err := hardware.ValidateSpec(c.spec)
		if err != nil {
			t.Fatalf("ValidateSpec(%q): %v", c.spec, err)
		}
		for _, w := range c.want {
			if !strings.Contains(got, w) {
				t.Errorf("ValidateSpec(%q) = %q, want it to contain %q", c.spec, got, w)
			}
		}
	}
}

func TestValidateSpecRefusesWhatTheEngineWouldRefuse(t *testing.T) {
	for _, spec := range []string{"banana", "cuda:", "auto:0", "cuda:0=nonsense"} {
		if got, err := hardware.ValidateSpec(spec); err == nil {
			t.Errorf("ValidateSpec(%q) = %q, want an error", spec, got)
		}
	}
}

func TestValidateBudgetEmptyIsTheDefaultAndNotAnError(t *testing.T) {
	got, err := hardware.ValidateBudget("  ", 8<<30, 6<<30)
	if err != nil {
		t.Fatalf("empty budget: %v", err)
	}
	if !strings.Contains(got, "default") || !strings.Contains(got, "8.00 GiB") {
		t.Errorf("ValidateBudget(\"\") = %q, want the default and its size", got)
	}
}

// The warning is the point of the field: a budget over the collector's usable
// heap makes the engine spend much of its CPU in GC. The under-cap arm makes
// this a gate rather than a constant.
func TestValidateBudgetWarnsOnlyWhenItIsOverTheCollectorsCap(t *testing.T) {
	over, err := hardware.ValidateBudget("9G", 8<<30, 7<<30)
	if err != nil {
		t.Fatalf("9G: %v", err)
	}
	if !strings.Contains(over, "collector") {
		t.Errorf("ValidateBudget(9G, cap 7 GiB) = %q, want the collector warning", over)
	}

	under, err := hardware.ValidateBudget("4G", 8<<30, 7<<30)
	if err != nil {
		t.Fatalf("4G: %v", err)
	}
	if strings.Contains(under, "collector") {
		t.Errorf("ValidateBudget(4G, cap 7 GiB) = %q, want NO warning", under)
	}
	if !strings.Contains(under, "4.00 GiB") {
		t.Errorf("ValidateBudget(4G) = %q, want the parsed size", under)
	}
}

func TestValidateBudgetRefusesSomethingThatIsNotAByteCount(t *testing.T) {
	if got, err := hardware.ValidateBudget("plenty", 8<<30, 0); err == nil {
		t.Errorf("ValidateBudget(\"plenty\") = %q, want an error", got)
	}
}

// The two kernel columns must stay two columns: on a pre-VNNI host the
// row-major family declines every format while the packed family is fully
// generated, so copying one into the other misreports the machine.
func TestMachineKeepsTheRowMajorAndPackedColumnsApart(t *testing.T) {
	r := hardware.Report{
		Probed: true,
		Arch:   "amd64",
		Kernels: []hardware.Kernel{
			{Format: "Q4_K", Native: false, NativeNote: "none", Packed: true, PackedNote: "vex"},
			{Format: "Q6_K", Native: true, NativeNote: "pack4 accs2", Packed: true, PackedNote: "vnni"},
		},
		GPUs: []hardware.GPU{{API: "ptx", Name: "a card", Slots: 30720, Free: 1 << 30, Total: 4 << 30, Unified: false}},
	}
	m := r.Machine()

	if len(m.Kernels) != 2 {
		t.Fatalf("Machine().Kernels = %d rows, want 2", len(m.Kernels))
	}
	if m.Kernels[0].RowMajor {
		t.Error("Q4_K row-major should be false: the packed column must not leak into it")
	}
	if !m.Kernels[0].Packed {
		t.Error("Q4_K packed should be true: the row-major column must not leak into it")
	}
	if !m.Kernels[1].RowMajor || !m.Kernels[1].Packed {
		t.Error("Q6_K should be generated in both families")
	}
}

// Mem is the device-local heap, and it is the total rather than the free
// figure: a report that quietly published free under a field named for the heap
// would read as a much smaller card.
func TestMachineCarriesTheDeviceHeapAndNotTheFreeFigure(t *testing.T) {
	r := hardware.Report{
		Probed: true,
		GPUs:   []hardware.GPU{{API: "ptx", Name: "a card", Free: 1 << 30, Total: 4 << 30, Unified: true}},
	}
	m := r.Machine()
	if len(m.GPUs) != 1 {
		t.Fatalf("Machine().GPUs = %d, want 1", len(m.GPUs))
	}
	if m.GPUs[0].Mem != 4<<30 {
		t.Errorf("GPUs[0].Mem = %d, want the 4 GiB heap", m.GPUs[0].Mem)
	}
	if !m.GPUs[0].Unified {
		t.Error("GPUs[0].Unified lost: a shared heap must be SUBTRACTED from the host budget, and a caller cannot know to do that if the flag does not survive")
	}
}

func TestFormatNamesMatchesTheFormatList(t *testing.T) {
	names := hardware.FormatNames()
	if len(names) != len(hardware.Formats) {
		t.Fatalf("FormatNames() = %d, Formats = %d", len(names), len(hardware.Formats))
	}
	for i, n := range names {
		if n == "" {
			t.Errorf("FormatNames()[%d] is empty", i)
		}
	}
}
