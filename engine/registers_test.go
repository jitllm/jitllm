package engine

import (
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// hotFloor is the occupancy a hot kernel family must keep on a device, as a
// count the driver reports rather than a timing (RULE 4).
//
// The streaming kernels (matvec, norms, quantize, activations) are bound by
// memory and hide its latency with resident threads, so they keep at least
// half the multiprocessor. The tiled ones (the batched GEMM and attention)
// trade occupancy for registers that hold a tile, which is the point of a
// tile; they keep at least two resident workgroups, so one's barrier or
// global load is covered by the other.
type hotFloor struct {
	prefixes  []string
	occupancy float64 // minimum ResidentThreads/MaxThreads
	groups    int     // minimum resident workgroups
}

func hotFloors() []hotFloor {
	return []hotFloor{
		{prefixes: []string{"matvec_", "rmsnorm", "normpart", "normapply", "quantize", "act"}, occupancy: 0.5},
		{prefixes: []string{"gemm", "flash"}, groups: 2},
	}
}

// hotKernelFaults is every hot kernel in ks that ran and spills or sits below its
// family's floor, and how many hot kernels it checked.
func hotKernelFaults(ks []backend.KernelResources) (faults []string, checked int) {
	for _, k := range ks {
		if !k.Ran {
			continue // compiled and passed over: a tile refused for spilling
		}
		for _, f := range hotFloors() {
			hot := false
			for _, p := range f.prefixes {
				hot = hot || strings.HasPrefix(k.Name, p)
			}
			if !hot {
				continue
			}
			checked++
			if k.SpillBytes != 0 {
				faults = append(faults, fmt.Sprintf("%s spills %d bytes per thread at %d registers",
					k.Name, k.SpillBytes, k.Regs))
			}
			if f.occupancy > 0 && k.Occupancy() < f.occupancy {
				faults = append(faults, fmt.Sprintf("%s holds %.0f%% of a multiprocessor (%d regs, %d B shared), floor %.0f%%",
					k.Name, 100*k.Occupancy(), k.Regs, k.SharedBytes, 100*f.occupancy))
			}
			if f.groups > 0 && k.Resident < f.groups {
				faults = append(faults, fmt.Sprintf("%s fits %d workgroup(s) per multiprocessor (%d regs, %d B shared), floor %d",
					k.Name, k.Resident, k.Regs, k.SharedBytes, f.groups))
			}
		}
	}
	return faults, checked
}

// runForRegisters runs decode and the batched prefill on a CUDA tier opened
// with opts and returns what each kernel it compiled cost the card.
func runForRegisters(t *testing.T, m *model.Model, io []Option, opts ...gpu.Option) []backend.KernelResources {
	t.Helper()
	g, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(testdevices.Like(t, "cuda:0")), gpu.WithDeviceTune(gpu.TuneOff)}, opts...)...)
	if err != nil || g == nil {
		noDevice(t, "CUDA device", err)
	}
	defer g.Close()
	st := testInstance(t, m, 64, io...)
	defer st.Close()
	if err := st.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	if st.GPULayers() == 0 {
		t.Fatal("the card took no blocks: no kernel this gate means to check ran")
	}
	prompt := m.Vocab.Encode(strings.Repeat("The capital of France is Paris. ", 5), true)
	lg, err := st.Prefill(prompt)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 4; i++ {
		if lg, err = st.Forward(Greedy(lg)); err != nil {
			t.Fatal(err)
		}
	}
	rs := g.KernelResources()
	if len(rs) != 1 || !rs[0].Known {
		t.Fatalf("the CUDA tier did not report its kernels' resources: %+v", rs)
	}
	return rs[0].Kernels
}

// TestHotKernelsKeepTheirRegisters holds the hot device kernels a model runs --
// decode's matvecs, norms and attention, and the batched prefill's GEMM and
// flash attention -- to no register spill and a stated occupancy floor, read
// from the driver (cuFuncGetAttribute, cuOccupancyMaxActiveBlocksPerMultiprocessor).
// It is run against a violation first: the same model with every kernel capped
// at 32 registers must fail it.
func TestHotKernelsKeepTheirRegisters(t *testing.T) {
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS) -- this gate proved nothing", err)
	}
	m, err := model.Open(jlmOf(t, p))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	t.Run("cpu", func(t *testing.T) {
		st := testInstance(t, m, 64)
		defer st.Close()
		lg, err := st.Prefill(m.Vocab.Encode(strings.Repeat("The capital of France is Paris. ", 5), true))
		if err != nil {
			t.Fatal(err)
		}
		if _, err = st.Forward(Greedy(lg)); err != nil {
			t.Fatal(err)
		}
		ks := st.eng.cpu.Kernels(m.Id)
		if len(ks) == 0 {
			t.Fatal("the model holds no composed CPU kernel: the gate checked nothing")
		}
		var spill []string
		for _, k := range ks {
			if k.Bytes == 0 || k.Forms == 0 {
				t.Errorf("%s is recorded with %d bytes over %d forms", k.Name, k.Bytes, k.Forms)
			}
			if k.Vec > k.File {
				t.Errorf("%s names %d vector registers of a %d-register file", k.Name, k.Vec, k.File)
			}
			if k.Stack > 0 {
				spill = append(spill, fmt.Sprintf("%s (%d)", k.Name, k.Stack))
			}
		}
		t.Logf("%d composed CPU kernels; addressing the stack: %s", len(ks), strings.Join(spill, ", "))
	})
	t.Run("violation", func(t *testing.T) {
		faults, checked := hotKernelFaults(runForRegisters(t, m, nil, gpu.WithCUDAMaxRegs(32)))
		if checked == 0 {
			t.Fatal("no hot kernel was compiled: the gate checked nothing")
		}
		spills := 0
		for _, f := range faults {
			if strings.Contains(f, "spills") {
				spills++
			}
		}
		if spills == 0 {
			t.Fatalf("every kernel capped at 32 registers and the gate saw no spill among %d hot kernels", checked)
		}
		t.Logf("capped at 32 registers: %d faults among %d hot kernels, %d spilling, e.g. %s",
			len(faults), checked, spills, faults[0])
	})
	t.Run("shipped", func(t *testing.T) {
		ks := runForRegisters(t, m, nil)
		faults, checked := hotKernelFaults(ks)
		if checked == 0 {
			t.Fatal("no hot kernel was compiled: the gate checked nothing")
		}
		for _, f := range faults {
			t.Error(f)
		}
		t.Logf("%d hot kernels of %d compiled checked", checked, len(ks))
	})
}

// TestHotKernelFloorsFire runs the occupancy floors against kernels that break
// them -- a register cap cannot force an occupancy below a floor without also
// spilling, so the floors are shown to fire here on the driver's own fields.
func TestHotKernelFloorsFire(t *testing.T) {
	ks := []backend.KernelResources{
		{Name: "matvec_q4_k_2048x2048_s32", Ran: true, Group: 128, Regs: 128, Resident: 3, ResidentThreads: 384, MaxThreads: 1536},
		{Name: "flashkvpaged", Ran: true, Group: 128, Regs: 255, Resident: 1, ResidentThreads: 128, MaxThreads: 1536},
		{Name: "add", Ran: true, Group: 128, Regs: 255, Resident: 1, ResidentThreads: 128, MaxThreads: 1536},
	}
	faults, checked := hotKernelFaults(ks)
	if checked != 2 || len(faults) != 2 {
		t.Fatalf("checked %d hot kernels with %d faults, want 2 and 2 (a cold kernel is not held to a floor): %q",
			checked, len(faults), faults)
	}
}
