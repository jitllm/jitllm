package model

import (
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/cpu"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
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

var hotFloors = []hotFloor{
	{prefixes: []string{"matvec_", "rmsnorm", "normpart", "normapply", "quantize", "act"}, occupancy: 0.5},
	{prefixes: []string{"gemm", "flash"}, groups: 2},
}

// hostStagers are the host emitters that address the stack by design: the
// k-quant matvecs keep loop-invariant bases and a block's unpacked scales in
// the System V red zone (jit_old/cpu/matvec.go, "The loop-invariant bases live in
// the RED ZONE"), and the SSE packed tile stages its scale planes there
// (jit_old/cpu/sse_packed.go). Every other host kernel holds its values in
// registers, and one that starts addressing the stack has outgrown the file.
// The set is every emitter in jit_old/cpu that writes At(RSP, ...).
var hostStagers = map[string]bool{
	"cpu.emitMatVec4":              true,
	"cpu.emitQ3KMatVec":            true,
	"cpu.emitQ6KMatVec":            true,
	"cpu.emitKQuantMatVec":         true,
	"cpu.EmitPackedMatVecSSE":      true,
	"cpu.EmitPackedMatVecFusedSSE": true,
}

// hotKernelFaults is every hot kernel in ks that ran and spills or sits below its
// family's floor, and how many hot kernels it checked.
func hotKernelFaults(ks []backend.KernelResources) (faults []string, checked int) {
	for _, k := range ks {
		if !k.Ran {
			continue // compiled and passed over: a tile refused for spilling
		}
		for _, f := range hotFloors {
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
func runForRegisters(t *testing.T, m *Model, opts ...tier.Option) []backend.KernelResources {
	t.Helper()
	g, err := openTier(t, append([]tier.Option{tier.WithDevices(testdevices.Like(t, "cuda:0")), tier.WithDeviceTune(tier.TuneOff)}, opts...)...)
	if err != nil || g == nil {
		noDevice(t, "CUDA device", err)
	}
	defer g.Close()
	st := m.NewState(64)
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
	m, err := Open(jlmOf(t, p), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	t.Run("host", func(t *testing.T) {
		st := m.NewState(64)
		defer st.Close()
		lg, err := st.Prefill(m.Vocab.Encode(strings.Repeat("The capital of France is Paris. ", 5), true))
		if err != nil {
			t.Fatal(err)
		}
		if _, err = st.Forward(Greedy(lg)); err != nil {
			t.Fatal(err)
		}
		ks := cpu.KernelRegisters()
		if len(ks) == 0 {
			t.Fatal("no host kernel was recorded: the gate checked nothing")
		}
		for _, k := range ks {
			if k.VecUsed > k.File {
				t.Errorf("%s names %d vector registers of a %d-register file", k.Emitter, k.VecUsed, k.File)
			}
			if k.StackRefs > 0 && !hostStagers[k.Emitter] {
				t.Errorf("%s addresses the stack %d times: a hand-allocated kernel that outgrew %d %s registers",
					k.Emitter, k.StackRefs, k.File, k.ISA)
			}
		}
		t.Logf("%d host emitters checked", len(ks))
	})
	t.Run("violation", func(t *testing.T) {
		faults, checked := hotKernelFaults(runForRegisters(t, m, tier.WithCUDAMaxRegs(32)))
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
		ks := runForRegisters(t, m)
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
