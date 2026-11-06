//go:build amd64 && jitllmtest

package engine

import (
	"github.com/jitllm/jitllm/engine/device/cpudev"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestEveryModelRunsPreVNNI decodes and prefills every model under 2 GiB on a
// CPU device without VNNI (cpu:novnni), as a pre-VNNI x86 runs. The tiled prefill kernel has no
// VEX form, so prefill uses the fused per-token kernel; a shape the pre-VNNI
// emitters cannot serve is an error, so running every model is the assertion.
func TestEveryModelRunsPreVNNI(t *testing.T) {
	if cpudev.Target(cpu.Probe().ISA(), 1).ISA&uint64(cpu.ISAVNNI) == 0 {
		t.Skip("this host is already pre-VNNI; the forced arm would not differ")
	}
	paths := modelFiles()
	if len(paths) == 0 {
		t.Skip("no models present")
	}
	sort.Strings(paths)

	ran := 0
	for _, p := range paths {
		if fi, err := os.Stat(p); err != nil || fi.Size() > 2<<30 {
			continue
		}
		t.Run(filepath.Base(p), func(t *testing.T) {
			instOpts := []Option{WithDevices("cpu:novnni")}
			m, err := model.Open(jlmOf(t, p))
			if err != nil {
				t.Skipf("cannot load: %v", err)
			}
			defer m.Close()
			noVNNI(t, m, instOpts...)

			ids := make([]int32, 16)
			for i := range ids {
				ids[i] = int32(1 + i)
			}
			s := testInstance(t, m, 64, instOpts...)
			defer s.Close()
			if _, err := s.Prefill(ids); err != nil {
				t.Fatal(err)
			}
			d := testInstance(t, m, 64, instOpts...)
			defer d.Close()
			for i := 0; i < 3; i++ {
				if _, err := d.Forward(int32(1 + i)); err != nil {
					t.Fatal(err)
				}
			}
			ran++
		})
	}
	if ran == 0 {
		t.Fatal("no model ran -- this gate proved nothing")
	}
	t.Logf("%d model(s) prefilled and decoded on a CPU without VNNI: a decline "+
		"is an error now, so running is the assertion", ran)
}

// TestExactGatesRunPreVNNI runs the gates that hold batch, prefill and decode
// bit-identical, and the CPU arms of the decode allocation gate, on a CPU
// device without VNNI (cpu:novnni): the kernel family a pre-VNNI host runs, which no gate here
// otherwise reaches on a VNNI host (RULE 11d). Two of them failed on a pre-VNNI
// host and passed here -- tinyllama's batch-vs-decode at NMSE 5.87e-14 and
// olmoe's decode at two allocations a token -- and both turned out to be the
// pool's WIDTH (14 workers against 6) rather than the kernel family; this arm
// is what showed the family exact.
func TestExactGatesRunPreVNNI(t *testing.T) {
	if cpudev.Target(cpu.Probe().ISA(), 1).ISA&uint64(cpu.ISAVNNI) == 0 {
		t.Skip("this host is already pre-VNNI; the plain gates run this family")
	}
	dev := WithDevices("cpu:novnni")
	t.Run("ForwardBatchFaultsItsBlocksIn", func(t *testing.T) { forwardBatchFaultsGate(t, dev) })
	t.Run("BatchMatchesForward", func(t *testing.T) { batchMatchesForward(t, dev) })
	t.Run("RaggedBatchMatchesForward", func(t *testing.T) { raggedBatchMatchesForward(t, dev) })
	t.Run("PrefillMatchesForward", func(t *testing.T) { prefillMatchesForward(t, dev) })
	t.Run("PrefillSeqMatchesForward", func(t *testing.T) { prefillSeqMatchesForward(t, dev) })
	t.Run("StepRunsOnTheCPUSharesOnePass", func(t *testing.T) { stepRunsOnTheCPUSharesOnePass(t, dev) })
	t.Run("DecodeDoesNotAllocate", func(t *testing.T) {
		for _, name := range []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf",
			"olmoe-1b-7b-0924-instruct-Q4_K_M.gguf", "qwen35/Qwen3.5-0.8B-Q4_K_M.gguf"} {
			t.Run(name, func(t *testing.T) {
				instOpts := []Option{dev}
				m, err := model.Open(jlmOf(t, testmodels.Path(name)))
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				noVNNI(t, m, instOpts...)
				decodeAllocs(t, m, nil, instOpts...)
			})
		}
	})
}

// noVNNI fails t where the CPU device an instance of m opens with io states
// VNNI: the arm would compose the VNNI forms the plain gates already run and
// prove nothing.
func noVNNI(t *testing.T, m *model.Model, io ...Option) {
	t.Helper()
	s := testInstance(t, m, 8, io...)
	defer s.Close()
	if s.cpuISA()&cpu.ISAVNNI != 0 {
		t.Fatal("a model opened on cpu:novnni has a CPU device stating VNNI")
	}
}
