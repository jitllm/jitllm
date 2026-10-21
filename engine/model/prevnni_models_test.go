//go:build amd64 && jitllmtest

package model

import (
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestEveryModelRunsPreVNNI decodes and prefills every model under 2 GiB with
// VNNI forced absent, as a pre-VNNI x86 runs. The tiled prefill kernel has no
// VEX form, so prefill uses the fused per-token kernel; a shape the pre-VNNI
// emitters cannot serve is an error, so running every model is the assertion.
func TestEveryModelRunsPreVNNI(t *testing.T) {
	if cpu.HostDotKind() == cpu.DotVEX {
		t.Skip("this host is already pre-VNNI; the forced arm would not differ")
	}
	paths := modelFiles()
	if len(paths) == 0 {
		t.Skip("no models present")
	}
	sort.Strings(paths)

	// Force before any model opens: nn.NewJIT builds its kernels at
	// construction.
	old := cpu.ForceNoVNNIForTest(true)
	defer cpu.ForceNoVNNIForTest(old)
	if cpu.HostDotKind() != cpu.DotVEX {
		t.Fatal("ForceNoVNNIForTest did not make HostDotKind report DotVEX, so " +
			"this gate would test the VNNI path twice and prove nothing")
	}

	ran := 0
	for _, p := range paths {
		if fi, err := os.Stat(p); err != nil || fi.Size() > 2<<30 {
			continue
		}
		t.Run(filepath.Base(p), func(t *testing.T) {
			m, err := Open(jlmOf(t, p))
			if err != nil {
				t.Skipf("cannot load: %v", err)
			}
			defer m.Close()

			ids := make([]int32, 16)
			for i := range ids {
				ids[i] = int32(1 + i)
			}
			s := m.NewState(64)
			defer s.Close()
			if _, err := s.Prefill(ids); err != nil {
				t.Fatal(err)
			}
			d := m.NewState(64)
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
	t.Logf("%d model(s) prefilled and decoded with VNNI forced absent: a decline "+
		"is an error now, so running is the assertion", ran)
}

// TestExactGatesRunPreVNNI runs the gates that hold batch, prefill and decode
// bit-identical, and the host arms of the decode allocation gate, with VNNI
// forced absent: the kernel family a pre-VNNI host runs, which no gate here
// otherwise reaches on a VNNI host (RULE 11d). Two of them failed on a pre-VNNI
// host and passed here -- tinyllama's batch-vs-decode at NMSE 5.87e-14 and
// olmoe's decode at two allocations a token -- and both turned out to be the
// pool's WIDTH (14 workers against 6) rather than the kernel family; this arm
// is what showed the family exact, and the width cases are gated where they
// live (nn.TestGEMMExactDecodeSumsTheWholeK, the host-ksliced arm).
func TestExactGatesRunPreVNNI(t *testing.T) {
	if cpu.HostDotKind() == cpu.DotVEX {
		t.Skip("this host is already pre-VNNI; the plain gates run this family")
	}
	old := cpu.ForceNoVNNIForTest(true)
	defer cpu.ForceNoVNNIForTest(old)
	if cpu.HostDotKind() != cpu.DotVEX {
		t.Fatal("ForceNoVNNIForTest did not make HostDotKind report DotVEX, so " +
			"this gate would test the VNNI path twice and prove nothing")
	}
	t.Run("ForwardBatchFaultsItsBlocksIn", TestForwardBatchFaultsItsBlocksIn)
	t.Run("BatchMatchesForward", TestBatchMatchesForward)
	t.Run("RaggedBatchMatchesForward", TestRaggedBatchMatchesForward)
	t.Run("PrefillMatchesForward", TestPrefillMatchesForward)
	t.Run("PrefillSeqMatchesForward", TestPrefillSeqMatchesForward)
	t.Run("StepRunsOnTheHostSharesOnePass", TestStepRunsOnTheHostSharesOnePass)
	t.Run("DecodeDoesNotAllocate", func(t *testing.T) {
		for _, name := range []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf",
			"olmoe-1b-7b-0924-instruct-Q4_K_M.gguf", "qwen35/Qwen3.5-0.8B-Q4_K_M.gguf"} {
			t.Run(name, func(t *testing.T) {
				p := jlmOf(t, testmodels.Path(name))
				for _, arm := range []struct {
					name   string
					opts   []Option
					sliced bool
				}{
					{"host", []Option{noTune}, false},
					{"host-ksliced", []Option{noTune, WithJITOptions(nn.WithFusedKSlices(2))}, true},
				} {
					t.Run(arm.name, func(t *testing.T) {
						m, err := Open(p, arm.opts...)
						if err != nil {
							t.Fatal(err)
						}
						defer m.Close()
						if dw := decodeAllocs(t, m, nil); arm.sliced && dw.ksliced == 0 {
							t.Fatal("no matvec of the window was split over k: this arm measured the plain host decode twice")
						}
					})
				}
			})
		}
	})
}
