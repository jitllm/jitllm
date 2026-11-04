//go:build arm64 && jitllmtest

package model

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// arm64 without FEAT_DotProd at the model level (engine/device/cpu/sdotemu.go): a
// Cortex-A53/A57/A72/A73 widens every int8 dot to SMULL/ADDP/SADDLP. Each gate
// forces the probe before a model opens (the CPU device's forms are composed
// when an instance is made) and asserts the device the composer formed for
// states no FEAT_DotProd, the bit its arm64 lowering widens by.

// nodotForce makes the probe answer "absent" for the rest of t.
func nodotForce(t *testing.T) {
	t.Helper()
	old := cpu.ForceNoDotProdForTest(true)
	t.Cleanup(func() { cpu.ForceNoDotProdForTest(old) })
}

// nodotFormed fails t unless the CPU device the composer forms for states
// FEAT_DotProd as want says.
func nodotFormed(t *testing.T, want bool) {
	t.Helper()
	if got := composedCPUDevice(1).ISA&uint64(cpu.ISADotProd) != 0; got != want {
		t.Fatalf("the CPU device states FEAT_DotProd %v, want %v: the arm formed the other int8 dots", got, want)
	}
}

// TestNoDotProdModelGates runs the generated-code audit and the llama.cpp
// golden gate with FEAT_DotProd forced absent.
func TestNoDotProdModelGates(t *testing.T) {
	nodotForce(t)
	nodotFormed(t, false)
	t.Run("EveryModelRunsGenerated", TestEveryModelRunsGenerated)
	t.Run("GreedyMatchesLlamaCpp", TestGreedyMatchesLlamaCpp)
}

// TestNoDotProdIsBitIdentical holds the widened kernels to the SDOT ones on
// every model under 2 GiB, in one process: the widening computes SDOT's int32
// lanes exactly, so decode and prefill logits are the same bits, not a band.
func TestNoDotProdIsBitIdentical(t *testing.T) {
	var models []string
	for _, p := range languageModels(t) {
		if fi, err := os.Stat(p); err == nil && fi.Size() <= 2<<30 {
			models = append(models, p)
		}
	}
	if len(models) == 0 {
		t.Fatal("no model under 2 GiB -- this gate proved nothing")
	}
	ids := make([]int32, 12)
	for i := range ids {
		ids[i] = int32(1 + i)
	}
	run := func(t *testing.T, path string) (forced [][]float32, prefill []float32) {
		t.Helper()
		m, err := Open(path)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		s := m.NewInstance(len(ids) + 1)
		for _, id := range ids {
			l, err := s.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			forced = append(forced, append([]float32(nil), l...))
		}
		s.Close()
		p := m.NewInstance(len(ids) + 1)
		l, err := p.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		prefill = append([]float32(nil), l...)
		p.Close()
		return forced, prefill
	}
	same := func(a, b []float32) int {
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				return i
			}
		}
		return -1
	}
	for _, src := range models {
		t.Run(filepath.Base(src), func(t *testing.T) {
			path := jlmOf(t, src)
			nodotFormed(t, true)
			dotF, dotP := run(t, path)
			old := cpu.ForceNoDotProdForTest(true)
			defer cpu.ForceNoDotProdForTest(old)
			nodotFormed(t, false)
			wideF, wideP := run(t, path)
			for pos := range dotF {
				if i := same(dotF[pos], wideF[pos]); i >= 0 {
					t.Fatalf("pos %d logit %d: SDOT %v, widened %v", pos, i, dotF[pos][i], wideF[pos][i])
				}
			}
			if i := same(dotP, wideP); i >= 0 {
				t.Fatalf("prefill logit %d: SDOT %v, widened %v", i, dotP[i], wideP[i])
			}
		})
	}
}

// BenchmarkNoDotProdDecode decodes JITLLM_BENCH_MODEL (a GGUF or container
// path, or a name under JITLLM_MODELS) greedily with the probe forced to "no
// FEAT_DotProd" ("widened") and as the chip reports it ("sdot"), one sub-
// benchmark each, so a run on a chip with the feature gives the widened kernels
// and the SDOT ceiling in one process. It is a harness for the A/B, not a gate:
// it asserts nothing about time (RULE 4).
func BenchmarkNoDotProdDecode(b *testing.B) {
	p := os.Getenv("JITLLM_BENCH_MODEL")
	if p == "" {
		b.Skip("JITLLM_BENCH_MODEL names no model")
	}
	path := jlmOf(b, testmodels.Resolve(p))
	for _, arm := range []string{"widened", "sdot"} {
		b.Run(arm, func(b *testing.B) {
			old := cpu.ForceNoDotProdForTest(arm == "widened")
			defer cpu.ForceNoDotProdForTest(old)
			if arm == "sdot" && !cpu.HasDotProd() {
				b.Skip("no FEAT_DotProd on this chip: the widened arm alone")
			}
			m, err := Open(path)
			if err != nil {
				b.Fatal(err)
			}
			defer m.Close()
			ids := m.Vocab.Encode("The capital of France is", true)
			st := m.NewInstance(len(ids) + b.N + 2)
			defer st.Close()
			var lg []float32
			for _, id := range ids {
				if lg, err = st.Forward(id); err != nil {
					b.Fatal(err)
				}
			}
			if widened := st.composedCPU().ISA&uint64(cpu.ISADotProd) == 0; widened != (arm == "widened") {
				b.Fatalf("arm %s: widened=%v -- the arm formed the other int8 dots", arm, widened)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if lg, err = st.Forward(Greedy(lg)); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "tok/s")
		})
	}
}
