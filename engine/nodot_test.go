//go:build arm64 && jitllmtest

package engine

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// arm64 without FEAT_DotProd at the model level (engine/device/cpu/sdotemu.go): a
// Cortex-A53/A57/A72/A73 widens every int8 dot to SMULL/ADDP/SADDLP. Each gate
// opens its models on the CPU device without FEAT_DotProd (cpu:nodotprod) and
// asserts the device the composer formed for states none, the bit its arm64
// lowering widens by.

// noDot is the device list of a CPU without FEAT_DotProd.
func noDot() Option { return WithDevices("cpu:nodotprod") }

// nodotFormed fails t unless the CPU device an instance of m opens with io
// states FEAT_DotProd as want says.
func nodotFormed(t *testing.T, m *model.Model, want bool, io ...Option) {
	t.Helper()
	s := testInstance(t, m, 8, io...)
	defer s.Close()
	if got := s.cpu.x.ISA()&cpu.ISADotProd != 0; got != want {
		t.Fatalf("the CPU device states FEAT_DotProd %v, want %v: the arm formed the other int8 dots", got, want)
	}
}

// TestNoDotProdModelGates runs the generated-code audit and the llama.cpp
// golden gate on a CPU device without FEAT_DotProd.
func TestNoDotProdModelGates(t *testing.T) {
	instOpts := []Option{noDot()}
	m, err := model.Open(jlmOf(t, benchModel(t)))
	if err != nil {
		t.Fatal(err)
	}
	nodotFormed(t, m, false, instOpts...)
	m.Close()
	t.Run("EveryModelRunsGenerated", func(t *testing.T) { everyModelRunsGenerated(t, noDot()) })
	t.Run("GreedyMatchesLlamaCpp", func(t *testing.T) { greedyMatchesLlamaCpp(t, noDot()) })
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
	run := func(t *testing.T, path string, dot bool, opts ...Option) (forced [][]float32, prefill []float32) {
		t.Helper()
		instOpts := opts
		m, err := model.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		nodotFormed(t, m, dot, instOpts...)
		s := testInstance(t, m, len(ids)+1, instOpts...)
		for _, id := range ids {
			l, err := s.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			forced = append(forced, append([]float32(nil), l...))
		}
		s.Close()
		p := testInstance(t, m, len(ids)+1, instOpts...)
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
			dotF, dotP := run(t, path, true)
			wideF, wideP := run(t, path, false, noDot())
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
// path, or a name under JITLLM_MODELS) greedily on a CPU device without
// FEAT_DotProd ("widened") and as the chip reports it ("sdot"), one sub-
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
			var opts []Option
			if arm == "widened" {
				opts = append(opts, noDot())
			}
			if arm == "sdot" && !cpu.HasDotProd() {
				b.Skip("no FEAT_DotProd on this chip: the widened arm alone")
			}
			instOpts := opts
			m, err := model.Open(path)
			if err != nil {
				b.Fatal(err)
			}
			defer m.Close()
			ids := m.Vocab.Encode("The capital of France is", true)
			st := testInstance(b, m, len(ids)+b.N+2, instOpts...)
			defer st.Close()
			var lg []float32
			for _, id := range ids {
				if lg, err = st.Forward(id); err != nil {
					b.Fatal(err)
				}
			}
			if widened := st.cpu.x.Composer().Device().ISA&uint64(cpu.ISADotProd) == 0; widened != (arm == "widened") {
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
