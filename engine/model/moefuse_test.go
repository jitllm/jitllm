package model

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestMoEFusedEpiloguesOnDevice is the end-to-end gate on tier.fuseMoE: a
// mixture decoded on the device with the expert matvecs applying their own
// biases and the up projection writing act(gate)*up, against the separate
// IndexedBiasAdd and ActMul launches. The fused epilogue is the same float ops
// in the same order (see backend.TestIndexedGateAndBiasMatchTheSeparateKernels),
// so the bar is equality; the model checks the buffer wiring. synth-gptoss has
// all three biases and swiglu-oai, Qwen3-MoE none. Stats.MoEFused is the
// selection check.
func TestMoEFusedEpiloguesOnDevice(t *testing.T) {
	for _, name := range []string{"synth-gptoss.gguf", "Qwen3-MOE-4x0.6B-Q4_K_M.gguf"} {
		t.Run(name, func(t *testing.T) {
			path, ok := existingModel(testmodels.Path(name))
			if !ok {
				testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path(name)+" (set JITLLM_MODELS) (RULE 11)")
			}
			m, err := Open(jlmOf(t, path))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			ids := m.Vocab.Encode("The capital of France is Paris, and the capital of", true)
			run := func(fuse bool) ([][]float32, tier.Stats) {
				opts := []tier.Option{tier.WithDevices(testdevices.Like(t, "cuda:0")), tier.WithDeviceTune(tier.TuneOff),
					tier.WithSplit(4), tier.WithConfig(func(c *tier.Config) {
						c.ForceIndexedGroup = true
						c.NoMoEFuse = !fuse
					})}
				gpu, err := openTier(t, opts...)
				if err != nil || gpu == nil {
					t.Skipf("no cuda device (%v)", err)
				}
				defer gpu.Close()
				st := m.NewState(len(ids) + 1)
				defer st.Close()
				st.SetDeviceLayers(gpu, -1)
				if n := st.GPULayers(); n != m.Cfg.NLayer {
					t.Fatalf("the device took %d of %d blocks: %v", n, m.Cfg.NLayer, gpu.Err())
				}
				got := runIDs(t, st, ids)
				if n := st.GPULayers(); n != m.Cfg.NLayer {
					t.Fatalf("the session fell to the host: %v", gpu.Err())
				}
				return got, gpu.Stats()
			}
			want, ws := run(false)
			got, gs := run(true)
			if ws.MoEFused != 0 || gs.MoEFused == 0 {
				t.Fatalf("selection: %d fused expert launch(es) in the separate arm, %d in the fused arm",
					ws.MoEFused, gs.MoEFused)
			}
			if worst, worstD := worstOf(want, got); worst != 0 {
				t.Fatalf("fused epilogues move the logits: NMSE %.3e, |dlogit| %.3f", worst, worstD)
			}
			t.Logf("%s: %d fused expert launch(es), logits identical to the separate launches",
				name, gs.MoEFused)
		})
	}
}
