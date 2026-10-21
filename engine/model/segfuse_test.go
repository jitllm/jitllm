package model

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestSegFusedPairsOnDevice is the end-to-end gate on the device tier's fused
// pairs (tier.fuseSegs): MLA's query projection with kv_a, and a shared
// expert's gate with up, each as one MatVecSegments launch over the shared
// activation. Both arms pin split 1, so the arithmetic is identical and the bar
// is equality; what can differ is the buffer wiring and the grid.
// Stats.SegFused is the selection check.
func TestSegFusedPairsOnDevice(t *testing.T) {
	ids := []int32{5, 11, 23, 41, 67, 89, 101, 127}
	for _, name := range []string{"synth-deepseek", "synth-deepseek-lite"} {
		t.Run(name, func(t *testing.T) {
			// Q8: MatVecSegments takes quantized weights only, and the
			// fixture's own matrices are float.
			dir := testmodels.Path(name)
			if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
				testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS) (RULE 11)", dir)
			}
			dst := filepath.Join(t.TempDir(), name+jlm.Ext)
			if _, err := convert.FromSafetensors(dir, dst, jlm.Fingerprint{Host: "test"}, convert.WithQ8()); err != nil {
				t.Fatalf("convert %s: %v", dir, err)
			}
			m, err := Open(dst, noTune, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			run := func(fuse bool) ([][]float32, tier.Stats) {
				opts := []tier.Option{tier.WithDevices(testdevices.Like(t, "cuda:0")), tier.WithDeviceTune(tier.TuneOff),
					tier.WithSplit(1)}
				if !fuse {
					opts = append(opts, tier.WithConfig(func(c *tier.Config) { c.NoSegFuse = true }))
				}
				gpu, err := openTier(t, opts...)
				if err != nil || gpu == nil {
					t.Skipf("no cuda device (%v)", err)
				}
				defer gpu.Close()
				st := m.NewState(len(ids) + 1)
				defer st.Close()
				st.SetDeviceLayers(gpu, -1)
				if n := st.GPULayers(); n != m.Cfg.NLayer {
					t.Fatalf("the device took %d of %d blocks: %v", n, m.Cfg.NLayer, declineList(st))
				}
				got := runIDs(t, st, ids)
				if n := st.GPULayers(); n != m.Cfg.NLayer {
					t.Fatalf("the session fell to the host: %v", gpu.Err())
				}
				return got, gpu.Stats()
			}
			want, ws := run(false)
			got, gs := run(true)
			if ws.SegFused != 0 || gs.SegFused == 0 {
				t.Fatalf("selection: %d fused pair launch(es) in the separate arm, %d in the fused arm",
					ws.SegFused, gs.SegFused)
			}
			worst, worstD := worstOf(want, got)
			if worst != 0 {
				t.Fatalf("fused pairs move the logits: NMSE %.3e, |dlogit| %.3f against separate launches",
					worst, worstD)
			}
			t.Logf("%s: %d fused pair launch(es), logits identical to separate launches", name, gs.SegFused)
		})
	}
}
