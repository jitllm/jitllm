package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestIndexedGroupSplitOnDevice is the device gate on the expert matvecs'
// in-group split reduction: a whole mixture model decoded on the device with
// its gate/up/down matvecs reducing in threadgroup memory, against the same
// model with them writing partials that a Reduce launch sums.
//
// The arms differ in the expert reduction and nothing else, so the bar is
// equality. ForceIndexedGroup, not ForceGroupSplit (which also rebuilds dense
// matvecs and moves logits enough to flip a near-tied expert). The in-group sum
// adds partials in the same order as Reduce, so any difference is the tier's
// wiring; the kernel itself is gated in
// backend.TestIndexedGroupSplitMatchesTheReduce.
//
// Stats.IndexedGroup is the selection check: without it the "group" arm could
// silently have built the partial form and compared two identical runs.
func TestIndexedGroupSplitOnDevice(t *testing.T) {
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
			run := func(group bool) ([][]float32, tier.Stats) {
				opts := []tier.Option{tier.WithDevices(testdevices.Like(t, "cuda:0")), tier.WithDeviceTune(tier.TuneOff),
					tier.WithSplit(4)}
				if group {
					opts = append(opts, tier.WithConfig(func(c *tier.Config) { c.ForceIndexedGroup = true }))
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
					t.Fatalf("the device took %d of %d blocks: %v", n, m.Cfg.NLayer, gpu.Err())
				}
				var out [][]float32
				for _, id := range ids {
					lg, err := st.Forward(id)
					if err != nil {
						t.Fatal(err)
					}
					out = append(out, append([]float32(nil), lg...))
				}
				if n := st.GPULayers(); n != m.Cfg.NLayer {
					t.Fatalf("the session fell to the host: %v", gpu.Err())
				}
				return out, gpu.Stats()
			}
			want, ws := run(false)
			got, gs := run(true)
			if ws.IndexedGroup != 0 || gs.IndexedGroup == 0 {
				t.Fatalf("selection: %d in-group expert matvecs in the partial arm and %d in the "+
					"in-group arm -- the arms did not run the configurations they name",
					ws.IndexedGroup, gs.IndexedGroup)
			}
			for p := range want {
				var num, den float64
				for i := range want[p] {
					d := float64(got[p][i] - want[p][i])
					num, den = num+d*d, den+float64(want[p][i])*float64(want[p][i])
				}
				if nmse := num / den; math.IsNaN(nmse) || den == 0 || nmse != 0 {
					t.Fatalf("pos %d: logit NMSE %.3e between the in-group and the reduce arms", p, nmse)
				}
			}
			t.Logf("%d positions, %d in-group expert matvecs, logits identical to the reduce arm",
				len(want), gs.IndexedGroup)
		})
	}
}
