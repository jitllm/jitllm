//go:build jitllmtest

package model

import (
	"os"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestShadowDenseDecode holds the composer's post-attention slice (the
// output projection, the residual add, the FFN norm, the gated MLP and its
// add) to what the tier's own kernels write, block by block, at every
// decoded position, on every device: every block of a plain dense model is
// compared, the slice must form, and the composer's distance from the
// tier's may be at most a quarter of the slice's int8 band: the tier's own
// distance from the slice with float activations (tier.ShadowLog). A block that is not that slice is refused by name, never
// compared; the gate's violation binds gate and up the wrong way round.
func TestShadowDenseDecode(t *testing.T) {
	// The composer's distance from the tier, as a fraction of the slice's
	// int8 band: 0.069 at worst on these models (Qwen3-0.6B, CUDA).
	const ratio = 0.25
	for _, c := range []struct {
		file   string
		refuse bool // every block refused by name: not the plain slice
	}{
		{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", false},
		{"Qwen3-0.6B-Q8_0.gguf", false},
		{"tinyllama-1.1b-q3_K_M.gguf", false},
		{"synth-gemma3n.gguf", true},
	} {
		t.Run(c.file, func(t *testing.T) {
			path := testmodels.Path(c.file)
			if _, err := os.Stat(path); err != nil {
				t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS)", path)
			}
			m, err := Open(jlmOf(t, path), noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			ids := m.Vocab.Encode("The capital of France is", true)
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					run := func(swap bool) *tier.ShadowLog {
						log := &tier.ShadowLog{Ratio: ratio, Floor: 1e-9, SwapGateUp: swap}
						restore := tier.ShadowForTest(log)
						defer restore()
						g, err := tier.OpenWith(slices.Concat([]tier.Option{tier.WithDevices(dev),
							tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t))...)
						if err != nil || g == nil {
							noDevice(t, dev, err)
						}
						defer g.Close()
						st := m.NewState(len(ids) + 1)
						defer st.Close()
						if err := st.SetDevice(g); err != nil {
							t.Fatal(err)
						}
						if st.devCount() != m.Cfg.NLayer {
							t.Fatalf("placed %d of %d blocks: %v %s", st.devCount(), m.Cfg.NLayer, st.DeviceDeclines(), g.Err())
						}
						for _, id := range ids {
							if _, err := st.Forward(id); err != nil {
								t.Fatal(err)
							}
						}
						return log
					}
					log := run(false)
					t.Logf("%d blocks x %d positions on %s: %d compared; worst NMSE composer-tier %.3g, tier-float %.3g, composer-float %.3g, worst composer-tier over the slice's band %.3g; refused %v; unformed %v",
						m.Cfg.NLayer, len(ids), dev, log.Compared, log.Worst, log.WorstTier, log.WorstComposer, log.WorstRatio, log.Refused, log.Unformed)
					if c.refuse {
						if log.Compared != 0 || len(log.Refused) == 0 {
							t.Fatalf("compared %d blocks and refused %v: this model's blocks are not the plain slice and must be refused by name", log.Compared, log.Refused)
						}
						return
					}
					if len(log.Unformed) > 0 || len(log.Refused) > 0 {
						t.Fatalf("the slice did not form or was refused: unformed %v, refused %v", log.Unformed, log.Refused)
					}
					if want := m.Cfg.NLayer * len(ids); log.Compared != want {
						t.Fatalf("compared %d slices, want every block at every position (%d)", log.Compared, want)
					}
					if len(log.Failed) > 0 {
						t.Fatalf("%d of %d slices parted from the tier: %v", len(log.Failed), log.Compared, log.Failed[:min(4, len(log.Failed))])
					}
					bad := run(true)
					if len(bad.Failed) == 0 {
						t.Fatalf("with gate and up bound the wrong way round no compare failed (worst %.3g): the gate cannot see a wrong slice", bad.Worst)
					}
					t.Logf("violation: %d of %d slices failed, worst NMSE %.3g", len(bad.Failed), bad.Compared, bad.Worst)
				})
			}
		})
	}
}
