//go:build amd64 || arm64

package model

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// prefillWavesModels are the packed mixtures the waves gate runs; a model whose
// residual is several streams is not here, since its chunk is not batched on
// the device (layer.go, prepBatch).
var prefillWavesModels = []struct {
	name, path string
	blocks     int
	band       float64
}{
	{"deepseek-v2-lite", "DeepSeek-V2-Lite.Q4_K_M.jlm", 6, 2e-3},
}

// prefillWavesCase runs ids as one step of rows (the batched chunk a prompt's
// prefill is) on a device running the model's experts as hybrid blocks with an
// expert cache, and returns the last row's logits with the device's counters.
// noWaves is the control arm: the chunk's experts on the host, as before they
// ran on the card.
func prefillWavesCase(t *testing.T, m *Model, ids []int32, blocks int, noWaves bool) ([]float32, tier.Stats) {
	t.Helper()
	c := m.Cfg
	s := m.NewState(len(ids) + 2)
	defer s.Close()
	g, err := tier.OpenWith(tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff),
		tier.WithConfig(func(cf *tier.Config) {
			cf.StreamExperts, cf.HybridExperts = true, true
			cf.SplitAdmit, cf.NoPrefillExperts = c.NExpertUsed, noWaves
		}))
	if err != nil || g == nil {
		noDevice(t, "gpu:0", err)
	}
	defer g.Close()
	placed := c.NLayer
	if blocks > 0 {
		placed = blocks
	}
	s.SetDeviceLayers(g, placed)
	if s.GPULayers() != placed {
		t.Skipf("CARD TOO SMALL: %d of %d blocks placed (%s) -- this arm proved nothing",
			s.GPULayers(), placed, g.Stats().LastErr)
	}
	if g.Stats().StreamCacheSize == 0 {
		t.Skip("CARD TOO SMALL: placement left no room for an expert cache -- this arm proved nothing")
	}
	defer m.enterPager()()
	out := rowsOut{logits: make([]float32, len(ids)*c.NVocab), hidden: make([]float32, len(ids)*c.ResidW())}
	if err := s.stepRows(embedSrc(s, ids), ids, &out); err != nil {
		t.Fatal(err)
	}
	t.Logf("device: %v", g.Err())
	last := out.logits[(len(ids)-1)*c.NVocab : len(ids)*c.NVocab]
	return append([]float32(nil), last...), g.Stats()
}

// TestHybridPrefillWavesMatchTheHost holds a prompt chunk of a hybrid block,
// its experts run on the card in waves through the expert cache
// (streamBank.prefillWaves), to the host's prefill of the same tokens, and
// demands the waves ran (blocks, waves and sheets counted, RULE 10). The
// control arm keeps the chunk's experts on the host and must count none.
func TestHybridPrefillWavesMatchTheHost(t *testing.T) {
	for _, md := range prefillWavesModels {
		t.Run(md.name, func(t *testing.T) {
			path, ok := existingModel(testmodels.Path(md.path))
			if !ok {
				t.Skipf("MODEL MISSING: %s -- this gate proved nothing", md.path)
			}
			m, err := Open(jlmOf(t, path), noTune, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			base := m.Vocab.Encode("The capital of France is Paris, and the capital of Italy is Rome; Spain's capital is Madrid, Portugal's Lisbon, and Germany's Berlin; ", true)
			var ids []int32
			for len(ids) < 64 {
				ids = append(ids, base...)
			}
			band := md.band
			ids = ids[:prefillWaveRowsForTest]
			want := func() []float32 {
				s := m.NewState(len(ids) + 2)
				defer s.Close()
				l, err := s.Prefill(ids)
				if err != nil {
					t.Fatal(err)
				}
				return append([]float32(nil), l...)
			}()
			for _, arm := range []struct {
				name    string
				noWaves bool
			}{{"waves", false}, {"host", true}} {
				t.Run(arm.name, func(t *testing.T) {
					got, st := prefillWavesCase(t, m, ids, md.blocks, arm.noWaves)
					var num, den float64
					for i := range want {
						d := float64(got[i] - want[i])
						num += d * d
						den += float64(want[i]) * float64(want[i])
					}
					nmse := num / den
					t.Logf("%s: %d tokens, NMSE %.2e (band %.0e); %d block-chunks in %d waves, %d sheets sent, %d hits; %d hybrid block-steps",
						arm.name, len(ids), nmse, band, st.PrefillBlocks, st.PrefillWaves, st.PrefillUploaded, st.PrefillHits, st.HybridRuns)
					if !(nmse <= band) || argmaxOf(got) != argmaxOf(want) {
						t.Fatalf("prefill NMSE %.2e against a band of %.0e, argmax %d against the host's %d",
							nmse, band, argmaxOf(got), argmaxOf(want))
					}
					if arm.noWaves {
						if st.PrefillBlocks != 0 || st.PrefillWaves != 0 {
							t.Fatalf("%d block-chunks ran in waves with the waves off", st.PrefillBlocks)
						}
						return
					}
					if st.PrefillBlocks == 0 || st.PrefillWaves < st.PrefillBlocks || st.PrefillUploaded == 0 {
						t.Fatalf("the waves did not run: %d block-chunks, %d waves, %d sheets sent",
							st.PrefillBlocks, st.PrefillWaves, st.PrefillUploaded)
					}
				})
			}
		})
	}
}

// prefillWaveRowsForTest is the chunk's width: the tier's threshold, the narrowest chunk that takes the waves.
const prefillWaveRowsForTest = 64
