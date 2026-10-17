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
	{"kimik3", "kimik3/Kimi-K3-0.40B.Q8_0.gguf", 0, 3e-3},
}

// prefillWavesCase runs ids as one step of rows (the batched chunk a prompt's
// prefill is) on a device running the model's experts as hybrid blocks with an
// expert cache, and returns the last row's logits with the device's counters.
// noWaves is the control arm: the chunk's experts on the host, as before they
// ran on the card.
func prefillWavesCase(t *testing.T, m *Model, ids []int32, blocks, slots int, noWaves bool) ([]float32, tier.Stats) {
	last, st, _ := prefillWavesAllocs(t, m, ids, blocks, slots, noWaves)
	return last, st
}

// prefillWavesAllocs is prefillWavesCase that also runs the chunk a second,
// warm time and returns the heap allocations that second run made.
func prefillWavesAllocs(t *testing.T, m *Model, ids []int32, blocks, slots int, noWaves bool) ([]float32, tier.Stats, uint64) {
	t.Helper()
	c := m.Cfg
	s := m.NewState(len(ids) + 2)
	defer s.Close()
	g, err := tier.OpenWith(tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff),
		tier.WithConfig(func(cf *tier.Config) {
			cf.StreamExperts, cf.HybridExperts = true, true
			cf.SplitAdmit, cf.NoPrefillExperts = c.NExpertUsed, noWaves
			cf.StreamCacheSlots = slots
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
	last := append([]float32(nil), out.logits[(len(ids)-1)*c.NVocab:len(ids)*c.NVocab]...)
	// The same chunk again from the same position: warm (the cache is filled,
	// the scratch grown). One unmeasured run first; the measured one counts the
	// engine's own allocations (countAllocs sets the runtime's per-P caches
	// apart, which refill at random under every device call): a warm step makes
	// none.
	s.Reset()
	if err := warmWavesStep(s, embedSrc(s, ids), ids, &out); err != nil {
		t.Fatal(err)
	}
	s.Reset()
	src := embedSrc(s, ids)
	var stepErr error
	w := countAllocs(func() { stepErr = warmWavesStep(s, src, ids, &out) })
	if stepErr != nil {
		t.Fatal(stepErr)
	}
	if w.engine > 0 {
		t.Logf("engine allocations at: %s", w.where)
	}
	fewest := uint64(w.engine)
	return last, g.Stats(), fewest
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
			band := md.band
			ids := prefillWavesIDs(m)
			want := prefillWavesHost(t, m, ids)
			armAllocs, blockChunks := map[string]uint64{}, map[string]int{}
			for _, arm := range []struct {
				name    string
				slots   int
				noWaves bool
			}{{"waves", 0, false}, {"waves-small-cache", 2 * m.Cfg.NExpertUsed, false}, {"host", 0, true}} {
				t.Run(arm.name, func(t *testing.T) {
					got, st, allocs := prefillWavesAllocs(t, m, ids, md.blocks, arm.slots, arm.noWaves)
					armAllocs[arm.name], blockChunks[arm.name] = allocs, st.PrefillBlocks
					t.Logf("%s: %d heap allocations in the warm run", arm.name, allocs)
					nmse := wavesNMSE(got, want)
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
					if arm.slots > 0 && st.PrefillWaves <= st.PrefillBlocks {
						t.Fatalf("a %d-slot cache ran %d waves over %d block-chunks: no chunk needed several", arm.slots, st.PrefillWaves, st.PrefillBlocks)
					}
					if st.PrefillBlocks == 0 || st.PrefillWaves < st.PrefillBlocks || st.PrefillUploaded == 0 {
						t.Fatalf("the waves did not run: %d block-chunks, %d waves, %d sheets sent",
							st.PrefillBlocks, st.PrefillWaves, st.PrefillUploaded)
					}
				})
			}
			// Warm, a chunk on the waves makes no heap allocation, whether the cache
			// holds its experts or churns sheets through the pager's Sheet path on
			// every wave (the admission path decode shares). The host arm is the
			// control: it is logged.
			for _, name := range []string{"waves", "waves-small-cache"} {
				if a := armAllocs[name]; a != 0 {
					t.Errorf("%s allocated %d times warm over %d block-chunks; a warm step allocates nothing", name, a, blockChunks[name])
				}
			}
		})
	}
}

// prefillWaveRowsForTest is the chunk's width: the tier's threshold, the narrowest chunk that takes the waves.
const prefillWaveRowsForTest = 64

// prefillWavesIDs is a prompt of the tier's threshold width.
func prefillWavesIDs(m *Model) []int32 {
	base := m.Vocab.Encode("The capital of France is Paris, and the capital of Italy is Rome; Spain's capital is Madrid, Portugal's Lisbon, and Germany's Berlin; ", true)
	var ids []int32
	for len(ids) < prefillWaveRowsForTest {
		ids = append(ids, base...)
	}
	return ids[:prefillWaveRowsForTest]
}

// prefillWavesHost is the host's prefill of ids: the last row's logits.
func prefillWavesHost(t *testing.T, m *Model, ids []int32) []float32 {
	t.Helper()
	s := m.NewState(len(ids) + 2)
	defer s.Close()
	l, err := s.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	return append([]float32(nil), l...)
}

// wavesNMSE is got's normalised mean squared error against want.
func wavesNMSE(got, want []float32) float64 {
	var num, den float64
	for i := range want {
		d := float64(got[i] - want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
	}
	return num / den
}

//go:noinline
func warmWavesStep(s *State, src []func(dst []float32) error, ids []int32, out *rowsOut) error {
	return s.stepRows(src, ids, out)
}
