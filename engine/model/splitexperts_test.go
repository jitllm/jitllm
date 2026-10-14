//go:build linux

package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestSplitExpertsMatchTheHost holds the split -- a hybrid block whose card
// keeps an expert cache, running the cached experts there while the host runs
// the rest of the selection -- to the host-only run: the same greedy token at
// every position and logits within the device band (RULE 11c), with the
// cache counted as running experts on the card and taking experts in (RULE
// 10). Three cache sizes: one sheet more than the selection (every token
// splits a few and admits), auto-sized from the room after placement, and
// none (the plain hybrid path, which must count no split). Kimi-K3 is the
// latent mixture; Moonlight the packed one with a shared expert, which the
// card runs while the host runs its share. The qwen4exp fixture's banks are float, which have no
// packed sheets to send: it must fall back to the plain hybrid path in every
// arm, counted, with the same answer.
func TestSplitExpertsMatchTheHost(t *testing.T) {
	models := []struct {
		name  string
		split bool
		// band is the device's logit NMSE against the host at this model's
		// depth and quantization (RULE 11c); the plain hybrid arm reads the
		// same, so it is not the split's.
		band float64
		// blocks is how many blocks go on the card, every one at 0: a
		// partial seam keeps a big model's arms on one placement on a small
		// card, whatever else holds its memory.
		blocks int
		open   func(t *testing.T) (*Model, []int32)
	}{
		{"kimik3", true, 1e-5, 0, func(t *testing.T) (*Model, []int32) {
			path, ok := existingModel(testmodels.Path("kimik3/Kimi-K3-0.40B.Q8_0.gguf"))
			if !ok {
				t.Skip("MODEL MISSING: kimik3/Kimi-K3-0.40B.Q8_0.gguf -- this gate proved nothing")
			}
			m, err := Open(jlmOf(t, path), noTune, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			return m, []int32{1008, 10484, 318, 15383, 387, 17374, 13, 646, 1008, 10484, 318, 15383}
		}},
		{"moonlight", true, 1e-3, 6, func(t *testing.T) (*Model, []int32) {
			// A packed mixture with a shared expert and no latent: the host's
			// share runs after the shared expert is queued on the card.
			path, ok := existingModel(testmodels.Path("Moonlight-16B-A3B-Instruct-Q4_K_M.gguf"))
			if !ok {
				t.Skip("MODEL MISSING: Moonlight-16B-A3B-Instruct-Q4_K_M.gguf -- this gate proved nothing")
			}
			m, err := Open(jlmOf(t, path), noTune, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			return m, m.Vocab.Encode("The capital of France is Paris, and the capital of Italy is", true)
		}},
		{"synth-qwen4exp", false, 1e-5, 0, func(t *testing.T) (*Model, []int32) {
			m, g := openQ4E(t, "synth-qwen4exp")
			return m, g.IDs
		}},
	}
	for _, md := range models {
		t.Run(md.name, func(t *testing.T) {
			m, ids := md.open(t)
			defer m.Close()
			want := make([][]float32, 0, len(ids))
			func() {
				s := m.NewState(len(ids) + 2)
				defer s.Close()
				for _, id := range ids {
					l, err := s.Forward(id)
					if err != nil {
						t.Fatal(err)
					}
					want = append(want, append([]float32(nil), l...))
				}
			}()
			k := m.Cfg.NExpertUsed
			for _, arm := range []struct {
				name  string
				slots int
			}{{"tight", k + 1}, {"auto", 0}, {"none", -1}} {
				t.Run(arm.name, func(t *testing.T) {
					var st tier.Stats
					var got [][]float32
					var placed int
					func() {
						s := m.NewState(len(ids) + 2)
						defer s.Close()
						g, err := tier.OpenWith(tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff),
							tier.WithConfig(func(c *tier.Config) {
								c.StreamExperts, c.HybridExperts, c.StreamCacheSlots = true, true, arm.slots
								c.SplitAdmit = k
							}))
						if err != nil || g == nil {
							noDevice(t, "gpu:0", err)
						}
						defer g.Close()
						want := m.Cfg.NLayer
						if md.blocks > 0 {
							want = md.blocks
						}
						s.SetDeviceLayers(g, want)
						placed = s.GPULayers()
						if placed != want {
							t.Fatalf("%d of %d blocks placed: %s", placed, want, g.Stats().LastErr)
						}
						if s.GPULayers() == 0 {
							t.Fatalf("no block placed: %s", g.Stats().LastErr)
						}
						for _, id := range ids {
							l, err := s.Forward(id)
							if err != nil {
								t.Fatal(err)
							}
							got = append(got, append([]float32(nil), l...))
						}
						st = g.Stats()
					}()
					t.Logf("%d blocks placed; stats: hybrid %d, split %d, hits %d, cache %d/%d, short %d, size %d",
						placed, st.HybridRuns, st.SplitRuns, st.SplitHits, st.StreamCacheHits, st.StreamCacheMisses,
						st.StreamCacheShort, st.StreamCacheSize)
					if arm.slots == 0 && md.split && st.StreamCacheSize == 0 {
						t.Skip("CARD TOO SMALL: placement left no room for an auto-sized cache -- this arm proved nothing")
					}
					if arm.slots < 0 || !md.split {
						if st.SplitRuns != 0 || st.HybridRuns == 0 {
							t.Fatalf("no cache: %d split steps, %d hybrid steps", st.SplitRuns, st.HybridRuns)
						}
					} else if st.SplitRuns == 0 || st.SplitHits == 0 || st.SplitAdmits == 0 {
						t.Fatalf("%d split steps, %d experts on the card, %d admitted: the cache ran nothing",
							st.SplitRuns, st.SplitHits, st.SplitAdmits)
					}
					worst := 0.0
					for p := range want {
						var num, den float64
						for i := range want[p] {
							if math.IsNaN(float64(got[p][i])) || math.IsInf(float64(got[p][i]), 0) {
								t.Fatalf("pos %d logit %d is not finite", p, i)
							}
							d := float64(got[p][i] - want[p][i])
							num += d * d
							den += float64(want[p][i]) * float64(want[p][i])
						}
						nmse := num / den
						worst = math.Max(worst, nmse)
						if nmse > md.band || argmaxOf(got[p]) != argmaxOf(want[p]) {
							t.Fatalf("pos %d: NMSE %.2e, argmax %d split against %d host",
								p, nmse, argmaxOf(got[p]), argmaxOf(want[p]))
						}
					}
					t.Logf("%d split steps, %d experts on the card, %d admitted, %d hybrid steps; worst NMSE %.2e",
						st.SplitRuns, st.SplitHits, st.SplitAdmits, st.HybridRuns, worst)
				})
			}
		})
	}
}
