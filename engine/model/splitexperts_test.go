//go:build linux

package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestSplitExpertsMatchTheHost holds the split -- a hybrid block whose card
// keeps an expert cache, running the cached experts there while the host runs
// the rest of the selection -- to the host-only run: the same greedy token at
// every position and logits within the device band (RULE 11c), with the
// cache counted as running experts on the card and taking experts in (RULE
// 10). Three cache sizes: one sheet more than the selection (every token
// splits a few and admits), auto-sized from the room after placement, and
// none (the plain hybrid path, which must count no split). Kimi-K3 is the
// latent mixture; DeepSeek-V2-Lite the packed one with a shared expert, which the
// card runs while the host runs its share. The qwen4exp fixture's banks are float, which have no
// packed sheets to send: it must fall back to the plain hybrid path in every
// arm, counted, with the same answer.
func TestSplitExpertsMatchTheHost(t *testing.T) {
	models := []struct {
		name  string
		split bool
		// band is the device's logit NMSE against the host at this model's
		// depth and quantization (RULE 11c); the plain hybrid arm reads the
		// same, so it is not the split's. DeepSeek-V2-Lite's plain hybrid arm
		// reads 5.1e-4 and the predict arm, with the most experts on the
		// card, 1.04e-3 at one position, the same value with every launch
		// serialised (CUDA_LAUNCH_BLOCKING=1), so it is the order the card's
		// sums round in, not a copy that landed late.
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
		{"deepseek-v2-lite", true, 2e-3, 6, func(t *testing.T) (*Model, []int32) {
			// A packed mixture with a shared expert and no latent: the host's
			// share runs after the shared expert is queued on the card.
			path, ok := existingModel(testmodels.Path("DeepSeek-V2-Lite.Q4_K_M.gguf"))
			if !ok {
				t.Skip("MODEL MISSING: DeepSeek-V2-Lite.Q4_K_M.gguf -- this gate proved nothing")
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
				// yield is auto with the budget cut under what the card holds
				// halfway through: the caches go before any block does.
				yield bool
				// predict sends each block's predicted experts ahead of its
				// router (tier.Config.SplitPredict).
				predict int
			}{{"tight", k + 1, false, 0}, {"auto", 0, false, 0}, {"yield", 0, true, 0}, {"predict", 0, false, k},
				{"none", -1, false, 0}} {
				t.Run(arm.name, func(t *testing.T) {
					if arm.yield && !md.split {
						t.Skip("a float bank keeps no cache to give back")
					}
					var st tier.Stats
					var got [][]float32
					var placed int
					func() {
						s := m.NewState(len(ids) + 2)
						defer s.Close()
						g, err := openTier(t, tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff),
							tier.WithConfig(func(c *tier.Config) {
								c.StreamExperts, c.HybridExperts, c.StreamCacheSlots = true, true, arm.slots
								c.SplitAdmit = k
								c.SplitPredict = arm.predict
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
							t.Skipf("CARD TOO SMALL: %d of %d blocks placed (%s) -- this arm proved nothing",
								placed, want, g.Stats().LastErr)
						}
						if s.GPULayers() == 0 {
							t.Fatalf("no block placed: %s", g.Stats().LastErr)
						}
						for i, id := range ids {
							if arm.yield && i == len(ids)/2 {
								if g.Stats().StreamCacheSize == 0 {
									t.Skip("CARD TOO SMALL: no auto-sized cache to give back -- this arm proved nothing")
								}
								// Half of what the caches hold: anything smaller
								// the unpack staging alone can cover.
								if _, err := g.SetBudget(g.Bytes() - g.Stats().CacheBytes/2); err != nil {
									t.Fatal(err)
								}
								if g.Stats().CachesDropped == 0 || s.GPULayers() != placed {
									t.Fatalf("the budget cut dropped %d caches and left %d of %d blocks placed",
										g.Stats().CachesDropped, s.GPULayers(), placed)
								}
							}
							l, err := s.Forward(id)
							if err != nil {
								t.Fatal(err)
							}
							got = append(got, append([]float32(nil), l...))
						}
						st = g.Stats()
						// One source of routing counts: the card's selections on
						// every hybrid block are the session's hotness, so each
						// placed mixture block's uses sum to the selection.
						uses, tokens := s.ExpertUses()
						c := m.Cfg
						for li := 0; li < placed && st.HybridRuns > 0; li++ {
							if !c.MoEAt(li) {
								continue
							}
							var sum float64
							for e := 0; e < c.NExpert; e++ {
								sum += uses[li*c.NExpert+e]
							}
							if math.Abs(sum-float64(c.NExpertUsed)) > 1e-9 {
								t.Fatalf("block %d's uses sum to %.4f over %d tokens, want %d: the card's selections are not the session's counts",
									li, sum, tokens, c.NExpertUsed)
							}
						}
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
					} else if arm.predict > 0 && st.SplitPredicted == 0 {
						t.Fatalf("the predict arm sent no expert ahead")
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
						// A different argmax passes only on a near tie in the
						// host's own logits.
						a, b := argmaxOf(got[p]), argmaxOf(want[p])
						if nmse > md.band || a != b && want[p][b]-want[p][a] > 0.05 {
							t.Fatalf("pos %d: NMSE %.2e, argmax %d split against %d host",
								p, nmse, argmaxOf(got[p]), argmaxOf(want[p]))
						}
					}
					t.Logf("%d split steps, %d experts on the card, %d admitted (%d on the copy queue), %d hybrid steps; worst NMSE %.2e",
						st.SplitRuns, st.SplitHits, st.SplitAdmits, st.SplitQueued, st.HybridRuns, worst)
				})
			}
		})
	}
}

// TestHybridShortPromptTakesTheDecodePath: once a device has run a model's
// experts on the host, a prompt of up to hostRowsByRow tokens runs through
// the decode path (where the split runs) and a longer one stays a chunk;
// before any such call, a short prompt is a chunk too.
func TestHybridShortPromptTakesTheDecodePath(t *testing.T) {
	path, ok := existingModel(testmodels.Path("kimik3/Kimi-K3-0.40B.Q8_0.gguf"))
	if !ok {
		t.Skip("MODEL MISSING: kimik3/Kimi-K3-0.40B.Q8_0.gguf -- this gate proved nothing")
	}
	m, err := Open(jlmOf(t, path), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	g, err := openTier(t, tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff),
		tier.WithConfig(func(c *tier.Config) { c.StreamExperts, c.HybridExperts = true, true }))
	if err != nil || g == nil {
		noDevice(t, "gpu:0", err)
	}
	defer g.Close()
	short := []int32{1008, 10484, 318, 15383, 387}
	long := append(append([]int32(nil), short...), 17374, 13, 646, 1008, 10484, 318, 15383)
	prompt := func(ids []int32) int {
		s := m.NewState(len(ids) + 2)
		defer s.Close()
		s.SetDeviceLayers(g, m.Cfg.NLayer)
		if s.GPULayers() == 0 {
			t.Fatalf("no block placed: %s", g.Stats().LastErr)
		}
		if _, err := s.Prefill(ids); err != nil {
			t.Fatal(err)
		}
		return s.forwardPrompts
	}
	if n := prompt(short); n != 0 {
		t.Fatalf("before any host expert call a short prompt ran %d prompts token by token", n)
	}
	if !m.hostExperts.Load() {
		t.Fatal("a chunk on hybrid blocks did not mark the model")
	}
	if n := prompt(long); n != 0 {
		t.Fatalf("a %d-token prompt ran token by token", len(long))
	}
	if n := prompt(short); n != 1 {
		t.Fatalf("after host expert calls a %d-token prompt ran %d prompts token by token, want 1", len(short), n)
	}
}
