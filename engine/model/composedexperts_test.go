package model

import (
	"math"
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// pagedExpertModels are the mixtures whose banks the composed block serves
// from pages on a card (TestComposedExpertPagesServe): the router families'
// fixtures with their matrices in Q8_0 (a float bank has no packed sheets to
// send, and the tier refuses to stream it by that name) and the checkpoints
// one card of 16 GiB runs.
var pagedExpertModels = []string{"Qwen3-MOE-4x0.6B-Q4_K_M.gguf", "olmoe-1b-7b-0924-instruct-Q4_K_M.gguf",
	"synth-glm4moe-hf" + q8Suffix, "synth-qwen2moe-hf" + q8Suffix, "synth-ernie45moe-hf" + q8Suffix,
	"synth-hunyuanmoe-hf" + q8Suffix, "synth-dots1-hf" + q8Suffix, "synth-deepseek" + q8Suffix,
	"synth-deepseek-lite" + q8Suffix, "synth-llama4-q8.gguf", "synth-gptoss.gguf"}

// TestComposedExpertPagesServe serves each mixture's blocks composed with
// its routed banks in pages on the card (tier.Config.StreamExperts): every
// token's selection read back, its experts sent to the block's pages from
// the host's pager, and the composed form reading each through the step's
// page table. A compact bank of the selection, re-read every token, and an
// expert cache holding a quarter of the experts, evicting the least
// recently used, under the host pager's budget of one page too (every
// expert faulted in from the file and evicted again): each must give the
// whole banks' composed logits, every block served and none declined, the
// fills, the cache's misses and the host's page-ins counted. With a page
// read from the wrong expert, or an eviction not re-read, it must not.
func TestComposedExpertPagesServe(t *testing.T) {
	models := pagedExpertModels
	if name := os.Getenv("JITLLM_PAGED_EXPERTS"); name != "" {
		models = append(models, name)
	}
	ran := 0
	for _, file := range models {
		t.Run(file, func(t *testing.T) {
			path := testmodels.Path(file)
			if _, err := os.Stat(fixtureSource(path)); err != nil {
				t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS)", path)
			}
			src := jlmOf(t, path)
			m, err := Open(src, noTune)
			if err != nil {
				t.Fatal(err)
			}
			// The prompt is one row, the rest of it teacher-forced: every
			// later row a decode step the composer serves, so both arms read
			// one history (a prompt of several rows runs the tier's own
			// batched kernels, over the whole banks or the streamed ones).
			prompt := m.Vocab.Encode("The capital of France is", true)
			ids := prompt[:1]
			toks, _ := trainedToks(t, m, prompt, 24)
			toks = append(append([]int32(nil), prompt[1:]...), toks...)
			// The prompt's one row is a decode step too.
			steps := len(toks) + 1
			nExp, used, nLayer := int(m.Cfg.NExpert), int(m.Cfg.NExpertUsed), m.Cfg.NLayer
			moe := 0
			for li := range nLayer {
				if m.Cfg.MoEAt(li) {
					moe++
				}
			}
			m.Close()
			if nExp == 0 || used == 0 {
				t.Fatalf("%s states no routed experts", file)
			}
			slots := max(used+1, nExp/4)
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					decode := func(t *testing.T, budget bool, opts ...tier.Option) ([][]float32, tier.Stats, int64) {
						t.Helper()
						mo := []Option{noTune}
						if budget {
							mo = append(mo, WithPageBudget(1))
						}
						m, err := Open(src, mo...)
						if err != nil {
							t.Fatal(err)
						}
						defer m.Close()
						got, st, _, _ := composedDecode(t, m, dev, ids, toks, tier.ComposedBlock, false, false, opts...)
						_, in, _ := m.PageStats()
						return got, st, in
					}
					want, whole, _ := decode(t, false)
					if n, d, _ := whole.ServedPart(tier.ComposedBlock); n == 0 || d != 0 {
						t.Fatalf("the whole banks served %d block steps and declined %d (%s)", n, d, declinesOf(whole))
					}
					stream := func(slots int) tier.Option {
						return tier.WithConfig(func(c *tier.Config) {
							c.StreamExperts, c.StreamCacheSlots, c.HybridExperts, c.NoHybrid = true, slots, false, true
						})
					}
					for _, arm := range []struct {
						name   string
						slots  int
						budget bool
					}{{"compact", -1, false}, {"cache", slots, false}, {"cache under one page", slots, true}} {
						t.Run(arm.name, func(t *testing.T) {
							got, st, pageIns := decode(t, arm.budget, stream(arm.slots))
							served, declined, _ := st.ServedPart(tier.ComposedBlock)
							switch {
							case st.StreamBlocks < moe:
								t.Fatalf("%d blocks placed streamed of the %d mixtures: the configuration under test was not selected", st.StreamBlocks, moe)
							case declined != 0 || st.BlockDeclines[jit.DeclineExperts] != 0:
								t.Fatalf("%d block steps declined (%s)", declined, declinesOf(st))
							case served != steps*nLayer:
								t.Fatalf("%d block steps served of %d", served, steps*nLayer)
							case st.StreamFills != steps*moe:
								t.Fatalf("%d fills where %d steps of %d streamed blocks read their selection", st.StreamFills, steps, moe)
							case arm.slots > 0 && (st.StreamCacheMisses == 0 || st.StreamCacheHits == 0):
								t.Fatalf("the cache of %d slots hit %d and missed %d: it did not evict and keep", arm.slots, st.StreamCacheHits, st.StreamCacheMisses)
							}
							worst := 0.0
							for i := range want {
								e := logitNMSE(want[i], got[i])
								if math.IsNaN(e) {
									t.Fatalf("step %d: NaN logits", i)
								}
								worst = max(worst, e)
							}
							if worst > 1e-10 {
								t.Fatalf("the banks in pages are %.3e from the whole banks' logits", worst)
							}
							t.Logf("%d block steps served from pages, %d fills, cache %d hits %d misses, %d host page-ins: worst NMSE %.3e against the whole banks",
								served, st.StreamFills, st.StreamCacheHits, st.StreamCacheMisses, pageIns, worst)
							if arm.budget && pageIns <= int64(moe*nExp) {
								t.Fatalf("a one-page budget took %d page-ins for %d experts' pages: the host evicted nothing", pageIns, moe*nExp)
							}
							if arm.budget {
								return
							}
							for _, f := range []tier.PageFault{tier.PageFaultSwap, tier.PageFaultStale} {
								bad, _, _ := decode(t, false, stream(arm.slots), tier.WithPageFault(f))
								far := 0.0
								for i := range want {
									far = max(far, logitNMSE(want[i], bad[i]))
								}
								if !(far > 1e-6) {
									t.Fatalf("with page fault %d the logits are %.3e from the whole banks': the gate cannot see it", f, far)
								}
							}
						})
					}
					ran++
				})
			}
		})
	}
	if ran == 0 {
		t.Fatal("no mixture served its banks from pages: this gate proved nothing")
	}
}

// TestComposedExpertPagesMove moves every block of a mixture, its banks in
// pages on the card (WithExperts("card")), from one card to the other
// mid-sequence and back, composed: the logits are the unmoved control's in
// the band of the two backends, every block step served and none declined,
// and the moved blocks fill their pages from the host's pager on the card
// they land on.
func TestComposedExpertPagesMove(t *testing.T) {
	for _, name := range []string{"synth-glm4moe.gguf", "synth-qwen2moe.gguf", "Qwen3-MOE-4x0.6B-Q4_K_M.gguf"} {
		t.Run(name, func(t *testing.T) {
			path := jlmOf(t, testmodels.Path(name))
			blocksMoveBetweenDevicesWith(t, path, moveWhole, tier.ComposedBlock, func(st tier.Stats) {
				if st.StreamFills == 0 || st.StreamBlocks == 0 {
					t.Fatalf("%d streamed placements and %d fills: the banks were not in pages", st.StreamBlocks, st.StreamFills)
				}
				t.Logf("%d streamed placements, %d fills from pages", st.StreamBlocks, st.StreamFills)
			}, WithExperts("card"), WithStreamTrial(false))
		})
	}
}
