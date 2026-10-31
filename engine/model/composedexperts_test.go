package model

import (
	"math"
	"math/rand"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
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
	"synth-deepseek-lite" + q8Suffix, "synth-gptoss.gguf", "synth-llama4-q8.gguf"}

// plainBankModels are the mixtures whose streamed banks the tier keeps plain
// whatever cache is stated (cacheSlotsFor): gpt-oss's experts carry biases,
// which the tier's own kernels index by the selection a cache rewrites as
// slots. Their pages are the compact bank's, re-read every token.
var plainBankModels = []string{"synth-gptoss.gguf"}

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
	// JITLLM_PAGED_EXPERTS adds checkpoints past one card of 16 GiB,
	// comma-separated (docs/testing.md).
	if names := os.Getenv("JITLLM_PAGED_EXPERTS"); names != "" {
		models = append(models, strings.Split(names, ",")...)
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
						mo := []Option{noTune, WithStreamTrial(false)}
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
					stream := func(slots int) tier.Option {
						return tier.WithConfig(func(c *tier.Config) {
							c.StreamExperts, c.StreamCacheSlots, c.HybridExperts, c.NoHybrid = true, slots, false, true
						})
					}
					// A card that does not hold every bank whole streams the
					// rest by default: the reference is then those blocks' banks
					// in pages, served like the whole ones.
					want, whole, _ := decode(t, false)
					if n, d, _ := whole.ServedPart(tier.ComposedBlock); n == 0 || d != 0 || whole.HybridRuns != 0 {
						t.Fatalf("the default placement served %d block steps, declined %d and ran %d on the host (%s)", n, d, whole.HybridRuns, declinesOf(whole))
					}
					if whole.StreamBlocks > 0 {
						t.Logf("the card holds %d blocks streamed by default, %d fills", whole.StreamBlocks, whole.StreamFills)
					}
					resident := map[int]int64{}
					for _, arm := range []struct {
						name   string
						slots  int
						budget bool
					}{{"compact", -1, false}, {"compact under one page", -1, true}, {"cache", slots, false}, {"cache under one page", slots, true}} {
						if arm.slots > 0 && slices.Contains(plainBankModels, file) {
							continue
						}
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
							if !arm.budget {
								resident[arm.slots] = pageIns
							} else if pageIns <= resident[arm.slots] {
								t.Fatalf("a one-page budget took %d page-ins against %d resident: the host evicted nothing", pageIns, resident[arm.slots])
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
// they land on. The prompt goes through the decode path in both runs, as
// TestStreamedPromptIsTheSameInASecondSession's does: the control's, a model's
// first, would be a chunk with its streamed experts on the host.
func TestComposedExpertPagesMove(t *testing.T) {
	for _, name := range []string{"synth-glm4moe-hf" + q8Suffix, "synth-qwen2moe-hf" + q8Suffix, "Qwen3-MOE-4x0.6B-Q4_K_M.gguf"} {
		// The tier's own kernels too: the moves are the tier's, whoever
		// runs the blocks.
		for _, parts := range []tier.Parts{0, tier.ComposedBlock} {
			t.Run(composedArm(name, parts), func(t *testing.T) {
				path := jlmOf(t, testmodels.Path(name))
				blocksMoveBetweenDevicesWith(t, path, moveWhole, parts, func(st tier.Stats) {
					if st.StreamFills == 0 || st.StreamBlocks == 0 {
						t.Fatalf("%d streamed placements and %d fills: the banks were not in pages", st.StreamBlocks, st.StreamFills)
					}
					t.Logf("%d streamed placements, %d fills from pages", st.StreamBlocks, st.StreamFills)
				}, WithExperts("card"), WithStreamTrial(false), WithPrefillForwardRows(16))
			})
		}
	}
}

// cpuExpertModels are the mixtures whose blocks the composer serves on the
// CPU device with their banks read in the host's expert pages
// (TestComposedCPUExpertPagesServe): the router families' fixtures, Llama
// 4's input-weighted top-1, dbrx's clamped float banks, phimoe's
// sparsemixer (its weights a stated softmax per pick), Ling 2.0's grouped
// sigmoid router and MiniMax-M2's among them, a checkpoint, and DeepSeek's
// mixtures beside latent attention: V3's grouped sigmoid router and V2's
// softmax one.
var cpuExpertModels = []string{"Qwen3-MOE-4x0.6B-Q4_K_M.gguf", "synth-glm4moe-hf" + q8Suffix, "synth-qwen2moe-hf" + q8Suffix,
	"synth-ernie45moe-hf" + q8Suffix, "synth-hunyuanmoe-hf" + q8Suffix, "synth-dots1-hf" + q8Suffix,
	"synth-llama4-q8.gguf", "synth-dbrx.gguf", "synth-gptoss.gguf", "synth-phimoe.gguf",
	"synth-bailingmoe2.gguf", "synth-minimaxm2.gguf",
	"synth-deepseek" + q8Suffix, "synth-deepseek-lite" + q8Suffix}

// TestComposedCPUExpertPagesServe runs each mixture's blocks on the CPU
// device composed, its banks read in the host's expert pages at a page's
// stride, teacher-forced, against main's kernels within main's own spread
// (TestComposedCPUBlocksServe's band): every block step served, none
// declined. Under a one-page budget every expert page is evicted and read
// again from the file between the steps that use it, and the composed
// logits are the resident run's bit for bit, the page-ins counted. With the
// pager made to hold another expert's page than the one a kernel reads, the
// logits leave the resident run's.
func TestComposedCPUExpertPagesServe(t *testing.T) {
	const steps, page = 40, 32
	names := cpuExpertModels
	if extra := os.Getenv("JITLLM_PAGED_EXPERTS"); extra != "" {
		names = append(names, strings.Split(extra, ",")...)
	}
	for _, file := range names {
		t.Run(file, func(t *testing.T) {
			path := jlmOf(t, testmodels.Path(file))
			ref, err := Open(path, noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer ref.Close()
			ref.setKVPageForTest(page)
			toks := cpuTokens(t, ref, steps)
			want := cpuLogits(t, ref, toks, cpuFaultNone, false)
			cm, err := Open(path, noTune, WithJITOptions(nn.WithActWindow(cpuControlWindow)))
			if err != nil {
				t.Fatal(err)
			}
			defer cm.Close()
			cm.setKVPageForTest(page)
			ctl := cpuLogits(t, cm, toks, cpuFaultNone, false)
			run := func(f cpuFault, opts ...Option) ([][]float32, int64) {
				m, err := Open(path, append([]Option{noTune, WithComposedCPU(true)}, opts...)...)
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				m.setKVPageForTest(page)
				got := cpuLogits(t, m, toks, f, true)
				_, in, _ := m.PageStats()
				return got, in
			}
			got, residentIn := run(cpuFaultNone)
			float := true
			for i := range ctl {
				float = float && slices.Equal(ctl[i], want[i])
			}
			if float {
				// A float model quantizes nothing: main's control is main, and
				// the composed CPU is held to it within f32's reduction order.
				worst := 0.0
				for i := range want {
					worst = max(worst, logitNMSE(got[i], want[i]))
				}
				t.Logf("over %d steps the composed CPU's float mixture is %.3g from main at the worst", steps, worst)
				if math.IsNaN(worst) || worst > 1e-9 {
					t.Fatalf("the composed CPU's float mixture is %.3g from main", worst)
				}
			} else {
				med, worst, widest := cpuBand(t, want, got, ctl)
				t.Logf("over %d steps the composed CPU is %.3f of main's spread from main at the median, worst NMSE %.3g where main's control reaches %.3g",
					steps, med, worst, widest)
				if !(med <= bandRatio) || !(worst <= bandWorst*widest) {
					t.Fatalf("the composed CPU mixture leaves main's band: median ratio %.3f (bound %.1f), worst %.3g (bound %.1f x %.3g)",
						med, bandRatio, worst, bandWorst, widest)
				}
			}
			paged, pagedIn := run(cpuFaultNone, WithPageBudget(1))
			if pagedIn <= residentIn {
				t.Fatalf("a one-page budget took %d page-ins against %d resident: nothing paged", pagedIn, residentIn)
			}
			for i := range got {
				if !slices.Equal(paged[i], got[i]) {
					t.Fatalf("step %d: under a one-page budget the composed logits differ from the resident run's (NMSE %.3e)", i, nmseF32(got[i], paged[i]))
				}
			}
			bad, _ := run(cpuFaultExpert, WithPageBudget(1))
			far := 0.0
			for i := range got {
				far = max(far, logitNMSE(got[i], bad[i]))
			}
			if !(far > 1e-6) && !math.IsNaN(far) {
				t.Fatalf("with another expert's page held the logits are %.3e from the resident run's: the gate cannot see it", far)
			}
			t.Logf("%d page-ins under one page against %d resident, every logit identical; the wrong page held moves them %.3e", pagedIn, residentIn, far)
		})
	}
}

// cpuExpertPromptModels are the mixtures whose prompts and batched steps
// TestComposedCPUExpertPromptsServe holds on the CPU device: cpuExpertModels
// and Gemma 4's mixture beside its dense MLP. JITLLM_PAGED_EXPERTS adds more.
var cpuExpertPromptModels = append(slices.Clone(cpuExpertModels), "synth-gemma4-moe.gguf")

// cpuMixtureRowsModels are the mixtures whose prompts TestPrefillMatchesForward
// and whose batched steps TestBatchMatchesForward hold on the composed CPU:
// qwen3moe's float fixture and a checkpoint, Gemma 4's mixture beside its
// dense MLP, phimoe's sparsemixer and gpt-oss's MXFP4 banks.
var cpuMixtureRowsModels = []string{"tiny-qwen3moe-f32.gguf", "Qwen3-MOE-4x0.6B-Q4_K_M.gguf", "synth-gemma4-moe.gguf",
	"synth-phimoe.gguf", "synth-gptoss.gguf"}

// TestComposedCPUExpertPromptsServe runs each mixture's prompts and batched
// steps on the CPU device composed: a prompt of more rows than a call of
// the rows' form holds and its decode, and a batch of three sequences, each
// prompted into its own slot and stepped together. Every call holds the
// experts its rows pick, each once (cpuExperts), every block step served
// and none declined, the logits within main's own spread (or, for a float
// mixture, within f32's reduction order); under a one-page budget the
// experts are evicted and read again mid-prompt and every logit is the
// resident run's bit for bit. Each violation -- another expert's page held
// than one a row reads, a row given another row's picks, each row's first
// pick's weight dropped -- moves the prompt's logits.
func TestComposedCPUExpertPromptsServe(t *testing.T) {
	const page, pn, decode = 32, 72, 4
	names := cpuExpertPromptModels
	if extra := os.Getenv("JITLLM_PAGED_EXPERTS"); extra != "" {
		names = append(names, strings.Split(extra, ",")...)
	}
	for _, file := range names {
		t.Run(file, func(t *testing.T) {
			path := jlmOf(t, testmodels.Path(file))
			open := func(opts ...Option) *Model {
				t.Helper()
				m, err := Open(path, append([]Option{noTune}, opts...)...)
				if err != nil {
					t.Fatal(err)
				}
				m.setKVPageForTest(page)
				return m
			}
			ref := open()
			defer ref.Close()
			// Tokens spread over the vocabulary: a random fixture's greedy
			// tokens repeat, and rows of one token pick one set of experts.
			v := ref.Cfg.NVocab
			rng := rand.New(rand.NewSource(7))
			ids := make([]int32, 140)
			for i := range ids {
				ids[i] = int32(v/8 + rng.Intn(v/2-v/8))
			}
			prompts := [][]int32{ids[:7], ids[10:43], ids[50 : pn+50]}
			steps := ids[pn+50 : pn+50+decode]
			// The prompt's last row and each batched sequence's prompt row:
			// logits the prompt's rows' form alone made.
			promptRows := []int{0}
			for i := range prompts {
				promptRows = append(promptRows, 1+decode+i*(1+decode))
			}
			prompt := func(m *Model, f cpuFault, composed bool) [][]float32 {
				t.Helper()
				out := cpuPrompt(t, m, ids, pn, decode, f, composed)
				return append(out, cpuBatch(t, m, prompts, steps, f, composed)...)
			}
			want := prompt(ref, cpuFaultNone, false)
			ctlm := open(WithJITOptions(nn.WithActWindow(cpuControlWindow)))
			ctl := prompt(ctlm, cpuFaultNone, false)
			ctlm.Close()
			run := func(f cpuFault, opts ...Option) ([][]float32, int64) {
				t.Helper()
				m := open(append([]Option{WithComposedCPU(true)}, opts...)...)
				defer m.Close()
				got := prompt(m, f, true)
				_, in, _ := m.PageStats()
				return got, in
			}
			got, residentIn := run(cpuFaultNone)
			float := true
			for i := range ctl {
				float = float && slices.Equal(ctl[i], want[i])
			}
			if float {
				// A float mixture quantizes nothing and main's control is main:
				// the rows' form is held to main within the float prompts'
				// bound (TestGemma4ServesComposedOnCPU's), its reduction order
				// over a call's rows not main's.
				const floatBound = 1e-5
				worst := 0.0
				for i := range want {
					worst = max(worst, logitNMSE(got[i], want[i]))
				}
				t.Logf("prompts of %d rows and a batch: the composed CPU's float mixture is %.3g from main at the worst", pn, worst)
				if math.IsNaN(worst) || !(worst < floatBound) {
					t.Fatalf("the composed CPU's float mixture's prompts are %.3g from main (bound %g)", worst, floatBound)
				}
			} else {
				med, worst, widest := cpuBand(t, want, got, ctl)
				t.Logf("prompts of %d rows and a batch: %.3f of main's spread at the median, worst NMSE %.3g where main's control reaches %.3g",
					pn, med, worst, widest)
				if !(med <= bandRatio) || !(worst <= bandWorst*widest) {
					t.Fatalf("the composed CPU's mixture prompts leave main's band: median ratio %.3f (bound %.1f), worst %.3g (bound %.1f x %.3g)",
						med, bandRatio, worst, bandWorst, widest)
				}
			}
			paged, pagedIn := run(cpuFaultNone, WithPageBudget(1))
			if pagedIn <= residentIn {
				t.Fatalf("a one-page budget took %d page-ins against %d resident: nothing paged", pagedIn, residentIn)
			}
			for i := range got {
				if !slices.Equal(paged[i], got[i]) {
					t.Fatalf("row %d: under a one-page budget the composed logits differ from the resident run's (NMSE %.3e)", i, nmseF32(got[i], paged[i]))
				}
			}
			t.Logf("%d page-ins under one page against %d resident, every logit identical", pagedIn, residentIn)
			for _, v := range []struct {
				name string
				f    cpuFault
				opts []Option
			}{
				{"another expert's page held", cpuFaultExpert, []Option{WithPageBudget(1)}},
				{"a row given another row's picks", cpuFaultPicks, nil},
				{"each row's first pick's weight dropped", cpuFaultWeight, nil},
			} {
				bad, _ := run(v.f, v.opts...)
				far := 0.0
				for _, i := range promptRows {
					far = max(far, logitNMSE(got[i], bad[i]))
				}
				t.Logf("violation %q: the prompts' rows move %.3e", v.name, far)
				if !(far > 1e-6) && !math.IsNaN(far) {
					t.Errorf("%s: the prompts' logits move %.3e: the gate cannot see it", v.name, far)
				}
			}
		})
	}
}

// TestComposedExpertPagesMoveToTheCPU moves every block of a mixture
// mid-sequence from a card, its banks in pages there, to the CPU device,
// its banks read in the host's expert pages, and back, composed on both:
// before the first move the logits are the unmoved run's bit for bit, after
// it in the band of the two devices' forms, every block step served on the
// card and on the CPU and none declined, the experts filled from pages on
// the card and read from them on the CPU.
func TestComposedExpertPagesMoveToTheCPU(t *testing.T) {
	const ntok, toCPU, back = 12, 4, 8
	for _, name := range []string{"synth-glm4moe-hf" + q8Suffix, "Qwen3-MOE-4x0.6B-Q4_K_M.gguf"} {
		for _, dev := range stepDevices() {
			t.Run(name+"/"+dev, func(t *testing.T) {
				g, err := tier.OpenWith(tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff), composedOpt(tier.ComposedBlock))
				if err != nil || g == nil {
					noDevice(t, dev, err)
				}
				defer g.Close()
				m, err := Open(jlmOf(t, testmodels.Path(name)), noTune, WithComposedCPU(true), WithExperts("card"), WithStreamTrial(false))
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				toks := cpuTokens(t, m, ntok+1)
				nl := m.Cfg.NLayer
				run := func(move bool) ([][]float32, int, [jit.NumDeclines]int) {
					st := m.NewState(ntok + 2)
					defer st.Close()
					st.SetDeviceFallback(false)
					if err := st.SetDevice(g); err != nil {
						t.Fatal(err)
					}
					card, _ := st.DeviceOf(0)
					if st.GPULayers() != nl || card == "" || onCPU(card) {
						t.Fatalf("%s took %d of %d blocks", dev, st.GPULayers(), nl)
					}
					var out [][]float32
					for i, tk := range toks[:ntok] {
						if move && (i == toCPU || i == back) {
							to := "cpu"
							if i == back {
								to = card
							}
							if _, err := st.MoveBlocks(0, nl, to); err != nil {
								t.Fatalf("to %s: %v", to, err)
							}
						}
						lg, err := st.Forward(tk)
						if err != nil {
							t.Fatal(err)
						}
						out = append(out, slices.Clone(lg))
					}
					served, declined := st.CPUBlockSteps()
					return out, served, declined
				}
				want, _, _ := run(false)
				// The two devices' spread: the sequence on the CPU alone.
				cs := m.NewState(ntok + 2)
				widest := 0.0
				for i, tk := range toks[:ntok] {
					lg, err := cs.Forward(tk)
					if err != nil {
						t.Fatal(err)
					}
					widest = max(widest, logitNMSE(want[i], lg))
				}
				cs.Close()
				f0 := g.Stats().StreamFills
				got, served, declined := run(true)
				for i, n := range declined {
					if n > 0 {
						t.Fatalf("the CPU declined %d block steps: %s", n, jit.DeclineName(i))
					}
				}
				if served != (back-toCPU)*nl {
					t.Fatalf("the CPU served %d block steps of %d", served, (back-toCPU)*nl)
				}
				if fills := g.Stats().StreamFills - f0; fills == 0 {
					t.Fatal("the card filled no pages: its banks were not in pages")
				}
				composedRan(t, tier.ComposedBlock, nil, g)
				worst := 0.0
				for i := range want {
					e := logitNMSE(want[i], got[i])
					switch {
					case i < toCPU && !slices.Equal(want[i], got[i]):
						t.Fatalf("token %d is before any move and its logits differ (%.3e)", i, e)
					case math.IsNaN(e) || e > bandWorst*widest:
						t.Fatalf("token %d: the moved logits are %.3e from the unmoved run's, past %.1f x the devices' spread %.3e", i, e, bandWorst, widest)
					}
					worst = max(worst, e)
				}
				t.Logf("%d block steps on the CPU from pages; worst logit NMSE after the moves %.3e, the devices' spread %.3e", served, worst, widest)
			})
		}
	}
}

// TestStreamedPromptIsTheSameInASecondSession prefills one prompt in three
// sessions one after the other on one card, every mixture block's experts
// sent to the card (WithExperts("card")): the second starts with the first's
// experts in the card's caches, and every session's logits must be the
// first's bit for bit, on the tier's kernels and composed.
//
// The prompt goes through the decode path in every session
// (WithPrefillForwardRows). Left unset, a model takes a chunk for the first
// prompt, its streamed experts run on the host, and the decode path from the
// second on, once a device has run host experts (State.forwardRowsFor): the
// two paths part by the host's and the card's arithmetic (5.7e-04 to 8.4e-03
// on the V100) and sessions two and three agree bit for bit.
func TestStreamedPromptIsTheSameInASecondSession(t *testing.T) {
	for _, name := range []string{"synth-glm4moe-hf" + q8Suffix, "Qwen3-MOE-4x0.6B-Q4_K_M.gguf"} {
		for _, dev := range stepDevices() {
			for _, parts := range []tier.Parts{0, tier.ComposedBlock} {
				t.Run(composedArm(name, parts)+"/"+dev, func(t *testing.T) {
					g, err := tier.OpenWith(tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff), composedOpt(parts))
					if err != nil || g == nil {
						noDevice(t, dev, err)
					}
					defer g.Close()
					m, err := Open(jlmOf(t, testmodels.Path(name)), noTune, WithExperts("card"), WithStreamTrial(false),
						WithPrefillForwardRows(16))
					if err != nil {
						t.Fatal(err)
					}
					defer m.Close()
					prompt := m.Vocab.Encode("The capital of France is a city that", true)
					var out [3][]float32
					for i := range out {
						st := m.NewState(len(prompt) + 2)
						st.SetDeviceFallback(false)
						if err := st.SetDevice(g); err != nil {
							t.Fatal(err)
						}
						lg, err := st.Prefill(prompt)
						if err != nil {
							t.Fatal(err)
						}
						out[i] = slices.Clone(lg)
						st.Close()
					}
					if g.Stats().StreamBlocks == 0 {
						t.Fatal("no block was streamed: the configuration under test was not selected")
					}
					for i := 1; i < len(out); i++ {
						if !slices.Equal(out[0], out[i]) {
							t.Fatalf("session %d's prompt is %.3e from the first's", i+1, logitNMSE(out[0], out[i]))
						}
					}
				})
			}
		}
	}
}

// streamArm is where a serving arm puts a mixture's routed experts: whole on
// the card (0), in pages under a budget the banks do not fit beside the
// blocks (the auto-stream default), or in an expert cache of half the
// experts and at least two rows' selections, so a prompt's calls of several
// rows fill their pages at once.
type streamArm int

const (
	streamBudget streamArm = iota + 1
	streamCache
)

// streamOpts is the tier options of arm a for mixture m on dev: for the
// budget, what m's blocks take there resident, with the arm's parts and graph
// setting and the narrowest prompt scratch, less half its routed banks. The
// placement halves the prompt width until the blocks fit, so a budget taken
// at a wider scratch is absorbed by the scratch and no bank ever misses room
// (a fixture's banks are kilobytes beside it).
func streamOpts(t *testing.T, m *Model, dev string, a streamArm, parts tier.Parts, noGraph bool) []tier.Option {
	t.Helper()
	switch a {
	case streamBudget:
		g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff),
			tier.WithConfig(func(c *tier.Config) { c.Composed, c.NoGraph, c.Census, c.PromptRows = parts, noGraph, true, 8 })},
			testTierOpts(t)...)...)
		if err != nil || g == nil {
			noDevice(t, dev, err)
		}
		defer g.Close()
		s := m.NewState(64)
		defer s.Close()
		s.SetDeviceFallback(false)
		if err := s.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		var bank uint64
		for li := range m.Cfg.NLayer {
			_, b := m.blockBytes(li)
			bank += b
		}
		used := g.Stats().BudgetUsed
		if s.GPULayers() != m.Cfg.NLayer || bank == 0 || used <= bank {
			t.Fatalf("%s holds %d of %d blocks in %d bytes, %d of them routed banks", dev, s.GPULayers(), m.Cfg.NLayer, used, bank)
		}
		t.Logf("experts under a budget: %d bytes resident with %d of routed banks; the budget %d", used, bank, used-bank/2)
		return []tier.Option{tier.WithBudget(used - bank/2)}
	case streamCache:
		e, k := int(m.Cfg.NExpert), int(m.Cfg.NExpertUsed)
		return []tier.Option{tier.WithConfig(func(c *tier.Config) {
			c.StreamExperts, c.StreamCacheSlots, c.HybridExperts, c.NoHybrid = true, min(e, max(2*k, e/2)), false, true
		})}
	}
	return nil
}

// streamSelected fails t unless a run of arm a streamed its mixture blocks,
// ran none of their experts on the host, and with a cache filled a call of
// several rows at once.
func streamSelected(t *testing.T, st tier.Stats, a streamArm) {
	t.Helper()
	switch {
	case st.StreamBlocks == 0 || st.StreamFills == 0:
		t.Fatalf("%d blocks placed streamed, %d fills: the configuration under test was not selected "+
			"(%d bytes used, %d declined (%s), %d page-ins, %d page-outs)",
			st.StreamBlocks, st.StreamFills, st.BudgetUsed, st.Declined, st.DeclineWhy, st.PageIns, st.PageOuts)
	case st.HybridRuns != 0:
		t.Fatalf("%d block steps ran their experts on the host", st.HybridRuns)
	case a == streamCache && st.StreamRowFills == 0:
		t.Fatalf("no call of several rows filled its pages (%d fills)", st.StreamFills)
	}
}
