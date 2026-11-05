package engine

import (
	"math"
	"math/rand"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/nn"
)

// pagedExpertModels are the mixtures whose banks the composed block serves
// from pages on a card (TestComposedExpertPagesServe): the router families'
// fixtures with their matrices in Q8_0 (a float bank has no packed sheets to
// send, and the tier refuses to stream it by that name) and the checkpoints
// one card of 16 GiB runs.
func pagedExpertModels() []string {
	return []string{"Qwen3-MOE-4x0.6B-Q4_K_M.gguf", "olmoe-1b-7b-0924-instruct-Q4_K_M.gguf",
		"synth-glm4moe-hf" + q8Suffix, "synth-qwen2moe-hf" + q8Suffix, "synth-ernie45moe-hf" + q8Suffix,
		"synth-hunyuanmoe-hf" + q8Suffix, "synth-dots1-hf" + q8Suffix, "synth-deepseek" + q8Suffix,
		"synth-deepseek-lite" + q8Suffix, "synth-gptoss.gguf", "synth-llama4-q8.gguf"}
}

// plainBankModels are the mixtures whose streamed banks the tier keeps plain
// whatever cache is stated (cacheSlotsFor): gpt-oss's experts carry biases,
// which the tier's own kernels index by the selection a cache rewrites as
// slots. Their pages are the compact bank's, re-read every token.
func plainBankModels() []string { return []string{"synth-gptoss.gguf"} }

// TestComposedExpertPagesServe serves each mixture's blocks composed with
// its routed banks in pages on the card (gpu.Config.StreamExperts): every
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
	models := pagedExpertModels()
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
			m, err := model.Open(src)
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
					decode := func(t *testing.T, budget bool, opts ...gpu.Option) ([][]float32, gpu.Stats, int64) {
						t.Helper()
						var mo []model.Option
						if budget {
							mo = append(mo, model.WithPageBudget(1))
						}
						instOpts := []Option{WithStreamTrial(false)}
						m, err := model.Open(src, mo...)
						if err != nil {
							t.Fatal(err)
						}
						defer m.Close()
						got, st, _ := composedDecode(t, m, instOpts, dev, ids, toks, nn.ComposedBlock, false, false, opts...)
						_, in, _ := m.PageStats()
						return got, st, in
					}
					stream := func(slots int) gpu.Option {
						return gpu.WithConfig(func(c *gpu.Config) {
							c.StreamExperts, c.StreamCacheSlots = true, slots
						})
					}
					// A card that does not hold every bank whole streams the
					// rest by default: the reference is then those blocks' banks
					// in pages, served like the whole ones.
					want, whole, _ := decode(t, false)
					if n, d := whole.ServedPart(nn.ComposedBlock); n == 0 || d != 0 {
						t.Fatalf("the default placement served %d block steps and declined %d (%s)", n, d, declinesOf(whole))
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
						if arm.slots > 0 && slices.Contains(plainBankModels(), file) {
							continue
						}
						t.Run(arm.name, func(t *testing.T) {
							got, st, pageIns := decode(t, arm.budget, stream(arm.slots))
							served, declined := st.ServedPart(nn.ComposedBlock)
							switch {
							case st.StreamBlocks < moe:
								t.Fatalf("%d blocks placed streamed of the %d mixtures: the configuration under test was not selected", st.StreamBlocks, moe)
							case declined != 0 || st.BlockDeclines()[nn.DeclineExperts] != 0:
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
							for _, f := range []gpu.PageFault{gpu.PageFaultSwap, gpu.PageFaultStale} {
								bad, _, _ := decode(t, false, stream(arm.slots), gpu.WithPageFault(f))
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
		for _, parts := range []nn.Parts{nn.ComposedBlock} {
			t.Run(name, func(t *testing.T) {
				path := jlmOf(t, testmodels.Path(name))
				blocksMoveBetweenDevicesWith(t, path, moveWhole, parts, func(st gpu.Stats) {
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
func cpuExpertModels() []string {
	return []string{"Qwen3-MOE-4x0.6B-Q4_K_M.gguf", "synth-glm4moe-hf" + q8Suffix, "synth-qwen2moe-hf" + q8Suffix,
		"synth-ernie45moe-hf" + q8Suffix, "synth-hunyuanmoe-hf" + q8Suffix, "synth-dots1-hf" + q8Suffix,
		"synth-llama4-q8.gguf", "synth-dbrx.gguf", "synth-gptoss.gguf", "synth-phimoe.gguf",
		"synth-bailingmoe2.gguf", "synth-minimaxm2.gguf",
		"synth-deepseek" + q8Suffix, "synth-deepseek-lite" + q8Suffix}
}

// TestComposedCPUExpertPagesServe runs each mixture's blocks on the CPU
// device composed, its banks read in the host's expert pages at a page's
// stride, teacher-forced, against the CPU's float reference within
// llama.cpp's spread from it (TestComposedCPUBlocksServe's band): every block step served, none
// declined. Under a one-page budget every expert page is evicted and read
// again from the file between the steps that use it, and the composed
// logits are the resident run's bit for bit, the page-ins counted. With the
// pager made to hold another expert's page than the one a kernel reads, the
// logits leave the resident run's.
func TestComposedCPUExpertPagesServe(t *testing.T) {
	const steps, page = 40, 32
	names := cpuExpertModels()
	if extra := os.Getenv("JITLLM_PAGED_EXPERTS"); extra != "" {
		names = append(names, strings.Split(extra, ",")...)
	}
	for _, file := range names {
		t.Run(file, func(t *testing.T) {
			path := jlmOf(t, testmodels.Path(file))
			fm, openOpts := cpuFloat(t, path, page)
			toks := cpuTokens(t, fm, steps, openOpts...)
			fm.Close()
			want, band := cpuReference(t, path, fixtureSource(testmodels.Path(file)), page, toks)
			run := func(f cpuFault, mo ...model.Option) ([][]float32, int64) {
				m, err := model.Open(path, mo...)
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				got := cpuLogits(t, m, toks, f, true, withKVPage(page))
				_, in, _ := m.PageStats()
				return got, in
			}
			got, residentIn := run(cpuFaultNone)
			med, worst, widest := cpuBandTo(t, want, got, band)
			t.Logf("over %d steps the composed CPU is %.3f of the band from the float reference at the median, worst NMSE %.3g where the band's widest is %.3g",
				steps, med, worst, widest)
			if !(med <= bandRatio) || !(worst <= bandWorst*widest) {
				t.Fatalf("the composed CPU mixture leaves the band: median ratio %.3f (bound %.1f), worst %.3g (bound %.1f x %.3g)",
					med, bandRatio, worst, bandWorst, widest)
			}
			paged, pagedIn := run(cpuFaultNone, model.WithPageBudget(1))
			if pagedIn <= residentIn {
				t.Fatalf("a one-page budget took %d page-ins against %d resident: nothing paged", pagedIn, residentIn)
			}
			for i := range got {
				if !slices.Equal(paged[i], got[i]) {
					t.Fatalf("step %d: under a one-page budget the composed logits differ from the resident run's (NMSE %.3e)", i, nmseF32(got[i], paged[i]))
				}
			}
			bad, _ := run(cpuFaultExpert, model.WithPageBudget(1))
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
func cpuExpertPromptModels() []string {
	return append(slices.Clone(cpuExpertModels()), "synth-gemma4-moe.gguf")
}

// cpuMixtureRowsModels are the mixtures whose prompts TestPrefillMatchesForward
// and whose batched steps TestBatchMatchesForward hold on the composed CPU:
// qwen3moe's float fixture and a checkpoint, Gemma 4's mixture beside its
// dense MLP, phimoe's sparsemixer and gpt-oss's MXFP4 banks.
func cpuMixtureRowsModels() []string {
	return []string{"tiny-qwen3moe-f32.gguf", "Qwen3-MOE-4x0.6B-Q4_K_M.gguf", "synth-gemma4-moe.gguf",
		"synth-phimoe.gguf", "synth-gptoss.gguf"}
}

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
	names := cpuExpertPromptModels()
	if extra := os.Getenv("JITLLM_PAGED_EXPERTS"); extra != "" {
		names = append(names, strings.Split(extra, ",")...)
	}
	for _, file := range names {
		t.Run(file, func(t *testing.T) {
			path := jlmOf(t, testmodels.Path(file))
			open := func(mo ...model.Option) (*model.Model, []Option) {
				t.Helper()
				instOpts := []Option{withKVPage(page)}
				m, err := model.Open(path, mo...)
				if err != nil {
					t.Fatal(err)
				}
				return m, instOpts
			}
			fm, openOpts := cpuFloat(t, path, page)
			defer fm.Close()
			// Tokens spread over the vocabulary: a random fixture's greedy
			// tokens repeat, and rows of one token pick one set of experts.
			v := fm.Cfg.NVocab
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
			prompt := func(m *model.Model, f cpuFault, composed bool, io ...Option) [][]float32 {
				t.Helper()
				out := cpuPrompt(t, m, ids, pn, decode, f, composed, io...)
				return append(out, cpuBatch(t, m, prompts, steps, f, composed, io...)...)
			}
			want := prompt(fm, cpuFaultNone, true, openOpts...)
			seqs := []lcppSeq{{ids[:pn], ids[pn : pn+decode], nil}}
			for _, p := range prompts {
				seqs = append(seqs, lcppSeq{p, steps, nil})
			}
			band := lcppBand(t, fixtureSource(testmodels.Path(file)), fm.Cfg.NLayer, want, seqs...)
			run := func(f cpuFault, mo ...model.Option) ([][]float32, int64) {
				t.Helper()
				m, openOpts2 := open(mo...)
				defer m.Close()
				got := prompt(m, f, true, openOpts2...)
				_, in, _ := m.PageStats()
				return got, in
			}
			got, residentIn := run(cpuFaultNone)
			med, worst, widest := cpuBandTo(t, want, got, band)
			t.Logf("prompts of %d rows and a batch: %.3f of the band at the median, worst NMSE %.3g where the band's widest is %.3g",
				pn, med, worst, widest)
			if !(med <= bandRatio) || !(worst <= bandWorst*widest) {
				t.Fatalf("the composed CPU's mixture prompts leave the band: median ratio %.3f (bound %.1f), worst %.3g (bound %.1f x %.3g)",
					med, bandRatio, worst, bandWorst, widest)
			}
			paged, pagedIn := run(cpuFaultNone, model.WithPageBudget(1))
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
				opts []model.Option
			}{
				{"another expert's page held", cpuFaultExpert, []model.Option{model.WithPageBudget(1)}},
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
				g, err := openGPU(t, gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff))
				if err != nil || g == nil {
					noDevice(t, dev, err)
				}
				defer g.Close()
				instOpts := []Option{WithExperts("card"), WithStreamTrial(false)}
				m, err := model.Open(jlmOf(t, testmodels.Path(name)))
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				toks := cpuTokens(t, m, ntok+1, instOpts...)
				nl := m.Cfg.NLayer
				run := func(move bool) ([][]float32, int, [nn.NumDeclines]int) {
					st := testInstance(t, m, ntok+2, instOpts...)
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
					served, declined := st.Account(nn.CPU).Steps.Of(nn.ComposedBlock)
					return out, served, declined
				}
				want, _, _ := run(false)
				// The two devices' spread: the sequence on the CPU alone.
				cs := testInstance(t, m, ntok+2, instOpts...)
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
						t.Fatalf("the CPU declined %d block steps: %s", n, nn.DeclineName(i))
					}
				}
				if served != (back-toCPU)*nl {
					t.Fatalf("the CPU served %d block steps of %d", served, (back-toCPU)*nl)
				}
				if fills := g.Stats().StreamFills - f0; fills == 0 {
					t.Fatal("the card filled no pages: its banks were not in pages")
				}
				composedRan(t, nn.ComposedBlock, nil, g)
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
// second on, once a device has run host experts (Instance.forwardRowsFor): the
// two paths part by the host's and the card's arithmetic (5.7e-04 to 8.4e-03
// on the V100) and sessions two and three agree bit for bit.
func TestStreamedPromptIsTheSameInASecondSession(t *testing.T) {
	for _, name := range []string{"synth-glm4moe-hf" + q8Suffix, "Qwen3-MOE-4x0.6B-Q4_K_M.gguf"} {
		for _, dev := range stepDevices() {
			t.Run(name+"/"+dev, func(t *testing.T) {
				g, err := openGPU(t, gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff))
				if err != nil || g == nil {
					noDevice(t, dev, err)
				}
				defer g.Close()
				instOpts := []Option{WithExperts("card"), WithStreamTrial(false), WithPrefillForwardRows(16)}
				m, err := model.Open(jlmOf(t, testmodels.Path(name)))
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				prompt := m.Vocab.Encode("The capital of France is a city that", true)
				var out [3][]float32
				for i := range out {
					st := testInstance(t, m, len(prompt)+2, instOpts...)
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
// budget, what m's blocks take placed as the arm places them -- its parts,
// graph setting, prompt width and context ctx -- less half of what
// streaming one mixture block saves on the card, measured: the blocks placed
// resident and placed streamed with no expert cache, the difference over the
// mixture blocks.
// Placement keeps the blocks that fit resident and streams the rest, the
// last first, so a deeper cut leaves that block home; and a fixture's banks
// are kilobytes, where the card's allocation pieces can make a streamed
// block cost what a resident one does, which no budget then tells apart.
// The head is placed after the blocks, so the probes leave it home: a cut it
// covered would land on it, and the arm runs it at home.
func streamOpts(t *testing.T, m *model.Model, dev string, a streamArm, parts nn.Parts, noGraph bool, ctx int, io ...Option) []gpu.Option {
	t.Helper()
	switch a {
	case streamBudget:
		probe := func(stream bool) gpu.Stats {
			g, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff),
				gpu.WithConfig(func(c *gpu.Config) {
					c.NoGraph, c.PromptRows = noGraph, 64
					// Streamed at its floor: no cache, each block's plain bank
					// of the selection's sheets. With no budget the caches
					// sizeAutoCaches gives would take the card's whole room,
					// every expert on Qwen3-MOE-4x0.6B; under the arm's budget
					// they take what the budget leaves.
					if stream {
						c.StreamExperts, c.StreamCacheSlots = true, -1
					}
				})},
				testTierOpts(t)...)...)
			if err != nil || g == nil {
				noDevice(t, dev, err)
			}
			defer g.Close()
			s := testInstance(t, m, ctx, append(append([]Option(nil), io...), WithHeadPlacement(false))...)
			defer s.Close()
			s.SetDeviceFallback(false)
			if err := s.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			st := g.Stats()
			if s.GPULayers() != m.Cfg.NLayer || stream && st.StreamBlocks == 0 {
				t.Fatalf("%s holds %d of %d blocks, %d streamed: %s", dev, s.GPULayers(), m.Cfg.NLayer, st.StreamBlocks, g.Err())
			}
			return st
		}
		mixed := 0
		for li := range m.Cfg.NLayer {
			if _, b := m.BlockBytes(li); b > 0 {
				mixed++
			}
		}
		rs, ss := probe(false), probe(true)
		resident, streamed := rs.BudgetUsed, ss.BudgetUsed
		if mixed == 0 || streamed >= resident {
			t.Fatalf("streaming saves the card nothing here, so no budget tells the two apart: %d bytes resident (scratch %d, composed %d, KV pool %d, rounding %d), "+
				"%d with every mixture block streamed (scratch %d, composed %d, KV pool %d, rounding %d, caches %d, %d streamed)",
				resident, rs.ScratchBytes, rs.ComposedBytes, rs.KVPoolBytes, rs.RoundingBytes,
				streamed, ss.ScratchBytes, ss.ComposedBytes, ss.KVPoolBytes, ss.RoundingBytes, ss.CacheBytes, ss.StreamBlocks)
		}
		cut := (resident - streamed) / uint64(mixed) / 2
		t.Logf("experts under a budget: %d bytes resident, %d streamed over %d mixture blocks; the budget %d", resident, streamed, mixed, resident-cut)
		return []gpu.Option{gpu.WithBudget(resident - cut)}
	case streamCache:
		e, k := int(m.Cfg.NExpert), int(m.Cfg.NExpertUsed)
		return []gpu.Option{gpu.WithConfig(func(c *gpu.Config) {
			c.StreamExperts, c.StreamCacheSlots = true, min(e, max(2*k, e/2))
		})}
	}
	return nil
}

// streamSelected fails t unless a run of arm a streamed its mixture blocks
// and with a cache filled a call of several rows at once.
func streamSelected(t *testing.T, st gpu.Stats, a streamArm) {
	t.Helper()
	switch {
	case st.StreamBlocks == 0 || st.StreamFills == 0:
		t.Fatalf("%d blocks placed streamed, %d fills: the configuration under test was not selected "+
			"(%d bytes used: scratch %d, composed %d, KV pool %d; %d declined (%s), %d page-ins, %d page-outs)",
			st.StreamBlocks, st.StreamFills, st.BudgetUsed, st.ScratchBytes, st.ComposedBytes, st.KVPoolBytes,
			st.Declined, st.DeclineWhy, st.PageIns, st.PageOuts)
	case a == streamCache && st.StreamRowFills == 0:
		t.Fatalf("no call of several rows filled its pages (%d fills)", st.StreamFills)
	}
}
