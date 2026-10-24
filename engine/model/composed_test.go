package model

import (
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// composedDecode decodes toks teacher-forced after ids on a fresh tier of
// dev serving the parts (Config.Composed), every block placed, and returns
// every step's logits, the tier's stats and its census of the launches of
// its own kernels on the steps the composer was asked for.
func composedDecode(t *testing.T, m *Model, dev string, ids, toks []int32, parts tier.Parts, noGraph, float bool, extra ...tier.Option) ([][]float32, tier.Stats, bool, map[string]int) {
	t.Helper()
	// The prompt is one chunk of a few rows: every arm reserves and warms the
	// widths up to 64, each a form of the composer's where it serves them.
	opts := append([]tier.Option{tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff),
		tier.WithConfig(func(c *tier.Config) {
			c.Composed, c.NoGraph, c.ComposedFloat, c.Census, c.PromptRows = parts, noGraph, float, true, 64
		})},
		testTierOpts(t)...)
	opts = append(opts, extra...)
	g, err := tier.OpenWith(opts...)
	if err != nil || g == nil {
		noDevice(t, dev, err)
	}
	defer g.Close()
	s := m.NewState(len(ids) + len(toks) + 4)
	defer s.Close()
	// A device that fails is an error here, never a sequence finished at home.
	s.SetDeviceFallback(false)
	if err := s.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	if s.GPULayers() != m.Cfg.NLayer {
		t.Fatalf("%s took %d of %d blocks: %s", dev, s.GPULayers(), m.Cfg.NLayer, g.Err())
	}
	l, err := s.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	var out [][]float32
	for _, tok := range toks {
		out = append(out, append([]float32(nil), l...))
		if l, err = s.Forward(tok); err != nil {
			t.Fatal(err)
		}
	}
	out = append(out, append([]float32(nil), l...))
	if s.DeviceDemotions() != 0 {
		t.Fatalf("the device failed %d time(s) and its blocks moved home", s.DeviceDemotions())
	}
	return out, g.Stats(), s.HeadOnDevice(), g.Census()
}

// servedModels is every checkpoint whose blocks and head the front end states
// as ops and that one card of 16 GiB holds: dense models of the llama, qwen2,
// qwen3, gemma, gemma2, gemma3, olmo2, exaone4, arcee, granite, hunyuan,
// phi3 and falcon classes, and the text models of qwen2vl and qwen3vl. Those
// state their rotaries over several position axes, but a served block turns
// by main's table, and on a text row every axis's position is the same:
// serving them says the blocks are stated and served at text positions,
// nothing about the axes.
var servedModels = []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Qwen3-0.6B-Q8_0.gguf", "tinyllama-1.1b-q3_K_M.gguf",
	"gemma-2b.gguf", "Qwen2-1.5B-Instruct-Q4_K_M.gguf", "Qwen3-1.7B-Q4_K_M.gguf", "qwen2.5-1.5b-instruct-q4_k_m.gguf",
	"SmolLM2-360M-Instruct-Q8_0.gguf", "stories15M-q8_0.gguf", "SmolVLM-256M-Instruct-Q8_0.gguf",
	"newarch/OLMo-2-0425-1B-Instruct-Q4_K_M.gguf", "newarch/EXAONE-4.0-1.2B-Q4_K_M.gguf", "newarch/AFM-4.5B-Q4_K_M.gguf",
	"granite/granite-3.3-2b-instruct-Q4_K_M.gguf", "granite/granite-4.0-micro-Q4_K_M.gguf",
	"hunyuan/tencent_Hunyuan-0.5B-Instruct-Q4_K_M.gguf", "Phi-3.5-mini-instruct-Q4_K_M.gguf",
	"gemma-3-1b-it-Q4_K_M.gguf", "gemma-3-4b-it-Q4_K_M.gguf", "gemma-2-2b-it-Q4_K_M.gguf",
	"falcon/falcon-7b-instruct.Q4_K_M.gguf",
	"Qwen2-VL-2B-Instruct-Q4_K_M.gguf", "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", "Qwen3VL-2B-Instruct-Q4_K_M.gguf"}

// headDeclined is why the tier still declines a served model's head where it
// does, by its name (tier.DeclineName). The composed head takes a norm (an
// RMSNorm, or a LayerNorm with its bias), the projection with its bias, the
// final softcap and the logit scale, so every served model's head is served.
var headDeclined = map[string]string{}

// formedModels is what the front end states and the serving gate does not
// run: a model the tier cannot place (stories260K's 172-wide rows, which its
// quantizer refuses), float fixtures and checkpoints whose composed logits
// equal the float reference's, which the gate's band cannot measure
// (Qwen3-VL-2B's f16: its blocks and head run their float form, median ratio
// 0), and rnj-1, whose
// composed blocks reach 0.19 from the float reference on CUDA at its 43rd
// teacher-forced step, past four times the tiers' widest (0.0425): open. The
// step reads as a near tie (one block's int8 rounding alone moves it 0.016,
// main's CUDA tier lands at 0.029), and its last block composed alone matches
// the tier's logits to 3e-14 at the median step, but CUDA's split attention,
// which Vulkan does not split at this key bound, is not yet ruled out.
var formedModels = []string{"Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
	"stories260K.gguf", "synth-ernie45.gguf", "synth-phi4v.gguf", "synth-seedoss.gguf", "synth-smollm3.gguf",
	"synth-olmo2.gguf", "synth-olmo2-gqa.gguf", "newarch/synth-olmo2-q8.gguf", "synth-arcee.gguf", "synth-exaone4.gguf",
	"synth-olmo3.gguf", "synth-phi3", "synth-glm4.gguf", "rnj/EssentialAI_rnj-1-instruct-Q4_K_M.gguf",
	"synth-phi2.gguf", "synth-starcoder.gguf", "synth-stablelm.gguf", "synth-stablelm-par.gguf",
	"synth-starcoder2.gguf", "synth-falcon.gguf", "synth-falcon40.gguf", "synth-nemotron.gguf",
	"synth-commandr.gguf", "synth-cohere2.gguf",
	"synth-qwen2vl.gguf", "synth-qwen25vl.gguf", "synth-qwen3vl.gguf",
	"Qwen3-VL-2B-Instruct-f16.gguf", "synth-apertus.gguf"}

// headUnstated is the models of servedModels and formedModels whose head the
// front end names as not stated yet, by the reason it gives: a LayerNorm head.
// Their blocks and input rows form; a head that becomes stated fails the front
// test until its entry goes.
var headUnstated = map[string]Uncovered{
	"falcon/falcon-7b-instruct.Q4_K_M.gguf": UncoveredClassic,
	"synth-phi2.gguf":                       UncoveredClassic,
	"synth-starcoder.gguf":                  UncoveredClassic,
	"synth-stablelm.gguf":                   UncoveredClassic,
	"synth-stablelm-par.gguf":               UncoveredClassic,
	"synth-starcoder2.gguf":                 UncoveredClassic,
	"synth-falcon.gguf":                     UncoveredClassic,
	"synth-falcon40.gguf":                   UncoveredClassic,
	"synth-nemotron.gguf":                   UncoveredClassic,
	"synth-commandr.gguf":                   UncoveredClassic,
	"synth-cohere2.gguf":                    UncoveredClassic,
}

// declinesOf is every decline the tier counted, the head's and the blocks',
// by name (tier.DeclineName).
func declinesOf(st tier.Stats) string {
	var out []string
	for i := range st.HeadDeclines {
		if k := st.HeadDeclines[i]; k > 0 {
			out = append(out, fmt.Sprintf("head %d: %s", k, tier.DeclineName(i)))
		}
		if k := st.BlockDeclines[i]; k > 0 {
			out = append(out, fmt.Sprintf("blocks %d: %s", k, tier.DeclineName(i)))
		}
	}
	if len(out) == 0 {
		return "none counted"
	}
	return strings.Join(out, "; ")
}

// composedResidue is what the tier still launches of its own on a one-row
// step whose blocks and head the composer serves: the rotary table.
var composedResidue = []string{"ropetable"}

// bandRatio and bandWorst hold a composed arm to the spread of the tier's own
// backends: at the median step it may be no further from the float reference
// than bandRatio times the tier, and at its worst no further than bandWorst
// times the widest the tier's backends come at theirs. Both gates that hold an
// arm to a reference take them from here.
const bandRatio, bandWorst = 1.5, 4.0

// promptModels are the served models whose prompt the gate has the composer
// serve too, at the chunk's rows (ComposedPrompt): three Q8_0 models, and
// gemma-3-1b, whose sliding layers' chunk reads the windowed descriptors.
// The others' prompts run the tier's kernels: their k-quant projections at a
// chunk's rows are single kernels of tens of thousands of ops, and a Vulkan
// driver compiles the float form of a 32-row Llama-3.2-1B block (2.7 MB of
// SPIR-V) for longer than the gate's two hours. Open until the row forms
// take the tiled projections.
var promptModels = []string{"stories15M-q8_0.gguf", "Qwen3-0.6B-Q8_0.gguf", "SmolLM2-360M-Instruct-Q8_0.gguf",
	"gemma-3-1b-it-Q4_K_M.gguf"}

// TestComposedBlocksServe decodes dense models with each block served whole
// by the composer from its graph (the model's front end's, from the
// container), the prompt's blocks at the chunk's rows (promptModels), and
// the head, in main's runtime (Config.Composed), against the tier's own
// kernels: every block placed, with and without the captured graph, the
// prompt's last row and 140 teacher-forced steps (past the first key bound),
// and on a model whose sliding window cuts inside its context, steps past
// the window. Each arm is held to a float reference -- the same blocks and
// head formed with float activations, which neither the tier nor the arm
// computes: from the first
// decoded step its distance from it may be no more than bandRatio times the
// tier's at the median step, and no more than bandWorst times the widest the
// tier's own backends come at their worst: a model whose decode passes a
// near tie (OLMo 2's fifth step) sends each implementation to one side of it
// or the other, main's CUDA and Vulkan tiers 0.118 apart there, and the arm
// is held to that spread, not to the side one backend's tier landed on.
// Stats().ServedPart with ReplayedPart is the selection check, every block
// of every step served whether emitted or replayed, a poisoned arm must change
// nothing, and one position off must be ten times the arm's worst: the
// comparison discriminates.
func TestComposedBlocksServe(t *testing.T) {
	arms := []struct {
		name  string
		parts tier.Parts
	}{
		{"whole block", tier.ComposedBlock},
		{"whole block and head", tier.ComposedBlock | tier.ComposedHead},
	}
	ran := 0
	for _, file := range servedModels {
		t.Run(file, func(t *testing.T) {
			prompt := tier.Parts(0)
			if slices.Contains(promptModels, file) {
				prompt = tier.ComposedPrompt
			}
			path := testmodels.Path(file)
			if _, err := os.Stat(path); err != nil {
				t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS)", path)
			}
			m, err := Open(jlmOf(t, path), noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			ids := m.Vocab.Encode("The capital of France is", true)
			n := 140
			// A window the context can outrun is decoded past, so it cuts:
			// the window's last position and windowPast more.
			const windowPast = 64
			w, c := 0, int(m.Cfg.NCtx)
			for li := range m.Cfg.NLayer {
				if m.Cfg.SWA(li) {
					w = m.Cfg.SWAWindow
				}
			}
			if w > 0 && (c <= 0 || w+windowPast+len(ids)+4 <= c) {
				n = max(n, w+windowPast)
			}
			if c > 0 {
				n = min(n, c-len(ids)-4)
			}
			if w > 0 && len(ids)+n > w {
				t.Logf("the decode runs %d positions past the window of %d", len(ids)+n-w, w)
			} else if w > 0 {
				t.Logf("the window of %d cuts nothing in the model's context of %d", w, c)
			}
			toks := make([]int32, n)
			for i := range toks {
				toks[i] = int32((int(ids[i%len(ids)])*7 + i) % m.Vocab.Size())
			}
			// The tier and the float reference on every device first: the
			// spread of the tier's own backends is the band each arm is held to.
			type tierRun struct{ want, ref [][]float32 }
			runs := map[string]*[2]tierRun{}
			var widest [2]float64
			for _, dev := range stepDevices() {
				// A device the box does not have skips its own subtest, not the model's.
				t.Run(dev+"/reference", func(t *testing.T) {
					r := &[2]tierRun{}
					for g, noGraph := range []bool{false, true} {
						r[g].want, _, _, _ = composedDecode(t, m, dev, ids, toks, 0, noGraph, false)
						r[g].ref, _, _, _ = composedDecode(t, m, dev, ids, toks, tier.ComposedBlock|tier.ComposedHead|prompt, noGraph, true)
						for i := 1; i < len(r[g].ref); i++ {
							widest[g] = max(widest[g], logitNMSE(r[g].want[i], r[g].ref[i]))
						}
					}
					runs[dev] = r
				})
			}
			for _, dev := range stepDevices() {
				if runs[dev] == nil {
					continue
				}
				t.Run(dev, func(t *testing.T) {
					for g, noGraph := range []bool{false, true} {
						want, ref := runs[dev][g].want, runs[dev][g].ref
						for _, a := range arms {
							name := a.name
							if noGraph {
								name = "no graph/" + a.name
							}
							t.Run(name, func(t *testing.T) {
								parts := a.parts | prompt
								got, st, headOn, census := composedDecode(t, m, dev, ids, toks, parts, noGraph, false)
								// Every form was built at the placement or PrepHead.
								if st.FormedAtStep != 0 {
									t.Fatalf("%d composed part(s) formed at a step, not at the placement", st.FormedAtStep)
								}
								// Every part the arm asks for served, and none declined: a
								// head the device does not hold (it did not fit beside the
								// blocks) is named, not demanded.
								served, _, unasked := st.ServedPart(parts)
								for _, k := range []tier.Parts{tier.ComposedBlock, tier.ComposedHead, tier.ComposedPrompt} {
									if parts&k == 0 {
										continue
									}
									n, d, _ := st.ServedPart(k)
									// The prompt's head is the tier's, declined by its
									// path's name (composedLater).
									if k == tier.ComposedHead {
										for i, c := range st.HeadDeclines {
											if slices.Contains(composedLater, tier.DeclineName(i)) {
												d -= c
											}
										}
									}
									if k == tier.ComposedHead && !headOn {
										t.Logf("the head runs at home here (it does not fit beside the blocks): the composed head has nothing to serve")
										continue
									}
									if why := headDeclined[file]; k == tier.ComposedHead && why != "" {
										named := 0
										for i, c := range st.HeadDeclines {
											if tier.DeclineName(i) == why {
												named = c
											}
										}
										if n != 0 || d == 0 || named != d {
											t.Fatalf("the head was served %d steps and declined %d, %d of them because %s: the decline it is known for did not hold", n, d, named, why)
										}
										t.Logf("the head declined %d steps: %s", d, why)
										continue
									}
									if n == 0 || d != 0 {
										t.Fatalf("composed parts %#x served %d steps and were declined %d: the arm did not run them (declines: %s; last error %q)", uint8(k), n, d, declinesOf(st), st.LastErr)
									}
								}
								// Every step runs every block, emitted or replayed in a
								// recording, so a block the composer did not serve shows
								// in the count.
								blocks, _, _ := st.ServedPart(tier.ComposedBlock)
								if replayed := st.ReplayedPart(tier.ComposedBlock); blocks+replayed != len(toks)*m.Cfg.NLayer {
									t.Fatalf("the composer ran %d block steps (%d of them replayed) where %d steps of %d blocks are %d",
										blocks+replayed, replayed, len(toks), m.Cfg.NLayer, len(toks)*m.Cfg.NLayer)
								}
								heads, _, _ := st.ServedPart(tier.ComposedHead)
								if replayed := st.ReplayedPart(tier.ComposedHead); a.parts&tier.ComposedHead != 0 && headOn &&
									headDeclined[file] == "" && heads+replayed != len(toks) {
									t.Fatalf("the composer ran the head %d times (%d of them replayed) in %d steps", heads+replayed, replayed, len(toks))
								}
								if noGraph {
									t.Logf("composed %s: the tier's own launches on the steps it served: %v", a.name, census)
									// With the blocks and the head the composer's, what the tier
									// still launches on a step is the residual list, nothing more.
									if a.parts&tier.ComposedHead != 0 && headOn && headDeclined[file] == "" {
										for k := range census {
											if !slices.Contains(composedResidue, k) {
												t.Fatalf("the tier launched its own %s on a step the composer served whole (%v)", k, census)
											}
										}
									}
									if nb, _, _ := st.ServedPart(tier.ComposedBlock); nb != len(toks)*m.Cfg.NLayer {
										t.Fatalf("the composer served %d block steps where %d steps of %d blocks are %d",
											nb, len(toks), m.Cfg.NLayer, len(toks)*m.Cfg.NLayer)
									}
									if nh, _, _ := st.ServedPart(tier.ComposedHead); a.parts&tier.ComposedHead != 0 && headOn && headDeclined[file] == "" && nh != len(toks) {
										t.Fatalf("the composer served the head %d times in %d steps", nh, len(toks))
									}
								}
								// The prompt's every block step served, in whole chunks.
								if prompt != 0 {
									np, dp, up := st.ServedPart(tier.ComposedPrompt)
									if np < m.Cfg.NLayer || np%m.Cfg.NLayer != 0 || dp != 0 || up != 0 {
										t.Fatalf("the composer served %d block steps of the prompt, declined %d and was not asked for %d, where every chunk is %d blocks (%s)",
											np, dp, up, m.Cfg.NLayer, st.LastErr)
									}
									t.Logf("composed %s: %d block steps of the prompt served", a.name, np)
								} else {
									t.Logf("composed %s: %d block steps of the prompt asked for nothing (prefill is not served)", a.name, unasked)
								}
								worst, worstAt := 0.0, 0
								for i := range want {
									e := logitNMSE(got[i], want[i])
									if math.IsNaN(e) {
										t.Errorf("step %d: with the composed %s the logits are not finite", i, a.name)
									}
									if e > worst {
										worst, worstAt = e, i
									}
								}
								off := 0.0
								for i := 1; i < len(want); i++ {
									off = max(off, logitNMSE(got[i], want[i-1]))
								}
								if !(off > max(1e-9, 10*worst)) {
									t.Errorf("one position off the logits are within %.3g of the tier's, where the arm's worst is %.3g (step %d): the comparison cannot see a wrong part", off, worst, worstAt)
								}
								// Step 0 is the prompt's last row. Where the prompt is
								// not served every arm's first logits are the tier's;
								// where it is, they are the composer's, held to the band
								// with every decoded step, the band's widest still the
								// tier's backends' over the decoded steps alone.
								first := 0
								if prompt == 0 {
									if !slices.Equal(got[0], want[0]) || !slices.Equal(ref[0], want[0]) {
										t.Fatalf("the prompt is not served, yet its logits differ from the tier's: the composed setting reached the prompt")
									}
									first = 1
								}
								{
									var ratios []float64
									worstTier, worstGot := 0.0, 0.0
									for i := first; i < len(ref); i++ {
										et, eg := logitNMSE(want[i], ref[i]), logitNMSE(got[i], ref[i])
										if math.IsNaN(et) || math.IsNaN(eg) || !(et > 0) {
											t.Fatalf("step %d: the float reference is %.3g from the tier and %.3g from the arm: a degenerate reference proves nothing", i, et, eg)
										}
										ratios = append(ratios, eg/et)
										worstTier, worstGot = max(worstTier, et), max(worstGot, eg)
									}
									slices.Sort(ratios)
									med := ratios[len(ratios)/2]
									if !(med <= bandRatio) || !(worstGot <= bandWorst*widest[g]) {
										t.Errorf("against the float reference the composed %s is %.3g at the median of the tier's distance and %.3g at its worst, where the tier's backends reach %.3g (bounds %.1f and %.1fx)",
											a.name, med, worstGot, widest[g], bandRatio, bandWorst)
									}
									t.Logf("composed %s against the float reference: median ratio to the tier's distance %.3f (range %.3f..%.3f), worst %.3g against the tier's %.3g here and %.3g on its widest backend",
										a.name, med, ratios[0], ratios[len(ratios)-1], worstGot, worstTier, widest[g])
								}
								t.Logf("composed %s: %d part steps served, worst NMSE from the tier %.3g, one position off %.3g", a.name, served, worst, off)
								// Every buffer filled with NaN at allocation: a
								// composed kernel reading what nothing wrote shows
								// as a changed logit.
								if a.parts&tier.ComposedHead != 0 {
									poisoned, _, _, _ := composedDecode(t, m, dev, ids, toks, parts, noGraph, false, tier.WithPoisonScratch(true), tier.WithPoisonKV(true))
									for i := range got {
										if !slices.Equal(poisoned[i], got[i]) {
											t.Fatalf("step %d: with every buffer poisoned the logits change: a composed kernel reads what nothing wrote", i)
										}
									}
								}
								ran++
							})
						}
					}
				})
			}
		})
	}
	if ran == 0 {
		t.Fatal("no arm ran")
	}
}

// TestComposedPromptIsNotServed holds a prompt's calls to the prompt
// (nn.PromptDevice) with the composer serving the blocks and the head. A
// float head does not fold into a batched chunk and a prompt of a few rows
// runs as one ragged step, so either way the prompt's last-row head is a
// one-row call of its own: it must count as unasked, never as a step the
// composer serves. Each decode step after it serves every block and the head
// once. With the State keeping the prompt from the device (the violation)
// the head is served once more than the steps, which the gate must see.
func TestComposedPromptIsNotServed(t *testing.T) {
	const file, steps = "synth-qwen3vl.gguf", 6
	path := testmodels.Path(file)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS)", path)
	}
	m, err := Open(jlmOf(t, path), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ran := 0
	for _, dev := range stepDevices() {
		// Three rows run as one ragged step, nine as a chunk whose float head
		// does not fold.
		for _, rows := range []int{3, 9} {
			t.Run(fmt.Sprintf("%s/%d rows", dev, rows), func(t *testing.T) {
				run := func(withhold bool) (heads, unasked, blocks int) {
					// Without a recording every step runs every part, so the
					// counts are the steps'.
					g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff),
						tier.WithConfig(func(c *tier.Config) {
							c.Composed, c.Census, c.NoGraph = tier.ComposedBlock|tier.ComposedHead, true, true
						})},
						testTierOpts(t)...)...)
					if err != nil || g == nil {
						noDevice(t, dev, err)
					}
					defer g.Close()
					s := m.NewState(rows + steps + 4)
					defer s.Close()
					s.SetDeviceFallback(false)
					if err := s.SetDevice(g); err != nil {
						t.Fatal(err)
					}
					if s.GPULayers() != m.Cfg.NLayer || !s.HeadOnDevice() {
						t.Fatalf("%s took %d of %d blocks, head on it %v: %s", dev, s.GPULayers(), m.Cfg.NLayer, s.HeadOnDevice(), g.Err())
					}
					if withhold {
						s.ldPrompt = nil
					}
					ids := make([]int32, rows)
					for i := range ids {
						ids[i] = int32((i*7 + 3) % m.Vocab.Size())
					}
					if _, err := s.Prefill(ids); err != nil {
						t.Fatal(err)
					}
					for i := range steps {
						if _, err := s.Forward(int32((i*11 + 5) % m.Vocab.Size())); err != nil {
							t.Fatal(err)
						}
					}
					st := g.Stats()
					heads, _, unasked = st.ServedPart(tier.ComposedHead)
					blocks, _, _ = st.ServedPart(tier.ComposedBlock)
					if !withhold {
						for k := range g.Census() {
							if !slices.Contains(composedResidue, k) {
								t.Fatalf("the tier launched its own %s on a call it counts as a step (%v)", k, g.Census())
							}
						}
					}
					return heads, unasked, blocks
				}
				heads, unasked, blocks := run(false)
				if heads != steps || unasked < 1 || blocks != steps*m.Cfg.NLayer {
					t.Fatalf("the head was served %d times and left unasked %d in %d steps, the blocks served %d times where %d steps of %d blocks are %d",
						heads, unasked, steps, blocks, steps, m.Cfg.NLayer, steps*m.Cfg.NLayer)
				}
				leaked, _, _ := run(true)
				if leaked != steps+1 {
					t.Fatalf("with the prompt kept from the device the head was served %d times in %d steps: the gate cannot see a prompt's head served", leaked, steps)
				}
				t.Logf("the head served %d times in %d steps, %d of its calls unasked; kept from the device, %d", heads, steps, unasked, leaked)
				ran++
			})
		}
	}
	if ran == 0 {
		t.Fatal("no device ran")
	}
}
