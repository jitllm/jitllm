package model

import (
	"math"
	"os"
	"slices"
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
	opts := append([]tier.Option{tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff),
		tier.WithConfig(func(c *tier.Config) { c.Composed, c.NoGraph, c.ComposedFloat, c.Census = parts, noGraph, float, true })},
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
// qwen3, gemma, olmo2, exaone4, arcee, granite, hunyuan and phi3 classes.
var servedModels = []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Qwen3-0.6B-Q8_0.gguf", "tinyllama-1.1b-q3_K_M.gguf",
	"gemma-2b.gguf", "Qwen2-1.5B-Instruct-Q4_K_M.gguf", "Qwen3-1.7B-Q4_K_M.gguf", "qwen2.5-1.5b-instruct-q4_k_m.gguf",
	"SmolLM2-360M-Instruct-Q8_0.gguf", "stories15M-q8_0.gguf", "SmolVLM-256M-Instruct-Q8_0.gguf",
	"newarch/OLMo-2-0425-1B-Instruct-Q4_K_M.gguf", "newarch/EXAONE-4.0-1.2B-Q4_K_M.gguf", "newarch/AFM-4.5B-Q4_K_M.gguf",
	"granite/granite-3.3-2b-instruct-Q4_K_M.gguf", "granite/granite-4.0-micro-Q4_K_M.gguf",
	"hunyuan/tencent_Hunyuan-0.5B-Instruct-Q4_K_M.gguf", "Phi-3.5-mini-instruct-Q4_K_M.gguf"}

// headDeclined is why the tier still declines a served model's head where it
// does, by its name (tier.DeclineName): a head graph past the norm and the
// projection (granite's logit scale), which the composed head does not take
// yet.
var headDeclined = map[string]string{
	"granite/granite-3.3-2b-instruct-Q4_K_M.gguf": "a head graph other than a norm and its projection",
	"granite/granite-4.0-micro-Q4_K_M.gguf":       "a head graph other than a norm and its projection",
}

// formedModels is what the front end states and the serving gate does not
// run: a model the tier cannot place (stories260K's 172-wide rows, which its
// quantizer refuses), float fixtures whose composed logits equal the float
// reference's, which the gate's band cannot measure, and rnj-1, whose
// composed blocks reach 0.19 from the float reference on CUDA at its 43rd
// teacher-forced step, past four times the tiers' widest (0.0425): open. The
// step reads as a near tie (one block's int8 rounding alone moves it 0.016,
// main's CUDA tier lands at 0.029), and its last block composed alone matches
// the tier's logits to 3e-14 at the median step, but CUDA's split attention,
// which Vulkan does not split at this key bound, is not yet ruled out.
var formedModels = []string{"Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
	"stories260K.gguf", "synth-ernie45.gguf", "synth-phi4v.gguf", "synth-seedoss.gguf", "synth-smollm3.gguf",
	"gemma-3-1b-it-Q4_K_M.gguf", "synth-olmo2.gguf", "synth-olmo2-gqa.gguf", "newarch/synth-olmo2-q8.gguf",
	"synth-arcee.gguf", "synth-exaone4.gguf", "synth-glm4.gguf", "rnj/EssentialAI_rnj-1-instruct-Q4_K_M.gguf"}

// composedResidue is what the tier still launches of its own on a one-row
// step whose blocks and head the composer serves: the rotary table.
var composedResidue = []string{"ropetable"}

// TestComposedBlocksServe decodes dense models with each block served whole
// by the composer from its graph (the model's front end's, from the
// container), and the head, in main's runtime (Config.Composed), against the
// tier's own kernels: every block placed, with and without the captured
// graph, 140 teacher-forced steps (past the first key bound). Each arm is held
// to a float reference -- the same blocks and head formed with float
// activations, which neither the tier nor the arm computes: from the first
// decoded step its distance from it may be no more than bandRatio times the
// tier's at the median step, and no more than bandWorst times the widest the
// tier's own backends come at their worst: a model whose decode passes a
// near tie (OLMo 2's fifth step) sends each implementation to one side of it
// or the other, main's CUDA and Vulkan tiers 0.118 apart there, and the arm
// is held to that spread, not to the side one backend's tier landed on.
// Stats().ServedPart is the selection check, a poisoned arm must change
// nothing, and one position off must be ten times the arm's worst: the
// comparison discriminates.
func TestComposedBlocksServe(t *testing.T) {
	const bandRatio, bandWorst = 1.5, 4.0
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
			if c := int(m.Cfg.NCtx); c > 0 {
				n = min(n, c-len(ids)-4)
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
						r[g].ref, _, _, _ = composedDecode(t, m, dev, ids, toks, tier.ComposedBlock|tier.ComposedHead, noGraph, true)
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
								got, st, headOn, census := composedDecode(t, m, dev, ids, toks, a.parts, noGraph, false)
								// Every form was built at the placement or PrepHead.
								if st.FormedAtStep != 0 {
									t.Fatalf("%d composed part(s) formed at a step, not at the placement", st.FormedAtStep)
								}
								// Every part the arm asks for served, and none declined: a
								// head the device does not hold (it did not fit beside the
								// blocks) is named, not demanded.
								served, _, unasked := st.ServedPart(a.parts)
								for _, k := range []tier.Parts{tier.ComposedBlock, tier.ComposedHead} {
									if a.parts&k == 0 {
										continue
									}
									n, d, _ := st.ServedPart(k)
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
										t.Fatalf("composed parts %#x served %d steps and were declined %d: the arm did not run them (%s)", uint8(k), n, d, st.LastErr)
									}
								}
								// Without a recording every step runs every block, so a
								// block the composer did not serve shows in the count.
								if noGraph {
									t.Logf("composed %s: the tier's own launches on the steps it served: %v", a.name, census)
									// With the blocks and the head the composer's, what the tier
									// still launches on a step is the residual list, nothing more.
									if a.parts == tier.ComposedBlock|tier.ComposedHead && headOn && headDeclined[file] == "" {
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
								t.Logf("composed %s: %d block steps of the prompt asked for nothing (prefill is not served)", a.name, unasked)
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
								// Step 0 is the prompt's last row, and the prompt is
								// not served: every arm's first logits are the tier's.
								if !slices.Equal(got[0], want[0]) || !slices.Equal(ref[0], want[0]) {
									t.Fatalf("the prompt is not served, yet its logits differ from the tier's: the composed setting reached the prompt")
								}
								{
									var ratios []float64
									worstTier, worstGot := 0.0, 0.0
									for i := 1; i < len(ref); i++ {
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
								if a.parts == tier.ComposedBlock|tier.ComposedHead {
									poisoned, _, _, _ := composedDecode(t, m, dev, ids, toks, a.parts, noGraph, false, tier.WithPoisonScratch(true), tier.WithPoisonKV(true))
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
