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
// as ops and that one card of 4 GiB holds: plain dense models of the llama,
// qwen2, qwen3, gemma, ernie4_5 and seed_oss classes, real and fixtures.
var servedModels = []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Qwen3-0.6B-Q8_0.gguf", "tinyllama-1.1b-q3_K_M.gguf",
	"gemma-2b.gguf", "Qwen2-1.5B-Instruct-Q4_K_M.gguf", "Qwen3-1.7B-Q4_K_M.gguf", "qwen2.5-1.5b-instruct-q4_k_m.gguf",
	"SmolLM2-360M-Instruct-Q8_0.gguf", "stories15M-q8_0.gguf", "SmolVLM-256M-Instruct-Q8_0.gguf"}

// formedModels is what the front end states and the serving gate does not
// run: a model the tier cannot place (stories260K's 172-wide rows, which its
// quantizer refuses), and float fixtures whose composed logits equal the float
// reference's, which the gate's band cannot measure.
var formedModels = []string{"Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
	"stories260K.gguf", "synth-ernie45.gguf", "synth-phi4v.gguf", "synth-seedoss.gguf", "synth-smollm3.gguf"}

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
// tier's at the median step and bandWorst times at the worst. Stats().ServedPart is the selection check,
// a poisoned arm must change nothing, and one position off must fail: the
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
			for _, dev := range stepDevices() {
				// A device the box does not have skips its own subtest, not the model's.
				t.Run(dev, func(t *testing.T) {
					for _, noGraph := range []bool{false, true} {
						want, _, _, _ := composedDecode(t, m, dev, ids, toks, 0, noGraph, false)
						ref, _, _, _ := composedDecode(t, m, dev, ids, toks, tier.ComposedBlock|tier.ComposedHead, noGraph, true)
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
									if a.parts == tier.ComposedBlock|tier.ComposedHead && headOn {
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
									if nh, _, _ := st.ServedPart(tier.ComposedHead); a.parts&tier.ComposedHead != 0 && headOn && nh != len(toks) {
										t.Fatalf("the composer served the head %d times in %d steps", nh, len(toks))
									}
								}
								t.Logf("composed %s: %d block steps of the prompt asked for nothing (prefill is not served)", a.name, unasked)
								worst := 0.0
								for i := range want {
									e := logitNMSE(got[i], want[i])
									if math.IsNaN(e) {
										t.Errorf("step %d: with the composed %s the logits are not finite", i, a.name)
									}
									worst = max(worst, e)
								}
								off := 0.0
								for i := 1; i < len(want); i++ {
									off = max(off, logitNMSE(got[i], want[i-1]))
								}
								if !(off > max(1e-9, 100*worst)) {
									t.Fatalf("one position off the logits are within %.3g of the tier's, where the arm's worst is %.3g: the comparison cannot see a wrong part", off, worst)
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
									if !(med <= bandRatio) || !(worstGot <= bandWorst*worstTier) {
										t.Errorf("against the float reference the composed %s is %.3g at the median of the tier's distance and %.3g at its worst against the tier's %.3g (bounds %.1f and %.1fx)",
											a.name, med, worstGot, worstTier, bandRatio, bandWorst)
									}
									t.Logf("composed %s against the float reference: median ratio to the tier's distance %.3f (range %.3f..%.3f), worst %.3g against the tier's %.3g",
										a.name, med, ratios[0], ratios[len(ratios)-1], worstGot, worstTier)
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
