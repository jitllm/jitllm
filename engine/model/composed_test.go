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
// every step's logits and the tier's stats.
func composedDecode(t *testing.T, m *Model, dev string, ids, toks []int32, parts tier.Parts, noGraph, float bool, extra ...tier.Option) ([][]float32, tier.Stats, bool) {
	t.Helper()
	opts := append([]tier.Option{tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff),
		tier.WithConfig(func(c *tier.Config) { c.Composed, c.NoGraph, c.ComposedFloat = parts, noGraph, float })}, testTierOpts(t)...)
	opts = append(opts, extra...)
	g, err := tier.OpenWith(opts...)
	if err != nil || g == nil {
		noDevice(t, dev, err)
	}
	defer g.Close()
	s := m.NewState(len(ids) + len(toks) + 4)
	defer s.Close()
	if err := s.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	if s.GPULayers() == 0 {
		t.Fatalf("no block placed on %s", dev)
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
	return out, g.Stats(), s.HeadOnDevice()
}

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
	for _, file := range []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Qwen3-0.6B-Q8_0.gguf", "tinyllama-1.1b-q3_K_M.gguf", "gemma-2b.gguf",
		"Qwen2-1.5B-Instruct-Q4_K_M.gguf"} {
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
			toks := make([]int32, 140)
			for i := range toks {
				toks[i] = int32((int(ids[i%len(ids)])*7 + i) % m.Vocab.Size())
			}
			for _, dev := range stepDevices() {
				for _, noGraph := range []bool{false, true} {
					name := dev
					if noGraph {
						name += "/no graph"
					}
					want, _, _ := composedDecode(t, m, dev, ids, toks, 0, noGraph, false)
					ref, _, _ := composedDecode(t, m, dev, ids, toks, tier.ComposedBlock|tier.ComposedHead, noGraph, true)
					for _, a := range arms {
						t.Run(name+"/"+a.name, func(t *testing.T) {
							got, st, headOn := composedDecode(t, m, dev, ids, toks, a.parts, noGraph, false)
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
								poisoned, _, _ := composedDecode(t, m, dev, ids, toks, a.parts, noGraph, false, tier.WithPoisonScratch(true), tier.WithPoisonKV(true))
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
			}
		})
	}
	if ran == 0 {
		t.Fatal("no arm ran")
	}
}
