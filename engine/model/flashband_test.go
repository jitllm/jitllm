//go:build jitllmfault

package model

import (
	"math"
	"os"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestFlashProductsEndToEnd decodes the serving gate's teacher-forced steps,
// at the gate's prompt and at one past 2000 positions, with the tier, the
// float reference and the composer's whole-block arm, its flash form at five
// binary16 products and at one a side; and greedy answers of 16 tokens with
// the tier and both arms.
func TestFlashProductsEndToEnd(t *testing.T) {
	defer tier.SetFlashOneProduct(false)
	const dev = "cuda:0"
	for _, file := range []string{"Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "Qwen3-1.7B-Q4_K_M.gguf"} {
		t.Run(file, func(t *testing.T) {
			path := testmodels.Path(file)
			if _, err := os.Stat(fixtureSource(path)); err != nil {
				t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS)", path)
			}
			m, err := Open(jlmOf(t, path), noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			long := crossingPrompt + " Paris. "
			for len(m.Vocab.Encode(long, true)) < 2000 {
				long += crossingPrompt + " Paris. "
			}
			parts := tier.ComposedBlock | tier.ComposedPrompt
			for _, pr := range []struct {
				name string
				text string
				n    int
			}{{"gate prompt", crossingPrompt, 140}, {"long prompt", long + crossingPrompt, 32}} {
				ids := m.Vocab.Encode(pr.text, true)
				toks, _ := trainedToks(t, m, ids, pr.n)
				tier.SetFlashOneProduct(false)
				want, _, _, _ := composedDecode(t, m, dev, ids, toks, 0, false, false)
				ref, _, _, _ := composedDecode(t, m, dev, ids, toks, tier.ComposedBlock|tier.ComposedHead|tier.ComposedPrompt, false, true)
				worstTier := 0.0
				for i := range ref {
					worstTier = max(worstTier, logitNMSE(want[i], ref[i]))
				}
				t.Logf("E2E %s %s (%d tokens, %d steps): the tier against the float reference, worst %.3g", file, pr.name, len(ids), len(toks)+1, worstTier)
				for _, one := range []bool{false, true} {
					tier.SetFlashOneProduct(one)
					got, st, _, _ := composedDecode(t, m, dev, ids, toks, parts, false, false)
					np, dp, _ := st.ServedPart(tier.ComposedPrompt)
					var ratios []float64
					worst, fromTier, step0 := 0.0, 0.0, 0.0
					for i := range ref {
						et, eg := logitNMSE(want[i], ref[i]), logitNMSE(got[i], ref[i])
						ratios = append(ratios, eg/et)
						worst = max(worst, eg)
						fromTier = max(fromTier, logitNMSE(got[i], want[i]))
						if i == 0 {
							step0 = eg
						}
					}
					slices.Sort(ratios)
					t.Logf("E2E %s %s products=%d: against the float reference median ratio to the tier's %.3f (range %.3f..%.3f), worst %.3g, prompt's last row %.3g; worst from the tier %.3g; prompt block steps %d served, %d declined",
						file, pr.name, map[bool]int{false: 5, true: 2}[one], ratios[len(ratios)/2], ratios[0], ratios[len(ratios)-1], worst, step0, fromTier, np, dp)
				}
				tier.SetFlashOneProduct(false)
			}
			chat := "Question: Name three rivers of Europe and say which sea each flows into.\nAnswer:"
			for _, pr := range []struct{ name, text string }{{"short question", chat}, {"long question", long + "\n" + chat}} {
				ids := m.Vocab.Encode(pr.text, true)
				var answers [3][]int32
				for a, arm := range []struct {
					parts tier.Parts
					one   bool
				}{{0, false}, {parts, false}, {parts, true}} {
					tier.SetFlashOneProduct(arm.one)
					answers[a] = greedyOn(t, m, dev, ids, arm.parts, 16)
				}
				tier.SetFlashOneProduct(false)
				same := func(a, b []int32) int {
					n := 0
					for n < len(a) && n < len(b) && a[n] == b[n] {
						n++
					}
					return n
				}
				t.Logf("ANS %s %s (%d tokens): main %q", file, pr.name, len(ids), m.Vocab.Decode(answers[0]))
				t.Logf("ANS %s %s: composed products=5 agrees with main for %d of 16 tokens: %q", file, pr.name, same(answers[0], answers[1]), m.Vocab.Decode(answers[1]))
				t.Logf("ANS %s %s: composed products=2 agrees with main for %d of 16 tokens, with products=5 for %d: %q", file, pr.name,
					same(answers[0], answers[2]), same(answers[1], answers[2]), m.Vocab.Decode(answers[2]))
			}
		})
	}
}

// greedyOn is n greedy tokens after ids with the tier on dev, parts composed.
func greedyOn(t *testing.T, m *Model, dev string, ids []int32, parts tier.Parts, n int) []int32 {
	t.Helper()
	opts := append([]tier.Option{tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff),
		tier.WithConfig(func(c *tier.Config) { c.Composed, c.PromptRows = parts, 64 })}, testTierOpts(t)...)
	g, err := tier.OpenWith(opts...)
	if err != nil || g == nil {
		noDevice(t, dev, err)
	}
	defer g.Close()
	s := m.NewState(len(ids) + n + 4)
	defer s.Close()
	s.SetDeviceFallback(false)
	if err := s.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	l, err := s.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	var out []int32
	for range n {
		best := 0
		for i, x := range l {
			if x > l[best] || math.IsNaN(float64(l[best])) {
				best = i
			}
		}
		out = append(out, int32(best))
		if l, err = s.Forward(int32(best)); err != nil {
			t.Fatal(err)
		}
	}
	if s.DeviceDemotions() != 0 {
		t.Fatalf("the device failed %d time(s)", s.DeviceDemotions())
	}
	return out
}
