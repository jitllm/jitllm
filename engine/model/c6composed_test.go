package model

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// classicFixtures are the classic-transformer fixtures (LayerNorm, a
// parallel attention and FFN, biases, ungated MLPs), dbrx's mixture with its
// clamped q, k and v among them, and mistral3's attention temperature.
var classicFixtures = []string{"synth-phi2", "synth-starcoder", "synth-stablelm", "synth-stablelm-par",
	"synth-starcoder2", "synth-falcon", "synth-falcon40", "synth-nemotron", "synth-commandr", "synth-cohere2",
	"synth-dbrx", "synth-mistral3", "synth-mistral3-af"}

// coveredBlocks is how many of m's blocks the front end states as ops.
func coveredBlocks(m *Model) int {
	n := 0
	for li := range m.Cfg.NLayer {
		if m.Uncovered(li) == Covered {
			n++
		}
	}
	return n
}

// TestC6ServesComposed holds the composer to each classic-transformer
// fixture's blocks and head on every device present: every block the front end
// states served whole at every decode step -- served block steps equal steps
// times the blocks the front end covers, windowed ones among them, none
// declined -- and the LayerNorm head served at each, the logits
// within the device band of the host's, and each feature taken out of the
// device session alone, as the composer reads it too, moving them: the composed
// arm of TestC6OnEveryDevice, asked for in any build.
func TestC6ServesComposed(t *testing.T) {
	// The prompt's rows come from the device's batched prefill, whose
	// attention tiles take Q K^T through binary16 where the device has them,
	// and every decoded step reads them: the band a prefilled prompt gets in
	// fixtureOnEveryDevice (1e-5), where a decode from the first token reads
	// 1e-12. The violations below land at 1e-4 and up.
	const bound = 1e-5
	ran := 0
	for _, name := range classicFixtures {
		t.Run(name, func(t *testing.T) {
			m, g := openC6(t, name)
			defer m.Close()
			ran += servesComposed(t, m, g.IDs, bound, append(slices.Clone(c6DevViolations[name]), statedViolations(m)...), true)
		})
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}

// hostLogits is the host's logits at each of ids, one row at a time.
func hostLogits(t *testing.T, m *Model, ids []int32) [][]float32 {
	t.Helper()
	host := m.NewState(len(ids) + 1)
	defer host.Close()
	var out [][]float32
	for _, id := range ids {
		lg, err := host.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	return out
}

// servesComposed is TestC6ServesComposed's arms for one model on every
// device present, decoding ids after a prompt of three: every block and the
// head served at every step, within bound of the host's logits with no
// argmax flip, and each violation of vs, taken out of the model while the
// device runs, moving them past 100 times the clean arm's worst and past
// 1e-4; with whole, every part of each step too (wholeStepC6). It returns
// how many devices ran.
func servesComposed(t *testing.T, m *Model, ids []int32, bound float64, vs []c6Violation, whole bool) int {
	t.Helper()
	const promptLen = 3
	ran := 0
	want := hostLogits(t, m, ids)
	// The prompt takes the batched path, which the composer is not asked
	// for; each token after it is a one-row step it is.
	prompt, toks := ids[:promptLen], ids[promptLen:]
	run := func(t *testing.T, dev string) (worst float64, flips int) {
		t.Helper()
		covered := coveredBlocks(m)
		if covered != m.Cfg.NLayer {
			t.Fatalf("the front end states %d of %d blocks: %s %v", covered, m.Cfg.NLayer, m.Uncovered(0), m.UncoveredErr(0))
		}
		got, st, _, _ := composedDecode(t, m, dev, prompt, toks, tier.ComposedBlock|tier.ComposedHead, true, false)
		served, declined, _ := st.ServedPart(tier.ComposedBlock)
		if want := len(toks) * covered; served != want || declined != 0 {
			t.Fatalf("%d block steps served and %d declined, where %d steps of %d covered blocks are %d served",
				served, declined, len(toks), covered, want)
		}
		// Every step's head, and the prompt's last row's.
		if heads, hd, _ := st.ServedPart(tier.ComposedHead); heads != len(toks)+1 || hd != 0 {
			t.Fatalf("%d head steps served and %d declined, where the %d steps and the prompt's last row are %d (%s)", heads, hd, len(toks), len(toks)+1, declinesOf(st))
		}
		for i, k := range st.BlockDeclines {
			if k > 0 {
				t.Fatalf("%d block step(s) declined because %s", k, tier.DeclineName(i))
			}
		}
		if st.FormedAtStep != 0 {
			t.Fatalf("%d composed block(s) formed at a step, not at the placement", st.FormedAtStep)
		}
		t.Logf("%s: %d of %d block steps served; the head served at all %d steps",
			dev, served, len(toks)*covered, len(toks))
		for i := range got {
			p := promptLen - 1 + i
			e := logitNMSE(got[i], want[p])
			if math.IsNaN(e) {
				t.Fatalf("pos %d: the logits are not finite", p)
			}
			worst = max(worst, e)
			if Greedy(got[i]) != Greedy(want[p]) {
				flips++
			}
		}
		return worst, flips
	}
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			worst, flips := run(t, dev)
			if !(worst < bound) || flips > 0 {
				t.Fatalf("worst logit NMSE %.3e, %d argmax flips, against the host", worst, flips)
			}
			t.Logf("composed blocks on %s: worst logit NMSE %.3e", dev, worst)
			for _, v := range vs {
				undo := v.mut(m)
				bad, _ := run(t, dev)
				undo()
				if !(bad > 100*worst) || !(bad > 10*bound) {
					t.Errorf("%s: the composed arm cannot see it -- worst NMSE %.3e against a clean %.3e", v.name, bad, worst)
				}
				t.Logf("violation %q: worst NMSE %.3e", v.name, bad)
			}
			if whole {
				wholeStepC6(t, m, dev, prompt, toks, want, bound)
			}
			ran++
		})
	}
	return ran
}

// wholeStepC6 is TestC6ServesComposed's whole-step arm: the input rows, the
// tables, the blocks, the head and the token the composer's, every decode
// row and the prompt's rows gathered on the device, a learned position
// table's rows among them (starcoder's), within bound of the host's logits
// want. Each row gathered at the next token's, and on a model with a
// position table each row at the next position's, must fall outside it.
func wholeStepC6(t *testing.T, m *Model, dev string, prompt, toks []int32, want [][]float32, bound float64) {
	t.Helper()
	decode := func(after func(*State)) (worst float64, flips int) {
		got, st, _, _ := composedDecodeWith(t, m, dev, prompt, toks, tier.ComposedStep, true, false, after)
		ne, de, _ := st.ServedPart(tier.ComposedEmbed)
		np, dp, _ := st.ServedPart(tier.PromptEmbed)
		if ne != len(toks) || de != 0 || np == 0 || dp != 0 {
			t.Fatalf("the composer gathered %d of %d steps' rows (%d declined) and the prompt's %d times (%d declined): %s",
				ne, len(toks), de, np, dp, declinesOf(st))
		}
		for i := range got {
			p := len(prompt) - 1 + i
			e := logitNMSE(got[i], want[p])
			if math.IsNaN(e) {
				t.Fatalf("pos %d: the logits are not finite", p)
			}
			worst = max(worst, e)
			if Greedy(got[i]) != Greedy(want[p]) {
				flips++
			}
		}
		return worst, flips
	}
	worst, flips := decode(nil)
	if !(worst < bound) || flips > 0 {
		t.Fatalf("the whole step: worst logit NMSE %.3e, %d argmax flips, against the host", worst, flips)
	}
	t.Logf("the whole step on %s: worst logit NMSE %.3e, every row gathered by the composer", dev, worst)
	type violation struct {
		name  string
		after func(*State)
	}
	vs := []violation{{"every row gathered at the next token's", shiftEmbed(1, 0, m.Vocab.Size())}}
	if m.posEmbd.e != nil {
		vs = append(vs, violation{"every row's position table row the next position's", shiftEmbed(0, 1, m.Vocab.Size())})
	}
	for _, v := range vs {
		bad, _ := decode(v.after)
		if !(bad > 100*worst) || !(bad > 10*bound) {
			t.Errorf("%s: the whole step cannot see it -- worst NMSE %.3e against a clean %.3e", v.name, bad, worst)
		}
		t.Logf("violation %q: worst NMSE %.3e", v.name, bad)
	}
}
