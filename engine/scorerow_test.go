package engine

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/nn"
)

// TestBatchedPrefillInAStateSizedToThePrompt prefills on the device with a
// Instance only four positions longer than the prompt, and demands the host's
// logits. The tight sizing is the point: a batched scores kernel writes its last
// key tile past the causal width, and the score row pad (maxKeyTile) must cover
// the widest tile (sm_70's AttnScoresMMA70 tiles 64 keys), or near MaxSeq the
// overshoot lands in the next row. The lengths are not multiples of 64.
func TestBatchedPrefillInAStateSizedToThePrompt(t *testing.T) {
	prefillAgainstHost(t, []int{8, 200, 336}, false)
}

// TestBatchedPrefillWithABinary16History is the same comparison with the
// instances' history binary16 (WithKVF16), keys and values: the card's pool
// holds it as the CPU's does, and the prompt's attention reads it there.
func TestBatchedPrefillWithABinary16History(t *testing.T) {
	prefillAgainstHost(t, []int{200, 336}, true)
}

// TestDeepPrefillSplitsItsHistory is the comparison on a prompt read past
// the positions where the card's attention deals each tile's history over
// workgroups (composer.Flash's split) and merges their partials: Qwen3-4B's
// 32 heads of 128 at a 512-row chunk leave a V100's second wave a fifth
// idle, so its chunks past some 930 positions take the split, and the
// gate holds the logits to the host's and counts the split's launches.
func TestDeepPrefillSplitsItsHistory(t *testing.T) {
	prefillAgainstHostOf(t, "Qwen3-4B-Q4_K_M.gguf", []int{2000}, true, true)
}

func prefillAgainstHost(t *testing.T, lengths []int, f16 bool) {
	prefillAgainstHostOf(t, "Llama-3.2-1B-Instruct-Q4_K_M.gguf", lengths, f16, false)
}

// prefillAgainstHostOf is the comparison on the named model; split asks
// that the card's attention took its split at every length.
func prefillAgainstHostOf(t *testing.T, name string, lengths []int, f16, split bool) {
	p := testmodels.Path(name)
	if _, err := os.Stat(p); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS) -- this gate proved nothing", err)
	}
	instOpts := []Option{WithKVF16(f16)}
	m, err := model.Open(jlmOf(t, p))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	g, err := openGPU(t)
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()
	text := strings.Repeat("The river rose every spring until the bridges were islands, "+
		"and the clerk counted barrels of salt on the upper floor. ", 120)
	all := m.Vocab.Encode(text, true)
	for _, n := range lengths {
		ids := all[:n]
		host := testInstance(t, m, n+4, instOpts...)
		want, err := host.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		want = append([]float32(nil), want...)
		host.Close()

		splits, wholes := g.Stats().SplitAttends, g.Stats().WholeAttends
		dev := testInstance(t, m, n+4, instOpts...)
		dev.SetDevice(g)
		if dev.GPULayers() != m.Cfg.NLayer {
			dev.Close()
			t.Skipf("the device took %d of %d blocks", dev.GPULayers(), m.Cfg.NLayer)
		}
		got, err := dev.Prefill(ids)
		// The selection check: a device that failed and handed the prompt to
		// the host answers with the host's own logits, which would pass.
		demoted, placed := dev.DeviceDemotions(), dev.GPULayers()
		dev.Close()
		if err != nil {
			t.Fatal(err)
		}
		if demoted != 0 || placed != m.Cfg.NLayer {
			t.Fatalf("n=%d: %d demotion(s), %d of %d blocks left on the device: the host answered",
				n, demoted, placed, m.Cfg.NLayer)
		}
		var num, den float64
		for i := range want {
			if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
				t.Fatalf("n=%d: logit %d is %v on the device", n, i, got[i])
			}
			d := float64(got[i] - want[i])
			num += d * d
			den += float64(want[i]) * float64(want[i])
		}
		splits, wholes = g.Stats().SplitAttends-splits, g.Stats().WholeAttends-wholes
		t.Logf("n=%d: logit NMSE %.3e, argmax %d (host %d), %d attention launches on the matrix unit, %d of a split and its merge, %d whole",
			n, num/den, argmaxID(got), argmaxID(want), g.Stats().AttendPart(nn.EveryPart), splits, wholes)
		switch {
		case split && splits == 0:
			t.Errorf("n=%d: no attention took its split: the gate compared the whole form alone", n)
		case split && 2*wholes >= splits:
			// Past the crossing a call launches the split alone: a call
			// launching both forms runs as many whole as split.
			t.Errorf("n=%d: %d whole launches beside %d split calls: the calls past the crossing launched both forms", n, wholes, splits/2)
		}
		if num/den > 1e-2 {
			t.Fatalf("n=%d: logit NMSE %.3e against the host", n, num/den)
		}
	}
}
