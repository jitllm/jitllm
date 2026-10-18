package model

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestBatchedPrefillInAStateSizedToThePrompt prefills on the device with a
// State only four positions longer than the prompt, and demands the host's
// logits. The tight sizing is the point: a batched scores kernel writes its last
// key tile past the causal width, and the score row pad (maxKeyTile) must cover
// the widest tile (sm_70's AttnScoresMMA70 tiles 64 keys), or near MaxSeq the
// overshoot lands in the next row. The lengths are not multiples of 64.
func TestBatchedPrefillInAStateSizedToThePrompt(t *testing.T) {
	prefillAgainstHost(t, []int{8, 200, 336})
}

// TestBatchedPrefillWithABinary16VCache is the same comparison with the
// device's V cache in binary16 (tier.Config.KVF16, JITLLM_KV_F16=1). Such a
// chunk goes down the per-row device path, which reads the layout it wrote;
// the batched scratch reads V as float32.
func TestBatchedPrefillWithABinary16VCache(t *testing.T) {
	prefillAgainstHost(t, []int{336}, tier.WithConfig(func(c *tier.Config) { c.KVF16 = true }))
}

func prefillAgainstHost(t *testing.T, lengths []int, opts ...tier.Option) {
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS) -- this gate proved nothing", err)
	}
	m, err := Open(jlmOf(t, p), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	g, err := openTier(t, opts...)
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()
	text := strings.Repeat("The river rose every spring until the bridges were islands, "+
		"and the clerk counted barrels of salt on the upper floor. ", 40)
	all := m.Vocab.Encode(text, true)
	for _, n := range lengths {
		ids := all[:n]
		host := m.NewState(n + 4)
		want, err := host.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		want = append([]float32(nil), want...)
		host.Close()

		dev := m.NewState(n + 4)
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
		t.Logf("n=%d: logit NMSE %.3e, argmax %d (host %d), batched attention on the matrix unit: %v",
			n, num/den, argmaxID(got), argmaxID(want), g.Stats().AttnMMA)
		if num/den > 1e-2 {
			t.Fatalf("n=%d: logit NMSE %.3e against the host", n, num/den)
		}
	}
}
