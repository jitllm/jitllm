package model

import (
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestComposedStepPromptRunsPrefillAttention prefills a two-chunk prompt on a
// device whose composer serves a step's parts and none of a prompt's
// (tier.ComposedStep, the jitllmcomposed build's default): the prompt's
// blocks are the tier's, so its attention is the tier's prefill kernels,
// a chunk's rows against the pages a launch, never the decode plan a row
// at a time (Stats.PagedPrefillLaunches, every block of every chunk), and
// its logits are the host's within the device band.
func TestComposedStepPromptRunsPrefillAttention(t *testing.T) {
	const name = "Qwen3-0.6B-Q8_0"
	src := testmodels.Path(name + ".gguf")
	if !fileExists(src) {
		t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS to the model directory)", src)
	}
	m, err := Open(jlmOf(t, src), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	text := strings.Repeat("The river rose every spring until the bridges were islands, "+
		"and the clerk counted barrels of salt on the upper floor. ", 40)
	ids := m.Vocab.Encode(text, true)[:700]

	host := m.NewState(len(ids) + 4)
	want, err := host.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	want = append([]float32(nil), want...)
	host.Close()

	g, err := tier.OpenWith(tier.WithDeviceTune(tier.TuneOff), tier.WithConfig(func(c *tier.Config) { c.Composed = tier.ComposedStep }))
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()
	st := m.NewState(len(ids) + 4)
	defer st.Close()
	st.SetDevice(g)
	if st.GPULayers() != m.Cfg.NLayer {
		t.Fatalf("placed %d of %d blocks: %s", st.GPULayers(), m.Cfg.NLayer, g.Err())
	}
	s0 := g.Stats()
	got, err := st.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	if st.DeviceDemotions() != 0 {
		t.Fatalf("the device demoted: %s", g.Err())
	}
	s1 := g.Stats()
	n := s1.PagedPrefillLaunches - s0.PagedPrefillLaunches
	chunks := (len(ids) + nn.MaxDevicePrefillChunk - 1) / nn.MaxDevicePrefillChunk
	if n != chunks*m.Cfg.NLayer {
		t.Fatalf("the prompt's attention ran the prefill kernels %d times, not once a block of each of its %d chunks (%d)",
			n, chunks, chunks*m.Cfg.NLayer)
	}
	var num, den float64
	for i := range want {
		if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
			t.Fatalf("logit %d is %g", i, got[i])
		}
		d := float64(got[i] - want[i])
		num, den = num+d*d, den+float64(want[i])*float64(want[i])
	}
	if nmse := num / den; !(nmse < 2e-3) || Greedy(got) != Greedy(want) {
		t.Fatalf("the prompt's logits NMSE %.3e against the host, argmax %d against %d", nmse, Greedy(got), Greedy(want))
	}
	t.Logf("%d prefill attention launches, %d chunks of %d blocks, logits NMSE %.3e against the host", n, chunks, m.Cfg.NLayer, num/den)
}
