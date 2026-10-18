package model

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestFlashPrefillMatchesThreeKernels prefills a two-chunk prompt on the device
// with the chunk's attention as one kernel (kernels.FlashPrefill70, or
// FlashPrefill80 and its m16n8k8 twin where the card has m16n8) and with
// the scores/softmax/accumulate trio it replaces (tier.Config.NoFlashPrefill),
// and holds the two to each other and the flash arm to the host.
//
// Two bounds, because the arms are not the same arithmetic: the one kernel
// rounds P = exp(s - running max) where the trio rounds the normalised
// probability, so they part in the last bits; the bound between them is the
// device-against-host band, and the argmax must agree. Stats.FlashPrefills must
// count every layer of every chunk on the flash arm and nothing on the other. A
// card with no one-kernel form (sm_70's, or Metal's tile kernel) skips by name.
func TestFlashPrefillMatchesThreeKernels(t *testing.T) {
	for _, name := range []string{"Llama-3.2-1B-Instruct-Q4_K_M", "Qwen3-0.6B-Q8_0"} {
		t.Run(name, func(t *testing.T) { flashPrefillMatches(t, name) })
	}
}

func flashPrefillMatches(t *testing.T, name string) {
	p := testmodels.Path(name + ".jlm")
	if src := testmodels.Path(name + ".gguf"); fileExists(src) {
		p = jlmOf(t, src)
	}
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := Open(p, noTune, WithKVF16(false))
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

	run := func(noFlash bool) ([]float32, int, int) {
		g, err := openTier(t, tier.WithDeviceTune(tier.TuneOff), tier.WithConfig(func(c *tier.Config) { c.NoFlashPrefill = noFlash }))
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
		lg, err := st.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		if st.DeviceDemotions() != 0 {
			t.Fatalf("the device demoted: %s", g.Err())
		}
		return append([]float32(nil), lg...), g.Stats().FlashPrefills, g.Stats().FlashPrefill80
	}
	trio, trioN, _ := run(true)
	flash, flashN, flash80 := run(false)
	if flashN == 0 && trioN == 0 {
		t.Skip("this device builds no one-kernel prompt attention: there is no arm to compare")
	}
	chunks := (len(ids) + 511) / 512
	if trioN != 0 || flashN != chunks*m.Cfg.NLayer {
		t.Fatalf("flash launches: %d on the three-kernel arm, %d on the flash arm (want %d)", trioN, flashN, chunks*m.Cfg.NLayer)
	}
	nmse := func(got, w []float32) float64 {
		var num, den float64
		for i := range w {
			if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
				return math.Inf(1)
			}
			d := float64(got[i] - w[i])
			num += d * d
			den += float64(w[i]) * float64(w[i])
		}
		return num / den
	}
	a, h := nmse(flash, trio), nmse(flash, want)
	if !(a < 2e-3) || !(h < 2e-3) {
		t.Fatalf("flash prefill logits NMSE %.3e against the three kernels, %.3e against the host", a, h)
	}
	if Greedy(flash) != Greedy(trio) || Greedy(flash) != Greedy(want) {
		t.Fatalf("argmax: flash %d, three kernels %d, host %d", Greedy(flash), Greedy(trio), Greedy(want))
	}
	t.Logf("%d flash launches (%d of them FlashPrefill80, m16n8k16 or sm_75's m16n8k8); logits NMSE %.3e against the three kernels, %.3e against the host (three kernels: %.3e)",
		flashN, flash80, a, h, nmse(trio, want))
}
