package engine

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestStreamedExpertsTakeAPrompt prefills synth-gptoss, whose head is held
// in float on the card, with every block's routed experts streamed and no
// expert cache: a block's pages hold one row's selections, so a prompt's
// call is cut to calls of a row. The prompt runs on the card -- no block is
// declined or moved to the CPU, the streamed sheets were sent -- and agrees
// with the CPU within the card's band.
func TestStreamedExpertsTakeAPrompt(t *testing.T) {
	m, err := model.Open(jlmOf(t, testmodels.Path("synth-gptoss.gguf")))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	g, err := openGPU(t, gpu.WithDeviceTune(gpu.TuneOff),
		gpu.WithConfig(func(c *gpu.Config) { c.StreamExperts = true; c.StreamCacheSlots = -1 }))
	if err != nil || g == nil {
		noDevice(t, "cuda:0", err)
	}
	defer g.Close()
	ids := make([]int32, 13)
	for i := range ids {
		ids[i] = int32((1000 + i*7) % m.Cfg.NVocab)
	}
	st := testInstance(t, m, 64, WithExperts("card"))
	defer st.Close()
	if err := st.SetDeviceLayers(g, -1); err != nil {
		t.Fatal(err)
	}
	if n := st.GPULayers(); n != m.Cfg.NLayer {
		t.Fatalf("the card took %d of %d blocks: %v", n, m.Cfg.NLayer, st.DeviceDeclines())
	}
	before := g.Stats()
	got, err := st.Prefill(ids)
	if err != nil {
		t.Fatalf("the prompt on the card: %v (%s)", err, g.Err())
	}
	got = append([]float32(nil), got...)
	after := g.Stats()
	if st.GPULayers() != m.Cfg.NLayer || st.DeviceDemotions() != 0 {
		t.Fatalf("the prompt left %d of %d blocks on the card (%d demotions: %+v): %s", st.GPULayers(), m.Cfg.NLayer,
			st.DeviceDemotions(), st.Evacuations(), g.Err())
	}
	if sent := after.StreamGathered + after.StreamDirect - before.StreamGathered - before.StreamDirect; sent == 0 {
		t.Fatalf("the prompt sent no streamed sheet: the streamed path did not run")
	}
	ref := testInstance(t, m, 64, WithKVF16(false))
	defer ref.Close()
	want, err := ref.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	e, w := nmseOf(got, want)
	t.Logf("a %d-row prompt with streamed experts: %.3e NMSE from the CPU (worst logit %.3g)", len(ids), e, w)
	// The card census's band for a card against the CPU.
	const band = 5e-2
	if math.IsNaN(e) || !(e < band) {
		t.Fatalf("the streamed prompt is %.3e NMSE from the CPU (bound %.0e)", e, band)
	}
}
