package engine

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestPlacedBlocksGiveTheirHostPagesBack places every block of a dense model
// and of a mixture and demands that the host holds no page once the device
// does: not the blocks' own, and not a mixture's expert pages. The release
// dropped a block's page and not its expert pages, which since container v26
// are nearly all of a mixture's weights, so a large mixture stayed resident
// on the host with every block on the cards.
//
// It runs a prompt and a decode step after placement too: a placed block the
// host read back would show here.
func TestPlacedBlocksGiveTheirHostPagesBack(t *testing.T) {
	for _, name := range []string{"Llama-3.2-1B-Instruct-Q4_K_M", "Qwen3-MOE-4x0.6B-Q4_K_M"} {
		t.Run(name, func(t *testing.T) {
			p := testmodels.Path(name + ".jlm")
			if src := testmodels.Path(name + ".gguf"); fileExists(src) {
				p = jlmOf(t, src)
			}
			if _, err := os.Stat(p); err != nil {
				testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
			}
			m, err := model.Open(p)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			g, err := openGPU(t, gpu.WithDeviceTune(gpu.TuneOff))
			if err != nil || g == nil {
				noDevice(t, "device", err)
			}
			defer g.Close()
			st := testInstance(t, m, 64)
			defer st.Close()
			st.SetDevice(g)
			if st.GPULayers() != m.Cfg.NLayer {
				if gs := g.Stats(); gs.Declined+gs.NoRoom > 0 || g.Counters().AllocRefused() > 0 {
					t.Skipf("CARD TOO SMALL: %d of %d blocks placed -- this gate proved nothing on this device", st.GPULayers(), m.Cfg.NLayer)
				}
				t.Fatalf("placed %d of %d blocks: %s", st.GPULayers(), m.Cfg.NLayer, g.Err())
			}
			if n := m.Container.ResidentPages(); n != 0 {
				t.Fatalf("every block is on the device and the host still holds %d page(s), %d bytes", n, m.Container.ResidentBytes())
			}
			// And the frames those pages lived in are handed back, not kept on
			// a free list nothing will take from: what is still mapped is the
			// dense region.
			if fb := m.Container.FreeBytes(); fb != 0 {
				t.Fatalf("placement left %d bytes of frames on the free list", fb)
			}
			if mb, dense := m.Container.MappedBytes(), m.Container.H.DenseLen; mb > dense+1<<16 {
				t.Fatalf("placement left %d bytes mapped against a %d-byte dense region", mb, dense)
			}
			ids := m.Vocab.Encode("The capital of France is", true)
			lg, err := st.Prefill(ids)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := st.Forward(Greedy(lg)); err != nil {
				t.Fatal(err)
			}
			if n := m.Container.ResidentPages(); n != 0 {
				t.Fatalf("a prompt and a decode step on the device read %d page(s) back to the host", n)
			}
		})
	}
}
