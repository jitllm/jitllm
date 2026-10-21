package model

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/vulkan"
)

// TestClosingADeviceLeavesNoKernelOpen runs a latent-attention mixture and a
// hybrid on each Vulkan device JITLLM_TEST_DEVICES names (vulkan:0 by default)
// with every block placed, closes the tier, and demands that the device's own
// Close found no kernel still open. vulkan.Ctx keeps a ledger of the kernels it
// compiled, and a kernel nobody closed is a shader module, a pipeline, two
// layouts and a descriptor pool that destroying the device leaks
// (VUID-vkDestroyDevice-device-05137). An expert bank's matvec was one: the
// tier compiled it per block and nothing closed it.
func TestClosingADeviceLeavesNoKernelOpen(t *testing.T) {
	var devs []string
	for _, d := range stepDevices() {
		if strings.HasPrefix(d, "vulkan") {
			devs = append(devs, d)
		}
	}
	if len(devs) == 0 {
		t.Skip("JITLLM_TEST_DEVICES names no Vulkan device, and the kernel ledger is Vulkan's -- this gate proved nothing")
	}
	for _, name := range []string{"synth-deepseek", "synth-kimilinear"} {
		t.Run(name, func(t *testing.T) {
			m := openLatent(t, name)
			defer m.Close()
			ids := m.Vocab.Encode("The capital of France is", true)
			for _, dev := range devs {
				t.Run(dev, func(t *testing.T) {
					before := len(vulkan.Orphans())
					g := stepTier(t, m, dev, false)
					st := m.NewState(len(ids) + 8)
					if err := st.SetDevice(g); err != nil {
						st.Close()
						g.Close()
						t.Fatal(err)
					}
					if st.GPULayers() != m.Cfg.NLayer {
						n := st.GPULayers()
						st.Close()
						g.Close()
						t.Skipf("CARD TOO SMALL: %d of %d blocks -- this gate proved nothing here", n, m.Cfg.NLayer)
					}
					if _, err := st.Prefill(ids); err != nil {
						t.Fatal(err)
					}
					for _, id := range []int32{5, 9, 3, 7} {
						if _, err := st.Forward(id); err != nil {
							t.Fatal(err)
						}
					}
					st.Close()
					g.Close()
					if left := vulkan.Orphans()[before:]; len(left) > 0 {
						t.Fatalf("closing %s found %d kernel(s) still open: %v", dev, len(left), left)
					}
				})
			}
		})
	}
}
