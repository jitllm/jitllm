//go:build jitllmfault

package model

import (
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestGemma3nTowerPagesThroughTheDeviceStale is the violation of
// TestGemma3nTowerPagesThroughTheDevice: a page-in that uploads the bytes its
// host slices held at admission, without asking the host for them again,
// reads whatever block the host pager has since put in that frame, and the
// rows must part from the resident tower's -- or the encode fail outright,
// where the wrong bytes priced a page-in the budget cannot hold.
func TestGemma3nTowerPagesThroughTheDeviceStale(t *testing.T) {
	_, g := openGemma3nV(t)
	ran := 0
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) {
			ref, ok := g3nvStreamed(t, spec, g, 0, true)
			if !ok {
				t.Skipf("%s: not present", spec)
			}
			ran++
			got, budget := g3nvPaged(t, spec, g, ref, tier.WithFault("paged", "stale"))
			if got.err != "" {
				t.Logf("%s, %d bytes, stale page-ins: the encode failed: %s", spec, budget, got.err)
				return
			}
			nm, _ := nmse32(ref.out, got.out)
			t.Logf("%s, %d bytes, stale page-ins: NMSE %.3e against resident over %d page-ins", spec, budget, nm, got.ins)
			if !(nm > 1e-3) {
				t.Fatalf("%s: stale page-ins read %.3e against the resident tower: the gate cannot see them", spec, nm)
			}
		})
	}
	if ran == 0 {
		t.Skip("no GPU backend on this host")
	}
}
