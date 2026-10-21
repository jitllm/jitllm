package model

import (
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestStagedPagedPrefillMatchesDecode prefills a prompt on the device as one
// chunk and, in a second State, a token at a time, and demands the same last
// logits, on a model whose attention head (256 wide) has no one-kernel flash
// prefill on any backend here, so the chunk runs the staged paged kernels:
// scores, a per-split softmax and an accumulate (tier/pagedprefill.go).
//
// The softmax's launch width is the tier's to give and the kernel's to
// declare. Metal runs a launch at the width it is given, so the 64 the tier
// used to pass for a 32-lane softmax put two subgroups on one item, and every
// prompt over one token prefilled wrong there while CUDA and Vulkan, which keep
// the declared width, were right: Qwen3.5-0.8B read 0.38 there against
// 3e-4 elsewhere. Every kernel gate was green, since each launches its
// kernel at the kernel's own width. Stats.PagedPrefillLaunches with no
// FlashPrefills is the selection check that the staged form ran.
func TestStagedPagedPrefillMatchesDecode(t *testing.T) {
	m := openStepHybrid(t)
	defer m.Close()
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			w := stagedPrefillGap(t, m, dev)
			if !(w < mlaDeviceNMSE) {
				t.Fatalf("a staged paged prefill disagrees with the same prompt a token at a time: NMSE %.3e", w)
			}
		})
	}
}

// stagedPrefillGap prefills the gate's prompt on dev in one chunk and a token
// at a time, every block placed, and returns the last logits' NMSE.
func stagedPrefillGap(t *testing.T, m *Model, dev string) float64 {
	t.Helper()
	ids := m.Vocab.Encode("The clerk counted barrels of salt on the upper floor, and", true)
	g, err := openTier(t, tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff))
	if err != nil || g == nil {
		noDevice(t, dev, err)
	}
	defer g.Close()
	plan := runsPlan{}
	chunk := plan.placed(t, m, g, len(ids)+1)
	defer chunk.Close()
	s0 := g.Stats()
	lc, err := chunk.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	s1 := g.Stats()
	if n, f := s1.PagedPrefillLaunches-s0.PagedPrefillLaunches, s1.FlashPrefills-s0.FlashPrefills; n == 0 || f != 0 {
		t.Fatalf("%d paged prefill attentions and %d flash prefills: the chunk did not run the staged form", n, f)
	}
	one := plan.placed(t, m, g, len(ids)+1)
	defer one.Close()
	var lo []float32
	for _, id := range ids {
		if lo, err = one.Forward(id); err != nil {
			t.Fatal(err)
		}
	}
	w := logitNMSE(lc, lo)
	t.Logf("%d tokens: the chunk against a token at a time, logit NMSE %.3e", len(ids), w)
	return w
}
