package model

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestGroupedMixtureRunsAtARaggedWidth: a prompt chunk through a mixture whose
// embedding is 8 wide (tiny-qwen3moe) runs grouped by expert on the device,
// every block placed, and gives the host's logits. Its width is not a block of
// the activation quantizer, so the grouped path used to decline it by name.
//
// VIOLATION SIGNATURE. Put the shape decline back in moeGroupWhyNot and
// Stats.GroupedMoE reads 0 (the chunk takes another route); quantize the
// sorted rows dense and the logits part from the host's.
func TestGroupedMixtureRunsAtARaggedWidth(t *testing.T) {
	m, err := Open(graphModel(t, "tiny-qwen3moe-f32"), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.Cfg.NEmbd%32 == 0 {
		t.Fatalf("n_embd %d is a whole number of blocks: this gate proved nothing", m.Cfg.NEmbd)
	}
	ids := m.Vocab.Encode("Once upon a time in a land far away there lived a king", true)
	hs := m.NewState(64)
	defer hs.Close()
	want, err := hs.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	want = slices.Clone(want)

	g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices("cuda")}, testTierOpts(t)...)...)
	if err != nil || g == nil {
		noDevice(t, "cuda", err)
	}
	defer g.Close()
	s := m.NewState(64)
	defer s.Close()
	if err := s.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	if s.GPULayers() != m.Cfg.NLayer {
		t.Fatalf("%d of %d blocks placed", s.GPULayers(), m.Cfg.NLayer)
	}
	got, err := s.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	if n := g.Stats().GroupedMoE; n == 0 {
		t.Fatalf("no mixture block ran grouped by expert (%d tokens)", len(ids))
	}
	worst, scale := 0.0, 0.0
	for i := range want {
		if math.IsNaN(float64(got[i])) {
			t.Fatalf("logit %d is NaN", i)
		}
		worst = math.Max(worst, math.Abs(float64(got[i]-want[i])))
		scale = math.Max(scale, math.Abs(float64(want[i])))
	}
	t.Logf("%d tokens, %d grouped mixture blocks; worst logit difference %.3g of %.3g", len(ids), g.Stats().GroupedMoE, worst, scale)
	if worst > 0.02*scale+1e-3 {
		t.Fatalf("device logits part from the host's by %.3g (largest logit %.3g)", worst, scale)
	}
}
