package model

import (
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestTheSameWorkCrossesTheSeamEverySession asserts that two sessions on one
// tier make the SAME placement decision for the same matvecs.
//
// A counter, not a tolerance: when the crossing verdict was latched after
// sampling the first offers, whether a matvec ran on the card depended on how
// many had been offered since the tier opened, and both answers sat inside the
// f32 reduction-order band.
func TestTheSameWorkCrossesTheSeamEverySession(t *testing.T) {
	m := hybridModelOpt(t, hyOpt{moe: true, deep: true})
	defer m.Close()
	ids := []int32{1, 2, 3, 4, 5, 6, 7, 8}

	// The crossing is forced on (WithMinMatVecBytes(0)): at the shipping
	// threshold this fixture's matvecs all stay on the host and 0 == 0 proves
	// nothing.
	g, err := tier.OpenWith(tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff),
		tier.WithMinMatVecBytes(0))
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()

	// served counts the matvecs one fresh State sends across the seam for one
	// token, bracketed so a previous session's traffic cannot be counted.
	served := func() int {
		s := m.NewState(16)
		defer s.Close()
		s.SetDeviceLayers(g, 2)
		if s.GPULayers() != 2 {
			t.Fatalf("wanted two blocks, got %d: %v", s.GPULayers(), g.Err())
		}
		before := g.Stats().Served
		for _, id := range ids {
			if _, err := s.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		return g.Stats().Served - before
	}
	first := served()
	if first == 0 {
		t.Fatal("no matvec crossed the seam even with the threshold at zero -- this gate " +
			"would pass on any regression, since every session would serve nothing")
	}
	for i := 2; i <= 3; i++ {
		if n := served(); n != first {
			t.Fatalf("session %d served %d matvecs on the device where the first served %d -- "+
				"the same tokens through the same blocks must cross the seam the same way, "+
				"or two sessions compute one model with different kernels", i, n, first)
		}
	}
	t.Logf("every session served %d matvecs across the seam", first)
}

// TestPoisonedScratchChangesNothing: no kernel may read a scratch region it did
// not write. A fresh device buffer is usually zero, the identity for a sum, so
// such a read is latent; tier.WithPoisonScratch fills block scratch and
// crossing buffers with NaN so it fails inside one session.
func TestPoisonedScratchChangesNothing(t *testing.T) {
	m := hybridModelOpt(t, hyOpt{moe: true, deep: true})
	defer m.Close()
	ids := []int32{1, 2, 3, 4, 5, 6, 7, 8}

	run := func(poison bool) [][]float32 {
		opts := []tier.Option{tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff)}
		if poison {
			opts = append(opts, tier.WithPoisonScratch(true))
		}
		g, err := tier.OpenWith(opts...)
		if err != nil || g == nil {
			noDevice(t, "device", err)
		}
		defer g.Close()
		s := m.NewState(16)
		defer s.Close()
		s.SetDeviceLayers(g, 2)
		if s.GPULayers() != 2 {
			t.Fatalf("wanted two blocks, got %d: %v", s.GPULayers(), g.Err())
		}
		out := make([][]float32, len(ids))
		for i, id := range ids {
			l, err := s.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			out[i] = append([]float32(nil), l...)
		}
		return out
	}
	clean, poisoned := run(false), run(true)
	for p := range clean {
		for i := range clean[p] {
			if clean[p][i] != poisoned[p][i] {
				t.Fatalf("pos %d logit %d: %g with a clean scratch and %g with a poisoned one -- "+
					"a kernel is reading scratch it did not write, and it is only correct "+
					"today because a fresh buffer happens to be zero (RULE 13)",
					p, i, clean[p][i], poisoned[p][i])
			}
		}
	}
}
