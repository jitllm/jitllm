//go:build jitllmfault

package model

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestWindowReleaseGateDiscriminates runs the device arm of
// TestWindowedLayersHoldTheirWindow with the page each window starts in
// released too (tier.SetPagedFault "winrelease"), and with the window lost --
// its layers reading the full descriptors ("window") -- and demands that the
// logits move or the decode fail: a gate that passes it cannot tell a window
// read from a released page or from the whole history. Where the composer
// serves the blocks (a jitllmcomposed build), it ran every block of every
// step of both.
func TestWindowReleaseGateDiscriminates(t *testing.T) {
	specs := testdevices.Specs("cuda", "vulkan", "metal")
	m, err := Open(jlmOf(t, testmodels.Path("synth-exaone4.gguf")), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const maxSeq = 1024
	decode := func(g *tier.GPU) ([]uint64, error) {
		s := m.NewState(maxSeq)
		defer s.Close()
		if err := s.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if s.GPULayers() != m.Cfg.NLayer {
			t.Skipf("CARD TOO SMALL: %d of %d blocks -- this arm proved nothing", s.GPULayers(), m.Cfg.NLayer)
		}
		var sums []uint64
		before := composedBlockSteps(g)
		for i := 0; i < maxSeq-8; i++ {
			lg, err := s.Forward(int32(3 + (i*7)%61))
			if err != nil {
				return sums, err
			}
			sums = append(sums, hashLogits(lg))
		}
		composedBlocksRan(t, composedOf(m, g), before, (maxSeq-8)*m.Cfg.NLayer, g)
		return sums, nil
	}
	for _, spec := range specs {
		t.Run(spec, func(t *testing.T) {
			open := func() *tier.GPU {
				g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(spec),
					tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t)...)...)
				if err != nil || g == nil {
					noDevice(t, spec, err)
				}
				return g
			}
			g := open()
			want, err := decode(g)
			g.Close()
			if err != nil {
				t.Fatal(err)
			}
			defer tier.SetPagedFault("")
			for _, fault := range []string{"winrelease", "window"} {
				tier.SetPagedFault(fault)
				g := open()
				got, err := decode(g)
				released := g.Stats().KVWindowReleased
				g.Close()
				tier.SetPagedFault("")
				if released == 0 {
					t.Fatalf("%s: no windowed page was released: the violation was never exercised", fault)
				}
				t.Logf("violation %q: %s", fault, violationWhat(err, got, want))
				if err == nil && equalSums(got, want) {
					t.Fatalf("%s: the violation changed no logit on the device", fault)
				}
			}
		})
	}
}
