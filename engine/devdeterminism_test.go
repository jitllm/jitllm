package engine

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
)

// TestPoisonedScratchChangesNothing: no kernel may read a scratch region it did
// not write. A fresh device buffer is usually zero, the identity for a sum, so
// such a read is latent; gpu.WithPoisonScratch fills block scratch and
// crossing buffers with NaN so it fails inside one session.
func TestPoisonedScratchChangesNothing(t *testing.T) {
	m, openOpts := hybridModelOpt(t, hyOpt{moe: true, deep: true})
	defer m.Close()
	ids := []int32{1, 2, 3, 4, 5, 6, 7, 8}

	run := func(poison bool) [][]float32 {
		opts := []gpu.Option{gpu.WithDevices("gpu:0"), gpu.WithDeviceTune(gpu.TuneOff)}
		if poison {
			opts = append(opts, gpu.WithPoisonScratch(true))
		}
		g, err := openGPU(t, opts...)
		if err != nil || g == nil {
			noDevice(t, "device", err)
		}
		defer g.Close()
		s := testInstance(t, m, 16, openOpts...)
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
