package engine

import (
	"image"
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/model"
)

// towerRunsOnTheCard places m's tower on each device -- every block, then
// half of them with the rest on the CPU device -- and holds img's rows,
// with no part formed at a step, to the CPU device's within the
// reduction-order band (an NMSE of 1e-3); or, where ref is given, to the
// reference within bound, the bound the CPU device's rows meet: over a deep
// tower each device's int8 activations part it from the other by more than
// the band, and the falsifiable statement is each device's distance from
// the reference.
func towerRunsOnTheCard(t *testing.T, m *model.Model, img image.Image, ref []float32, bound float64, opts ...Option) {
	t.Helper()
	host := testInstance(t, m, 64, opts...)
	want, err := host.EncodePicture(img)
	if err != nil {
		t.Fatal(err)
	}
	want = slices.Clone(want)
	host.Close()
	what := "the CPU device's"
	if ref != nil {
		cpu, _ := nmseOf(want, ref)
		t.Logf("the CPU device's rows: NMSE %.3e from the reference", cpu)
		want, what = ref, "the reference"
	} else {
		bound = 1e-3
	}
	nb := len(m.Tower.Blocks)
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			g, err := openGPU(t, gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff))
			if err != nil || g == nil {
				noDevice(t, dev, err)
			}
			defer g.Close()
			for _, blocks := range []int{nb, nb / 2} {
				s := testInstance(t, m, 64, opts...)
				s.SetDeviceFallback(false)
				if err := s.tower.s.SetDeviceLayers(g, blocks); err != nil {
					s.Close()
					t.Fatal(err)
				}
				if n := s.tower.s.GPULayers(); n != blocks {
					s.Close()
					t.Fatalf("%d of the tower's %d blocks placed on %s, want %d: %s", n, nb, dev, blocks, g.Err())
				}
				got, err := s.EncodePicture(img)
				s.Close()
				if err != nil {
					t.Fatal(err)
				}
				if st := g.Stats(); st.UnformedAtStep != 0 {
					t.Fatalf("%d of the tower's parts formed at a step on %s, not at the placement", st.UnformedAtStep, dev)
				}
				e, _ := nmseOf(got, want)
				t.Logf("%s: %d of %d tower blocks on the card: NMSE %.3e from %s", dev, blocks, nb, e, what)
				if math.IsNaN(e) || e > bound {
					t.Fatalf("%d of %d tower blocks on %s: the rows sit at NMSE %.3e from %s (bound %.0e)", blocks, nb, dev, e, what, bound)
				}
			}
		})
	}
}
