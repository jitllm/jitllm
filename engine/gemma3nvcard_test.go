package engine

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
)

// TestGemma3nTowerRunsOnTheCard places Gemma 3n's chained MobileNet-V5
// tower on each device, every block and the first half of them, each
// placed block a part of its own on the card (PlaceTowerChain), and holds
// the soft rows to the reference's at the engine's arithmetic within the
// CPU device's own distance from it, three times over, or 1e-3. No part
// forms at a picture. The blocks placed back on the CPU device read the CPU
// device's rows exactly.
func TestGemma3nTowerRunsOnTheCard(t *testing.T) {
	m := openGemma3nV(t)
	defer m.Close()
	g, img := readG3nv(t)
	D, soft := int(m.Container.Vision().ProjDim), g.Image.Soft
	host := testInstance(t, m, 64)
	want, err := host.EncodePicture(img)
	if err != nil {
		t.Fatal(err)
	}
	want = append([]float32(nil), want...)
	host.Close()
	cpu, _ := nmseOf(want[:soft*D], g.Image.EmbdA8)
	bound := max(3*cpu, 1e-3)
	t.Logf("the CPU device's soft rows: NMSE %.3e from the reference", cpu)
	nb := len(m.Tower.Blocks)
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			gd, err := openGPU(t, gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff))
			if err != nil || gd == nil {
				noDevice(t, dev, err)
			}
			defer gd.Close()
			for _, blocks := range []int{nb, nb / 2} {
				s := testInstance(t, m, 64)
				if err := s.PlaceTowerChain(gd, blocks); err != nil {
					s.Close()
					t.Fatalf("%d of %d blocks on %s: %v (%s)", blocks, nb, dev, err, gd.Err())
				}
				formed := gd.Stats().UnformedAtStep
				got, err := s.EncodePicture(img)
				if err != nil {
					s.Close()
					t.Fatalf("%v (%s)", err, gd.Err())
				}
				got = append([]float32(nil), got...)
				if err := s.PlaceTowerChain(gd, 0); err != nil {
					s.Close()
					t.Fatal(err)
				}
				back, err := s.EncodePicture(img)
				s.Close()
				if err != nil {
					t.Fatal(err)
				}
				if st := gd.Stats(); st.UnformedAtStep != formed {
					t.Fatalf("%d of the tower's parts formed at a picture on %s, not at the placement", st.UnformedAtStep-formed, dev)
				}
				e, _ := nmseOf(got[:soft*D], g.Image.EmbdA8)
				t.Logf("%s: %d of %d blocks on the card: NMSE %.3e from the reference", dev, blocks, nb, e)
				if math.IsNaN(e) || e > bound {
					t.Fatalf("%d of %d blocks on %s: NMSE %.3e from the reference (bound %.1e)", blocks, nb, dev, e, bound)
				}
				for j := range back {
					if back[j] != want[j] {
						t.Fatalf("placed back on the CPU device, element %d is %v, the CPU device's %v", j, back[j], want[j])
					}
				}
			}
		})
	}
}
