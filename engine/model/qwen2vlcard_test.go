package model

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
)

// TestQwen2VLTowerRunsOnTheCard places the Qwen2-VL family's towers on each
// device -- every block, then half of them with the rest on the CPU device --
// and holds the picture's rows to the CPU device's: the blocks' grid rotary
// read from the card's own table and, on Qwen2.5-VL's, the windowed blocks'
// row windows (graph.Attrs.RowWindows) beside the whole-picture blocks.
func TestQwen2VLTowerRunsOnTheCard(t *testing.T) {
	for _, name := range qvlTowers {
		t.Run(name, func(t *testing.T) {
			m, g := openQVLTower(t, name)
			defer m.Close()
			img := rgbPicture(g.Image.RGB, g.Image.W, g.Image.H)
			host := m.NewInstance(64)
			want, err := host.EncodePicture(img)
			if err != nil {
				t.Fatal(err)
			}
			want = slices.Clone(want)
			wantDeep := slices.Clone(host.EncodedTaps())
			host.Close()
			nb := len(m.tower.blocks)
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					d, err := openGPU(t, gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff))
					if err != nil || d == nil {
						noDevice(t, dev, err)
					}
					defer d.Close()
					var whole []float32
					for _, blocks := range []int{nb, nb / 2} {
						s := m.NewInstance(64)
						s.SetDeviceFallback(false)
						if err := s.towerReady(); err != nil {
							t.Fatal(err)
						}
						if err := s.tower.s.SetDeviceLayers(d, blocks); err != nil {
							t.Fatal(err)
						}
						if n := s.tower.s.GPULayers(); n != blocks {
							s.Close()
							t.Fatalf("%d of the tower's %d blocks placed on %s, want %d: %s", n, nb, dev, blocks, d.Err())
						}
						got, err := s.EncodePicture(img)
						got = slices.Clone(got)
						deep := slices.Clone(s.EncodedTaps())
						s.Close()
						if err != nil {
							t.Fatalf("%d blocks on %s: %v", blocks, dev, err)
						}
						// Every part the picture ran was formed at the placement.
						if st := d.Stats(); st.FormedAtStep != 0 {
							t.Fatalf("%d of the tower's parts formed at a step on %s, not at the placement", st.FormedAtStep, dev)
						}
						e, worst := nmseOf(got, want)
						t.Logf("%s, %d of %d tower blocks: NMSE %.3e, max|d| %.3g from the CPU device's rows", dev, blocks, nb, e, worst)
						if math.IsNaN(e) || e > 1e-3 {
							t.Fatalf("%d of the tower's blocks on %s sit at NMSE %.3e from the CPU device's rows", blocks, dev, e)
						}
						// Qwen3-VL's taps are the rows the blocks before them left on the card.
						if len(deep) != len(wantDeep) {
							t.Fatalf("%d tap values on %s, %d on the CPU device", len(deep), dev, len(wantDeep))
						}
						if len(deep) > 0 {
							de, dw := nmseOf(deep, wantDeep)
							t.Logf("%s, %d of %d tower blocks: taps at NMSE %.3e, max|d| %.3g", dev, blocks, nb, de, dw)
							if math.IsNaN(de) || de > 1e-3 {
								t.Fatalf("the taps over %d of the tower's blocks on %s sit at NMSE %.3e from the CPU device's", blocks, dev, de)
							}
						}
						if whole == nil {
							whole = got
							continue
						}
						// The two seams run different blocks on the card: rows
						// equal to the bit would be one placement measured twice.
						e, worst = nmseOf(got, whole)
						t.Logf("%s: the half seam at NMSE %.3e, max|d| %.3g from the whole seam's rows", dev, e, worst)
						if slices.Equal(got, whole) {
							t.Fatalf("on %s the half seam's rows are the whole seam's to the bit", dev)
						}
					}
				})
			}
		})
	}
}
