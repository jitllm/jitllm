package engine

import (
	"image"
	"image/color"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// TestATowerMovedBackFormsItsWidths: synth-gemma4v's tower blocks placed on
// a card, moved to the CPU and back (MoveBlocks), encode a picture as they
// did before the move, bit for bit, and nothing is formed at a step. The move
// back is a placement, and the card forms what a picture of each width reads
// when it ends; it did not, and the picture's step met the grid table at its
// 3232 rows unformed and declined the block.
func TestATowerMovedBackFormsItsWidths(t *testing.T) {
	text, mm := testmodels.Path("synth-gemma4v.gguf"), testmodels.Path("mmproj-synth-gemma4v.gguf")
	if !sourcePresent(text, mm) {
		testmodels.Missing(t, "MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", text, mm)
	}
	m, err := model.Open(jlmOfPair(t, text, mm))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	img := image.NewRGBA(image.Rect(0, 0, 300, 200))
	for y := range 200 {
		for x := range 300 {
			img.Set(x, y, color.RGBA{uint8(x * 255 / 299), uint8(y * 255 / 199), uint8((x + y) % 256), 255})
		}
	}
	nb, base := len(m.Tower.Blocks), m.TowerBase
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			g, err := openGPU(t, gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff))
			if err != nil || g == nil {
				noDevice(t, dev, err)
			}
			defer g.Close()
			s := testInstance(t, m, 64)
			defer s.Close()
			s.SetDeviceFallback(false)
			ts := s.tower.s
			if err := ts.SetDeviceLayers(g, nb); err != nil {
				t.Fatal(err)
			}
			if n := ts.GPULayers(); n != nb {
				t.Fatalf("%d of the tower's %d blocks placed on %s: %s", n, nb, dev, g.Err())
			}
			card, err := s.EncodePicture(img)
			if err != nil {
				t.Fatal(err)
			}
			card = slices.Clone(card)
			if mv, err := ts.MoveBlocks(base, base+nb, nn.CPU); err != nil || len(mv.Moved) != nb {
				t.Fatalf("%d of the tower's %d blocks moved to the CPU: %v", len(mv.Moved), nb, err)
			}
			if mv, err := ts.MoveBlocks(base, base+nb, dev); err != nil || len(mv.Moved) != nb || ts.GPULayers() != nb {
				t.Fatalf("%d of the tower's %d blocks moved back to %s, %d on it: %v", len(mv.Moved), nb, dev, ts.GPULayers(), err)
			}
			again, err := s.EncodePicture(img)
			if err != nil {
				t.Fatalf("the picture after the tower moved back to %s: %v", dev, err)
			}
			if !slices.Equal(again, card) {
				e, _ := nmseOf(again, card)
				t.Fatalf("the rows after the tower moved back to %s are %.3e from its first encode's", dev, e)
			}
			if st := g.Stats(); st.UnformedAtStep != 0 {
				t.Fatalf("%d of the tower's parts formed at a step on %s, not at the placement", st.UnformedAtStep, dev)
			}
			t.Logf("%s: %d tower blocks to the CPU and back, the picture's %d values bit for bit", dev, nb, len(card))
		})
	}
}
