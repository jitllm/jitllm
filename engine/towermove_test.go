package engine

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/nn"
)

// TestTowerBlocksMoveBetweenDevices moves a stated tower's blocks between a
// card and the CPU device between pictures (Instance.MoveBlocks on the
// tower's instance): every block on the card, all of them to the CPU, then
// back. Each picture's rows are its placement's -- on the CPU device the CPU
// instance's to the bit, back on the card the card's first rows to the bit --
// so a move carries nothing a picture leaves (the run's history is its own,
// kvCache.spent) and composes nothing at a step. Qwen2-VL's, Qwen2.5-VL's
// windowed and Qwen3-VL's tapped towers; the card's rows must not be the
// CPU's to the bit, or the gate could not tell the placements apart.
func TestTowerBlocksMoveBetweenDevices(t *testing.T) {
	for _, name := range []string{"synth-qwen2vl", "synth-qwen25vl", "synth-qwen3vl"} {
		t.Run(name, func(t *testing.T) {
			m, g, opts := openQVLTower(t, name)
			defer m.Close()
			img := rgbPicture(g.Image.RGB, g.Image.W, g.Image.H)
			host := testInstance(t, m, 64, opts...)
			want, err := host.EncodePicture(img)
			if err != nil {
				t.Fatal(err)
			}
			want = slices.Clone(want)
			wantDeep := slices.Clone(host.EncodedTaps())
			host.Close()
			nb := len(m.Tower.Blocks)
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					d, err := openGPU(t, gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff))
					if err != nil || d == nil {
						noDevice(t, dev, err)
					}
					defer d.Close()
					s := testInstance(t, m, 64, opts...)
					defer s.Close()
					s.SetDeviceFallback(false)
					if err := s.towerReady(); err != nil {
						t.Fatal(err)
					}
					ts := s.tower.s
					if err := ts.SetDeviceLayers(d, nb); err != nil {
						t.Fatal(err)
					}
					if n := ts.GPULayers(); n != nb {
						t.Fatalf("%d of the tower's %d blocks placed on %s: %s", n, nb, dev, d.Err())
					}
					at, _ := ts.DeviceOf(ts.lo)
					encode := func(where string) ([]float32, []float32) {
						t.Helper()
						rows, err := s.EncodePicture(img)
						if err != nil {
							t.Fatalf("%s: %v", where, err)
						}
						return slices.Clone(rows), slices.Clone(s.EncodedTaps())
					}
					card, cardDeep := encode("every block on " + dev)
					if e, _ := nmseOf(card, want); math.IsNaN(e) || e > 1e-3 {
						t.Fatalf("the tower on %s sits at NMSE %.3e from the CPU device's rows", dev, e)
					}
					if slices.Equal(card, want) {
						t.Fatalf("the tower's rows on %s are the CPU device's to the bit: the gate cannot tell them apart", dev)
					}
					if _, err := ts.MoveBlocks(ts.lo, ts.hi, nn.CPU); err != nil {
						t.Fatal(err)
					}
					if n := ts.GPULayers(); n != 0 {
						t.Fatalf("%d of the tower's blocks are left on %s after a move to the CPU", n, dev)
					}
					cpu, cpuDeep := encode("moved to the CPU")
					if !slices.Equal(cpu, want) || !slices.Equal(cpuDeep, wantDeep) {
						e, _ := nmseOf(cpu, want)
						t.Fatalf("moved off %s to the CPU device, the rows sit at NMSE %.3e from a CPU instance's", dev, e)
					}
					mv, err := ts.MoveBlocks(ts.lo, ts.hi, at)
					if err != nil {
						t.Fatal(err)
					}
					if n := ts.GPULayers(); n != nb {
						t.Fatalf("%d of the tower's %d blocks came back to %s (declined %v)", n, nb, at, mv.Declined)
					}
					back, backDeep := encode("moved back to " + dev)
					if !slices.Equal(back, card) || !slices.Equal(backDeep, cardDeep) {
						e, _ := nmseOf(back, card)
						t.Fatalf("back on %s, the rows sit at NMSE %.3e from the card's first rows", dev, e)
					}
					if st := d.Stats(); st.UnformedAtStep != 0 {
						t.Fatalf("%d of the tower's parts formed at a step on %s", st.UnformedAtStep, dev)
					}
					t.Logf("%s: %d blocks to the CPU and back, each picture its placement's rows to the bit", dev, nb)
				})
			}
		})
	}
}
