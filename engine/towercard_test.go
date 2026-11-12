package engine

import (
	"image"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// towerRunsOnTheCard places m's tower on each device -- every block, then
// half of them with the rest on the CPU device -- and holds img's rows,
// with no part formed at a step, to the CPU device's within the
// reduction-order band (an NMSE of 1e-3); or, where ref is given, to the
// reference within bound, the bound the CPU device's rows meet (zero: twice
// the CPU device's own distance): over a deep tower each device's int8
// activations part it from the other by more than the band, and the
// falsifiable statement is each device's distance from the reference. The
// violation: a grey picture of img's size, its rows on the CPU device ten
// times past the bound.
func towerRunsOnTheCard(t *testing.T, m *model.Model, img image.Image, ref []float32, bound float64, opts ...Option) {
	t.Helper()
	host := testInstance(t, m, 64, opts...)
	want, err := host.EncodePicture(img)
	if err != nil {
		t.Fatal(err)
	}
	want = slices.Clone(want)
	grey := image.NewGray(img.Bounds())
	for i := range grey.Pix {
		grey.Pix[i] = 128
	}
	blank, err := host.EncodePicture(grey)
	if err != nil {
		t.Fatal(err)
	}
	blank = slices.Clone(blank)
	host.Close()
	what := "the CPU device's"
	if ref != nil {
		if len(ref) != len(want) {
			t.Fatalf("the reference holds %d values, the CPU device's rows %d", len(ref), len(want))
		}
		cpu, _ := nmseOf(want, ref)
		if bound == 0 {
			// No device further from the reference than twice the CPU
			// device is.
			bound = 2 * cpu
		}
		t.Logf("the CPU device's rows: NMSE %.3e from the reference (bound %.3e)", cpu, bound)
		want, what = ref, "the reference"
	} else {
		bound = 1e-3
	}
	if len(blank) != len(want) {
		t.Fatalf("a grey picture of the same size writes %d values, the picture %d", len(blank), len(want))
	}
	if bad, _ := nmseOf(blank, want); !(bad > 10*bound) {
		t.Fatalf("VIOLATION: a grey picture's rows sit at NMSE %.3e from %s, within ten times the bound %.0e", bad, what, bound)
	} else {
		t.Logf("violation (a grey picture): NMSE %.3e from %s", bad, what)
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
				if m.Tower.Chain {
					// A chain's blocks are forms over their own grids, placed
					// as the chain (PlaceTowerChain), not as instance blocks.
					if err := s.PlaceTowerChain(g, blocks); err != nil {
						s.Close()
						t.Fatalf("%d of the chain's %d blocks on %s: %v (%s)", blocks, nb, dev, err, g.Err())
					}
				} else {
					if err := s.tower.s.SetDeviceLayers(g, blocks); err != nil {
						s.Close()
						t.Fatal(err)
					}
					if n := s.tower.s.GPULayers(); n != blocks {
						s.Close()
						t.Fatalf("%d of the tower's %d blocks placed on %s, want %d: %s", n, nb, dev, blocks, g.Err())
					}
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

// TestCheckpointTowersRunOnTheCard places Gemma 3's, InternVL3's and llava's
// stated towers -- SigLIP's 27 blocks and the pooled projector, InternViT's
// 24 over each tile, CLIP's 23 and the two-layer projector -- on each device,
// every block and then half of them, and holds the quadrant picture's rows to
// transformers' own (scripts/visiongold.py), each device within twice the CPU
// device's distance from them (towerRunsOnTheCard): over towers this deep the
// devices' int8 activations part them from each other past the
// reduction-order band.
func TestCheckpointTowersRunOnTheCard(t *testing.T) {
	for _, c := range []struct{ name, text, mm, gold string }{
		{"gemma3", "gemma-3-4b-it-Q4_K_M.gguf", "vlm/mmproj-gemma-3-4b-it-f16.gguf", "gemma3"},
		{"internvl3", "vlm/InternVL3-1B-Instruct-Q8_0.gguf", "vlm/mmproj-InternVL3-1B-Instruct-Q8_0.gguf", "internvl"},
		{"llava", "llava-phi-3-mini-int4.gguf", "llava-phi-3-mini-mmproj-f16.gguf", "llava"},
	} {
		t.Run(c.name, func(t *testing.T) {
			text, mm := testmodels.Path(c.text), testmodels.Path(c.mm)
			if !sourcePresent(text, mm) {
				testmodels.Missing(t, "MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", text, mm)
			}
			gold := testmodels.Path(filepath.Join("vlm", "goldens", c.gold, "quad-and-disc.features.f32"))
			if _, err := os.Stat(gold); err != nil {
				testmodels.Missing(t, "GOLDEN MISSING: %v (scripts/visiongold.py %s writes it)", err, c.gold)
			}
			m, err := model.Open(jlmOfPair(t, text, mm))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			towerRunsOnTheCard(t, m, auditPicture(t), readF32(t, gold), 0)
		})
	}
}
