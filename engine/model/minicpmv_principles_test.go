package model

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// The five principles on a sliced tower: its blocks page under a budget and
// come back bit-identical, they relocate host -> device -> host with the same
// answer, and a decode after a picture allocates nothing.

// mcvEncodePhoto encodes the photo's overview, a 45x23-patch grid, and returns
// the projector's output.
func mcvEncodePhoto(t *testing.T, tw *Tower, s *State) []float32 {
	t.Helper()
	px, gh, gw := mcvOverview(t, tw, s)
	out, err := s.EncodeGrid(px, gh, gw)
	if err != nil {
		t.Fatal(err)
	}
	return append([]float32(nil), out...)
}

// mcvOverview is the photo's overview, resampled and normalised by the
// engine's own picture path: a 45x23-patch grid.
func mcvOverview(t testing.TB, tw *Tower, s *State) ([]float32, int, int) {
	t.Helper()
	img := decodeImage(t, mcvPhoto)
	lay, err := tw.Layout(img.Bounds().Dx(), img.Bounds().Dy())
	if err != nil {
		t.Fatal(err)
	}
	px, err := tw.Pieces(img, lay)
	if err != nil {
		t.Fatal(err)
	}
	gh, gw := lay.Pieces[0].Grid(tw.Cfg.PatchSz)
	return px[0], gh, gw
}

// TestMiniCPMVTowerPagesUnderABudget encodes with every tower block resident
// and again under budgets of a few vision pages, which must evict and re-read
// tower blocks mid-encode, and demands the same bits.
func TestMiniCPMVTowerPagesUnderABudget(t *testing.T) {
	m := openMiniCPMV(t)
	vpage := m.container.PageBytes(int(m.container.H.NBlocks))
	full := func() []float32 {
		s := m.Tower().testState()
		defer s.Close()
		return mcvEncodePhoto(t, m.Tower(), s)
	}()
	m.Close()
	for _, k := range []uint64{1, 3} {
		mb := openMiniCPMV(t, WithPageBudget(k*vpage))
		s := mb.Tower().testState()
		got := mcvEncodePhoto(t, mb.Tower(), s)
		s.Close()
		_, ev := mb.container.Faults()
		mb.Close()
		if ev == 0 {
			t.Fatalf("a budget of %d vision page(s) evicted nothing: it was not the binding constraint", k)
		}
		for i := range full {
			if got[i] != full[i] {
				t.Fatalf("budget of %d vision page(s): element %d is %v, want %v -- a page-in bound a stale span",
					k, i, got[i], full[i])
			}
		}
		t.Logf("a budget of %d vision page(s): %d eviction(s), every element identical", k, ev)
	}
}

// TestMiniCPMVTowerRelocates moves the tower's blocks onto each device, home,
// and back, encoding at every stop: the device runs are bit-identical to each
// other (the same kernels on the same input) and both within the host's band.
func TestMiniCPMVTowerRelocates(t *testing.T) {
	m := openMiniCPMV(t)
	defer m.Close()
	tw := m.Tower()
	ran := 0
	for _, spec := range stepDevices() {
		g, err := openTier(t, tier.WithDevices(spec))
		if err != nil || g == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer g.Close()
			s := tw.testState()
			defer s.Close()
			s.SetDevice(g)
			if s.GPUBlocks() != tw.Cfg.NLayer {
				t.Fatalf("%d of %d blocks placed", s.GPUBlocks(), tw.Cfg.NLayer)
			}
			dev1 := mcvEncodePhoto(t, tw, s)
			s.SetDevice(nil)
			host := mcvEncodePhoto(t, tw, s)
			s.SetDevice(g)
			if s.GPUBlocks() != tw.Cfg.NLayer {
				t.Fatalf("back on the device: %d of %d blocks placed", s.GPUBlocks(), tw.Cfg.NLayer)
			}
			dev2 := mcvEncodePhoto(t, tw, s)
			for i := range dev1 {
				if dev1[i] != dev2[i] {
					t.Fatalf("element %d: %v on the first visit, %v on the second", i, dev1[i], dev2[i])
				}
			}
			n := visNMSE(t, "device against host", dev1, host)
			t.Logf("%s -> host -> %s: the two device encodes identical, NMSE %.3e against the host's", spec, spec, n)
			if n > 1e-2 {
				t.Fatalf("NMSE %.3e between the device's and the host's", n)
			}
		})
	}
	if ran == 0 {
		t.Skip("no GPU backend on this host")
	}
}

// TestMiniCPMVDecodeAfterAPictureDoesNotAllocate prefills a picture -- the
// overview and both slices, in their markers -- and then holds a warm decode
// window to zero engine allocations, on the host and on each device: the
// tower's scratch and the picture's embeddings are the prompt's, not a
// token's.
func TestMiniCPMVDecodeAfterAPictureDoesNotAllocate(t *testing.T) {
	m := openMiniCPMV(t, noTune)
	defer m.Close()
	tw := m.Tower()
	img := decodeImage(t, mcvPhoto)
	run := func(t *testing.T, g *tier.GPU) {
		lay, err := tw.Layout(img.Bounds().Dx(), img.Bounds().Dy())
		if err != nil {
			t.Fatal(err)
		}
		pieces, err := tw.PicturePieces(img, lay)
		if err != nil {
			t.Fatal(err)
		}
		pic, err := m.PictureSpans(lay, 0, true, pieces)
		if err != nil {
			t.Fatal(err)
		}
		msgs := []ChatMessage{{Role: "user", Content: "What is this?", Images: 1}}
		spans, err := m.ChatSpansParts(msgs, [][]Span{pic}, true)
		if err != nil {
			t.Fatal(err)
		}
		const warm, n = 96, 48
		s := m.NewState(SpanPositions(spans, m.Cfg.NEmbd) + warm + n + 8)
		defer s.Close()
		if g != nil {
			if err := s.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			if s.GPULayers() == 0 {
				t.Skip("no text block placed on the device -- this arm proved nothing")
			}
		}
		if _, err := s.PrefillMixed(spans...); err != nil {
			t.Fatal(err)
		}
		for i := range warm {
			if _, err := s.Forward(int32(5 + i)); err != nil {
				t.Fatal(err)
			}
		}
		var c0 tier.Stats
		if g != nil {
			c0 = g.Stats()
		}
		r0 := m.container.Reads()
		w := countAllocs(func() {
			for i := range n {
				if _, err := s.Forward(int32(3 + i%64)); err != nil {
					t.Fatal(err)
				}
			}
		})
		captures := 0
		if g != nil {
			captures = g.Stats().Captures - c0.Captures
		}
		allocVerdict(t, fmt.Sprintf("after a %d-piece picture, %d of %d blocks placed", len(lay.Pieces),
			s.GPULayers(), m.Cfg.NLayer), w, n, m.container.Reads()-r0, captures)
	}
	t.Run("host", func(t *testing.T) { run(t, nil) })
	for _, spec := range testdevices.Specs("cuda", "vulkan", "metal") {
		t.Run(spec, func(t *testing.T) {
			g, err := openTier(t, append([]tier.Option{tier.WithDevices(spec)}, testTierOpts(t)...)...)
			if err != nil || g == nil {
				noDevice(t, spec, err)
			}
			defer g.Close()
			run(t, g)
		})
	}
}
