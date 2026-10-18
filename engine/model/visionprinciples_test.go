package model

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// The five principles for the gemma3 and internvl towers (AGENTS.md): their
// blocks page under a budget below the tower with the same answer, relocate
// (visiondevice_test.go), and a warm text decode after a picture allocates
// nothing on the host or a device. JIT and no Go compute are the tower's own
// generated kernels and are held by the package-wide gates.

// TestVisionTowerPagesUnderABudget encodes with a budget of a few vision pages
// -- below the tower, so every block is evicted and read again -- and demands
// the unbudgeted answer bit for bit, which needs eviction, re-read and re-bind
// all to be right.
func TestVisionTowerPagesUnderABudget(t *testing.T) {
	for _, fam := range visionFamilies {
		t.Run(fam.name, func(t *testing.T) {
			var px []float32
			encode := func(budget uint64) ([]float32, int64) {
				var opts []Option
				if budget > 0 {
					opts = append(opts, WithPageBudget(budget))
				}
				m, tw := fam.open(t, opts...)
				defer m.Close()
				if px == nil {
					px = rainbow(tw.Cfg.ImageSz)
				}
				s := tw.testState()
				defer s.Close()
				out, err := s.Encode(px)
				if err != nil {
					t.Fatalf("budget %d: %v", budget, err)
				}
				_, ev := m.container.Faults()
				return append([]float32(nil), out...), int64(ev)
			}
			full, _ := encode(0)
			m, tw := fam.open(t)
			base := int(m.container.H.NBlocks)
			vpage := m.container.PageBytes(base)
			tower := uint64(tw.Cfg.NLayer) * vpage
			m.Close()
			got, ev := encode(2 * vpage)
			if ev == 0 {
				t.Fatalf("a budget of two vision pages against a %d-page tower evicted nothing: "+
					"it was not the binding constraint", tw.Cfg.NLayer)
			}
			if len(got) != len(full) {
				t.Fatalf("%d values under the budget, %d without", len(got), len(full))
			}
			for i := range full {
				if got[i] != full[i] {
					t.Fatalf("element %d is %v under a %d-byte budget (tower %d), %v without: a page-in "+
						"bound a stale span", i, got[i], 2*vpage, tower, full[i])
				}
			}
			t.Logf("two pages of %d MiB against a %d-page tower: %d eviction(s), bit-identical",
				vpage>>20, tw.Cfg.NLayer, ev)
		})
	}
}

// TestVisionDecodeDoesNotAllocate holds a warm text decode AFTER a picture to
// zero engine allocations, on the host and on every device backend present: the
// picture's spans, the tower and a bidirectional run must leave nothing behind
// that a token pays for.
func TestVisionDecodeDoesNotAllocate(t *testing.T) {
	specs := append([]string{"host"}, testdevices.Specs("cuda", "vulkan", "metal")...)
	for _, fam := range visionFamilies {
		t.Run(fam.name, func(t *testing.T) {
			m, tw := fam.open(t, noTune)
			defer m.Close()
			ts := tw.testState()
			emb, err := ts.Encode(rainbow(tw.Cfg.ImageSz))
			if err != nil {
				t.Fatal(err)
			}
			emb = append([]float32(nil), emb...)
			ts.Close()
			spans, err := m.ChatSpans([]ChatMessage{{Role: "user", Content: "Describe this image.",
				Images: 1}}, [][]float32{emb}, true)
			if err != nil {
				t.Fatal(err)
			}
			for _, spec := range specs {
				t.Run(spec, func(t *testing.T) {
					var g *tier.GPU
					if spec != "host" {
						if g, err = openTier(t, tier.WithDevices(spec)); err != nil || g == nil {
							t.Skipf("%s: not present (%v)", spec, err)
						}
						defer g.Close()
					}
					// Positions stay inside one 256-row KV page and one
					// 128-position score grain over the window (decodeAllocs).
					const warm, n = 24, 64
					pos := SpanPositions(spans, m.Cfg.NEmbd)
					s := m.NewState(pos + warm + n + 8)
					defer s.Close()
					if g != nil {
						if err := s.SetDevice(g); err != nil {
							t.Fatal(err)
						}
						if s.GPULayers() == 0 {
							t.Skipf("no block placed on %s -- this arm proved nothing", spec)
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
					allocVerdict(t, fmt.Sprintf("after a %d-row picture at position %d, %d of %d blocks placed",
						len(emb)/m.Cfg.NEmbd, pos, s.GPULayers(), m.Cfg.NLayer), w, n, m.container.Reads()-r0, captures)
				})
			}
		})
	}
}
