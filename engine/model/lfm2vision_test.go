package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// LFM2-VL against transformers (scripts/lfm2visiongold.py): the processor's
// plan for many sizes with no model, then d1-3B's tower and projector on real
// pictures from its own checkpoint in float32.

// lfm2Defaults is the tower the converter describes for an lfm2 mmproj of
// SigLIP2's 16-pixel patches and a 2x2 merge.
func lfm2Defaults() TowerConfig {
	return TowerConfig{Kind: jlm.ProjLFM2, Projector: "lfm2", PatchSz: 16, Scale: 2, ImageSz: 256,
		MinTiles: 2, MaxTiles: 10, MinPixels: 64 * 1024, MaxPixels: 256 * 1024, TileSize: 512, Tolerance: 2,
		Bicubic: true}
}

// TestLFM2LayoutMatchesProcessor holds the plan -- whether a picture is cut,
// the grid of tiles, the smart-resized size and every piece's patch grid --
// to Lfm2VlImageProcessor's for sizes below the floor, inside the budget,
// past the tolerance and of extreme aspect.
func TestLFM2LayoutMatchesProcessor(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "lfm2-layout.json"))
	if err != nil {
		t.Fatalf("GOLDEN MISSING: %v (scripts/lfm2visiongold.py --model-free)", err)
	}
	var g struct {
		Sizes []struct {
			W, H, Rows, Cols int
			Size             [2]int // height, width
			Pieces           [][2]int
		}
	}
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	if len(g.Sizes) == 0 {
		t.Fatal("the golden is empty")
	}
	c := lfm2Defaults()
	tiled := 0
	for _, s := range g.Sizes {
		cols, rows, tw, th, err := c.lfm2Layout(s.W, s.H)
		if err != nil {
			t.Fatal(err)
		}
		if cols != s.Cols || rows != s.Rows || tw != s.Size[1] || th != s.Size[0] {
			t.Errorf("%dx%d: %dx%d tiles and %dx%d, the processor's %dx%d and %dx%d",
				s.W, s.H, cols, rows, tw, th, s.Cols, s.Rows, s.Size[1], s.Size[0])
		}
		n := 1
		if cols*rows > 1 {
			n = cols*rows + 1
			tiled++
		}
		if n != len(s.Pieces) {
			t.Errorf("%dx%d: %d pieces, the processor's %d", s.W, s.H, n, len(s.Pieces))
		}
		want := 0
		for _, p := range s.Pieces {
			want += p[0] * p[1] / 4
		}
		if got := c.lfm2Tokens(s.W, s.H); got != want {
			t.Errorf("%dx%d: %d rows, the processor's %d", s.W, s.H, got, want)
		}
	}
	if tiled == 0 {
		t.Fatal("no golden size is cut into tiles, so the grid was never exercised")
	}
	t.Logf("%d sizes, %d of them tiled, every one the processor's", len(g.Sizes), tiled)
}

var (
	d1Text = testmodels.Path("d1/d1-3B-Q8_0.gguf")
	d1Proj = testmodels.Path("d1/mmproj-d1-3B-F16.gguf")
	d1Gold = testmodels.Path("d1/gold")
)

func openD1VL(t testing.TB, opts ...Option) *Model {
	t.Helper()
	if !vlmAvailable(d1Text, d1Proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (RULE 11)", d1Text)
	}
	m, err := Open(jlmOfPair(t, d1Text, d1Proj), append([]Option{noTune, WithKVF16(false)}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	if m.Tower() == nil {
		m.Close()
		t.Fatal("the container carries no tower, so this gate proves nothing")
	}
	return m
}

// lfm2Encode runs every piece of the picture through s's tower: the pixels,
// the tower's rows before the projector, and the projector's rows, each
// piece's back to back.
func lfm2Encode(t *testing.T, tw *Tower, s *State, path string) (px, vit, emb []float32, pieces int) {
	t.Helper()
	_, _, pcs, err := tw.LFM2Pieces(decodeImage(t, path))
	if err != nil {
		t.Fatal(err)
	}
	s.vis.beforeProj = func(x []float32) { vit = append(vit, x...) }
	defer func() { s.vis.beforeProj = nil }()
	for _, p := range pcs {
		px = append(px, p.Pixels...)
		out, err := s.EncodeGrid(p.Pixels, p.GH, p.GW)
		if err != nil {
			t.Fatal(err)
		}
		emb = append(emb, out...)
	}
	return px, vit, emb, len(pcs)
}

// TestLFM2RealMatchesTransformers pins the tower the container describes,
// then holds d1-3B's pictures to its checkpoint: the processor's pixels to
// the bit, the tower's rows, the projector alone on the reference's rows,
// and the projector's rows from the pixels; then the violations, and every
// tower block on each device backend against the host.
func TestLFM2RealMatchesTransformers(t *testing.T) {
	m := openD1VL(t)
	defer m.Close()
	tw := m.Tower()
	c := tw.Cfg
	for _, tc := range []struct {
		name      string
		got, want int
	}{
		{"NLayer", c.NLayer, 27}, {"NEmbd", c.NEmbd, 1152}, {"PatchSz", c.PatchSz, 16}, {"Scale", c.Scale, 2},
		{"table side", c.ImageSz / c.PatchSz, 16}, {"MinPixels", c.MinPixels, 64 * 1024},
		{"MaxPixels", c.MaxPixels, 256 * 1024}, {"TileSize", c.TileSize, 512}, {"ProjDim", c.ProjDim, m.Cfg.NEmbd},
	} {
		if tc.got != tc.want {
			t.Errorf("%s = %d, want %d", tc.name, tc.got, tc.want)
		}
	}
	if !c.Bicubic || c.Tolerance != 2 || c.MinTiles != 2 || c.MaxTiles != 10 {
		t.Errorf("bicubic %v, tolerance %g, tiles %d..%d: not the file's and the class's", c.Bicubic, c.Tolerance,
			c.MinTiles, c.MaxTiles)
	}
	for _, pic := range []string{"quad-and-disc", "photo", "photo-896", "perf_speed"} {
		t.Run(pic, func(t *testing.T) {
			base := filepath.Join(d1Gold, pic)
			raw, err := os.ReadFile(base + ".json")
			if err != nil {
				testmodels.Missing(t, "GOLDEN MISSING: %v -- run scripts/lfm2visiongold.py (RULE 11)", err)
			}
			var g struct {
				Rows, Cols int
				Pieces     [][2]int
			}
			if err := json.Unmarshal(raw, &g); err != nil {
				t.Fatal(err)
			}
			path := base + ".png"
			s := tw.testState()
			px, vit, emb, n := lfm2Encode(t, tw, s, path)
			s.Close()
			if n != len(g.Pieces) {
				t.Fatalf("%d pieces, the processor's %d", n, len(g.Pieces))
			}
			wantPx, wantVit, wantEmb := readF32(t, base+".px"), readF32(t, base+".vit"), readF32(t, base+".emb")
			if len(px) != len(wantPx) {
				t.Fatalf("%d pixel values, the processor's %d", len(px), len(wantPx))
			}
			nd, worst := pixelDiff(px, wantPx)
			nv, _ := nmse32(wantVit, vit)
			ne, we := nmse32(wantEmb, emb)
			t.Logf("%dx%d tiles, %d piece(s): %d of %d pixel values differ (worst %.3g); tower NMSE %.3e, "+
				"projector NMSE %.3e (max|d| %.4f)", g.Cols, g.Rows, n, nd, len(wantPx), worst, nv, ne, we)
			if nd != 0 {
				t.Errorf("%d of %d pixel values differ from the processor's", nd, len(wantPx))
			}
			if math.IsNaN(nv) || math.IsNaN(ne) || nv > 1e-2 || ne > 1e-2 {
				t.Fatalf("tower %.3e, projector %.3e against transformers", nv, ne)
			}

			// The projector alone, on the reference's own tower rows, piece
			// by piece: the unshuffle, the matrices and the erf GELU, without
			// the tower's int8 in front of them.
			proj := func() []float32 {
				ps := tw.testState()
				defer ps.Close()
				var out []float32
				at := 0
				for _, p := range g.Pieces {
					rows := p[0] * p[1] * c.NEmbd
					ps.vis.gh, ps.vis.gw = p[0], p[1]
					pj, err := ps.project(append([]float32(nil), wantVit[at:at+rows]...))
					if err != nil {
						t.Fatal(err)
					}
					out = append(out, pj...)
					at += rows
				}
				return out
			}
			pn, _ := nmse32(wantEmb, proj())
			lfm2TanhGELU = true
			pt, _ := nmse32(wantEmb, proj())
			lfm2TanhGELU = false
			was := tw.Cfg.fault
			tw.Cfg.fault = towerFaultShuffleSwap
			ps, _ := nmse32(wantEmb, proj())
			tw.Cfg.fault = was
			t.Logf("the projector alone on transformers' tower rows: NMSE %.3e; llama.cpp's tanh GELU %.3e; "+
				"the unshuffle transposed %.3e", pn, pt, ps)
			if !(ps > 20*pn) {
				t.Errorf("the transposed unshuffle reads %.3e against a clean %.3e", ps, pn)
			}
			// The bound is the int8 activations: the reference projector with
			// its inputs rounded to q8 per 32 and f16 weights reads 2.7e-3
			// against itself on the photo. Under that the erf and tanh GELUs
			// cannot be told apart (they differ by under 1e-3 of a value); the
			// form is the reference's by construction, and logged here.
			if math.IsNaN(pn) || pn > 1e-2 {
				t.Fatalf("the projector alone reads NMSE %.3e against transformers'", pn)
			}

			// The violations, from the pixels.
			vio := func(name string, set *bool) {
				*set = true
				s := tw.testState()
				vpx, _, vemb, _ := lfm2Encode(t, tw, s, path)
				s.Close()
				*set = false
				if len(vpx) != len(wantPx) || len(vemb) != len(wantEmb) {
					t.Logf("violation %q: %d values against %d -- caught by shape", name, len(vemb), len(wantEmb))
					return
				}
				d, _ := pixelDiff(vpx, wantPx)
				t.Logf("violation %q: %d pixel values differ", name, d)
				if d == 0 {
					t.Errorf("violation %q moved no pixel: the gate cannot see it", name)
				}
			}
			// A picture already at its planned size is not resampled, so the
			// filter is not in it.
			if b := decodeImage(t, path).Bounds(); g.Cols*g.Rows > 1 ||
				b.Dy() != g.Pieces[0][0]*c.PatchSz || b.Dx() != g.Pieces[0][1]*c.PatchSz {
				vio("the bilinear resize", &lfm2Bilinear)
			}
			if g.Cols*g.Rows > 1 {
				vio("no thumbnail", &lfm2NoThumb)
			}
			if p := g.Pieces[len(g.Pieces)-1]; p[0] != p[1] {
				phi4PosTransposed = true
				s := tw.testState()
				_, _, bad, _ := lfm2Encode(t, tw, s, path)
				s.Close()
				phi4PosTransposed = false
				nb, _ := nmse32(wantEmb, bad)
				t.Logf("violation, the table resized to the grid's transpose: NMSE %.3e", nb)
				if !(nb > 10*ne) {
					t.Errorf("a transposed position table reads %.3e against a clean %.3e", nb, ne)
				}
			}

			for _, spec := range stepDevices() {
				gpu, err := tier.OpenWith(tier.WithDevices(spec))
				if err != nil || gpu == nil {
					t.Logf("%s: not present (%v)", spec, err)
					continue
				}
				t.Run(spec, func(t *testing.T) {
					defer gpu.Close()
					s := tw.testState()
					defer s.Close()
					s.SetDevice(gpu)
					if n := s.GPUBlocks(); n != c.NLayer {
						t.Fatalf("%s took %d of %d tower blocks (%s)", spec, n, c.NLayer, gpu.Err())
					}
					_, _, got, _ := lfm2Encode(t, tw, s, path)
					dn, _ := nmseOf(got, emb)
					tn, _ := nmse32(wantEmb, got)
					t.Logf("%s, all %d blocks: NMSE %.3e against the host, %.3e against transformers",
						spec, c.NLayer, dn, tn)
					if math.IsNaN(dn) || dn > 1e-2 || tn > 1e-2 {
						t.Fatalf("%s: NMSE %.3e against the host, %.3e against transformers", spec, dn, tn)
					}
				})
			}
		})
	}
}
