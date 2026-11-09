package engine

import (
	"fmt"
	"image"
	"image/color"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// layoutsOf is gray picture g held in each sample layout a frame decodes to
// -- gray, paletted over a gray palette, YCbCr at neutral chroma, RGBA --
// every one of them the same colours.
func layoutsOf(g *image.Gray) []image.Image {
	r := g.Bounds()
	pal := make(color.Palette, 256)
	for i := range pal {
		pal[i] = color.Gray{Y: uint8(i)}
	}
	p := image.NewPaletted(r, pal)
	y := image.NewYCbCr(r, image.YCbCrSubsampleRatio444)
	n := image.NewNRGBA(r)
	for j := r.Min.Y; j < r.Max.Y; j++ {
		for i := r.Min.X; i < r.Max.X; i++ {
			v := g.GrayAt(i, j).Y
			p.SetColorIndex(i, j, v)
			y.Y[y.YOffset(i, j)] = v
			y.Cb[y.COffset(i, j)], y.Cr[y.COffset(i, j)] = 128, 128
			n.SetNRGBA(i, j, color.NRGBA{R: v, G: v, B: v, A: 255})
		}
	}
	return []image.Image{g, p, y, n}
}

// TestAVideoPairMixesLayouts encodes a video pair whose frames decode to
// different sample layouts: each of Clef's VL fixture's first pair's frames
// made gray and held gray, paletted, YCbCr and RGBA, every ordered pair of
// the four through its own entry (model.PairKey), each frame reading its
// own layout's words. All sixteen encode the rows the RGBA pair does, bit
// for bit; a frame read in the other's layout would not.
func TestAVideoPairMixesLayouts(t *testing.T) {
	src, ok := existingModel(testmodels.Path("clef/fixture-vl/clef-fixture-vl.gguf"))
	if !ok {
		testmodels.Missing(t, "MODEL MISSING: %s (scripts/clefgold.py makes it)", testmodels.Path("clef/fixture-vl/clef-fixture-vl.gguf"))
	}
	m, err := model.Open(jlmOfPair(t, src, towerOf(src)))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	spans, err := m.VideoSpans(decisionVideo(t, "video"), 0)
	if err != nil {
		t.Fatal(err)
	}
	at := slices.IndexFunc(spans, func(sp model.Span) bool { return sp.Frame != nil })
	if at < 0 {
		t.Fatal("the video has no pair")
	}
	gray := func(img image.Image) *image.Gray {
		g := image.NewGray(img.Bounds())
		for j := g.Rect.Min.Y; j < g.Rect.Max.Y; j++ {
			for i := g.Rect.Min.X; i < g.Rect.Max.X; i++ {
				g.Set(i, j, img.At(i, j))
			}
		}
		return g
	}
	f0, f1 := layoutsOf(gray(spans[at].Picture)), layoutsOf(gray(spans[at].Frame))
	s := testInstance(t, m, 4096)
	defer s.Close()
	rows := func(a, b image.Image) []float32 {
		sp := slices.Clone(spans)
		sp[at].Picture, sp[at].Frame = a, b
		enc, err := s.encodePictures(sp, 0)
		if err != nil {
			t.Fatalf("%T beside %T: %v", a, b, err)
		}
		return slices.Clone(enc[at].Embd)
	}
	want := rows(f0[3], f1[3])
	if slices.Equal(want, rows(f1[3], f0[3])) {
		t.Fatal("the pair's frames swapped encode alike: the comparison cannot see a frame read wrong")
	}
	for i := range f0 {
		for j := range f1 {
			name := fmt.Sprintf("%T then %T", f0[i], f1[j])
			if got := rows(f0[i], f1[j]); !slices.Equal(got, want) {
				t.Errorf("%s: the pair's rows are not the RGBA pair's", name)
			}
		}
	}
}
