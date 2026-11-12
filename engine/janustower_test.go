package engine

import (
	"encoding/binary"
	"image"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestJanusTowerMatchesTransformers holds Janus-Pro-1B's stated tower to
// transformers' own JanusImageProcessorPil, JanusVisionModel and aligner
// (scripts/janusgold.py) on two pictures, one wide enough to letterbox: the
// picture op's pixels -- resized by the longer side and centred on the
// background's square -- within a byte of the processor's, and the
// aligner's rows at the bound the Q8_0 mmproj and the engine's int8
// activations over 24 blocks part them by (5e-2, the sum of the three
// parts the deleted per-family gate measured). The box filled black in
// place of the processor's grey is the violation.
func TestJanusTowerMatchesTransformers(t *testing.T) {
	text, proj := testmodels.Path("minicpmv/Janus-Pro-1B-Q4_K_M.gguf"), testmodels.Path("minicpmv/mmproj-Janus-Pro-1B-Q8_0.gguf")
	gold := testmodels.Path("minicpmv/hf/Janus-Pro-1B/gold")
	if !sourcePresent(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (set JITLLM_MODELS)", text)
	}
	m, err := model.Open(jlmOfPair(t, text, proj))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	tg := m.Tower
	if tg == nil {
		t.Fatal("the model has no tower")
	}
	wide, _ := squareOf(t, tg, 1000, 500, 0)
	thin, _ := squareOf(t, tg, 1000, 10, 0)
	if wide.Ow != 384 || wide.Oh != 192 || thin.Oh != 14 {
		t.Fatalf("1000x500 resizes to %dx%d and 1000x10 to a side of %d: want a 384-pixel box, a side at least 14",
			wide.Ow, wide.Oh, thin.Oh)
	}
	s := testInstance(t, m, 64, WithKVF16(false))
	defer s.Close()
	readF32 := func(path string, n int) []float32 {
		raw, err := os.ReadFile(path)
		if err != nil {
			testmodels.Missing(t, "GOLDEN MISSING: %v -- run scripts/janusgold.py", err)
		}
		if len(raw) != 4*n {
			t.Fatalf("%s holds %d bytes, want %d", path, len(raw), 4*n)
		}
		out := make([]float32, n)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		}
		return out
	}
	readPNG := func(path string) image.Image {
		f, err := os.Open(path)
		if err != nil {
			testmodels.Missing(t, "PICTURE MISSING: %v", err)
		}
		defer f.Close()
		img, err := png.Decode(f)
		if err != nil {
			t.Fatal(err)
		}
		return img
	}
	v := m.Container.Vision()
	for _, pic := range []struct{ name, path string }{
		{"photo896x448", testmodels.Path("minicpmv/photo896x448.png")},
		{"quad-and-disc", filepath.Join("testdata", "quad-and-disc.png")},
	} {
		t.Run(pic.name, func(t *testing.T) {
			img := readPNG(pic.path)
			base := filepath.Join(gold, pic.name)
			want := readF32(base+".px", 384*384*3)
			px, _ := towerPixels(t, s, img)
			off, worst := 0, 0.0
			for i, w := range want {
				d := math.Abs(float64(px[i] - w))
				if d > 1e-6 {
					off++
				}
				worst = math.Max(worst, d)
			}
			t.Logf("%d of %d pixel values off the processor's, worst by %.5f", off, len(want), worst)
			if worst > 1.01/127.5 || off*10000 > len(want) {
				t.Fatalf("the letterbox is not the processor's: %d values off, worst by %.5f", off, worst)
			}
			rows, err := s.EncodePicture(img)
			if err != nil {
				t.Fatal(err)
			}
			emb := readF32(base+".emb", 576*int(v.ProjDim))
			ne, nw := nmseOf(rows, emb)
			t.Logf("the aligner's rows: NMSE %.3e, max|d| %.3g", ne, nw)
			if math.IsNaN(ne) || ne > 5e-2 {
				t.Fatalf("the aligner's rows are NMSE %.3e from transformers' (bound 5e-2)", ne)
			}
		})
	}
	// The violation: a black box.
	rec := v.Tower
	ci := slices.IndexFunc(rec.Pre, func(s jlm.PreStep) bool { return s.Kind == jlm.PreCanvas })
	if ci < 0 {
		t.Fatalf("the tower's steps %+v lay the picture on no canvas", rec.Pre)
	}
	fill := rec.Pre[ci].Ints[:3]
	keep := [3]int32{fill[0], fill[1], fill[2]}
	fill[0], fill[1], fill[2] = 0, 0, 0
	black, err := m.BuildTowerGraphs()
	fill[0], fill[1], fill[2] = keep[0], keep[1], keep[2]
	if err != nil {
		t.Fatal(err)
	}
	m.Tower = black
	bs := testInstance(t, m, 64, WithKVF16(false))
	bad, _ := towerPixels(t, bs, readPNG(testmodels.Path("minicpmv/photo896x448.png")))
	bs.Close()
	m.Tower = tg
	want := readF32(filepath.Join(gold, "photo896x448.px"), 384*384*3)
	off := 0
	for i, w := range want {
		if math.Abs(float64(bad[i]-w)) > 1.01/127.5 {
			off++
		}
	}
	if off == 0 {
		t.Fatal("VIOLATION: a black box writes the processor's pixels")
	}
	t.Logf("violation (a black box): %d pixel values off", off)
}

// TestJanusTowerRunsOnTheCard places Janus-Pro's tower on each device,
// every block and then half of them, and holds the letterboxed photo's rows
// to transformers' at the CPU device's bound.
func TestJanusTowerRunsOnTheCard(t *testing.T) {
	text, proj := testmodels.Path("minicpmv/Janus-Pro-1B-Q4_K_M.gguf"), testmodels.Path("minicpmv/mmproj-Janus-Pro-1B-Q8_0.gguf")
	if !sourcePresent(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (set JITLLM_MODELS)", text)
	}
	m, err := model.Open(jlmOfPair(t, text, proj))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	f, err := os.Open(testmodels.Path("minicpmv/photo896x448.png"))
	if err != nil {
		testmodels.Missing(t, "PICTURE MISSING: %v", err)
	}
	img, err := png.Decode(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(testmodels.Path("minicpmv/hf/Janus-Pro-1B/gold"), "photo896x448.emb"))
	if err != nil {
		testmodels.Missing(t, "GOLDEN MISSING: %v -- run scripts/janusgold.py", err)
	}
	ref := make([]float32, len(raw)/4)
	for i := range ref {
		ref[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	towerRunsOnTheCard(t, m, img, ref, 5e-2, WithKVF16(false))
}
