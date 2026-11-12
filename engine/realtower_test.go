package engine

import (
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

// TestRealTowersMatchTransformers holds real checkpoints' stated towers to
// transformers' own tower and head in float32 on the photo
// (scripts/gemma4real.py, gemma3nreal.py, pixtralreal.py; only the towers'
// tensors fetched): the processor's pixels, the rows the text model reads
// within 3e-2 of transformers -- the towers' matrices are the converter's
// and their activations int8, which is what separates the two -- and then
// every block on each card and half of them, held to transformers within
// twice the CPU device's distance, a grey picture of the photo's size the
// violation.
func TestRealTowersMatchTransformers(t *testing.T) {
	for _, c := range []struct{ name, golden, text, proj string }{
		{"gemma-4-E2B-it", "vlmb/hf/gemma-4-E2B-it/goldens", "vlmb/gemma-4-E2B-it-Q4_K_M.gguf", "vlmb/mmproj-gemma-4-E2B-it-BF16.gguf"},
		{"gemma-3n-E2B-it", "vlmb/hf/gemma-3n-E2B-it/goldens", "gemma3n/gemma-3n-E2B-it-Q4_K_M.gguf", "gemma3n/mmproj-gemma-3n-E2B-it-f16.gguf"},
		{"Ministral-3-3B", "vlmb/hf/Ministral-3-3B/goldens", "newarch/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf", "vlmb/mmproj-Ministral-3-3B-BF16.gguf"},
	} {
		t.Run(c.name, func(t *testing.T) {
			dir := testmodels.Path(c.golden)
			if _, err := os.Stat(filepath.Join(dir, "photo.features.f32")); err != nil {
				testmodels.Missing(t, "GOLDEN MISSING: %v -- run scripts/*real.py (RULE 11)", err)
			}
			text, proj := testmodels.Path(c.text), testmodels.Path(c.proj)
			if !sourcePresent(text, proj) {
				testmodels.Missing(t, "MODEL MISSING: %s and %s (RULE 11)", text, proj)
			}
			m, err := model.Open(jlmOfPair(t, text, proj))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			f, err := os.Open(testmodels.Path("vlm/goldens/photo.png"))
			if err != nil {
				testmodels.Missing(t, "PICTURE MISSING: %v", err)
			}
			img, err := png.Decode(f)
			f.Close()
			if err != nil {
				t.Fatal(err)
			}
			wantPx := readF32File(t, filepath.Join(dir, "photo.pixels.f32"))
			want := readF32File(t, filepath.Join(dir, "photo.features.f32"))
			opts := []Option{WithKVF16(false)}
			s := testInstance(t, m, 64, opts...)
			px, sq := towerPixels(t, s, img)
			P := m.Tower.Patch
			if len(px) != len(wantPx) {
				t.Fatalf("the photo resized to %dx%d, %d samples; the processor's %d", sq.Gw*P, sq.Gh*P, len(px), len(wantPx))
			}
			pn, pw := nmseOf(px, wantPx)
			t.Logf("pixels: %dx%d, NMSE %.3e, max|d| %.2e against the processor", sq.Gw*P, sq.Gh*P, pn, pw)
			if math.IsNaN(pn) || pn > 1e-10 || pw > 1e-5 {
				t.Fatalf("the photo's pixels are not the processor's: NMSE %.3e, max|d| %.2e", pn, pw)
			}
			rows, err := s.EncodePicture(img)
			s.Close()
			if err != nil {
				t.Fatal(err)
			}
			D, tail := int(m.Container.Vision().ProjDim), 0
			for _, op := range m.Container.Vision().Tower.Head.Ops {
				if op.Code == jlm.TOpAppendRow {
					tail++
				}
			}
			full := rows
			rows = gridFeatures(m.Tower, sq, rows[:len(rows)-tail*D], D)
			nm, worst := nmseOf(rows, want)
			t.Logf("%d values: NMSE %.3e, max|d| %.4f against transformers' f32 tower", len(rows), nm, worst)
			if len(rows) != len(want) || math.IsNaN(nm) || nm > 3e-2 {
				t.Fatalf("the real tower's %d values read NMSE %.3e against transformers' %d", len(rows), nm, len(want))
			}
			towerRunsOnTheCard(t, m, img, withFeatures(m.Tower, sq, full, want, D), 0, opts...)
		})
	}
}

// withFeatures is rows with the features gridFeatures reads in them
// replaced by feats: transformers' features among the rows the head lays
// between and after them, which are stored rows the same on every device.
func withFeatures(tg *model.TowerGraphs, sq model.Square, rows, feats []float32, D int) []float32 {
	out := slices.Clone(rows)
	if tg.Frame == 0 {
		copy(out, feats)
		return out
	}
	w, h := sq.Gw/tg.Frame, sq.Gh/tg.Frame
	at := tg.Ends * D
	for r := range h {
		copy(out[at:at+w*D], feats[r*w*D:(r+1)*w*D])
		at += (w + tg.Line) * D
	}
	return out
}

// gridFeatures is a square's rows of D values without the rows the head lays
// between them (tg.Line after each merged row, tg.Trim fewer, tg.Ends around
// them): the features alone, as transformers' tower writes them and the
// prompt carries the rest as tokens.
func gridFeatures(tg *model.TowerGraphs, sq model.Square, rows []float32, D int) []float32 {
	if tg.Frame == 0 {
		return rows
	}
	w, h := sq.Gw/tg.Frame, sq.Gh/tg.Frame
	out := make([]float32, 0, w*h*D)
	at := tg.Ends * D
	for range h {
		out = append(out, rows[at:at+w*D]...)
		at += (w + tg.Line) * D
	}
	return out
}

// TestQwenVLRealMatchesTransformers holds real Qwen-VL checkpoints and
// HunyuanOCR to transformers on the quadrant picture (scripts/qwenvlreal.py,
// hunyuanvlreal.py): the text model teacher-forced on transformers' tower
// rows, its logits within the checkpoint's bound -- the F16 checkpoints
// against float32 within 1e-3, Qwen2.5-VL-3B's Q4_K_M against bfloat16 (its
// float32 copy does not fit in memory) within 2e-2 and one argmax -- and the
// rows at sequential positions, as an engine without M-RoPE lays them, the
// violation where the arm is precise enough to see it; then the whole model
// from the picture, the engine's own preprocessing and tower, within 2e-2
// and one argmax; then the tower on each card against transformers' rows.
func TestQwenVLRealMatchesTransformers(t *testing.T) {
	for _, c := range []struct {
		name, text, proj string
		textNMSE         float64
		textFlips        int
		mrope            bool
	}{
		{"Qwen2-VL-2B-Instruct", "Qwen2-VL-2B-Instruct-f16.gguf", "mmproj-Qwen2-VL-2B-Instruct-f16.gguf", 1e-3, 0, true},
		{"Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf", 2e-2, 1, false},
		{"Qwen3-VL-2B-Instruct", "Qwen3-VL-2B-Instruct-f16.gguf", "mmproj-Qwen3-VL-2B-Instruct-f16.gguf", 1e-3, 0, true},
		{"Qwen3.5-0.8B", "Qwen3.5-0.8B-f16.gguf", "mmproj-Qwen3.5-0.8B-f16.gguf", 1e-3, 0, true},
		{"HunyuanOCR", "HunyuanOCR-f16.gguf", "mmproj-HunyuanOCR-f16.gguf", 1e-3, 0, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			g := qvlGoldenOf(t, "real-"+c.name)
			text, proj := testmodels.Path(c.text), testmodels.Path(c.proj)
			if !sourcePresent(text, proj) {
				testmodels.Missing(t, "MODEL MISSING: %s and %s -- QVL_GGUF=1 scripts/qwenvlreal.py writes them (RULE 11)", text, proj)
			}
			m, err := model.Open(jlmOfPair(t, text, proj))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			opts := []Option{WithKVF16(false)}
			run := func(img model.Span) [][]float32 {
				t.Helper()
				s := testInstance(t, m, len(g.Pre)+img.N(m.Cfg.NEmbd)+len(g.Post)+len(g.Cont)+1, opts...)
				defer s.Close()
				lg, err := s.PrefillMixed(model.Span{Tokens: g.Pre}, img, model.Span{Tokens: g.Post})
				if err != nil {
					t.Fatal(err)
				}
				out := [][]float32{slices.Clone(lg)}
				for _, id := range g.Cont[:len(g.Cont)-1] {
					if lg, err = s.Forward(id); err != nil {
						t.Fatal(err)
					}
					out = append(out, slices.Clone(lg))
				}
				return out
			}
			after, flips := qvlWorst(g, run(model.Span{Embd: g.Image.Embd, Deep: g.Image.Deep, Grid: g.grid()}))
			before, bflips := qvlWorst(g, run(model.Span{Embd: g.Image.Embd, Deep: g.Image.Deep}))
			t.Logf("transformers' (%s) tower rows, grid %v, %d rows teacher-forced: worst logit NMSE %.3e, %d argmax flips; "+
				"at sequential positions %.3e, %d flips", g.Dtype, g.Image.Grid, len(g.Logits), after, flips, before, bflips)
			if math.IsNaN(after) || after > c.textNMSE || flips > c.textFlips {
				t.Errorf("the text model on transformers' tower rows: worst NMSE %.3e, %d flips (bound %.0e, %d)",
					after, flips, c.textNMSE, c.textFlips)
			}
			if c.mrope && !(before > 10*after) {
				t.Errorf("VIOLATION: sequential positions read %.3e against %.3e: the gate cannot see M-RoPE", before, after)
			}
			f, err := os.Open(filepath.Join("testdata", "quad-and-disc.png"))
			if err != nil {
				t.Fatal(err)
			}
			img, err := png.Decode(f)
			f.Close()
			if err != nil {
				t.Fatal(err)
			}
			picture, err := m.PictureSpan(img)
			if err != nil {
				t.Fatal(err)
			}
			if picture.Grid != g.grid() {
				t.Fatalf("the picture's grid is %+v, transformers' %+v", picture.Grid, g.grid())
			}
			worst, wflips := qvlWorst(g, run(picture))
			t.Logf("the whole model from the picture: worst logit NMSE %.3e, %d argmax flips", worst, wflips)
			if math.IsNaN(worst) || worst > 2e-2 || wflips > 1 {
				t.Errorf("the whole model from the picture: worst NMSE %.3e, %d flips (bound 2e-2, 1)", worst, wflips)
			}
			towerRunsOnTheCard(t, m, img, g.Image.Embd, 0, opts...)
		})
	}
}
