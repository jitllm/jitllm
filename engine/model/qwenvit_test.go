package model

import (
	"image"
	"image/color"
	"math"

	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// goldenImage is a golden's RGB bytes as an image.
func goldenImage(w, h int, rgb []byte) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			o := (y*w + x) * 3
			img.Set(x, y, color.RGBA{rgb[o], rgb[o+1], rgb[o+2], 255})
		}
	}
	return img
}

// nmse32 is the NMSE of got against want and the largest absolute difference.
func nmse32(want, got []float32) (float64, float64) {
	var num, den, worst float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		num, den = num+d*d, den+float64(want[i])*float64(want[i])
		worst = math.Max(worst, math.Abs(d))
	}
	return num / den, worst
}

// qvlEncode preprocesses and encodes the golden's picture on a fresh state of
// tw, after mut (nil for none) has had its say over the state.
func qvlEncode(t *testing.T, tw *Tower, g *qvlGolden, mut func(*State)) ([]float32, ImageGrid) {
	t.Helper()
	ts := tw.testState()
	defer ts.Close()
	px, err := ts.PreprocessImage(goldenImage(g.Image.W, g.Image.H, g.Image.RGB))
	if err != nil {
		t.Fatal(err)
	}
	if mut != nil {
		mut(ts)
	}
	out, err := ts.Encode(px)
	if err != nil {
		t.Fatal(err)
	}
	return append([]float32(nil), out...), ts.Grid()
}

// towerBound is how far the tower's output may sit from transformers'. The
// tower's matrices are Q8_0 (convert.quantizeTower) and its matmuls'
// activations int8, which moves transformers' own tower by 5-9e-4 on these
// fixtures (scripts/qwenvlgold.py's embd_a8); the violations below land at
// 2e-1 and up.
const towerBound = 2e-3

// TestQwenTowerMatchesTransformers holds each Qwen-VL tower's output, row for
// row in the order the language model reads them, to transformers' own
// get_image_features on the same picture -- and, for Qwen2.5-VL, each of its
// tower's features taken out of the engine's copy must move it far past the
// bound: the 2-D rotary, the window attention, the window partition and the
// merger's put-back into raster order.
func TestQwenTowerMatchesTransformers(t *testing.T) {
	for _, name := range towerFixtures() {
		t.Run(name, func(t *testing.T) {
			m, g := openQVL(t, name)
			defer m.Close()
			tw := m.Tower()
			if tw == nil {
				t.Fatal("the fixture's container carries no tower")
			}
			got, grid := qvlEncode(t, tw, g, nil)
			if grid != g.grid() {
				t.Fatalf("the tower's grid is %v, transformers' %v", grid, g.grid())
			}
			clean, worst := nmse32(g.Image.EmbdA8, got)
			f32, _ := nmse32(g.Image.Embd, got)
			q8, _ := nmse32(g.Image.Embd, g.Image.EmbdA8)
			t.Logf("%d rows of %d: NMSE %.3e, max|d| %.4f against transformers at the engine's int8 "+
				"(%.3e against its f32 tower, where the int8 alone moves transformers by %.3e)",
				grid.Rows(), tw.Cfg.ProjDim, clean, worst, f32, q8)
			if math.IsNaN(clean) || clean > towerBound {
				t.Fatalf("the tower's output is not transformers': NMSE %.3e (bound %.0e)", clean, towerBound)
			}
			if len(tw.Cfg.Deep) > 0 {
				qwen3TowerGate(t, tw, g, got, clean)
				return
			}
			if tw.glm != nil {
				glmTowerGate(t, tw, g, clean)
				return
			}
			if tw.Cfg.Kind == jlm.ProjKimiVL {
				kimiTowerGate(t, tw, g, clean)
				return
			}
			if tw.Cfg.Kind == jlm.ProjHunyuanVL {
				hyTowerGate(t, tw, g, clean)
				return
			}
			if !tw.Cfg.RMS {
				return
			}
			for _, v := range []struct {
				name string
				mut  func(*State)
				cfg  func(c *TowerConfig) func()
			}{
				{"no 2-D rotary", func(s *State) {
					gh, gw := s.patchGrid()
					s.vis.ropeSC = make([]float32, gh*gw*tw.Cfg.HeadDim)
					for i := 0; i < len(s.vis.ropeSC); i += 2 {
						s.vis.ropeSC[i] = 1
					}
					s.vis.ropeAt = [2]int{gh, gw}
				}, nil},
				{"full attention where windowed", nil, func(c *TowerConfig) func() {
					was := c.WinPattern
					c.WinPattern = 1
					return func() { c.WinPattern = was }
				}},
				{"the wrong window partition (half the window)", nil, func(c *TowerConfig) func() {
					was := c.WinSize
					c.WinSize /= 2
					return func() { c.WinSize = was }
				}},
				{"the merged rows left in window order", nil, func(c *TowerConfig) func() {
					c.fault = faultQwenNoUnwindow
					return func() { c.fault = towerFaultNone }
				}},
			} {
				undo := func() {}
				if v.cfg != nil {
					undo = v.cfg(&tw.Cfg)
				}
				bad, _ := qvlEncode(t, tw, g, v.mut)
				undo()
				nb, _ := nmse32(g.Image.EmbdA8, bad)
				t.Logf("violation %q: NMSE %.3e", v.name, nb)
				if !(nb > 10*towerBound) {
					t.Errorf("%s reads %.3e against a clean %.3e: the gate cannot see it", v.name, nb, clean)
				}
			}
		})
	}
}

// TestQwenVLMatchesTransformersEndToEnd is the whole model on the golden's
// picture: the engine's own preprocessing and tower, the merged rows spliced
// at their M-RoPE grid, and the continuation decoded, against transformers'
// Qwen2VLForConditionalGeneration / Qwen2_5_VLForConditionalGeneration from
// the pixels on. The tower's int8 (TestQwenTowerMatchesTransformers) is the
// only arithmetic the two do not share, and these toy text models amplify its
// 5e-4 to 3-8e-3 of the logits, so that is the bound, with every argmax held.
func TestQwenVLMatchesTransformersEndToEnd(t *testing.T) {
	for _, name := range towerFixtures() {
		t.Run(name, func(t *testing.T) {
			m, g := openQVL(t, name)
			defer m.Close()
			emb, grid := qvlEncode(t, m.Tower(), g, nil)
			n := len(g.Pre) + grid.Rows() + len(g.Post) + len(g.Cont) + 1
			run := func(img Span) [][]float32 {
				st := m.NewState(n)
				defer st.Close()
				lg, err := st.PrefillMixed(Span{Tokens: g.Pre}, img, Span{Tokens: g.Post})
				if err != nil {
					t.Fatal(err)
				}
				out := [][]float32{append([]float32(nil), lg...)}
				for _, id := range g.Cont[:len(g.Cont)-1] {
					if lg, err = st.Forward(id); err != nil {
						t.Fatal(err)
					}
					out = append(out, append([]float32(nil), lg...))
				}
				return out
			}
			embd, deep := splitDeep(emb, grid.Rows(), m.Tower().Cfg.ProjDim)
			rows := run(Span{Embd: embd, Grid: grid, Deep: deep})
			w, _ := qvlWorst(g, rows)
			// A flip at a tie is the tower's int8, not a fault: synth-qwen35vl's
			// 248k-wide head has two logits 0.04 apart at one row.
			flips := qvlFlipsPastTie(g, rows, 0.1)
			t.Logf("the engine's tower and text model: worst logit NMSE %.3e against transformers, %d argmax flips", w, flips)
			if math.IsNaN(w) || w > 2e-2 || flips > 0 {
				t.Fatalf("end to end: worst NMSE %.3e, %d argmax flips against transformers", w, flips)
			}
			// The violation: the same picture laid out at sequential positions,
			// which is where a single-axis text model (Kimi-VL's) lays it.
			if len(m.rope.Runs) == 0 {
				return
			}
			bad, _ := qvlWorst(g, run(Span{Embd: embd, Deep: deep}))
			t.Logf("sequential image positions (the violation): %.3e", bad)
			if !(bad > 20*w) {
				t.Fatalf("sequential positions read %.3e against %.3e: the gate cannot see M-RoPE", bad, w)
			}
			// And for a deepstack tower, the taps dropped: the text model's
			// first blocks with nothing added to the image's rows.
			if deep != nil {
				nd, _ := qvlWorst(g, run(Span{Embd: embd, Grid: grid}))
				t.Logf("no deepstack rows (the violation): %.3e", nd)
				if !(nd > 20*w) {
					t.Fatalf("dropping the deepstack rows reads %.3e against %.3e: the gate cannot see them", nd, w)
				}
			}
		})
	}
}

// qvlFlipsPastTie counts the rows whose argmax is not transformers' by more
// than tie: the engine's own logit at transformers' choice is that far below
// its maximum.
func qvlFlipsPastTie(g *qvlGolden, rows [][]float32, tie float32) int {
	n := 0
	for i, lg := range rows {
		if a := Greedy(lg); a != g.Logits[i].Argmax && lg[a]-lg[g.Logits[i].Argmax] > tie {
			n++
		}
	}
	return n
}

// TestQwenPreprocessMatchesTheProcessor holds Preprocess to transformers'
// Qwen2VLImageProcessor on a picture it must resize: the grid smart_resize
// picks, and the pixels PIL's BICUBIC and the normalisation produce. The
// resize is Pillow's own fixed point (rgb8.resizePIL on generated code), 8 bits
// after each pass as Pillow rounds, so every sample is the processor's; what is
// left is the normalisation's float rounding -- the table's (v - 255m)/(255s)
// against the processor's v/255 then (x - m)/s -- an ulp, which the bound is.
// A float resize read NMSE 7.4e-06, max|d| 0.0143.
func TestQwenPreprocessMatchesTheProcessor(t *testing.T) {
	// Qwen2-VL's processor at patch 14 and CLIP's mean and std, and Qwen3-VL's
	// at patch 16 and 0.5 (Qwen2.5-VL's is Qwen2-VL's).
	for _, name := range []string{"synth-qwen2vl", "synth-qwen3vl", "synth-glm4v", "synth-kimivl", "synth-hunyuanvl"} {
		t.Run(name, func(t *testing.T) { qwenPreprocessMatchesTheProcessor(t, name) })
	}
}

func qwenPreprocessMatchesTheProcessor(t *testing.T, name string) {
	m, g := openQVL(t, name)
	defer m.Close()
	tw := m.Tower()
	p := g.Prep
	// The processor ran under the golden's pixel ceiling, which is under the
	// tower's own: the reduction is the point. Its floor was Qwen2-VL's.
	defer func(was, floor int) { tw.Cfg.MaxPixels, tw.Cfg.MinPixels = was, floor }(tw.Cfg.MaxPixels, tw.Cfg.MinPixels)
	tw.Cfg.MaxPixels, tw.Cfg.MinPixels = p.Max, qwenMinPixels
	if p.Min > 0 {
		tw.Cfg.MinPixels = p.Min
	}
	// Kimi-VL's processor ran under a lowered patch limit, past which it
	// resizes, and the tower holds what that limit pads to.
	if tw.Cfg.Kind == jlm.ProjKimiVL {
		defer func(was int) { tw.Cfg.TokenLimit = was }(tw.Cfg.TokenLimit)
		tw.Cfg.MaxPixels = kimiCapacity(p.Max) * tw.Cfg.PatchSz * tw.Cfg.PatchSz
		tw.Cfg.TokenLimit = p.Max
	}
	ts := tw.testState()
	defer ts.Close()
	got, err := ts.PreprocessImage(goldenImage(p.W, p.H, p.RGB))
	if err != nil {
		t.Fatal(err)
	}
	gh, gw := ts.patchGrid()
	if gh*tw.Cfg.PatchSz != p.OutH || gw*tw.Cfg.PatchSz != p.OutW {
		t.Fatalf("a %dx%d picture resized to %dx%d, the processor's smart_resize says %dx%d",
			p.W, p.H, gw*tw.Cfg.PatchSz, gh*tw.Cfg.PatchSz, p.OutW, p.OutH)
	}
	nm, worst := nmse32(p.PX, got)
	t.Logf("%dx%d -> %dx%d: NMSE %.3e, max|d| %.4f against the processor", p.W, p.H, p.OutW, p.OutH, nm, worst)
	if math.IsNaN(nm) || nm > 1e-10 || worst > 1e-5 {
		t.Fatalf("the preprocessed picture is not the processor's: NMSE %.3e, max|d| %.4f", nm, worst)
	}
	// The violation: the bilinear resize the other towers take.
	bl, err := ts.preprocessWith(goldenImage(p.W, p.H, p.RGB), bilinearTaps)
	if err != nil {
		t.Fatal(err)
	}
	bad, _ := nmse32(p.PX, bl)
	if !(bad > 10*nm) {
		t.Fatalf("a bilinear resize reads %.3e against bicubic's %.3e: the gate cannot see the filter", bad, nm)
	}
	t.Logf("bilinear (the violation): NMSE %.3e", bad)
}
