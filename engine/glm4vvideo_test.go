package engine

import (
	"encoding/json"
	"image"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/model"
)

// TestGLM4VVideoMatchesTransformers runs scripts/qwenvlvideogold.py's video
// through GLM-4.1V's fixture, on the CPU device and on each card with every
// block placed, and holds it to transformers' Glm4vProcessor and
// Glm4vForConditionalGeneration: the frames sample_frames takes; the prompt
// id for id -- <|begin_of_video|>, each pair a picture between its markers
// followed by its first frame's whole seconds, <|end_of_video|>; every row's
// position as get_rope_index lays it, each pair a picture of its own; each
// pair's tower rows at the engine's arithmetic; and the prompt's last row
// and each decoded row's logits at the picture gate's bound. Each part has
// its violation: each pair's time written before it, the pairs' rows
// advancing as text, and each pair's frames swapped.
func TestGLM4VVideoMatchesTransformers(t *testing.T) {
	const name = "synth-glm4v"
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "qwenvl", name+".video.json"))
	if err != nil {
		t.Fatalf("GOLDEN MISSING: %v -- run scripts/qwenvlvideogold.py %s", err, name)
	}
	g := &qvlVideoGolden{}
	if err := json.Unmarshal(raw, g); err != nil {
		t.Fatal(err)
	}
	if len(g.IDs) == 0 || len(g.Video.Frames) == 0 {
		t.Fatal("the golden holds no processor prompt or sampled frames: it proves nothing of the plan")
	}
	m, _, openOpts := openQVLTower(t, name)
	defer m.Close()
	sz := g.Video.H * g.Video.W * 3
	frames := make([]image.Image, g.Video.N)
	for i := range frames {
		frames[i] = rgbPicture(g.Video.RGB[i*sz:(i+1)*sz], g.Video.W, g.Video.H)
	}
	plan, err := m.Tower.VideoPlan(len(frames), g.Video.W, g.Video.H, g.Video.Fps)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(plan.Frames, g.Video.Frames) {
		t.Fatalf("the plan samples frames %v, the processor %v", plan.Frames, g.Video.Frames)
	}
	video, err := m.VideoSpans(frames, g.Video.Fps)
	if err != nil {
		t.Fatal(err)
	}
	pairs := g.Video.Grid[0]
	var pics []int
	for i, sp := range video {
		if sp.Picture != nil {
			pics = append(pics, i)
		}
	}
	if len(pics) != pairs {
		t.Fatalf("the video holds %d pairs, the processor's %d", len(pics), pairs)
	}
	pad := m.Vocab.EncodeSpecial("<|image|>", false)
	if len(pad) != 1 {
		t.Fatalf("<|image|> encodes as %v", pad)
	}
	ids := func(vid []model.Span) []int32 {
		out := slices.Clone(g.Pre)
		for _, sp := range vid {
			if sp.Picture == nil && sp.Embd == nil {
				out = append(out, sp.Tokens...)
				continue
			}
			for range sp.N(m.Cfg.NEmbd) {
				out = append(out, pad[0])
			}
		}
		return append(out, g.Post...)
	}
	if got := ids(video); !slices.Equal(got, g.IDs) {
		t.Fatalf("the prompt\n  got  %v\n  want %v", got, g.IDs)
	}
	// The violation: each pair's time before its picture, as Qwen3-VL's
	// processor writes it.
	v := m.Tower.Video
	v.StampLast = false
	before, err := m.VideoSpans(frames, g.Video.Fps)
	v.StampLast = true
	if err != nil {
		t.Fatal(err)
	}
	if slices.Equal(ids(before), g.IDs) {
		t.Fatal("VIOLATION: each pair's time before its picture writes the processor's prompt")
	}
	prompt := func(vid []model.Span) []model.Span {
		return append(append([]model.Span{{Tokens: g.Pre}}, vid...), model.Span{Tokens: g.Post})
	}
	layout := func(vid []model.Span) ([][4]int, int) {
		t.Helper()
		st := testInstance(t, m, 64, openOpts...)
		defer st.Close()
		sp := prompt(vid)
		next, err := st.spanRope(0, 0, sp)
		if err != nil {
			t.Fatal(err)
		}
		return marksLayout(&st.rmarks, 0, 0, spanRows(m, sp)), next
	}
	rows, next := layout(video)
	if len(rows) != len(g.Pos) {
		t.Fatalf("%d prompt rows, transformers lays out %d", len(rows), len(g.Pos))
	}
	for i, r := range rows {
		if [3]int(r[:3]) != g.Pos[i] {
			t.Fatalf("row %d at %v, transformers %v", i, r, g.Pos[i])
		}
	}
	if d := next - len(rows); d != g.Delta {
		t.Fatalf("the decode resumes %d off its cache position, transformers' rope delta is %d", d, g.Delta)
	}
	asText := func(model.ImageGrid, int) model.ImageGrid { return model.ImageGrid{} }
	flat := slices.Clone(video)
	for _, at := range pics {
		flat[at].Grid = model.ImageGrid{}
	}
	if bad, _ := layout(flat); slices.Equal(bad, rows) {
		t.Fatal("VIOLATION: the pairs' rows advancing as text lay out as pictures of their own do")
	}
	t.Logf("%d ids, the processor's; frames %v; %d pairs of %dx%d; the positions get_rope_index's, the decode at %d",
		len(g.IDs), plan.Frames, pairs, g.Video.Grid[1], g.Video.Grid[2], g.Delta)

	arm := func(t *testing.T, dev *gpu.GPU) {
		mk := func(n int) *Instance {
			s := testInstance(t, m, n, openOpts...)
			if dev != nil {
				if err := s.SetDevice(dev); err != nil {
					t.Fatal(err)
				}
				if s.GPULayers() != m.Cfg.NLayer {
					t.Fatalf("the card took %d of %d blocks: %s", s.GPULayers(), m.Cfg.NLayer, dev.Err())
				}
			}
			return s
		}
		room := len(g.Pos) + len(g.Cont) + 1
		run := func(vid []model.Span) [][]float32 {
			t.Helper()
			s := mk(room)
			defer s.Close()
			lg, err := s.PrefillMixed(prompt(vid)...)
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
		worst := func(rows [][]float32) (float64, int) {
			w, flips := 0.0, 0
			for i, lg := range rows {
				n := llama4Cmp(g.Logits[i].Head, lg)
				if math.IsNaN(n) {
					return math.Inf(1), len(rows)
				}
				w = math.Max(w, n)
				if Greedy(lg) != g.Logits[i].Argmax {
					flips++
				}
			}
			return w, flips
		}
		// Each pair's tower rows, the pairs in order.
		encode := func(vid []model.Span) []float32 {
			t.Helper()
			s := mk(64)
			defer s.Close()
			enc, err := s.encodePictures(slices.Clone(vid), 0)
			if err != nil {
				t.Fatal(err)
			}
			var rows []float32
			for _, at := range pics {
				rows = append(rows, enc[at].Embd...)
			}
			return rows
		}
		rows := encode(video)
		if len(rows) != len(g.Video.EmbdA8) {
			t.Fatalf("%d values, transformers' tower %d", len(rows), len(g.Video.EmbdA8))
		}
		nmse, wd := nmseOf(rows, g.Video.EmbdA8)
		e, _ := nmseOf(rows, g.Video.Embd)
		t.Logf("the pairs' %d rows: NMSE %.3e, max|d| %.3g at the engine's arithmetic (%.3e against the f32 tower)",
			len(rows)/m.Cfg.NEmbd, nmse, wd, e)
		if !(nmse < 2e-3) {
			t.Fatalf("the pairs' rows are NMSE %.3e from transformers' (bound 2e-3)", nmse)
		}
		swapped := slices.Clone(video)
		for _, at := range pics {
			swapped[at].Picture, swapped[at].Frame = swapped[at].Frame, swapped[at].Picture
		}
		if nb, _ := nmseOf(encode(swapped), g.Video.EmbdA8); !(nb > 100*math.Max(nmse, 1e-12)) {
			t.Fatalf("VIOLATION: each pair's frames swapped read NMSE %.3e against a clean %.3e", nb, nmse)
		} else {
			t.Logf("violation (each pair's frames swapped): NMSE %.3e", nb)
		}
		// The pairs carrying given rows, each its span's grid as grid makes it.
		given := func(src []float32, grid func(model.ImageGrid, int) model.ImageGrid) []model.Span {
			out := slices.Clone(video)
			n := len(src) / pairs
			for i, at := range pics {
				out[at] = model.Span{Embd: src[i*n : (i+1)*n], Grid: grid(video[at].Grid, i)}
			}
			return out
		}
		same := func(gr model.ImageGrid, _ int) model.ImageGrid { return gr }
		arith, aflips := worst(run(given(g.Video.EmbdA8, same)))
		a8 := arith
		for k := range 8 {
			rng := rand.New(rand.NewPCG(uint64(k), 0x9e3779b97f4a7c15))
			n, f := worst(run(given(perturbed(rng, g.Video.Embd, e), same)))
			arith, aflips = math.Max(arith, n), max(aflips, f)
		}
		clean, flips := worst(run(video))
		t.Logf("%d rows: worst NMSE %.3e against transformers, %d argmax flips; a tower error of the engine's size (%.3e) moves them up to %.3e, flipping up to %d (the reference's int8 draw %.3e)",
			len(g.Logits), clean, flips, e, arith, aflips, a8)
		if !(arith > 0) || !(clean < 2*arith) || flips > aflips {
			t.Fatalf("worst NMSE %.3e against transformers (bound twice the worst draw's %.3e), %d argmax flips where the draws flip up to %d",
				clean, arith, flips, aflips)
		}
		bound := 2 * arith
		bad, _ := worst(run(given(g.Video.EmbdA8, asText)))
		if !(bad > 10*bound) {
			t.Fatalf("VIOLATION: the pairs' rows advancing as text read %.3e against the bound %.3e", bad, bound)
		}
		t.Logf("violation (the pairs' rows advancing as text): %.3e, %.0fx the bound", bad, bad/bound)
	}
	t.Run("cpu", func(t *testing.T) { arm(t, nil) })
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) {
			dev, err := openGPU(t, gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff))
			if err != nil || dev == nil {
				noDevice(t, spec, err)
			}
			defer dev.Close()
			arm(t, dev)
		})
	}
}
