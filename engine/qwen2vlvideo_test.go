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

// qvlVideoGolden is scripts/qwenvlvideogold.py's record of a video through a
// Qwen2-VL family fixture.
type qvlVideoGolden struct {
	Pre  []int32 `json:"pre"`
	Post []int32 `json:"post"`
	Cont []int32 `json:"cont"`
	// IDs is the whole prompt as the processor writes it, where the golden
	// is built through it (GLM-4V's).
	IDs   []int32 `json:"ids"`
	Video struct {
		N    int     `json:"n"`
		H    int     `json:"h"`
		W    int     `json:"w"`
		Fps  float64 `json:"fps"`
		RGB  []byte  `json:"rgb"`
		Grid [3]int  `json:"grid"`
		// Frames is the frames the processor samples, where it samples.
		Frames []int `json:"frames"`
		// Embd is every pair's tower rows in f32, EmbdA8 at the engine's
		// arithmetic (the matrices at Q8_0, their inputs at int8).
		Embd   []float32 `json:"embd"`
		EmbdA8 []float32 `json:"embd_a8"`
	} `json:"video"`
	Pos    [][3]int `json:"pos"`
	Delta  int      `json:"delta"`
	Logits []struct {
		Argmax int32     `json:"argmax"`
		Head   []float64 `json:"head"`
	} `json:"logits"`
}

// TestQwen2VLVideoMatchesTransformers runs scripts/qwenvlvideogold.py's video
// through Qwen2-VL's and Qwen2.5-VL's fixtures, on the CPU device and on each
// card with every block placed, and holds it to transformers': the spans,
// the processor's markers around one grid of the pairs, each pair its frame
// and the temporal axis's step between pairs get_rope_index's; each pair's
// tower rows to get_video_features at the engine's arithmetic (embd_a8); and
// the prompt's last row and each decoded row's logits, at the bound the
// picture's gate takes (TestQwen2VLMatchesTransformers): twice the worst of
// transformers' tower at the engine's arithmetic and eight draws of noise
// the size of the engine tower's error. Qwen2.5-VL's pairs stand eight
// positions apart, so the temporal axis passes the prompt's text and the
// first decoded row takes one past the prompt's highest position. Each part
// has its violation: each pair's frames swapped, the pairs as pictures of
// their own, the pairs at one temporal position, and, where the decode
// jumps, the first decoded token run as the prompt's text.
func TestQwen2VLVideoMatchesTransformers(t *testing.T) {
	for _, name := range []string{"synth-qwen2vl", "synth-qwen25vl"} {
		t.Run(name, func(t *testing.T) { qwen2VLVideoMatchesTransformers(t, name) })
	}
}

func qwen2VLVideoMatchesTransformers(t *testing.T, name string) {
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "qwenvl", name+".video.json"))
	if err != nil {
		t.Fatal(err)
	}
	g := &qvlVideoGolden{}
	if err := json.Unmarshal(raw, g); err != nil {
		t.Fatal(err)
	}
	m, _, openOpts := openQVLTower(t, name)
	defer m.Close()
	sz := g.Video.H * g.Video.W * 3
	frames := make([]image.Image, g.Video.N)
	for i := range frames {
		frames[i] = rgbPicture(g.Video.RGB[i*sz:(i+1)*sz], g.Video.W, g.Video.H)
	}
	video, err := m.VideoSpans(frames, g.Video.Fps)
	if err != nil {
		t.Fatal(err)
	}
	pairs := g.Video.Grid[0]
	if len(video) != pairs+2 || len(video[0].Tokens) != 1 || len(video[len(video)-1].Tokens) != 1 {
		t.Fatalf("the video is %d spans, want its start, %d pairs and its end", len(video), pairs)
	}
	// The temporal axis's step between pairs, as get_rope_index lays them.
	at := len(g.Pre) + 1
	per := g.Video.Grid[1] * g.Video.Grid[2]
	stride := g.Pos[at+per][0] - g.Pos[at][0]
	for i, sp := range video[1 : pairs+1] {
		want := model.ImageGrid{T: pairs, H: g.Video.Grid[1], W: g.Video.Grid[2], Frame: i, Frames: 1, Stride: stride}
		if sp.Frame == nil || sp.Grid != want {
			t.Fatalf("pair %d's span holds a second frame %v and grid %+v, want %+v", i, sp.Frame != nil, sp.Grid, want)
		}
	}
	hi := 0
	for _, p := range g.Pos {
		hi = max(hi, p[0], p[1], p[2])
	}
	jumps := hi > g.Pos[len(g.Pos)-1][0]
	t.Logf("%d pairs of %dx%d, %d positions apart; the prompt's positions reach %d, its last row %d",
		pairs, g.Video.Grid[1], g.Video.Grid[2], stride, hi, g.Pos[len(g.Pos)-1][0])
	prompt := func(vid []model.Span, more ...model.Span) []model.Span {
		out := append([]model.Span{{Tokens: g.Pre}}, vid...)
		return append(append(out, model.Span{Tokens: g.Post}), more...)
	}
	// The prompt's rows on the rotary axes, row for row get_rope_index's,
	// and the decode resuming at its rope delta; the pairs as pictures of
	// their own lay out otherwise.
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
	apart := slices.Clone(video)
	for i := 1; i <= pairs; i++ {
		apart[i].Grid.T, apart[i].Grid.Frame, apart[i].Grid.Frames = 1, 0, 0
	}
	if bad, _ := layout(apart); slices.Equal(bad, rows) {
		t.Fatal("VIOLATION: the pairs as pictures of their own lay out as one grid does")
	}
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
			for _, sp := range enc[1 : pairs+1] {
				rows = append(rows, sp.Embd...)
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
			len(rows)/int(m.Container.Vision().ProjDim), nmse, wd, e)
		if !(nmse < 2e-3) {
			t.Fatalf("the pairs' rows are NMSE %.3e from transformers' (bound 2e-3)", nmse)
		}
		swapped := slices.Clone(video)
		for i := 1; i <= pairs; i++ {
			swapped[i].Picture, swapped[i].Frame = swapped[i].Frame, swapped[i].Picture
		}
		if nb, _ := nmseOf(encode(swapped), g.Video.EmbdA8); !(nb > 100*math.Max(nmse, 1e-12)) {
			t.Fatalf("VIOLATION: each pair's frames swapped read NMSE %.3e against a clean %.3e", nb, nmse)
		} else {
			t.Logf("violation (each pair's frames swapped): NMSE %.3e", nb)
		}
		// The pairs carrying given rows, each its span's grid unless grid
		// says otherwise.
		given := func(src []float32, grid func(model.ImageGrid) model.ImageGrid) []model.Span {
			out := slices.Clone(video)
			n := len(src) / pairs
			for i := 1; i <= pairs; i++ {
				out[i] = model.Span{Embd: src[(i-1)*n : i*n], Grid: grid(video[i].Grid)}
			}
			return out
		}
		same := func(gr model.ImageGrid) model.ImageGrid { return gr }
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
		for _, v := range []struct {
			what string
			grid func(model.ImageGrid) model.ImageGrid
		}{
			{"the pairs as pictures of their own", func(gr model.ImageGrid) model.ImageGrid {
				gr.T, gr.Frame, gr.Frames = 1, 0, 0
				return gr
			}},
			{"the pairs at one temporal position", func(gr model.ImageGrid) model.ImageGrid {
				gr.Stride = 1 - min(gr.Stride, 1)
				return gr
			}},
		} {
			bad, _ := worst(run(given(g.Video.EmbdA8, v.grid)))
			if !(bad > 10*bound) {
				t.Fatalf("VIOLATION: %s reads %.3e against the bound %.3e", v.what, bad, bound)
			}
			t.Logf("violation (%s): %.3e, %.0fx the bound", v.what, bad, bad/bound)
		}
		if !jumps {
			t.Logf("the text after the video passes its temporal axis: the decode continues the text")
			return
		}
		s := mk(room)
		defer s.Close()
		lg, err := s.PrefillMixed(prompt(given(g.Video.EmbdA8, same), model.Span{Tokens: g.Cont[:1]})...)
		if err != nil {
			t.Fatal(err)
		}
		if bad := llama4Cmp(g.Logits[1].Head, lg); !(bad > 10*bound) {
			t.Fatalf("VIOLATION: the first decoded token as the prompt's text reads %.3e against the bound %.3e", bad, bound)
		} else {
			t.Logf("violation (the first decoded token as the prompt's text): %.3e, %.0fx the bound", bad, bad/bound)
		}
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
