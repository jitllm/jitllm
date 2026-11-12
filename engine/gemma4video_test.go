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
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/model"
)

// gemma4VideoGolden is scripts/gemma4videogold.py's reference for a video
// on synth-gemma4v.
type gemma4VideoGolden struct {
	Pre        []int32 `json:"pre"`
	Post       []int32 `json:"post"`
	Cont       []int32 `json:"cont"`
	IDs        []int32 `json:"ids"`
	VideoToken int32   `json:"video_token"`
	Video      struct {
		N, H, W int
		Fps     float64   `json:"fps"`
		RGB     []byte    `json:"rgb"`
		Frames  []int     `json:"frames"`
		Sampled int       `json:"sampled"`
		Grid    [2]int    `json:"grid"`
		Soft    int       `json:"soft"`
		Embd    []float32 `json:"embd"`
		EmbdA8  []float32 `json:"embd_a8"`
	} `json:"video"`
	Logits       []e2eLogit `json:"logits"`
	LogitsCausal []e2eLogit `json:"logits_causal"`
}

// TestGemma4VideoMatchesTransformers runs scripts/gemma4videogold.py's
// video through synth-gemma4v, on the CPU device and on each card with every
// block of the text model and the tower placed, and holds it to
// transformers' Gemma4Processor and Gemma4ForConditionalGeneration: the 32
// frames sample_frames takes of forty; each frame's grid at its aspect
// within 70 soft tokens; the prompt id for id -- each frame's MM:SS, a space
// and its span, the frames a space apart; each frame's rows at the engine's
// arithmetic within the picture gate's bound; and the prompt's last row and
// each decoded row's logits within twice what a tower error of the engine's
// size moves them by, each frame's rows bidirectional on the sliding
// layers. Each part has its violation: the times in seconds, the frames'
// rows reversed, and the frames' rows causal -- which must read as
// transformers' own causal run does.
func TestGemma4VideoMatchesTransformers(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "gemma4v", "synth-gemma4v.video.json"))
	if err != nil {
		t.Fatalf("GOLDEN MISSING: %v -- run scripts/gemma4videogold.py", err)
	}
	g := &gemma4VideoGolden{}
	if err := json.Unmarshal(raw, g); err != nil {
		t.Fatal(err)
	}
	if len(g.IDs) == 0 || len(g.Video.Frames) == 0 || len(g.Logits) == 0 || len(g.LogitsCausal) != len(g.Logits) {
		t.Fatal("the golden holds no prompt, sampled frames or logits: it proves nothing")
	}
	m, _ := openGemma4V(t)
	defer m.Close()
	sz := g.Video.H * g.Video.W * 3
	frames := make([]image.Image, g.Video.N)
	for i := range frames {
		frames[i] = rgbPicture(g.Video.RGB[i*sz:(i+1)*sz], g.Video.W, g.Video.H)
	}
	plan, err := m.Tower.VideoPlan(g.Video.N, g.Video.W, g.Video.H, g.Video.Fps)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(plan.Frames, g.Video.Frames) {
		t.Fatalf("the plan samples frames %v, the processor %v", plan.Frames, g.Video.Frames)
	}
	if sq := m.Tower.FrameSquare(g.Video.W, g.Video.H); [2]int{sq.Gh, sq.Gw} != g.Video.Grid {
		t.Fatalf("a %dx%d frame is %dx%d patches, the processor's %v", g.Video.W, g.Video.H, sq.Gh, sq.Gw, g.Video.Grid)
	}
	video, err := m.VideoSpans(frames, g.Video.Fps)
	if err != nil {
		t.Fatal(err)
	}
	var pics []int
	for i, sp := range video {
		if sp.Picture != nil {
			pics = append(pics, i)
			if sp.PicRows != g.Video.Soft || !sp.Bidir {
				t.Fatalf("frame %d is %d rows, bidirectional %v; the processor's %d soft tokens, seen both ways", len(pics)-1, sp.PicRows, sp.Bidir, g.Video.Soft)
			}
		}
	}
	if len(pics) != g.Video.Sampled {
		t.Fatalf("the video holds %d frames, the processor's %d", len(pics), g.Video.Sampled)
	}
	ids := func(vid []model.Span) []int32 {
		out := slices.Clone(g.Pre)
		for _, sp := range vid {
			if sp.Picture == nil && sp.Embd == nil {
				out = append(out, sp.Tokens...)
				continue
			}
			for range sp.N(m.Cfg.NEmbd) {
				out = append(out, g.VideoToken)
			}
		}
		return append(out, g.Post...)
	}
	if got := ids(video); !slices.Equal(got, g.IDs) {
		t.Fatalf("the prompt\n  got  %v\n  want %v", got, g.IDs)
	}
	// The violation: the times written as seconds, the stated prompt's
	// clock rewritten as a decimal.
	clock := &videoGroups(t, m.Tower.Video).Body[0].Format
	was := *clock
	*clock = jlm.ValueFormat{Kind: jlm.FormatDecimal, Digits: 1}
	seconds, err := m.VideoSpans(frames, g.Video.Fps)
	*clock = was
	if err != nil {
		t.Fatal(err)
	}
	if slices.Equal(ids(seconds), g.IDs) {
		t.Fatal("VIOLATION: the times in seconds write the processor's prompt")
	}
	// The processor writes no BOS; the reference runs the prompt after one.
	prompt := func(vid []model.Span) []model.Span {
		return append(append([]model.Span{{Tokens: append([]int32{2}, g.Pre...)}}, vid...), model.Span{Tokens: g.Post})
	}
	t.Logf("%d ids, the processor's; frames %v; %d frames of %dx%d patches, %d rows each", len(g.IDs), plan.Frames,
		len(pics), g.Video.Grid[0], g.Video.Grid[1], g.Video.Soft)

	nb := 0
	if m.Tower != nil {
		nb = len(m.Tower.Blocks)
	}
	arm := func(t *testing.T, dev *gpu.GPU) {
		mk := func(n int) *Instance {
			s := testInstance(t, m, n, WithKVF16(false))
			if dev != nil {
				s.SetDeviceFallback(false)
				if err := s.SetDevice(dev); err != nil {
					t.Fatal(err)
				}
				if s.GPULayers() != m.Cfg.NLayer {
					t.Fatalf("the card took %d of %d blocks: %s", s.GPULayers(), m.Cfg.NLayer, dev.Err())
				}
				if err := s.tower.s.SetDeviceLayers(dev, nb); err != nil {
					t.Fatal(err)
				}
				if p := s.tower.s.GPULayers(); p != nb {
					t.Fatalf("the card took %d of the tower's %d blocks: %s", p, nb, dev.Err())
				}
			}
			return s
		}
		room := len(g.IDs) + len(g.Cont) + 2
		run := func(vid []model.Span) [][]float32 {
			t.Helper()
			s := mk(room)
			defer s.Close()
			lg, err := s.PrefillMixed(prompt(vid)...)
			if err != nil {
				t.Fatal(err)
			}
			out := [][]float32{slices.Clone(lg)}
			for _, id := range g.Cont[:len(g.Logits)-1] {
				if lg, err = s.Forward(id); err != nil {
					t.Fatal(err)
				}
				out = append(out, slices.Clone(lg))
			}
			return out
		}
		s := mk(64)
		enc, err := s.encodePictures(slices.Clone(video), 0)
		if err != nil {
			s.Close()
			t.Fatal(err)
		}
		var rows []float32
		for _, at := range pics {
			rows = append(rows, enc[at].Embd...)
		}
		s.Close()
		if len(rows) != len(g.Video.EmbdA8) {
			t.Fatalf("%d values, transformers' frames %d", len(rows), len(g.Video.EmbdA8))
		}
		clean, wd := nmseOf(rows, g.Video.EmbdA8)
		e, _ := nmseOf(rows, g.Video.Embd)
		t.Logf("the frames' %d rows: NMSE %.3e, max|d| %.3g at the engine's arithmetic (%.3e against the f32 tower)",
			len(rows)/m.Cfg.NEmbd, clean, wd, e)
		if !(clean < 2e-3) {
			t.Fatalf("the frames' rows are NMSE %.3e from transformers' (bound 2e-3)", clean)
		}
		per := len(rows) / len(pics)
		rev := make([]float32, 0, len(rows))
		for i := len(pics) - 1; i >= 0; i-- {
			rev = append(rev, rows[i*per:(i+1)*per]...)
		}
		if nb, _ := nmseOf(rev, g.Video.EmbdA8); !(nb > 100*math.Max(clean, 1e-12)) {
			t.Fatalf("VIOLATION: the frames' rows reversed read NMSE %.3e against a clean %.3e", nb, clean)
		} else {
			t.Logf("violation (the frames' rows reversed): NMSE %.3e", nb)
		}
		// The frames carrying given rows, each bidirectional or not.
		given := func(src []float32, bidir bool) []model.Span {
			out := slices.Clone(video)
			for i, at := range pics {
				out[at] = model.Span{Embd: src[i*per : (i+1)*per], Bidir: bidir}
			}
			return out
		}
		arith, aflips := worstAgainst(g.Logits, run(given(g.Video.EmbdA8, true)))
		a8 := arith
		for k := range 8 {
			rng := rand.New(rand.NewPCG(uint64(k), 0x9e3779b97f4a7c15))
			n, f := worstAgainst(g.Logits, run(given(perturbed(rng, g.Video.Embd, e), true)))
			arith, aflips = math.Max(arith, n), max(aflips, f)
		}
		got, flips := worstAgainst(g.Logits, run(video))
		t.Logf("%d rows: worst NMSE %.3e against transformers, %d argmax flips; a tower error of the engine's size moves them up to %.3e, flipping up to %d (the reference's int8 rows %.3e)",
			len(g.Logits), got, flips, arith, aflips, a8)
		if !(arith > 0) || !(got < 2*arith) || flips > aflips {
			t.Fatalf("worst NMSE %.3e against transformers (bound twice the worst draw's %.3e), %d argmax flips where the draws flip up to %d",
				got, arith, flips, aflips)
		}
		causal := run(given(g.Video.EmbdA8, false))
		ref, _ := worstAgainst(g.LogitsCausal, causal)
		bad, _ := worstAgainst(g.Logits, causal)
		t.Logf("violation (the frames' rows causal): %.3e against transformers' bidirectional run, %.3e against its causal one", bad, ref)
		// A mask on the sliding layers moves the logits less than the rows
		// reversed do, so the violation is held both ways: past the bound
		// from transformers' run, and within it of transformers' own causal
		// run.
		if !(bad > 2*arith) || !(ref < 2*arith) {
			t.Fatalf("VIOLATION: the frames' rows causal read %.3e against the bidirectional run (bound %.3e) and %.3e against transformers' causal run",
				bad, 2*arith, ref)
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
