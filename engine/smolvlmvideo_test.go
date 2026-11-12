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
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// smolvlmVideoGolden is scripts/smolvlmvideogold.py's reference for a video
// on synth-smolvlm2.
type smolvlmVideoGolden struct {
	PreText    string  `json:"pre_text"`
	PostText   string  `json:"post_text"`
	BOS        int32   `json:"bos"`
	IDs        []int32 `json:"ids"`
	ImageToken int32   `json:"image_token"`
	Cont       []int32 `json:"cont"`
	Video      struct {
		N, H, W    int
		Fps        float64   `json:"fps"`
		RGB        []byte    `json:"rgb"`
		Frames     []int     `json:"frames"`
		Sampled    int       `json:"sampled"`
		Soft       int       `json:"soft"`
		Embd       []float32 `json:"embd"`
		EmbdA8     []float32 `json:"embd_a8"`
		Divergence float64   `json:"divergence"`
	} `json:"video"`
	Logits []e2eLogit `json:"logits"`
}

// TestSmolVLMVideoMatchesTransformers runs scripts/smolvlmvideogold.py's
// video through synth-smolvlm2, on the CPU device and on each card with
// every block of the text model and the tower placed, and holds it to
// transformers' SmolVLMProcessor and SmolVLMForConditionalGeneration: the
// 64 frames sample_frames takes of seventy, a second in from either end;
// the prompt id for id -- the introduction counting them in words, each
// frame's "\nFrame from MM:SS:" and its global image, "\n\n" after the
// last, the text at either end tokenized with the prompt beside it; each
// frame's rows at the engine's arithmetic within the picture gate's bound;
// and the prompt's last row and each decoded row's logits within twice what
// a tower error of the engine's size moves them by. The frames are resized
// as the picture is, the golden's rows over the engine's pixels
// (docs/divergences.md). Each part has its violation: no second skipped,
// mtmd's picture markers, the video's text encoded alone, and the frames'
// rows reversed.
func TestSmolVLMVideoMatchesTransformers(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "smolvlm2", "synth-smolvlm2.video.json"))
	if err != nil {
		t.Fatalf("GOLDEN MISSING: %v -- run scripts/smolvlmvideogold.py", err)
	}
	g := &smolvlmVideoGolden{}
	if err := json.Unmarshal(raw, g); err != nil {
		t.Fatal(err)
	}
	if len(g.IDs) == 0 || len(g.Video.Frames) == 0 || len(g.Logits) == 0 {
		t.Fatal("the golden holds no prompt, sampled frames or logits: it proves nothing")
	}
	text, proj := testmodels.Path("synth-smolvlm2.gguf"), testmodels.Path("mmproj-synth-smolvlm2.gguf")
	if !sourcePresent(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj -- run scripts/smolvlmvideogold.py (RULE 11)", text)
	}
	m, err := model.Open(jlmOfPair(t, text, proj))
	if err != nil {
		t.Fatal(err)
	}
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
	// The violation: a sampler that skips no second, the stated index
	// rewritten as numpy's linspace over every frame.
	var every jlm.ExprBuilder
	k1 := every.Fn(jlm.FnSub, jlm.Operand(jlm.VideoK), jlm.Num(1))
	last := every.Fn(jlm.FnSub, jlm.Operand(jlm.VideoN), jlm.Num(1))
	every.Fn(jlm.FnWhere, every.Fn(jlm.FnLess, jlm.Operand(jlm.VideoAt), k1),
		every.Fn(jlm.FnMul, jlm.Operand(jlm.VideoAt), every.Fn(jlm.FnDiv, last, k1)), last)
	v := m.Tower.Video
	index := v.Index
	v.Index = jlm.Expr{Terms: every.T}
	plain, err := m.Tower.VideoPlan(g.Video.N, g.Video.W, g.Video.H, g.Video.Fps)
	v.Index = index
	if err != nil {
		t.Fatal(err)
	}
	if slices.Equal(plain.Frames, g.Video.Frames) {
		t.Fatal("VIOLATION: a sampler that skips no second takes the processor's frames")
	}
	enc := func(s string) []int32 { return m.Vocab.EncodeSpecial(s, false) }
	sq, err := m.Tower.FrameSquare(g.Video.W, g.Video.H)
	if err != nil {
		t.Fatal(err)
	}
	fake, okF := m.Vocab.ID("<fake_token_around_image>")
	global, okG := m.Vocab.ID("<global-img>")
	if !okF || !okG || sq.MarkN != 2 || sq.TrailN != 1 {
		t.Fatalf("a frame's markers are runs of %d and %d (%v, %v), want the processor's two and one", sq.MarkN, sq.TrailN, okF, okG)
	}
	// The prompt: the video's text at either end with the prompt's beside it.
	prompt := func() []model.Span {
		vp, err := m.VideoParts(frames, g.Video.Fps)
		if err != nil {
			t.Fatal(err)
		}
		head := enc(g.PreText + vp.Lead)
		if g.BOS >= 0 {
			head = append([]int32{g.BOS}, head...)
		}
		return append(append([]model.Span{{Tokens: head}}, vp.Spans...), model.Span{Tokens: enc(vp.Tail + g.PostText)})
	}
	// A frame's span is its markers' rows about its own, the processor's
	// <fake_token_around_image><global-img>, its image tokens and
	// <fake_token_around_image>.
	ids := func(sp []model.Span) []int32 {
		var out []int32
		for _, s := range sp {
			if s.Picture == nil && s.Embd == nil {
				out = append(out, s.Tokens...)
				continue
			}
			out = append(out, fake, global)
			for range s.N(m.Cfg.NEmbd) - 3 {
				out = append(out, g.ImageToken)
			}
			out = append(out, fake)
		}
		return out
	}
	full := prompt()
	var pics []int
	for i, sp := range full {
		if sp.Picture != nil {
			pics = append(pics, i)
			if sp.PicRows != g.Video.Soft+3 || sp.Bidir {
				t.Fatalf("frame %d is %d rows, bidirectional %v; the processor's %d and three markers, causal", len(pics)-1,
					sp.PicRows, sp.Bidir, g.Video.Soft)
			}
		}
	}
	if len(pics) != g.Video.Sampled {
		t.Fatalf("the video holds %d frames, the processor's %d", len(pics), g.Video.Sampled)
	}
	if got := ids(full); !slices.Equal(got, g.IDs) {
		t.Fatalf("the prompt\n  got  %v\n  want %v", got, g.IDs)
	}
	// The violation: each frame read as the picture's global view, whose
	// markers open with the newline after the tiles.
	own := v.Frames
	v.Frames = nil
	gsq, err := m.Tower.FrameSquare(g.Video.W, g.Video.H)
	v.Frames = own
	if err != nil {
		t.Fatal(err)
	}
	if gsq.MarkN == sq.MarkN && gsq.Mark == sq.Mark {
		t.Fatal("VIOLATION: a frame read as the picture's global view takes the processor's markers")
	}
	alone, err := m.VideoSpans(frames, g.Video.Fps)
	if err != nil {
		t.Fatal(err)
	}
	apart := append(append([]model.Span{{Tokens: enc(g.PreText)}}, alone...), model.Span{Tokens: enc(g.PostText)})
	if g.BOS >= 0 {
		apart[0].Tokens = append([]int32{g.BOS}, apart[0].Tokens...)
	}
	if slices.Equal(ids(apart), g.IDs) {
		t.Fatal("VIOLATION: the video's text encoded apart from the prompt's writes the processor's prompt")
	}
	// The chat, as the front ends build it (model.ChatSpansMedia): the
	// fixture's template writes the turn the golden's text is, and the
	// video's text at either end is encoded with the template's.
	vp, err := m.VideoParts(frames, g.Video.Fps)
	if err != nil {
		t.Fatal(err)
	}
	chat, err := m.ChatSpansMedia([]model.ChatMessage{{Role: "user", Content: "Describe the video.", Videos: 1}}, nil,
		[]model.VideoParts{vp}, nil, model.ToolChoice{}, true)
	if err != nil {
		t.Fatal(err)
	}
	if got := ids(chat); !slices.Equal(got, g.IDs) {
		t.Fatalf("the chat's prompt\n  got  %v\n  want %v", got, g.IDs)
	}
	t.Logf("%d ids, the processor's; frames %v; %d frames, %d rows each; the processor's own resize parts the rows by NMSE %.3e",
		len(g.IDs), plan.Frames, len(pics), g.Video.Soft, g.Video.Divergence)

	nb := len(m.Tower.Blocks)
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
		run := func(sp []model.Span) [][]float32 {
			t.Helper()
			s := mk(room)
			defer s.Close()
			lg, err := s.PrefillMixed(sp...)
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
		encd, err := s.encodePictures(slices.Clone(full), 0)
		if err != nil {
			s.Close()
			t.Fatal(err)
		}
		// Each frame's rows between its markers, which are the head's table's.
		d := m.Cfg.NEmbd
		table := markerTable(t, m, d)
		lead, tail := table[(sq.Mark-1)*d:(sq.Mark+1)*d], table[(sq.Trail-1)*d:sq.Trail*d]
		var rows []float32
		for _, at := range pics {
			e := encd[at].Embd
			n := len(e) / d
			if n != g.Video.Soft+3 || !slices.Equal(e[:2*d], lead) || !slices.Equal(e[(n-1)*d:], tail) {
				t.Fatalf("frame span %d is %d rows, its markers not the table's rows %d, %d and %d", at, n, sq.Mark, sq.Mark+1, sq.Trail)
			}
			rows = append(rows, e[2*d:(n-1)*d]...)
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
		given := func(src []float32) []model.Span {
			out := slices.Clone(full)
			for i, at := range pics {
				out[at] = model.Span{Embd: slices.Concat(lead, src[i*per:(i+1)*per], tail)}
			}
			return out
		}
		arith, aflips := worstAgainst(g.Logits, run(given(g.Video.EmbdA8)))
		a8 := arith
		for k := range 8 {
			rng := rand.New(rand.NewPCG(uint64(k), 0x9e3779b97f4a7c15))
			n, f := worstAgainst(g.Logits, run(given(perturbed(rng, g.Video.Embd, e))))
			arith, aflips = math.Max(arith, n), max(aflips, f)
		}
		got, flips := worstAgainst(g.Logits, run(full))
		t.Logf("%d rows: worst NMSE %.3e against transformers, %d argmax flips; a tower error of the engine's size moves them up to %.3e, flipping up to %d (the reference's int8 rows %.3e)",
			len(g.Logits), got, flips, arith, aflips, a8)
		if !(arith > 0) || !(got < 2*arith) || flips > aflips {
			t.Fatalf("worst NMSE %.3e against transformers (bound twice the worst draw's %.3e), %d argmax flips where the draws flip up to %d",
				got, arith, flips, aflips)
		}
		bad, _ := worstAgainst(g.Logits, run(given(rev)))
		if !(bad > 5*2*arith) {
			t.Fatalf("VIOLATION: the frames' rows reversed read %.3e against the bound %.3e", bad, 2*arith)
		}
		t.Logf("violation (the frames' rows reversed): %.3e, %.0fx the bound", bad, bad/(2*arith))
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
