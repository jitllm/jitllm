package engine

import (
	"encoding/json"
	"image"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// internvlVideoGolden is scripts/internvlvideogold.py's video.json.
type internvlVideoGolden struct {
	N, H, W    int
	RGB        []byte  `json:"rgb"`
	IDs        []int32 `json:"ids"`
	Pre        []int32 `json:"pre"`
	Post       []int32 `json:"post"`
	ImageToken int32   `json:"image_token"`
}

// TestInternVLVideoMatchesTransformers runs scripts/internvlvideogold.py's
// video through InternVL3-1B, on the CPU device and on each card with every
// tower block placed, and holds it to transformers' InternVLProcessor and
// InternVLModel: every frame read; the prompt id for id -- "Frame{n}: " and
// each frame's span, a line between one frame and the next; each frame's
// rows its picture's last square's, bit for bit -- the thumbnail, which
// transformers' video processor reads a frame as to the bit -- and within
// the tower's error of transformers'. Each part has its violation: no line
// between the frames, and each frame's first square in place of its last.
func TestInternVLVideoMatchesTransformers(t *testing.T) {
	text, mm := testmodels.Path("vlm/InternVL3-1B-Instruct-Q8_0.gguf"), testmodels.Path("vlm/mmproj-InternVL3-1B-Instruct-Q8_0.gguf")
	if !sourcePresent(text, mm) {
		testmodels.Missing(t, "MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", text, mm)
	}
	dir := testmodels.Path(filepath.Join("vlm", "goldens", "internvl"))
	raw, err := os.ReadFile(filepath.Join(dir, "video.json"))
	if err != nil {
		testmodels.Missing(t, "GOLDEN MISSING: %v (scripts/internvlvideogold.py writes it)", err)
	}
	g := &internvlVideoGolden{}
	if err := json.Unmarshal(raw, g); err != nil {
		t.Fatal(err)
	}
	if len(g.IDs) == 0 || g.N < 2 || len(g.RGB) != g.N*g.H*g.W*3 {
		t.Fatalf("the golden holds %d ids and %d bytes for %d frames of %dx%d: it proves nothing", len(g.IDs), len(g.RGB), g.N, g.W, g.H)
	}
	want := readF32(t, filepath.Join(dir, "video.features.f32"))
	if d, _ := nmseOf(want, readF32(t, filepath.Join(dir, "video-thumbs.features.f32"))); d != 0 {
		t.Fatalf("transformers' frames read NMSE %.3e from their pictures' thumbnails: a frame is no longer its picture's last square", d)
	}
	m, err := model.Open(jlmOfPair(t, text, mm))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	sz := g.H * g.W * 3
	frames := make([]image.Image, g.N)
	every := make([]int, g.N)
	for i := range frames {
		frames[i], every[i] = rgbPicture(g.RGB[i*sz:(i+1)*sz], g.W, g.H), i
	}
	plan, err := m.Tower.VideoPlan(g.N, g.W, g.H, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(plan.Frames, every) {
		t.Fatalf("the plan reads frames %v, the processor every one", plan.Frames)
	}
	video, err := m.VideoSpans(frames, 0)
	if err != nil {
		t.Fatal(err)
	}
	var pics []int
	for i, sp := range video {
		if sp.Picture != nil {
			pics = append(pics, i)
		}
	}
	if len(pics) != g.N {
		t.Fatalf("the video holds %d pictures, the processor's %d frames", len(pics), g.N)
	}
	ids := func(vid []model.Span) []int32 {
		out := slices.Clone(g.Pre)
		for _, sp := range vid {
			if sp.Picture == nil {
				out = append(out, sp.Tokens...)
				continue
			}
			for range sp.PicRows {
				out = append(out, g.ImageToken)
			}
		}
		return append(out, g.Post...)
	}
	if got := ids(video); !slices.Equal(got, g.IDs) {
		t.Fatalf("the prompt\n  got  %v\n  want %v", got, g.IDs)
	}
	grp := videoGroups(t, m.Tower.Video)
	between := grp.Between
	grp.Between = nil
	bare, err := m.VideoSpans(frames, 0)
	grp.Between = between
	if err != nil {
		t.Fatal(err)
	}
	if slices.Equal(ids(bare), g.IDs) {
		t.Fatal("VIOLATION: no line between the frames writes the processor's prompt")
	}

	D := int(m.Container.Vision().ProjDim)
	frame, err := m.Tower.LastSquare(g.W, g.H)
	if err != nil {
		t.Fatal(err)
	}
	n := m.Tower.SquareOut(frame) * D
	encode := func(s *Instance) []float32 {
		t.Helper()
		enc, err := s.encodePictures(slices.Clone(video), 0)
		if err != nil {
			t.Fatal(err)
		}
		var rows []float32
		for _, at := range pics {
			rows = append(rows, enc[at].Embd...)
		}
		return rows
	}
	host := testInstance(t, m, 64)
	got := encode(host)
	if len(got) != len(want) || len(got) != g.N*n {
		t.Fatalf("%d values, transformers' frames %d, %d frames of %d", len(got), len(want), g.N, n)
	}
	// Each frame as a picture: its last square is the frame read whole, its
	// first the violation.
	var last, first []float32
	for _, f := range frames {
		sp, err := m.PictureSpan(f)
		if err != nil {
			t.Fatal(err)
		}
		enc, err := host.encodePictures([]model.Span{sp}, 0)
		if err != nil {
			t.Fatal(err)
		}
		rows := enc[0].Embd
		if len(rows) < 2*n {
			t.Fatalf("a frame of %dx%d is %d rows as a picture: one square, so its first square is its last", g.W, g.H, len(rows)/D)
		}
		last, first = append(last, rows[len(rows)-n:]...), append(first, rows[:n]...)
	}
	host.Close()
	if !slices.Equal(got, last) {
		d, _ := nmseOf(got, last)
		t.Fatalf("the frames' rows are NMSE %.3e from their pictures' last squares: not the same rows", d)
	}
	cpu, worst := nmseOf(got, want)
	t.Logf("%d ids, the processor's; %d frames of %d rows, each its picture's last square to the bit: NMSE %.3e, max|d| %.3g from transformers'",
		len(g.IDs), g.N, n/D, cpu, worst)
	// The Q8_0 tower's distance from transformers' f32 one: the quadrant
	// picture's rows on the CPU device sit at 6.2e-3
	// (TestCheckpointTowersRunOnTheCard).
	const bound = 2e-2
	if !(cpu < bound) {
		t.Fatalf("the frames' rows are NMSE %.3e from transformers' (bound %.0e)", cpu, bound)
	}
	if bad, _ := nmseOf(first, want); !(bad > 10*bound) {
		t.Fatalf("VIOLATION: each frame's first square in place of its last reads NMSE %.3e, within ten times the bound %.0e", bad, bound)
	} else {
		t.Logf("violation (each frame's first square): NMSE %.3e", bad)
	}
	nb := len(m.Tower.Blocks)
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			gd, err := openGPU(t, gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff))
			if err != nil || gd == nil {
				noDevice(t, dev, err)
			}
			defer gd.Close()
			s := testInstance(t, m, 64)
			defer s.Close()
			s.SetDeviceFallback(false)
			if err := s.tower.s.SetDeviceLayers(gd, nb); err != nil {
				t.Fatal(err)
			}
			if p := s.tower.s.GPULayers(); p != nb {
				t.Fatalf("%d of the tower's %d blocks placed on %s: %s", p, nb, dev, gd.Err())
			}
			rows := encode(s)
			if st := gd.Stats(); st.UnformedAtStep != 0 {
				t.Fatalf("%d of the tower's parts formed at a step on %s, not at the placement", st.UnformedAtStep, dev)
			}
			e, _ := nmseOf(rows, want)
			t.Logf("%s: every tower block on the card: NMSE %.3e from transformers' (the CPU device's %.3e)", dev, e, cpu)
			if math.IsNaN(e) || e > 2*cpu {
				t.Fatalf("%s: the frames' rows are NMSE %.3e from transformers', past twice the CPU device's %.3e", dev, e, cpu)
			}
		})
	}
}

// videoGroups is the piece of v's prompt its groups are written by, which
// a violation rewrites.
func videoGroups(t *testing.T, v *jlm.Video) *jlm.VideoPiece {
	t.Helper()
	for i := range v.Prompt {
		if v.Prompt[i].Kind == jlm.PieceGroups {
			return &v.Prompt[i]
		}
	}
	t.Fatal("the video's prompt writes no groups")
	return nil
}
