package convert

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/tok"
)

// TestQwen3VLStatesItsVideo converts Qwen3-VL's fixture and reads the video
// its container states: Qwen3-VL's published sampling and bounds, the entry
// over two frames binding the second patch plane, and that plane in the
// container beside the planes' sum -- its own matrix in the sum's layout, not
// the sum again.
func TestQwen3VLStatesItsVideo(t *testing.T) {
	src := testmodels.Path("synth-qwen3vl.gguf")
	mm := testmodels.Path("mmproj-synth-qwen3vl.gguf")
	for _, p := range []string{src, mm} {
		if _, err := os.Stat(p); err != nil {
			testmodels.Missing(t, "MODEL MISSING: %v (run scripts/qwenvlgold.py synth-qwen3vl) -- this gate proved nothing", err)
		}
	}
	dst := filepath.Join(t.TempDir(), "qwen3vl.jlm")
	if _, err := FromGGUFs(src, mm, dst, jlm.Fingerprint{Writer: "gate"}); err != nil {
		t.Fatal(err)
	}
	f, err := jlm.Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	v := f.Vision()
	if v == nil || v.Tower == nil || v.Tower.Video == nil {
		t.Fatal("the container states no video")
	}
	vid := v.Tower.Video
	if err := vid.Valid(); err != nil {
		t.Fatal(err)
	}
	if got := vid.Pre[0].Ints; got[1] != 4096 || got[2] != 25165824 || jlm.Precision(got[4]) != jlm.PrecTorch {
		t.Fatalf("the video resizes with %v", got)
	}
	// Qwen3VLVideoProcessor's and Qwen3VLProcessor's own, on the bounds
	// Qwen3-VL publishes: with and without a stated rate, an odd count's
	// last frame repeated, sides rounded half to even, a side lifted to the
	// factor, a long video's frames shrunk to share the area.
	holdVideo(t, vid, []videoCase{
		{10, 160, 120, 0, []int{0, 3, 6, 9}, 160, 128, 0, []string{"0.1", "0.3"}},
		{2, 64, 48, 0, []int{0, 1}, 64, 64, 0, []string{"0.0"}},
		{7, 33, 600, 0, []int{0, 2, 4, 6}, 32, 608, 0, []string{"0.0", "0.2"}},
		{100, 640, 360, 0, []int{0, 14, 28, 42, 57, 71, 85, 99}, 640, 352, 0, []string{"0.3", "1.5", "2.7", "3.8"}},
		{90, 1280, 720, 30, []int{0, 18, 36, 53, 71, 89}, 1280, 704, 0, []string{"0.3", "1.5", "2.7"}},
		{37, 20, 20, 12.5, []int{0, 9, 18, 27, 36, 36}, 32, 32, 0, []string{"0.4", "1.8", "2.9"}},
		{5, 3000, 2000, 0, []int{0, 1, 3, 4}, 3008, 1984, 0, []string{"0.0", "0.1"}},
	}, secondsOf)
	// 3000 frames at 25 a second: 240 sampled, the times past 25 seconds
	// written as the processor rounds them.
	s, items := sampleVideo(t, vid, 3000, 25)
	switch got := secondsOf(items); {
	case len(s.Frames) != 240 || s.Frames[1] != 13 || s.Frames[239] != 2999:
		t.Errorf("3000 frames sample %d, the second %d, the last %d", len(s.Frames), s.Frames[1], s.Frames[239])
	case len(got) != 120 || got[25] != "25.3" || got[26] != "26.4" || got[119] != "119.7":
		t.Errorf("3000 frames' times: %d of them, %v", len(got), got)
	}
	// A video of one frame samples fewer than a pair, which the reference
	// refuses too.
	if s, err := model.SampleVideo(vid, 1, 0); err == nil {
		if _, _, err := model.VideoFrameSize(vid, s, 64, 64); err == nil {
			t.Error("a video of one frame was sized as a pair")
		}
	}
	var binds []jlm.Role
	for _, op := range vid.Entry.Ops {
		binds = append(binds, op.Bind...)
	}
	if !slices.Contains(binds, jlm.RoleVPatchEmbd2) || !slices.Contains(binds, jlm.RoleVPatchEmbd) {
		t.Fatalf("the video's entry binds %v", binds)
	}
	sum, ok1 := f.Find(jlm.RoleVPatchEmbd, jlm.DenseBlock, -1)
	second, ok2 := f.Find(jlm.RoleVPatchEmbd2, jlm.DenseBlock, -1)
	if !ok1 || !ok2 {
		t.Fatalf("the container holds the sum %v and the second plane %v", ok1, ok2)
	}
	if sum.Type != second.Type || sum.Dims != second.Dims {
		t.Fatalf("the second plane is %v %v and the sum %v %v", second.Type, second.Dims, sum.Type, sum.Dims)
	}
	a, err := f.ReadPayload(sum)
	if err != nil {
		t.Fatal(err)
	}
	b, err := f.ReadPayload(second)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(a, b) {
		t.Fatal("the second plane holds the sum")
	}
	if _, ok := f.Find(jlm.RoleVPatchEmbd, jlm.DenseBlock, 1); ok {
		t.Fatal("the second plane is held twice")
	}
}

// TestQwen2VLStatesItsVideo converts Qwen2-VL's and Qwen2.5-VL's fixtures
// and reads the video each container states: every frame, each resized
// alone within Qwen2VLVideoProcessor's bounds, the pairs one grid with no
// time, a frame apart on the temporal axis for Qwen2-VL and
// tokens_per_second apart a second for Qwen2.5-VL, the entry binding the
// second patch plane.
func TestQwen2VLStatesItsVideo(t *testing.T) {
	// Qwen2VLVideoProcessor on its class defaults and get_rope_index: every
	// frame, an odd count's last repeated, each frame sized alone by
	// smart_resize (sizes from transformers' own), a video of no stated rate
	// at 24 frames a second; stride is Qwen2.5-VL's, Qwen2-VL's pairs a
	// position apart.
	cases := []videoCase{
		{5, 160, 120, 0, []int{0, 1, 2, 3, 4, 4}, 392, 280, 0, nil},
		{4, 1280, 720, 30, []int{0, 1, 2, 3}, 1008, 560, 0, nil},
		{4, 640, 360, 2, []int{0, 1, 2, 3}, 644, 364, 2, nil},
		{3, 3000, 2000, 1, []int{0, 1, 2, 2}, 924, 616, 4, nil},
		{1, 20, 20, 0.5, []int{0, 0}, 336, 336, 8, nil},
		{2, 33, 600, 0, []int{0, 1}, 84, 1372, 0, nil},
	}
	for _, c := range []struct {
		name   string
		qwen25 bool
	}{{"synth-qwen2vl", false}, {"synth-qwen25vl", true}} {
		t.Run(c.name, func(t *testing.T) {
			src := testmodels.Path(c.name + ".gguf")
			mm := testmodels.Path("mmproj-" + c.name + ".gguf")
			for _, p := range []string{src, mm} {
				if _, err := os.Stat(p); err != nil {
					testmodels.Missing(t, "MODEL MISSING: %v (run scripts/qwenvlgold.py %s) -- this gate proved nothing", err, c.name)
				}
			}
			dst := filepath.Join(t.TempDir(), c.name+".jlm")
			if _, err := FromGGUFs(src, mm, dst, jlm.Fingerprint{Writer: "gate"}); err != nil {
				t.Fatal(err)
			}
			f, err := jlm.Open(dst)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			v := f.Vision()
			if v == nil || v.Tower == nil || v.Tower.Video == nil {
				t.Fatal("the container states no video")
			}
			vid := v.Tower.Video
			if got := vid.Pre[0].Ints; len(got) != 7 || got[1] != 100352 || got[2] != 602112 || got[6] != 1 ||
				jlm.Precision(got[4]) != jlm.PrecTorch {
				t.Fatalf("the video resizes with %v", got)
			}
			for _, vc := range cases {
				if !c.qwen25 {
					vc.stride = 1
				}
				holdVideo(t, vid, []videoCase{vc}, wholeSeconds)
			}
			var binds []jlm.Role
			for _, op := range vid.Entry.Ops {
				binds = append(binds, op.Bind...)
			}
			if !slices.Contains(binds, jlm.RoleVPatchEmbd2) || !slices.Contains(binds, jlm.RoleVPatchEmbd) {
				t.Fatalf("the video's entry binds %v", binds)
			}
			if _, ok := f.Find(jlm.RoleVPatchEmbd2, jlm.DenseBlock, -1); !ok {
				t.Fatal("the container holds no second plane")
			}
		})
	}
}

// TestGLM4VStatesItsVideo converts GLM-4.1V's and GLM-4.5V's fixtures and
// reads the video each container states: two frames a second in whole
// steps over at most 300 seconds, the frames sized over their count within
// Glm4vVideoProcessor's bounds, each pair a picture between
// <|begin_of_video|> and <|end_of_video|> followed by its first frame's
// whole seconds, the entry binding the second patch plane, the patches'
// norm and the resampled position table.
func TestGLM4VStatesItsVideo(t *testing.T) {
	for _, name := range []string{"synth-glm4v", "synth-glm4vmoe"} {
		t.Run(name, func(t *testing.T) {
			src := testmodels.Path(name + ".gguf")
			mm := testmodels.Path("mmproj-" + name + ".gguf")
			for _, p := range []string{src, mm} {
				if _, err := os.Stat(p); err != nil {
					testmodels.Missing(t, "MODEL MISSING: %v (run scripts/qwenvlgold.py %s) -- this gate proved nothing", err, name)
				}
			}
			dst := filepath.Join(t.TempDir(), name+".jlm")
			if _, err := FromGGUFs(src, mm, dst, jlm.Fingerprint{Writer: "gate"}); err != nil {
				t.Fatal(err)
			}
			f, err := jlm.Open(dst)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			v := f.Vision()
			if v == nil || v.Tower == nil || v.Tower.Video == nil {
				t.Fatal("the container states no video")
			}
			vid := v.Tower.Video
			if got := vid.Pre[0].Ints; len(got) != 6 || got[1] != 12544 || got[2] != 47040000 || got[5] != 2 {
				t.Fatalf("the video resizes with %v", got)
			}
			// Glm4vVideoProcessor.sample_frames, the times Glm4vProcessor
			// writes for its pairs and smart_resize's size over the sampled
			// count, each from transformers' own: two frames a second in
			// whole steps of the rate, every frame of a video past
			// max_duration that samples no fewer, each frame taken once and
			// an odd count's last repeated; each pair's first frame's whole
			// seconds.
			holdVideo(t, vid, []videoCase{
				{5, 160, 120, 24, []int{0, 4}, 168, 112, 0, []string{"0"}},
				{8, 1280, 720, 4, []int{0, 2, 4, 6, 7, 7}, 1288, 728, 0, []string{"0", "1", "1"}},
				{30, 640, 360, 30, []int{0, 15, 29, 29}, 644, 364, 0, []string{"0", "0"}},
				{7, 112, 112, 1, []int{0, 1, 2, 3, 4, 5, 6, 6}, 112, 112, 0, []string{"0", "2", "4", "6"}},
				{13, 1000, 50, 3, []int{0, 2, 3, 5, 6, 8, 9, 11, 12, 12}, 1008, 56, 0, []string{"0", "1", "2", "3", "4"}},
				{2, 20, 20, 24, []int{0, 1}, 84, 84, 0, []string{"0"}},
				{40, 33, 600, 0.1, []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
					24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39}, 28, 588, 0, []string{"0", "20", "40", "60",
					"80", "100", "120", "140", "160", "180", "200", "220", "240", "260", "280", "300", "320", "340", "360", "380"}},
			}, wholeSeconds)
			// Past max_duration with more frames than it samples: 600 at
			// linspace over the seconds, each taken once, the 599 left
			// filled to pairs.
			for _, c := range []struct {
				n                     int
				fps                   float64
				second, third, before int
				times                 []string
			}{{9000, 24, 16, 31, 8994, []string{"0", "1", "2", "372", "374", "374"}},
				{20000, 29.97, 34, 67, 19954, []string{"0", "2", "4", "662", "664", "666"}}} {
				s, items := sampleVideo(t, vid, c.n, c.fps)
				f, got := s.Frames, wholeSeconds(items)
				if len(f) != 600 || f[1] != c.second || f[2] != c.third || f[597] != c.before || f[599] != c.n-1 ||
					len(got) != 300 || !slices.Equal(append(got[:3:3], got[297:]...), c.times) {
					t.Errorf("%d frames at %v sample %d (%v ... %v), times %v", c.n, c.fps, len(f), f[:min(3, len(f))],
						f[max(len(f)-3, 0):], got)
				}
			}
			var binds []jlm.Role
			resampled := false
			for _, op := range vid.Entry.Ops {
				binds = append(binds, op.Bind...)
				resampled = resampled || op.Code == jlm.TOpPosResample
			}
			if !slices.Contains(binds, jlm.RoleVPatchEmbd2) || !slices.Contains(binds, jlm.RoleVEmbdNorm) || !resampled {
				t.Fatalf("the video's entry binds %v, its table resampled %v", binds, resampled)
			}
			voc, err := tok.New(f.Vocab())
			if err != nil {
				t.Fatal(err)
			}
			id := func(p string) int32 {
				i, ok := voc.ID(p)
				if !ok {
					t.Fatalf("%q is no token", p)
				}
				return i
			}
			// replace_video_token: the pairs as pictures between the video's
			// markers, each followed by its time.
			s, _ := sampleVideo(t, vid, 8, 4)
			want := fmt.Sprintf("[%d][%d]{0}[%d]0[%d]{1}[%d]1[%d]{2}[%d]1[%d]", id("<|begin_of_video|>"),
				id("<|begin_of_image|>"), id("<|end_of_image|>"), id("<|begin_of_image|>"), id("<|end_of_image|>"),
				id("<|begin_of_image|>"), id("<|end_of_image|>"), id("<|end_of_video|>"))
			if got := layoutOf(t, vid, s); got != want {
				t.Errorf("a video of three pairs reads %q, want %q", got, want)
			}
		})
	}
}

// TestGemma4StatesItsVideo converts synth-gemma4v's two files
// (scripts/gemma4visiongold.py) and holds the stated video to the
// checkpoints' Gemma4VideoProcessor and Gemma4Processor: 32 frames sampled
// uniformly, each read as a picture within 70 soft tokens' patches (630 at
// the 3x3 pool), its time as MM:SS at 24 frames a second where none is
// stated, a space after it and between the frames; its rows bidirectional
// as the picture's are, the text model seeing vision bidirectionally.
func TestGemma4StatesItsVideo(t *testing.T) {
	const name = "synth-gemma4v"
	src, mm := testmodels.Path(name+".gguf"), testmodels.Path("mmproj-"+name+".gguf")
	for _, p := range []string{src, mm} {
		if _, err := os.Stat(p); err != nil {
			testmodels.Missing(t, "MODEL MISSING: %v (run scripts/gemma4visiongold.py) -- this gate proved nothing", err)
		}
	}
	dst := filepath.Join(t.TempDir(), name+".jlm")
	var stated []jlm.PreStep
	if _, err := FromGGUFs(src, mm, dst, jlm.Fingerprint{Writer: "gate"}, statedSteps(&stated)); err != nil {
		t.Fatal(err)
	}
	f, err := jlm.Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	v := f.Vision()
	if v == nil || v.Tower == nil || v.Tower.Video == nil {
		t.Fatal("the container states no video")
	}
	vid := v.Tower.Video
	// The stater writes the picture's steps, then the frames' own.
	half := len(stated) / 2
	if !vid.AsPictures() || len(vid.Pre) != 0 || vid.Frames == nil || len(stated) != 2*half ||
		stated[half].Kind != PreAspectBudget || stated[half].Ints[0] != 630 {
		t.Fatalf("the video reads %+v, its frames' processor steps %+v", vid, stated[half:])
	}
	if !v.Tower.Span.Bidir {
		t.Fatal("the frames' rows are causal where the text model sees vision bidirectionally")
	}
	// torch.arange(0, 40, 40 / 32), truncated; at 24 frames a second the
	// last frame's time is 38/24 seconds.
	s, items := sampleVideo(t, vid, 40, 0)
	want := []int{0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27, 28, 30, 31, 32, 33,
		35, 36, 37, 38}
	if !slices.Equal(s.Frames, want) {
		t.Errorf("40 frames sample %v, want %v", s.Frames, want)
	}
	b, a := v.Tower.Span.Before[0], v.Tower.Span.After[0]
	got := layoutOf(t, vid, s)
	head := fmt.Sprintf("00:00 [%d]{0}[%d] 00:00 [%d]{1}[%d] ", b, a, b, a)
	tail := fmt.Sprintf(" 00:01 [%d]{31}[%d]", b, a)
	if !strings.HasPrefix(got, head) || !strings.HasSuffix(got, tail) || len(items) != 4*32 {
		t.Errorf("the prompt reads %q (%d items), want it to begin %q and end %q", got, len(items), head, tail)
	}
	if s, _ := sampleVideo(t, vid, 3000, 30); s.Frames[1] != 93 || s.Frames[31] != 2906 {
		t.Errorf("3000 frames sample %v", s.Frames)
	}
}

// TestSmolVLMStatesItsVideo converts synth-smolvlm2 and holds the stated
// video to SmolVLMVideoProcessor.sample_frames at SmolVLM2's
// video_sampling (the frames from transformers' own) and to
// SmolVLMProcessor.replace_video_token: a frame a second, at most 64, by
// linspace a second in from either end of a video past 66 seconds; the
// introduction's count in num2words' words and duration as timedelta writes
// it; each frame's "\nFrame from MM:SS:" and the frame between the global
// image's markers, without the line the picture's open with; "\n\n" after
// the last.
func TestSmolVLMStatesItsVideo(t *testing.T) {
	const name = "synth-smolvlm2"
	src, mm := testmodels.Path(name+".gguf"), testmodels.Path("mmproj-"+name+".gguf")
	for _, p := range []string{src, mm} {
		if _, err := os.Stat(p); err != nil {
			testmodels.Missing(t, "MODEL MISSING: %v (run scripts/smolvlmvideogold.py) -- this gate proved nothing", err)
		}
	}
	dst := filepath.Join(t.TempDir(), name+".jlm")
	if _, err := FromGGUFs(src, mm, dst, jlm.Fingerprint{Writer: "gate"}); err != nil {
		t.Fatal(err)
	}
	f, err := jlm.Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	v := f.Vision()
	if v == nil || v.Tower == nil || v.Tower.Video == nil {
		t.Fatal("the container states no video")
	}
	vid := v.Tower.Video
	if !vid.AsPictures() || len(vid.Pre) != 0 {
		t.Fatalf("the video reads %+v, want each frame the picture's square", vid)
	}
	// SmolVLMProcessor renders a conversation carrying a video by its own
	// template, the published one writing no video part.
	if body, ok := f.Vocab().Template("video"); !ok || body != smolvlmVideoTemplate {
		t.Fatalf("the container states the video template %v: want SmolVLMProcessor's", ok)
	}
	if _, ok := f.Vocab().Template("default"); !ok {
		t.Fatal("the model's own template is not named default beside the video's")
	}
	stamp := regexp.MustCompile(`\nFrame from (\d\d:\d\d):`)
	for _, c := range []struct {
		n           int
		fps         float64
		count       int
		head, tail  []int
		first, last string
		intro       string
	}{
		{10, 1, 10, []int{0, 1, 2, 3}, []int{7, 8, 9}, "00:00", "00:09", "ten frames from a 0:00:10 "},
		{300, 30, 10, []int{0, 33, 66, 99}, []int{232, 265, 299}, "00:00", "00:09", "ten frames from a 0:00:10 "},
		{2400, 24, 64, []int{24, 61, 98, 136}, []int{2301, 2338, 2376}, "00:01", "01:39", "sixty-four frames from a 0:01:40 "},
		{7, 2.5, 3, []int{0, 3, 6}, []int{0, 3, 6}, "00:00", "00:02", "three frames from a 0:00:02 "},
		{1, 1, 1, []int{0}, []int{0}, "00:00", "00:00", "one frames from a 0:00:01 "},
		{5, 24, 1, []int{0}, []int{0}, "00:00", "00:00", "one frames from a 0:00:00 "},
		{70, 1, 64, []int{1, 2, 3, 4}, []int{66, 67, 69}, "00:01", "01:09", "sixty-four frames from a 0:01:10 "},
		{4000, 24, 64, []int{24, 86, 149, 212}, []int{3850, 3913, 3976}, "00:01", "02:45", "sixty-four frames from a 0:02:46 "},
	} {
		s, items := sampleVideo(t, vid, c.n, c.fps)
		fr := s.Frames
		var times []string
		for _, it := range items {
			for _, m := range stamp.FindAllStringSubmatch(it.Text, -1) {
				times = append(times, m[1])
			}
		}
		if len(fr) != c.count || !slices.Equal(fr[:len(c.head)], c.head) || !slices.Equal(fr[len(fr)-len(c.tail):], c.tail) ||
			len(times) != c.count || times[0] != c.first || times[len(times)-1] != c.last ||
			!strings.HasPrefix(items[0].Text, "You are provided the following series of "+c.intro+"[H:MM:SS] video.\n") ||
			items[len(items)-1].Text != "\n\n" {
			t.Errorf("%d frames at %v: %d frames %v, times %v, opening %q; want %d, %v..%v, %s..%s, %q",
				c.n, c.fps, len(fr), fr, times, items[0].Text, c.count, c.head, c.tail, c.first, c.last, c.intro)
		}
	}
	span := v.Tower.Span
	s, _ := sampleVideo(t, vid, 2, 1)
	want := fmt.Sprintf("You are provided the following series of two frames from a 0:00:02 [H:MM:SS] video.\n"+
		"\nFrame from 00:00:%s{0}%s\nFrame from 00:01:%s{1}%s\n\n", ids(span.Before[1:]), ids(span.After), ids(span.Before[1:]),
		ids(span.After))
	if got := layoutOf(t, vid, s); len(span.Before) != 3 || got != want {
		t.Errorf("a video of two frames reads\n%q\nwant\n%q (the picture's markers %v ... %v)", got, want, span.Before, span.After)
	}
	for n, w := range map[int]string{0: "zero", 5: "five", 21: "twenty-one", 64: "sixty-four", 100: "one hundred",
		105: "one hundred and five", 999: "nine hundred and ninety-nine"} {
		if got := cardinal(n); got != w {
			t.Errorf("cardinal(%d) = %q, num2words %q", n, got, w)
		}
	}
}

// ids is token ids written as layoutOf writes them.
func ids(in []int32) string {
	var b strings.Builder
	for _, id := range in {
		fmt.Fprintf(&b, "[%d]", id)
	}
	return b.String()
}

// TestEveryVideoProcessorIsStated holds every projector the converter
// states a tower for to an entry saying whether its reference processor
// takes a video, and every one that does to a video the converter states
// or to a named gap: the work the model's refusal of a video is.
func TestEveryVideoProcessorIsStated(t *testing.T) {
	// The reference's video processor, "" for a processor that takes none
	// (MiniCPM-V's and Kimi-VL's read a video's frames as pictures).
	processor := map[string]string{
		"d1omni_v": "", "gemma3": "", "gemma3nv": "", "gemma4v": "Gemma4VideoProcessor",
		"glm4v": "Glm4vVideoProcessor", "hunyuanvl": "", "idefics3": "SmolVLMVideoProcessor",
		"internvl": "InternVLVideoProcessor", "janus_pro": "", "kimivl": "", "lfm2": "",
		"llama4": "", "mlp": "", "phi4": "", "pixtral": "",
		"qwen2.5vl_merger": "Qwen2VLVideoProcessor", "qwen2vl_merger": "Qwen2VLVideoProcessor",
		"qwen3vl_merger": "Qwen3VLVideoProcessor", "resampler": "",
	}
	gaps := map[string]bool{}
	for _, name := range Projectors() {
		p := projectorOf()[name]
		if !StatesTower(p) {
			continue
		}
		cls, ok := processor[name]
		if err := videoStatement(name, cls, ok, StatesVideo(p), gaps[name]); err != nil {
			t.Error(err)
		}
	}
	for name := range gaps {
		t.Logf("a gap: %s's %s takes a video the converter does not state", name, processor[name])
	}
	for _, v := range []struct {
		listed, states, gap bool
		cls                 string
	}{
		{false, false, false, ""},
		{true, true, false, ""},
		{true, true, true, "Qwen2VLVideoProcessor"},
		{true, false, false, "Qwen2VLVideoProcessor"},
	} {
		if videoStatement("p", v.cls, v.listed, v.states, v.gap) == nil {
			t.Errorf("the violation %+v was not seen", v)
		}
	}
}

// videoStatement is whether projector name's video is stated as its
// reference processor's class cls says (listed, an entry for it at all):
// stated where the processor takes one, unless it is a named gap.
func videoStatement(name, cls string, listed, states, gap bool) error {
	switch {
	case !listed:
		return fmt.Errorf("%s: no entry says whether its reference processor takes a video", name)
	case cls == "" && states:
		return fmt.Errorf("%s: the converter states a video for a processor that takes none", name)
	case cls != "" && states && gap:
		return fmt.Errorf("%s: stated, and still named a gap", name)
	case cls != "" && !states && !gap:
		return fmt.Errorf("%s: %s takes a video and the converter states none", name, cls)
	}
	return nil
}

// videoCase is a video and what its reference processor does with it: the
// frames it samples, an odd count's last repeated; the size it resizes
// them to; the positions its pairs stand apart on one grid; and the times
// its prompt writes.
type videoCase struct {
	n, w, h int
	fps     float64
	frames  []int
	ow, oh  int
	stride  int
	times   []string
}

// holdVideo holds the stated video vid to each case, the times read from
// its prompt laid out by times.
func holdVideo(t *testing.T, vid *jlm.Video, cases []videoCase, times func([]model.VideoItem) []string) {
	t.Helper()
	for _, c := range cases {
		s, items := sampleVideo(t, vid, c.n, c.fps)
		ow, oh, err := model.VideoFrameSize(vid, s, c.w, c.h)
		if err != nil {
			t.Errorf("%d frames of %dx%d at %v: %v", c.n, c.w, c.h, c.fps, err)
			continue
		}
		if got := times(items); !slices.Equal(s.Frames, c.frames) || ow != c.ow || oh != c.oh || s.Stride != c.stride ||
			!slices.Equal(got, c.times) {
			t.Errorf("%d frames of %dx%d at %v: frames %v, %dx%d, stride %d, times %q; want %v %dx%d %d %q", c.n, c.w, c.h,
				c.fps, s.Frames, ow, oh, s.Stride, got, c.frames, c.ow, c.oh, c.stride, c.times)
		}
	}
}

// sampleVideo is what vid samples of a video of n frames at fps frames a
// second, and its prompt laid out.
func sampleVideo(t *testing.T, vid *jlm.Video, n int, fps float64) (model.VideoSample, []model.VideoItem) {
	t.Helper()
	s, err := model.SampleVideo(vid, n, fps)
	if err != nil {
		t.Fatalf("%d frames at %v: %v", n, fps, err)
	}
	items, err := model.VideoLayout(vid, s)
	if err != nil {
		t.Fatalf("%d frames at %v: %v", n, fps, err)
	}
	return s, items
}

// layoutOf is vid's prompt for sample s written out: its text as it is,
// each id in brackets, each group's rows its number in braces.
func layoutOf(t *testing.T, vid *jlm.Video, s model.VideoSample) string {
	t.Helper()
	items, err := model.VideoLayout(vid, s)
	if err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	for _, it := range items {
		switch {
		case it.Group >= 0:
			fmt.Fprintf(&b, "{%d}", it.Group)
		case it.IDs != nil:
			for _, id := range it.IDs {
				fmt.Fprintf(&b, "[%d]", id)
			}
		default:
			b.WriteString(it.Text)
		}
	}
	return b.String()
}

// secondsOf is the times "<t seconds>" a prompt writes.
func secondsOf(items []model.VideoItem) []string {
	var out []string
	for _, it := range items {
		for _, m := range regexp.MustCompile(`<([0-9.]+) seconds>`).FindAllStringSubmatch(it.Text, -1) {
			out = append(out, m[1])
		}
	}
	return out
}

// wholeSeconds is the text a prompt writes between its ids and rows, each
// a time.
func wholeSeconds(items []model.VideoItem) []string {
	var out []string
	for _, it := range items {
		if it.Group < 0 && it.IDs == nil {
			out = append(out, it.Text)
		}
	}
	return out
}
