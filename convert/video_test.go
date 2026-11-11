package convert

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
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
	if vid.Fps != 2 || vid.DefaultFps != 24 || vid.MinFrames != 4 || vid.MaxFrames != 768 {
		t.Fatalf("the video samples %+v", vid)
	}
	if got := vid.Pre[0].Ints; got[1] != 4096 || got[2] != 25165824 || jlm.Precision(got[4]) != jlm.PrecTorch {
		t.Fatalf("the video resizes with %v", got)
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
	for _, c := range []struct {
		name      string
		perSecond int32
	}{{"synth-qwen2vl", 0}, {"synth-qwen25vl", 2}} {
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
			if !vid.EveryFrame || !vid.Joined || vid.PerSecond != c.perSecond || vid.DefaultFps != 24 || vid.StampDigits != 0 {
				t.Fatalf("the video reads %+v", vid)
			}
			if got := vid.Pre[0].Ints; len(got) != 7 || got[1] != 100352 || got[2] != 602112 || got[6] != 1 ||
				jlm.Precision(got[4]) != jlm.PrecTorch {
				t.Fatalf("the video resizes with %v", got)
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
			if !vid.Steps || vid.Fps != 2 || vid.MaxSeconds != 300 || vid.DefaultFps != 24 || !vid.StampFirst ||
				!vid.StampLast || vid.StampDigits != 0 || vid.Joined || vid.EveryFrame {
				t.Fatalf("the video reads %+v", vid)
			}
			if got := vid.Pre[0].Ints; len(got) != 6 || got[1] != 12544 || got[2] != 47040000 || got[5] != 2 {
				t.Fatalf("the video resizes with %v", got)
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
			if len(vid.Before) != 1 || len(vid.After) != 1 || len(v.Tower.Span.Before) != 1 || len(v.Tower.Span.After) != 1 {
				t.Fatalf("the video is %v ... %v, a pair %v ... %v", vid.Before, vid.After, v.Tower.Span.Before, v.Tower.Span.After)
			}
		})
	}
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
	gaps := map[string]bool{"gemma4v": true, "idefics3": true}
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
