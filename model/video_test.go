package model

import (
	"reflect"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestVideoPlanIsTheProcessors holds the plan to Qwen3VLVideoProcessor's
// own on the bounds Qwen3-VL publishes: the frames it samples, the size it
// resizes them to (as the patch grid) and each pair's time as
// Qwen3VLProcessor writes it -- with and without a stated rate, an odd
// count's last frame repeated, sides rounded half to even, a side lifted to
// the factor, a long video's frames shrunk to share the area. A video of
// one frame samples fewer than a pair, which the reference refuses too.
func TestVideoPlanIsTheProcessors(t *testing.T) {
	v := &jlm.Video{Fps: 2, DefaultFps: 24, MinFrames: 4, MaxFrames: 768, StampDigits: 1,
		Pre:   []jlm.PreStep{{Kind: jlm.PreVideoResize, Ints: []int32{32, 4096, 25165824, int32(jlm.FilterPILBicubic), int32(jlm.PrecTorch), 2}}},
		Entry: jlm.Program{NIn: 2}}
	tg := &TowerGraphs{Patch: 16, Video: v, Videos: []towerClass{{Pic: graph.Attrs{PicW: 8192}}}}
	cases := []struct {
		n, w, h int
		fps     float64
		frames  []int
		gw, gh  int
		stamps  []string
	}{
		{10, 160, 120, 0, []int{0, 3, 6, 9}, 10, 8, []string{"0.1", "0.3"}},
		{2, 64, 48, 0, []int{0, 1}, 4, 4, []string{"0.0"}},
		{7, 33, 600, 0, []int{0, 2, 4, 6}, 2, 38, []string{"0.0", "0.2"}},
		{100, 640, 360, 0, []int{0, 14, 28, 42, 57, 71, 85, 99}, 40, 22, []string{"0.3", "1.5", "2.7", "3.8"}},
		{90, 1280, 720, 30, []int{0, 18, 36, 53, 71, 89}, 80, 44, []string{"0.3", "1.5", "2.7"}},
		{37, 20, 20, 12.5, []int{0, 9, 18, 27, 36, 36}, 2, 2, []string{"0.4", "1.8", "2.9"}},
		{5, 3000, 2000, 0, []int{0, 1, 3, 4}, 188, 124, []string{"0.0", "0.1"}},
	}
	for _, c := range cases {
		p, err := tg.VideoPlan(c.n, c.w, c.h, c.fps)
		if err != nil {
			t.Fatalf("%d frames of %dx%d at %v: %v", c.n, c.w, c.h, c.fps, err)
		}
		if !reflect.DeepEqual(p.Frames, c.frames) || p.Square.Gw != c.gw || p.Square.Gh != c.gh || !reflect.DeepEqual(p.Stamps, c.stamps) {
			t.Errorf("%d frames of %dx%d at %v: frames %v grid %dx%d times %v, want %v %dx%d %v", c.n, c.w, c.h, c.fps,
				p.Frames, p.Square.Gw, p.Square.Gh, p.Stamps, c.frames, c.gw, c.gh, c.stamps)
		}
	}
	// 3000 frames at 25 a second: 240 sampled, the times past 25 seconds
	// written as the processor rounds them.
	p, err := tg.VideoPlan(3000, 320, 240, 25)
	switch {
	case err != nil:
		t.Fatal(err)
	case len(p.Frames) != 240 || p.Frames[1] != 13 || p.Frames[239] != 2999 || p.Square.Gw != 20 || p.Square.Gh != 16:
		t.Errorf("3000 frames sample %d, the second %d, the last %d, at %dx%d", len(p.Frames), p.Frames[1], p.Frames[239], p.Square.Gw, p.Square.Gh)
	case len(p.Stamps) != 120 || p.Stamps[25] != "25.3" || p.Stamps[26] != "26.4" || p.Stamps[119] != "119.7":
		t.Errorf("3000 frames' times: %d of them, %v", len(p.Stamps), p.Stamps)
	}
	if _, err := tg.VideoPlan(1, 64, 64, 0); err == nil {
		t.Error("a video of one frame was planned as a pair")
	}
}

// TestJoinedVideoPlanIsTheProcessors holds the plan to Qwen2VLVideoProcessor
// on its class defaults and to get_rope_index: every frame, an odd count's
// last repeated, each frame sized alone by smart_resize (sizes from
// transformers' own), and the pairs a position apart (Qwen2-VL) or
// tokens_per_second * int(2 / fps) apart (Qwen2.5-VL), a video of no stated
// rate at 24 frames a second.
func TestJoinedVideoPlanIsTheProcessors(t *testing.T) {
	cases := []struct {
		n, w, h   int
		fps       float64
		perSecond int32
		frames    []int
		gw, gh    int
		stride    int
	}{
		{5, 160, 120, 0, 0, []int{0, 1, 2, 3, 4, 4}, 28, 20, 1},
		{4, 1280, 720, 30, 2, []int{0, 1, 2, 3}, 72, 40, 0},
		{4, 640, 360, 2, 2, []int{0, 1, 2, 3}, 46, 26, 2},
		{3, 3000, 2000, 1, 2, []int{0, 1, 2, 2}, 66, 44, 4},
		{1, 20, 20, 0.5, 2, []int{0, 0}, 24, 24, 8},
		{2, 33, 600, 0, 2, []int{0, 1}, 6, 98, 0},
	}
	for _, c := range cases {
		v := &jlm.Video{DefaultFps: 24, EveryFrame: true, Joined: true, PerSecond: c.perSecond,
			Pre: []jlm.PreStep{{Kind: jlm.PreVideoResize, Ints: []int32{28, 100352, 602112,
				int32(jlm.FilterPILBicubic), int32(jlm.PrecTorch), 2, 1}}},
			Entry: jlm.Program{NIn: 2}}
		tg := &TowerGraphs{Patch: 14, Video: v, Videos: []towerClass{{Pic: graph.Attrs{PicW: 8192}}}}
		p, err := tg.VideoPlan(c.n, c.w, c.h, c.fps)
		if err != nil {
			t.Fatalf("%d frames of %dx%d at %v: %v", c.n, c.w, c.h, c.fps, err)
		}
		if !reflect.DeepEqual(p.Frames, c.frames) || p.Square.Gw != c.gw || p.Square.Gh != c.gh || p.Stride != c.stride || p.Stamps != nil {
			t.Errorf("%d frames of %dx%d at %v: frames %v grid %dx%d stride %d times %v, want %v %dx%d %d", c.n, c.w, c.h, c.fps,
				p.Frames, p.Square.Gw, p.Square.Gh, p.Stride, p.Stamps, c.frames, c.gw, c.gh, c.stride)
		}
	}
}

// TestStepVideoPlanIsTheProcessors holds a video sampled in steps
// (jlm.Video.Steps, GLM-4V's) to Glm4vVideoProcessor.sample_frames, the
// times Glm4vProcessor writes for its pairs and smart_resize's size over the
// sampled count, each from transformers' own: two frames a second in whole
// steps of the rate, every frame of a video past max_duration that samples
// no fewer, linspace over its seconds of one that does, each frame taken
// once and an odd count's last repeated; each pair's first frame's whole
// seconds.
func TestStepVideoPlanIsTheProcessors(t *testing.T) {
	cases := []struct {
		maxSeconds int32
		n          int
		fps        float64
		w, h       int
		frames     []int
		stamps     []string
		ow, oh     int
	}{
		{300, 5, 24, 160, 120, []int{0, 4}, []string{"0"}, 168, 112},
		{300, 8, 4, 1280, 720, []int{0, 2, 4, 6, 7, 7}, []string{"0", "1", "1"}, 1288, 728},
		{300, 30, 30, 640, 360, []int{0, 15, 29, 29}, []string{"0", "0"}, 644, 364},
		{300, 7, 1, 112, 112, []int{0, 1, 2, 3, 4, 5, 6, 6}, []string{"0", "2", "4", "6"}, 112, 112},
		{300, 13, 3, 1000, 50, []int{0, 2, 3, 5, 6, 8, 9, 11, 12, 12}, []string{"0", "1", "2", "3", "4"}, 1008, 56},
		{300, 2, 24, 20, 20, []int{0, 1}, []string{"0"}, 84, 84},
		{300, 40, 0.1, 33, 600, []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
			24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39}, []string{"0", "20", "40", "60", "80", "100",
			"120", "140", "160", "180", "200", "220", "240", "260", "280", "300", "320", "340", "360", "380"}, 28, 588},
		{5, 200, 24, 640, 480, []int{0, 24, 48, 72, 96, 120, 144, 168, 192, 199}, []string{"0", "2", "4", "6", "8"}, 644, 476},
		{5, 9, 1, 300, 200, []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 8}, []string{"0", "2", "4", "6", "8"}, 308, 196},
	}
	for _, c := range cases {
		v := &jlm.Video{Fps: 2, DefaultFps: 24, MinFrames: 1, MaxFrames: 2 * c.maxSeconds, Steps: true, MaxSeconds: c.maxSeconds,
			StampFirst: true, StampLast: true,
			Pre: []jlm.PreStep{{Kind: jlm.PreVideoResize, Ints: []int32{28, 12544, 47040000,
				int32(jlm.FilterPILBicubic), int32(jlm.PrecTorch), 2}}},
			Entry: jlm.Program{NIn: 2}}
		tg := &TowerGraphs{Patch: 14, Video: v, Videos: []towerClass{{Pic: graph.Attrs{PicW: 8192}}}}
		p, err := tg.VideoPlan(c.n, c.w, c.h, c.fps)
		if err != nil {
			t.Fatalf("%d frames of %dx%d at %v: %v", c.n, c.w, c.h, c.fps, err)
		}
		if !reflect.DeepEqual(p.Frames, c.frames) || !reflect.DeepEqual(p.Stamps, c.stamps) ||
			p.Square.Ow != c.ow || p.Square.Oh != c.oh || p.Stride != 0 {
			t.Errorf("%d frames of %dx%d at %v (%d s at most): frames %v times %v, %dx%d, stride %d; want %v %v %dx%d",
				c.n, c.w, c.h, c.fps, c.maxSeconds, p.Frames, p.Stamps, p.Square.Ow, p.Square.Oh, p.Stride,
				c.frames, c.stamps, c.ow, c.oh)
		}
	}
}
