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
		Pre: []jlm.PreStep{{Kind: jlm.PreVideoResize, Ints: []int32{32, 4096, 25165824, int32(jlm.FilterPILBicubic), int32(jlm.PrecTorch), 2}}},
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
