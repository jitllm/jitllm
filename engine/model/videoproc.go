package model

import (
	"fmt"
	"image"
	"math"
	"strconv"
)

// A video as a Qwen-VL processor takes it (transformers' Qwen3VLVideoProcessor
// and Qwen3VLProcessor), for the decision models that read one (Clef): frames
// sampled at the processor's rate, every frame resized to one size of whole
// merge units, the frames paired (temporal_patch_size 2, the last repeated to
// make the count even), and each pair one Picture of the tower -- its patches
// attend only to each other, as the reference's per-frame cu_seqlens -- laid
// out in the prompt as "<t seconds>", <|vision_start|>, its rows as
// <|video_pad|>, <|vision_end|>, with t the pair's two frames' mean time.
//
// A video arrives as its frames (Clef's "frame arrays"). With no frame rate
// beside them the reference assumes 24 frames a second, for the sampling and
// for the stamps (BaseVideoProcessor.sample_frames, replace_video_token), and
// so does this.

// VideoFrames is one video: its frames, every one the same size, and their
// rate, zero when the request gave none.
type VideoFrames struct {
	Frames []image.Image
	FPS    float64
}

// videoProc is a video processor's settings: the processor_config.json values
// the reference reads (Clef's, clefVideo).
type videoProc struct {
	fps                  float64 // frames sampled per second of video
	minFrames, maxFrames int
	minPixels, maxPixels int // shortest_edge and longest_edge: over every frame together
	temporal             int // temporal_patch_size
}

// clefVideo is Clef-Flash's video processor (its processor_config.json):
// 2 frames a second, 4 to 768 of them, 4096 to 25165824 pixels over the
// whole video. The container does not carry a processor's video settings;
// these are the Clef readout's, as its prompt is.
var clefVideo = videoProc{fps: 2, minFrames: 4, maxFrames: 768, minPixels: 4096, maxPixels: 25165824, temporal: 2}

// defaultVideoFPS is the rate the reference assumes for frames with none.
const defaultVideoFPS = 24

// sampleIndices is sample_frames: the frames kept, num_frames of them spread
// over the video by np.linspace and rounded half to even.
func (vp videoProc) sampleIndices(total int, fps float64) []int {
	if fps <= 0 {
		fps = defaultVideoFPS
	}
	n := int(float64(total) / fps * vp.fps)
	n = min(max(n, vp.minFrames), vp.maxFrames, total)
	// np.linspace's arithmetic: i times one step, the last point the end
	// itself; a different order of the same arithmetic moves a half.
	out := make([]int, n)
	if n == 0 {
		return out
	}
	step := 0.0
	if n > 1 {
		step = float64(total-1) / float64(n-1)
	}
	for i := range out {
		out[i] = int(math.RoundToEven(float64(i) * step))
	}
	if n > 1 {
		out[n-1] = total - 1
	}
	return out
}

// frameSize is the video smart_resize: the size every frame is resized to
// for n frames of w x h, each side a whole number of merge units f and the
// pixels over all frames inside the bounds.
func (vp videoProc) frameSize(n, w, h, f int) (int, int, error) {
	if n < vp.temporal {
		return 0, 0, fmt.Errorf("model: a video of %d frames; the processor takes at least %d", n, vp.temporal)
	}
	if h < f || w < f {
		scale := math.Max(float64(f)/float64(h), float64(f)/float64(w))
		h, w = int(float64(h)*scale), int(float64(w)*scale)
	}
	if float64(max(w, h))/float64(min(w, h)) > 200 {
		return 0, 0, fmt.Errorf("model: a %dx%d frame's aspect is over 200, which the reference refuses", w, h)
	}
	ff := float64(f)
	hb := int(math.RoundToEven(float64(h)/ff)) * f
	wb := int(math.RoundToEven(float64(w)/ff)) * f
	tb := int(math.RoundToEven(float64(n)/float64(vp.temporal))) * vp.temporal
	switch {
	case tb*hb*wb > vp.maxPixels:
		beta := math.Sqrt(float64(n) * float64(h) * float64(w) / float64(vp.maxPixels))
		hb = max(f, int(math.Floor(float64(h)/beta/ff))*f)
		wb = max(f, int(math.Floor(float64(w)/beta/ff))*f)
	case tb*hb*wb < vp.minPixels:
		beta := math.Sqrt(float64(vp.minPixels) / (float64(n) * float64(h) * float64(w)))
		hb = int(math.Ceil(float64(h)*beta/ff)) * f
		wb = int(math.Ceil(float64(w)*beta/ff)) * f
	}
	return wb, hb, nil
}

// stamps is _calculate_timestamps: each pair's two frames' mean time, the
// indices padded with the last to whole pairs.
func (vp videoProc) stamps(idx []int, fps float64) []float64 {
	if fps <= 0 {
		fps = defaultVideoFPS
	}
	ix := append([]int(nil), idx...)
	for len(ix)%vp.temporal != 0 {
		ix = append(ix, ix[len(ix)-1])
	}
	var out []float64
	for i := 0; i < len(ix); i += vp.temporal {
		a, b := float64(ix[i])/fps, float64(ix[i+vp.temporal-1])/fps
		out = append(out, (a+b)/2)
	}
	return out
}

// stampText is a pair's stamp as the processor writes it: f"<{t:.1f} seconds>".
func stampText(t float64) string { return "<" + strconv.FormatFloat(t, 'f', 1, 64) + " seconds>" }

// videoPictures preprocesses a video for this tower: its sampled frames
// resized and normalised, paired, each pair a Picture, with each pair's stamp.
func (t *Tower) videoPictures(vp videoProc, v VideoFrames) ([]*Picture, []float64, error) {
	c := t.Cfg
	if t.patchW1.e == nil {
		return nil, nil, fmt.Errorf("model: this container's tower has no second temporal plane, so it cannot " +
			"read a video; convert the model again with its mmproj")
	}
	if len(v.Frames) == 0 {
		return nil, nil, fmt.Errorf("model: a video with no frames")
	}
	b := v.Frames[0].Bounds()
	for i, f := range v.Frames {
		if f.Bounds().Dx() != b.Dx() || f.Bounds().Dy() != b.Dy() {
			return nil, nil, fmt.Errorf("model: frame %d is %dx%d and frame 0 %dx%d; a video's frames are one size",
				i, f.Bounds().Dx(), f.Bounds().Dy(), b.Dx(), b.Dy())
		}
	}
	idx := vp.sampleIndices(len(v.Frames), v.FPS)
	unit := c.PatchSz * max(c.Scale, 1)
	w, h, err := vp.frameSize(len(idx), b.Dx(), b.Dy(), unit)
	if err != nil {
		return nil, nil, err
	}
	gh, gw := h/c.PatchSz, w/c.PatchSz
	if gh*gw > c.MaxSeq() {
		return nil, nil, fmt.Errorf("model: a %dx%d frame is %d patches and this tower reads at most %d; "+
			"send smaller frames", w, h, gh*gw, c.MaxSeq())
	}
	frames := make([][]float32, 0, len(idx)+1)
	for _, i := range idx {
		px := make([]float32, w*h*3)
		toRGB8(v.Frames[i]).resize(w, h, aaBicubic).normalizeInto(px, c.Mean, c.Std)
		frames = append(frames, px)
	}
	for len(frames)%vp.temporal != 0 {
		frames = append(frames, frames[len(frames)-1])
	}
	var pics []*Picture
	for i := 0; i < len(frames); i += vp.temporal {
		second := frames[i+1]
		if t.Cfg.fault == faultVideoFirstFrame {
			second = frames[i]
		}
		pc, err := c.framePair(frames[i], second, gh, gw)
		if err != nil {
			return nil, nil, err
		}
		pics = append(pics, pc)
	}
	return pics, vp.stamps(idx, v.FPS), nil
}
