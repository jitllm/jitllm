package jlm

import "fmt"

// PreVideoResize resamples a video's frames to one size of the video's own,
// each side a multiple of Ints[0] pixels: a frame a side shorter than Ints[0]
// scaled up first so that side is Ints[0], each side truncated; each side
// rounded half to even to a multiple; then, where the frame count rounded
// half to even to a multiple of Ints[5] times that area passes Ints[2], both
// sides scaled down by sqrt(count * area / Ints[2]) of the unrounded sides
// and floored to a multiple, at least one, or where it falls short of
// Ints[1], scaled up by sqrt(Ints[1] / (count * area)) and ceiled to one
// (Qwen3-VL's video smart_resize); with filter Ints[3] in arithmetic Ints[4].
// Ints[6], where stated, is 1 for each frame sized alone: no side lifted
// and the count not read, the area held to [Ints[1], Ints[2]] (Qwen2-VL's
// video smart_resize, the picture's).
const PreVideoResize PreKind = 16

// Video is how a tower reads a video (Tower.Video): its frames sampled as
// Fps, DefaultFps, MinFrames and MaxFrames say, sized and normalised as Pre
// states, each two consecutive frames one picture of the tower -- Entry over
// the first frame's pixels (value 0) and the second's (value 1), the tower's
// blocks and head over its rows -- an odd count's last frame repeated. In
// the prompt a video is Before, then for each pair its time and the tower's
// span, then After; or, Joined, Before, every pair's rows, then After.
type Video struct {
	// A video of n frames at rate r (DefaultFps where the request states
	// none) samples int(n / r * Fps) frames, held to MinFrames and MaxFrames
	// and then to n, at linspace(0, n-1, count) rounded half to even
	// (numpy's).
	Fps, DefaultFps      float32
	MinFrames, MaxFrames int32
	// Pre is a PreVideoResize, then a PreNormalize or a PreNormalizeBF16.
	Pre   []PreStep
	Entry Program
	// A pair's time is the mean of its frames' (index / r) in seconds,
	// written to StampDigits places between StampBefore and StampAfter.
	StampBefore, StampAfter string
	StampDigits             int32
	Before, After           []int32
	// Joined is a video one grid of its pairs by the merged rows and
	// columns, written as Before, the pairs' rows and After with no time:
	// the pairs a frame apart on the text model's temporal axis, or,
	// PerSecond, PerSecond * int(2 / r) apart (transformers' get_rope_index,
	// tokens_per_second times the seconds a pair takes, truncated).
	Joined    bool
	PerSecond int32
	// EveryFrame reads each of a video's frames, none sampled
	// (do_sample_frames false): Fps, MinFrames and MaxFrames zero, each
	// frame resized alone.
	EveryFrame bool
}

// Valid says the record is one this build can read.
func (v *Video) Valid() error {
	sampled := v.Fps > 0 && v.MinFrames >= 1 && v.MaxFrames >= v.MinFrames
	if v.EveryFrame {
		sampled = v.Fps == 0 && v.MinFrames == 0 && v.MaxFrames == 0
	}
	if !sampled || !(v.DefaultFps > 0) || v.StampDigits < 0 || v.StampDigits > 9 {
		return fmt.Errorf("jlm: the video samples %v frames a second (%v unstated), %d to %d, its times to %d places",
			v.Fps, v.DefaultFps, v.MinFrames, v.MaxFrames, v.StampDigits)
	}
	if len(v.Pre) != 2 || v.Pre[0].Kind != PreVideoResize ||
		v.Pre[1].Kind != PreNormalize && v.Pre[1].Kind != PreNormalizeBF16 || len(v.Pre[1].Floats) != 6 {
		return fmt.Errorf("jlm: the video's steps are not a resize and a normalisation")
	}
	if s := v.Pre[0].Ints; len(s) != 6 && len(s) != 7 || s[0] <= 0 || s[1] <= 0 || s[2] < s[1] ||
		!resizes(Filter(s[3]), Precision(s[4])) || s[5] < 1 || len(s) == 7 && s[6] != 0 && s[6] != 1 {
		return fmt.Errorf("jlm: the video resizes with %v", s)
	}
	if v.EveryFrame && (len(v.Pre[0].Ints) < 7 || v.Pre[0].Ints[6] != 1) {
		// Every frame of a resize over the count shrinks without a floor.
		return fmt.Errorf("jlm: a video of every frame resized over their count")
	}
	if v.PerSecond < 0 || v.PerSecond > 0 && !v.Joined {
		return fmt.Errorf("jlm: a video of %d positions a second whose pairs are not one grid", v.PerSecond)
	}
	if v.Entry.NIn != 2 || len(v.Entry.Ops) == 0 || v.Entry.Ops[0].Code != TOpPatchify {
		return fmt.Errorf("jlm: the video's entry reads %d frames and does not start with their patches", v.Entry.NIn)
	}
	return v.Entry.valid("video entry")
}

func encodeVideo(w *wbuf, v *Video) {
	w.f32(v.Fps)
	w.f32(v.DefaultFps)
	w.i32(v.MinFrames)
	w.i32(v.MaxFrames)
	w.u32(uint32(len(v.Pre)))
	for _, s := range v.Pre {
		w.u16(uint16(s.Kind))
		w.u32(uint32(len(s.Ints)))
		for _, x := range s.Ints {
			w.i32(x)
		}
		w.u32(uint32(len(s.Floats)))
		for _, x := range s.Floats {
			w.f32(x)
		}
	}
	encodeProgram(w, &v.Entry)
	w.str(v.StampBefore)
	w.str(v.StampAfter)
	w.i32(v.StampDigits)
	for _, ids := range [][]int32{v.Before, v.After} {
		w.u32(uint32(len(ids)))
		for _, id := range ids {
			w.i32(id)
		}
	}
	// The tail, written only where it states something: a video of a
	// stamp a pair is the bytes it was.
	if v.Joined || v.EveryFrame || v.PerSecond != 0 {
		var f uint32
		if v.Joined {
			f |= videoJoined
		}
		if v.EveryFrame {
			f |= videoEveryFrame
		}
		w.u32(f)
		w.i32(v.PerSecond)
	}
}

// The video tail's flags.
const (
	videoJoined = 1 << iota
	videoEveryFrame
)

func decodeVideo(r *rbuf) *Video {
	v := &Video{Fps: r.f32(), DefaultFps: r.f32(), MinFrames: r.i32(), MaxFrames: r.i32()}
	for range r.count(10) {
		s := PreStep{Kind: PreKind(r.u16())}
		for range r.count(4) {
			s.Ints = append(s.Ints, r.i32())
		}
		for range r.count(4) {
			s.Floats = append(s.Floats, r.f32())
		}
		v.Pre = append(v.Pre, s)
	}
	decodeProgram(r, &v.Entry)
	v.StampBefore, v.StampAfter, v.StampDigits = r.str(), r.str(), r.i32()
	for _, ids := range []*[]int32{&v.Before, &v.After} {
		for range r.count(4) {
			*ids = append(*ids, r.i32())
		}
	}
	if r.err == nil && len(r.b) > 0 {
		f := r.u32()
		v.Joined, v.EveryFrame, v.PerSecond = f&videoJoined != 0, f&videoEveryFrame != 0, r.i32()
		// A flag a later build states is a video this one would read wrong.
		if f&^(videoJoined|videoEveryFrame) != 0 {
			r.err = fmt.Errorf("jlm: a video stated with flags %#x this build does not read", f)
		}
	}
	return v
}
