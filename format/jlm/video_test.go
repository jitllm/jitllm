package jlm

import (
	"reflect"
	"testing"
)

func testVideo() *Video {
	return &Video{
		Fps: 2, DefaultFps: 24, MinFrames: 4, MaxFrames: 768,
		Pre: []PreStep{
			{Kind: PreVideoResize, Ints: []int32{32, 4096, 25165824, int32(FilterPILBicubic), int32(PrecTorch), 2}},
			{Kind: PreNormalize, Floats: []float32{0.5, 0.5, 0.5, 0.5, 0.5, 0.5}},
		},
		Entry: Program{NIn: 2, Out: 7, Ops: []TowerOp{
			{Code: TOpPatchify, In: []uint16{0}, Span: 16, Width: 2},
			{Code: TOpPatchify, In: []uint16{1}, Span: 16, Width: 2},
			{Code: TOpSub, In: []uint16{3, 2}},
			{Code: TOpLinear, In: []uint16{2}, Bind: []Role{RoleVPatchEmbd, RoleVPatchBias}},
			{Code: TOpLinear, In: []uint16{4}, Bind: []Role{RoleVPatchEmbd2}},
			{Code: TOpAdd, In: []uint16{5, 6}},
		}},
		StampBefore: "<", StampAfter: " seconds>", StampDigits: 1,
		Before: []int32{248053}, After: []int32{248054},
	}
}

// testJoinedVideo is Qwen2.5-VL's: every frame, each sized alone, the
// pairs one grid two positions a second apart.
func testJoinedVideo() *Video {
	v := testVideo()
	v.Fps, v.MinFrames, v.MaxFrames = 0, 0, 0
	v.EveryFrame, v.Joined, v.PerSecond = true, true, 2
	v.Pre[0].Ints = []int32{28, 100352, 602112, int32(FilterPILBicubic), int32(PrecTorch), 2, 1}
	v.StampBefore, v.StampAfter, v.StampDigits = "", "", 0
	return v
}

// testStepVideo is GLM-4V's: two frames a second in whole steps over at
// most 300 seconds, each pair's first frame's whole seconds after it.
func testStepVideo() *Video {
	v := testVideo()
	v.MinFrames, v.MaxFrames, v.Steps, v.MaxSeconds = 1, 600, true, 300
	v.StampFirst, v.StampLast = true, true
	v.StampBefore, v.StampAfter, v.StampDigits = "", "", 0
	return v
}

// testWholeVideo is InternVL's: every frame read whole as the picture's
// last square, numbered from 1, a line between one frame and the next.
func testWholeVideo() *Video {
	return &Video{EveryFrame: true, Whole: true, StampIndex: true, StampBefore: "Frame", StampAfter: ": ", Between: "\n"}
}

// TestVideoRoundTrips writes a stated tower's video record into the vision
// section's tail and reads the same record back, behind an audio tower's
// and with none; a section with no video reads back with none.
func TestVideoRoundTrips(t *testing.T) {
	for _, audio := range []bool{false, true} {
		v := &Vision{NLayer: 12, NEmbd: 768, NHead: 12, NFFN: 3072, ImageSize: 512, PatchSize: 16,
			ProjDim: 576, Scale: 4, Eps: 1e-6, Projector: ProjIdefics3, Tower: sampleTower(t)}
		if audio {
			v.Audio = &Audio{NLayer: 1, NEmbd: 8, NHead: 2, NFFN: 16, ProjDim: 8, MaxSamples: 480000, Tower: sampleTower(t)}
		}
		got, err := decodeVision(encodeVision(v))
		if err != nil {
			t.Fatal(err)
		}
		if got.Tower.Video != nil {
			t.Fatalf("audio %v: a section with no video read one back", audio)
		}
		v.Tower.Video = testVideo()
		if got, err = decodeVision(encodeVision(v)); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got.Tower.Video, v.Tower.Video) {
			t.Fatalf("audio %v: the video read back as\n%+v\nnot\n%+v", audio, got.Tower.Video, v.Tower.Video)
		}
		v.Tower.Video = testJoinedVideo()
		if got, err = decodeVision(encodeVision(v)); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got.Tower.Video, v.Tower.Video) {
			t.Fatalf("audio %v: the joined video read back as\n%+v\nnot\n%+v", audio, got.Tower.Video, v.Tower.Video)
		}
		v.Tower.Video = testStepVideo()
		if got, err = decodeVision(encodeVision(v)); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got.Tower.Video, v.Tower.Video) {
			t.Fatalf("audio %v: the stepped video read back as\n%+v\nnot\n%+v", audio, got.Tower.Video, v.Tower.Video)
		}
		v.Tower.Video = testWholeVideo()
		if got, err = decodeVision(encodeVision(v)); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got.Tower.Video, v.Tower.Video) {
			t.Fatalf("audio %v: the whole frames read back as\n%+v\nnot\n%+v", audio, got.Tower.Video, v.Tower.Video)
		}
		if audio && (got.Audio == nil || got.Audio.Tower == nil) {
			t.Fatal("the audio tower did not survive the video after it")
		}
	}
	// A flag this build does not know is refused, not read past: the
	// tail's flags are the word before PerSecond, the first past the known
	// its second byte's lowest bit.
	var w wbuf
	encodeVideo(&w, testJoinedVideo())
	w.b[len(w.b)-7] |= 1
	r := rbuf{b: w.b}
	decodeVideo(&r)
	if r.err == nil {
		t.Fatal("a video of a flag this build does not read was read")
	}
}

// TestVideoRefusesWhatItCannotRead holds Valid to each field it bounds: a
// record broken in one of them is refused.
func TestVideoRefusesWhatItCannotRead(t *testing.T) {
	if err := testVideo().Valid(); err != nil {
		t.Fatal(err)
	}
	for name, breaks := range map[string]func(*Video){
		"no rate":         func(v *Video) { v.Fps = 0 },
		"frames reversed": func(v *Video) { v.MinFrames, v.MaxFrames = 8, 4 },
		"no resize":       func(v *Video) { v.Pre = v.Pre[1:] },
		"unknown filter":  func(v *Video) { v.Pre[0].Ints[3] = 99 },
		"one frame":       func(v *Video) { v.Entry.NIn = 1 },
		"no patches":      func(v *Video) { v.Entry.Ops[0].Code = TOpAdd },
		"unread value":    func(v *Video) { v.Entry.Ops[2].In = []uint16{3, 9} },
		"digits":          func(v *Video) { v.StampDigits = 12 },
	} {
		v := testVideo()
		breaks(v)
		if v.Valid() == nil {
			t.Errorf("%s: Valid accepted it", name)
		}
	}
	if err := testJoinedVideo().Valid(); err != nil {
		t.Fatal(err)
	}
	for name, breaks := range map[string]func(*Video){
		"every frame sampled":     func(v *Video) { v.Fps = 2 },
		"every frame bounded":     func(v *Video) { v.MaxFrames = 768 },
		"every frame by count":    func(v *Video) { v.Pre[0].Ints[6] = 0 },
		"every frame, count read": func(v *Video) { v.Pre[0].Ints = v.Pre[0].Ints[:6] },
		"alone of three kinds":    func(v *Video) { v.Pre[0].Ints[6] = 2 },
		"negative per second":     func(v *Video) { v.PerSecond = -1 },
		"per second, not joined":  func(v *Video) { v.Joined = false },
	} {
		v := testJoinedVideo()
		breaks(v)
		if v.Valid() == nil {
			t.Errorf("%s: Valid accepted it", name)
		}
	}
	if err := testStepVideo().Valid(); err != nil {
		t.Fatal(err)
	}
	for name, breaks := range map[string]func(*Video){
		"steps of every frame":   func(v *Video) { v.EveryFrame = true },
		"steps over no seconds":  func(v *Video) { v.MaxSeconds = 0 },
		"seconds with no steps":  func(v *Video) { v.Steps = false },
		"a time in one grid":     func(v *Video) { v.Joined = true },
		"a first time unsampled": func(v *Video) { v.Fps = 0 },
		"pairs numbered":         func(v *Video) { v.StampFirst, v.StampIndex = false, true },
		"pairs parted":           func(v *Video) { v.Between = "\n" },
	} {
		v := testStepVideo()
		breaks(v)
		if v.Valid() == nil {
			t.Errorf("%s: Valid accepted it", name)
		}
	}
	if err := testWholeVideo().Valid(); err != nil {
		t.Fatal(err)
	}
	for name, breaks := range map[string]func(*Video){
		"whole frames resized":     func(v *Video) { v.Pre = testVideo().Pre },
		"whole frames' own entry":  func(v *Video) { v.Entry = testVideo().Entry },
		"whole frames in one grid": func(v *Video) { v.Joined = true },
		"numbered and timed":       func(v *Video) { v.StampFirst = true },
		"sampled with no rate":     func(v *Video) { v.EveryFrame, v.Fps, v.MinFrames, v.MaxFrames = false, 2, 1, 8 },
	} {
		v := testWholeVideo()
		breaks(v)
		if v.Valid() == nil {
			t.Errorf("%s: Valid accepted it", name)
		}
	}
}
