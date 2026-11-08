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
		if audio && (got.Audio == nil || got.Audio.Tower == nil) {
			t.Fatal("the audio tower did not survive the video after it")
		}
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
}
