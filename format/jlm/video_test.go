package jlm

import (
	"math"
	"reflect"
	"testing"
)

// in is the expression whose value is operand i.
func in(i uint32) Expr { return Expr{Terms: []Term{{Fn: FnAdd, Args: [3]Arg{Operand(i), Num(0)}}}} }

// testVideo is a video of pairs: two frames a second, 4 to 768 of them,
// resized together, each pair's mean time to one place and its span.
func testVideo() *Video {
	var c, mean ExprBuilder
	c.Fn(FnMin, c.Fn(FnMax, c.Fn(FnTrunc, c.Fn(FnMul, c.Fn(FnDiv, Operand(VideoN), Operand(VideoRate)), Num(2))), Num(4)), Num(768))
	mean.Fn(FnDiv, mean.Fn(FnAdd, Operand(VideoT0), Operand(VideoT1)), Num(2))
	return &Video{
		DefaultFps: 24, Count: Expr{Terms: c.T}, Index: in(VideoAt), MaxFrames: 768, Group: 2, Sized: in(0),
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
		Prompt: []VideoPiece{{Kind: PieceIDs, IDs: []int32{248053}}, {Kind: PieceGroups, Body: []VideoPiece{
			{Kind: PieceText, Text: "<"},
			{Kind: PieceValue, Value: Expr{Terms: mean.T}, Format: ValueFormat{Kind: FormatDecimal, Digits: 1}},
			{Kind: PieceText, Text: " seconds>"},
			{Kind: PieceSpan},
		}}, {Kind: PieceIDs, IDs: []int32{248054}}},
	}
}

// testGridVideo reads every frame, each sized alone, the pairs one grid
// two positions a second apart.
func testGridVideo() *Video {
	v := testVideo()
	v.Count, v.Index, v.MaxFrames = in(VideoN), in(VideoAt), 0
	v.Pre[0].Ints = []int32{28, 100352, 602112, int32(FilterPILBicubic), int32(PrecTorch), 2, 1}
	var s ExprBuilder
	s.Fn(FnMul, Num(2), s.Fn(FnTrunc, s.Fn(FnDiv, Num(2), Operand(VideoRate))))
	v.Stride = &Expr{Terms: s.T}
	v.Prompt = []VideoPiece{{Kind: PieceIDs, IDs: []int32{1}}, {Kind: PieceGroups, Body: []VideoPiece{{Kind: PieceSpan}}},
		{Kind: PieceIDs, IDs: []int32{2}}}
	return v
}

// testPictureVideo reads each frame as a picture: its count in words from
// a table, its length as a clock with days, and each frame numbered from 1
// and timed as minutes and seconds, a line between one frame and the next.
func testPictureVideo() *Video {
	var j1 ExprBuilder
	j1.Fn(FnAdd, Operand(VideoGroup), Num(1))
	return &Video{Count: in(VideoN), Index: in(VideoAt), Group: 1, Sized: in(0), Prompt: []VideoPiece{
		{Kind: PieceText, Text: "A video of "},
		{Kind: PieceValue, Value: in(VideoK), Format: ValueFormat{Kind: FormatTable, Strings: []string{"zero", "one", "two"}}},
		{Kind: PieceText, Text: " frames, "},
		{Kind: PieceValue, Value: in(VideoSeconds), Format: ValueFormat{Kind: FormatClock, Clock: [3]int32{1, 2, 2}, Sep: ":",
			Days: [2]string{" day, ", " days, "}}},
		{Kind: PieceGroups, Between: []VideoPiece{{Kind: PieceText, Text: "\n"}}, Body: []VideoPiece{
			{Kind: PieceText, Text: "Frame"},
			{Kind: PieceValue, Value: Expr{Terms: j1.T}, Format: ValueFormat{Kind: FormatDecimal}},
			{Kind: PieceValue, Value: in(VideoT0), Format: ValueFormat{Kind: FormatClock, Clock: [3]int32{0, 2, 2}, Sep: ":"}},
			{Kind: PieceIDs, IDs: []int32{7}}, {Kind: PieceSpan}, {Kind: PieceIDs, IDs: []int32{8}},
		}},
		{Kind: PieceText, Text: "\n\n"},
	}}
}

// TestVideoRoundTrips writes a stated tower's video record into the vision
// section's tail and reads the same record back, behind an audio tower's
// and with none; a section with no video reads back with none. A record of
// the build before this one's statement, and one of a flag this build does
// not read, are refused rather than read.
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
		for name, vid := range map[string]*Video{"pairs": testVideo(), "grid": testGridVideo(), "pictures": testPictureVideo()} {
			v.Tower.Video = vid
			if got, err = decodeVision(encodeVision(v)); err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			if !reflect.DeepEqual(got.Tower.Video, v.Tower.Video) {
				t.Fatalf("audio %v: the %s video read back as\n%+v\nnot\n%+v", audio, name, got.Tower.Video, v.Tower.Video)
			}
		}
		if audio && (got.Audio == nil || got.Audio.Tower == nil) {
			t.Fatal("the audio tower did not survive the video after it")
		}
	}
	var w wbuf
	encodeVideo(&w, testGridVideo())
	b := append([]byte(nil), w.b...)
	// The build before read a rate where this record's NaN is.
	b[0], b[1], b[2], b[3] = 0, 0, 0x40, 0x40
	r := rbuf{b: b}
	if decodeVideo(&r); r.err == nil {
		t.Fatal("a record of the statement before this one was read")
	}
	// The flags are the byte after the record's revision, rate and count
	// and index expressions.
	b = append([]byte(nil), w.b...)
	r = rbuf{b: b}
	r.u32()
	r.u32()
	r.f32()
	var e Expr
	decodeExpr(&r, &e)
	decodeExpr(&r, &e)
	b[len(b)-len(r.b)] |= 4
	r = rbuf{b: b}
	if decodeVideo(&r); r.err == nil {
		t.Fatal("a video of a flag this build does not read was read")
	}
	if math.Float32bits(float32(math.NaN())) != 0x7fc00000 {
		t.Fatal("the marker is not the quiet NaN the record states")
	}
}

// TestVideoRefusesWhatItCannotRead holds Valid to each field it bounds: a
// record broken in one of them is refused.
func TestVideoRefusesWhatItCannotRead(t *testing.T) {
	for name, v := range map[string]*Video{"pairs": testVideo(), "grid": testGridVideo(), "pictures": testPictureVideo()} {
		if err := v.Valid(); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
	}
	for name, breaks := range map[string]func(*Video){
		"a negative rate":            func(v *Video) { v.DefaultFps = -1 },
		"groups of three":            func(v *Video) { v.Group = 3 },
		"a count of the index":       func(v *Video) { v.Count = in(VideoAt) },
		"an index of a group":        func(v *Video) { v.Index = in(VideoGroup) },
		"a sized count of three":     func(v *Video) { v.Sized = in(2) },
		"a stride of the count":      func(v *Video) { s := in(VideoK); v.Stride = &s },
		"a count of a stored vector": func(v *Video) { v.Count.Terms[0].Args[0] = Arg{Kind: ArgStored, Role: RoleVPosEmbd} },
		"a count reduced":            func(v *Video) { v.Count.Terms[0].Reduce = true },
		"no resize":                  func(v *Video) { v.Pre = v.Pre[1:] },
		"unknown filter":             func(v *Video) { v.Pre[0].Ints[3] = 99 },
		"one frame read":             func(v *Video) { v.Entry.NIn = 1 },
		"no patches":                 func(v *Video) { v.Entry.Ops[0].Code = TOpAdd },
		"unread value":               func(v *Video) { v.Entry.Ops[2].In = []uint16{3, 9} },
		"sized together, unbounded":  func(v *Video) { v.MaxFrames = 0 },
		"digits":                     func(v *Video) { v.Prompt[1].Body[1].Format.Digits = 12 },
		"an unknown format":          func(v *Video) { v.Prompt[1].Body[1].Format.Kind = 9 },
		"an unknown piece":           func(v *Video) { v.Prompt[0].Kind = 9 },
		"no ids":                     func(v *Video) { v.Prompt[0].IDs = nil },
		"no text":                    func(v *Video) { v.Prompt[1].Body[0].Text = "" },
		"rows outside a group":       func(v *Video) { v.Prompt = append(v.Prompt, VideoPiece{Kind: PieceSpan}) },
		"groups within a group": func(v *Video) {
			v.Prompt[1].Body = append(v.Prompt[1].Body, VideoPiece{Kind: PieceGroups})
		},
	} {
		v := testVideo()
		breaks(v)
		if v.Valid() == nil {
			t.Errorf("%s: Valid accepted it", name)
		}
	}
	for name, breaks := range map[string]func(*Video){
		"pictures in one grid":  func(v *Video) { v.Stride = testGridVideo().Stride },
		"pictures in pairs":     func(v *Video) { v.Group = 2 },
		"a clock of no fields":  func(v *Video) { v.Prompt[3].Format.Clock = [3]int32{} },
		"days with no hours":    func(v *Video) { v.Prompt[3].Format.Clock[0] = 0 },
		"one word for days":     func(v *Video) { v.Prompt[3].Format.Days[1] = "" },
		"an empty table":        func(v *Video) { v.Prompt[1].Format.Strings = nil },
		"a field of five":       func(v *Video) { v.Prompt[3].Format.Clock[1] = 5 },
		"a value of an operand": func(v *Video) { v.Prompt[1].Value = in(VideoOperands) },
	} {
		v := testPictureVideo()
		breaks(v)
		if v.Valid() == nil {
			t.Errorf("%s: Valid accepted it", name)
		}
	}
}

// TestPictureVideoStatesNoStepsOfItsOwn holds a video read as pictures to
// the picture's steps: a resize of its frames' own is their geometry
// (Frames), held to the geometry's check, and a step of its own is refused.
func TestPictureVideoStatesNoStepsOfItsOwn(t *testing.T) {
	for name, c := range map[string]struct {
		edit func(v *Video, tw *Tower)
		ok   bool
	}{
		"the picture's":         {func(v *Video, tw *Tower) {}, true},
		"a geometry of its own": {func(v *Video, tw *Tower) { v.Frames = testGeometry() }, true},
		"a step of its own": {func(v *Video, tw *Tower) {
			v.Pre = []PreStep{{Kind: PreResizeLive, Ints: []int32{int32(FilterPILBicubic), int32(PrecPIL)}}}
		}, false},
		"a geometry below a unit": {func(v *Video, tw *Tower) {
			v.Frames = testGeometry()
			v.Frames.Bounds.Unit = -1
		}, false},
	} {
		tw := sampleTower(t)
		v := testPictureVideo()
		c.edit(v, tw)
		tw.Video = v
		if err := tw.Valid(); (err == nil) != c.ok {
			t.Errorf("%s: the tower's Valid says %v", name, err)
		}
	}
}
