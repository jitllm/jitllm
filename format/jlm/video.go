package jlm

import (
	"fmt"
	"math"
)

// Video is how a tower reads a video (Tower.Video; docs/design/
// video-statement.md): which frames it samples, how many frames one picture
// of the tower reads, how they are sized, and what the prompt writes around
// them -- every one stated, none a processor's kind.
type Video struct {
	// DefaultFps is the rate a request that states none is read at, 0
	// where nothing the video states reads the rate.
	DefaultFps float32
	// Count is how many frames are sampled, of the video's frames N and
	// rate R (operands VideoN, VideoRate); Index is sampled frame i's index,
	// of N, R, the count K and i (VideoK, VideoAt), truncated toward zero
	// and held to [0, N-1]. Once drops a frame already sampled. MaxFrames
	// is the most Count gives, 0 for no bound (the frames then sized
	// alone).
	Count, Index Expr
	Once         bool
	MaxFrames    int32
	// Group is the frames one picture of the tower reads, an odd count's
	// last repeated to fill the last; Sized is the count the frames' resize
	// reads, of the sampled count K and the count filled to whole groups
	// (operands 0 and 1).
	Group int32
	Sized Expr
	// Pre and Entry are the frames' own resize and entry, a group of two
	// frames read by the entry as the conv3d's two planes; neither stated,
	// each frame (a group of one) is read as a picture is, at the picture's
	// last square.
	Pre   []PreStep
	Entry Program
	// Frames is how a frame read as a picture becomes the tower's squares
	// where its steps are its own, nil where they are the picture's. It is
	// written after the picture's geometry, in the vision section's tail.
	Frames *PictureGeometry
	// Stride, where stated, makes the groups one grid, group j at j times
	// its value on the text model's temporal axis, of N and R; where not,
	// each group is a picture span of its own.
	Stride *Expr
	// Prompt is what the prompt writes for the video, in order.
	Prompt []VideoPiece
}

// The operands of a video's expressions: Count reads VideoN and VideoRate,
// Index those and VideoK, Count's value, and VideoAt; a prompt value all of
// them but VideoAt, VideoK the frames sampled once Once has dropped its,
// and of the group it is in its number and its first and last frames'
// indices and seconds (0 outside a group), and the video's seconds N / R.
const (
	VideoN = iota
	VideoRate
	VideoK
	VideoAt
	VideoGroup
	VideoF0
	VideoT0
	VideoF1
	VideoT1
	VideoSeconds
	VideoOperands
)

// PieceKind is what a video's prompt piece writes.
type PieceKind uint8

const (
	// PieceIDs is its IDs, as stated: a marker no text encodes to.
	PieceIDs PieceKind = iota + 1
	// PieceText is its Text, encoded with the text beside it.
	PieceText
	// PieceValue is Value's value as Format writes it, encoded as text.
	PieceValue
	// PieceGroups is Body once per group, Between before every one but the
	// first.
	PieceGroups
	// PieceSpan is the group's rows.
	PieceSpan
)

// VideoPiece is one piece of a video's prompt (Video.Prompt).
type VideoPiece struct {
	Kind    PieceKind
	IDs     []int32
	Text    string
	Value   Expr
	Format  ValueFormat
	Body    []VideoPiece
	Between []VideoPiece
}

// FormatKind is how a value is written.
type FormatKind uint8

const (
	// FormatDecimal writes it to Digits places: correctly rounded, or,
	// Trunc, cut to them first.
	FormatDecimal FormatKind = iota + 1
	// FormatClock writes it as a count of seconds: Clock's fields (hours,
	// minutes, seconds), each of a width w > 0 written as an integer of w
	// digits at least, Sep between them, a field of width 0 not written and
	// its count carried into the one below; the first written field takes
	// all of the count above it, but where Days states its words (for one
	// day, for several) the days are written before the hours, the words
	// after their number, a day of none not written. The seconds are the
	// count's integer part.
	FormatClock
	// FormatTable writes Strings[int(value)].
	FormatTable
)

// ValueFormat is how a prompt value is written (VideoPiece.Format).
type ValueFormat struct {
	Kind    FormatKind
	Digits  int32
	Trunc   bool
	Clock   [3]int32
	Days    [2]string
	Sep     string
	Strings []string
}

// Valid says the record is one this build can read.
func (v *Video) Valid() error {
	if !(v.DefaultFps >= 0) || v.MaxFrames < 0 || v.Group < 1 || v.Group > 2 {
		return fmt.Errorf("jlm: the video is read at %v frames a second where unstated, at most %d frames, %d to a picture",
			v.DefaultFps, v.MaxFrames, v.Group)
	}
	for _, e := range []struct {
		name string
		x    *Expr
		in   int
	}{{"count", &v.Count, VideoK}, {"index", &v.Index, VideoAt + 1}, {"sized count", &v.Sized, 2}} {
		if err := e.x.ValidOver(e.in); err != nil {
			return fmt.Errorf("jlm: the video's %s: %w", e.name, err)
		}
		if err := scalar(e.x); err != nil {
			return fmt.Errorf("jlm: the video's %s: %w", e.name, err)
		}
	}
	if v.Stride != nil {
		if err := v.Stride.ValidOver(VideoRate + 1); err != nil {
			return fmt.Errorf("jlm: the video's stride: %w", err)
		}
		if err := scalar(v.Stride); err != nil {
			return fmt.Errorf("jlm: the video's stride: %w", err)
		}
	}
	if err := piecesValid(v.Prompt, false); err != nil {
		return err
	}
	if v.AsPictures() {
		// Its frames take the picture's steps; a resize of their own is
		// their geometry's (Frames).
		if v.Group != 1 || v.Stride != nil || len(v.Pre) != 0 {
			return fmt.Errorf("jlm: a video read as pictures of %d frames, one grid %v, %d steps of its own",
				v.Group, v.Stride != nil, len(v.Pre))
		}
		if v.Frames != nil {
			return v.Frames.Valid()
		}
		return nil
	}
	// Its frames' size is their geometry's: one square, of the frames
	// evaluated with the count they share the area among (PicFrames).
	if v.Frames == nil {
		return fmt.Errorf("jlm: a video of an entry of its own states no geometry of its frames")
	}
	if err := v.Frames.Valid(); err != nil {
		return fmt.Errorf("jlm: the video's frames: %w", err)
	}
	// Its classes are the squares its frames' resize reaches, in its unit.
	if v.Frames.Bounds.Unit < 1 {
		return fmt.Errorf("jlm: a video of an entry of its own sizes its frames in a unit of %d", v.Frames.Bounds.Unit)
	}
	if len(v.Pre) != 2 || v.Pre[0].Kind != PreResizeLive || len(v.Pre[0].Ints) != 2 ||
		!resizes(Filter(v.Pre[0].Ints[0]), Precision(v.Pre[0].Ints[1])) ||
		v.Pre[1].Kind != PreNormalize && v.Pre[1].Kind != PreNormalizeBF16 || len(v.Pre[1].Floats) != 6 {
		return fmt.Errorf("jlm: the video's steps are not a resize to the frames' size and a normalisation")
	}
	if v.Group != 2 || v.Entry.NIn != 2 || len(v.Entry.Ops) == 0 || v.Entry.Ops[0].Code != TOpPatchify {
		return fmt.Errorf("jlm: the video's entry reads %d frames of a group of %d and does not start with their patches",
			v.Entry.NIn, v.Group)
	}
	return v.Entry.valid("video entry")
}

// AsPictures reports a video whose frames are each read as a picture is,
// through the picture's entry and steps: it states neither of its own, its
// frames' own resize, where they have one, their geometry's (Frames).
func (v *Video) AsPictures() bool {
	return v.Entry.NIn == 0 && len(v.Entry.Ops) == 0
}

// scalar refuses an expression of sizes that reads a stored vector, the
// row's width or a reduction: it is a number of the video, not of a row.
func scalar(e *Expr) error {
	for k, t := range e.Terms {
		if t.Reduce {
			return fmt.Errorf("jlm: term %d reduces a row", k)
		}
		for _, a := range t.Args {
			if a.Kind == ArgStored || a.Kind == ArgWidth {
				return fmt.Errorf("jlm: term %d reads a row's %d operand", k, a.Kind)
			}
		}
	}
	return nil
}

func piecesValid(ps []VideoPiece, inGroup bool) error {
	for i, p := range ps {
		switch p.Kind {
		case PieceIDs:
			if len(p.IDs) == 0 {
				return fmt.Errorf("jlm: the video's piece %d writes no ids", i)
			}
		case PieceText:
			if p.Text == "" {
				return fmt.Errorf("jlm: the video's piece %d writes no text", i)
			}
		case PieceValue:
			if err := p.Value.ValidOver(VideoOperands); err != nil {
				return fmt.Errorf("jlm: the video's piece %d: %w", i, err)
			}
			if err := scalar(&p.Value); err != nil {
				return fmt.Errorf("jlm: the video's piece %d: %w", i, err)
			}
			if err := p.Format.valid(); err != nil {
				return fmt.Errorf("jlm: the video's piece %d: %w", i, err)
			}
		case PieceGroups:
			if inGroup {
				return fmt.Errorf("jlm: the video's piece %d repeats the groups within a group", i)
			}
			if err := piecesValid(p.Body, true); err != nil {
				return err
			}
			if err := piecesValid(p.Between, true); err != nil {
				return err
			}
		case PieceSpan:
			if !inGroup {
				return fmt.Errorf("jlm: the video's piece %d is a group's rows outside its groups", i)
			}
		default:
			return fmt.Errorf("jlm: the video's piece %d is of kind %d, which this build does not read", i, p.Kind)
		}
	}
	return nil
}

func (f *ValueFormat) valid() error {
	switch f.Kind {
	case FormatDecimal:
		if f.Digits < 0 || f.Digits > 9 {
			return fmt.Errorf("jlm: a value written to %d places", f.Digits)
		}
	case FormatClock:
		any := false
		for _, w := range f.Clock {
			if w < 0 || w > 4 {
				return fmt.Errorf("jlm: a clock of fields %v", f.Clock)
			}
			any = any || w > 0
		}
		if !any || (f.Days[0] == "") != (f.Days[1] == "") || f.Days[0] != "" && f.Clock[0] == 0 {
			return fmt.Errorf("jlm: a clock of fields %v and days %q", f.Clock, f.Days)
		}
	case FormatTable:
		if len(f.Strings) == 0 {
			return fmt.Errorf("jlm: a value written by a table of no strings")
		}
	default:
		return fmt.Errorf("jlm: a value written by format %d, which this build does not read", f.Kind)
	}
	return nil
}

// videoRecord is the record's revision: the word before it is a NaN where
// the record of a build before it read the frame rate, so that build's
// Valid refuses it rather than read it.
const videoRecord = 2

func encodeVideo(w *wbuf, v *Video) {
	w.u32(math.Float32bits(float32(math.NaN())))
	w.u32(videoRecord)
	w.f32(v.DefaultFps)
	encodeExpr(w, &v.Count)
	encodeExpr(w, &v.Index)
	var fl uint8
	if v.Once {
		fl |= 1
	}
	if v.Stride != nil {
		fl |= 2
	}
	w.u8(fl)
	w.i32(v.MaxFrames)
	w.i32(v.Group)
	encodeExpr(w, &v.Sized)
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
	if v.Stride != nil {
		encodeExpr(w, v.Stride)
	}
	encodePieces(w, v.Prompt)
}

func encodePieces(w *wbuf, ps []VideoPiece) {
	w.u32(uint32(len(ps)))
	for _, p := range ps {
		w.u8(uint8(p.Kind))
		switch p.Kind {
		case PieceIDs:
			w.u32(uint32(len(p.IDs)))
			for _, id := range p.IDs {
				w.i32(id)
			}
		case PieceText:
			w.str(p.Text)
		case PieceValue:
			encodeExpr(w, &p.Value)
			f := &p.Format
			w.u8(uint8(f.Kind))
			w.i32(f.Digits)
			var tr uint8
			if f.Trunc {
				tr = 1
			}
			w.u8(tr)
			for _, x := range f.Clock {
				w.i32(x)
			}
			w.str(f.Days[0])
			w.str(f.Days[1])
			w.str(f.Sep)
			w.u32(uint32(len(f.Strings)))
			for _, s := range f.Strings {
				w.str(s)
			}
		case PieceGroups:
			encodePieces(w, p.Body)
			encodePieces(w, p.Between)
		}
	}
}

func decodeVideo(r *rbuf) *Video {
	if m := r.u32(); r.err == nil && m != math.Float32bits(float32(math.NaN())) {
		r.err = fmt.Errorf("jlm: a video stated as a build before this one stated it: reconvert the model")
		return nil
	}
	if rev := r.u32(); r.err == nil && rev != videoRecord {
		r.err = fmt.Errorf("jlm: a video record of revision %d, where this build reads %d", rev, videoRecord)
		return nil
	}
	v := &Video{DefaultFps: r.f32()}
	decodeExpr(r, &v.Count)
	decodeExpr(r, &v.Index)
	fl := r.u8()
	v.Once = fl&1 != 0
	v.MaxFrames, v.Group = r.i32(), r.i32()
	decodeExpr(r, &v.Sized)
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
	if fl&2 != 0 {
		v.Stride = &Expr{}
		decodeExpr(r, v.Stride)
	}
	v.Prompt = decodePieces(r, 0)
	if fl&^3 != 0 && r.err == nil {
		r.err = fmt.Errorf("jlm: a video stated with flags %#x this build does not read", fl)
	}
	return v
}

func decodePieces(r *rbuf, depth int) []VideoPiece {
	n := r.count(1)
	if depth > 1 && n > 0 {
		r.err = fmt.Errorf("jlm: a video's prompt nested %d deep", depth)
		return nil
	}
	var ps []VideoPiece
	for range n {
		p := VideoPiece{Kind: PieceKind(r.u8())}
		switch p.Kind {
		case PieceIDs:
			for range r.count(4) {
				p.IDs = append(p.IDs, r.i32())
			}
		case PieceText:
			p.Text = r.str()
		case PieceValue:
			decodeExpr(r, &p.Value)
			f := &p.Format
			f.Kind, f.Digits, f.Trunc = FormatKind(r.u8()), r.i32(), r.u8() != 0
			for i := range f.Clock {
				f.Clock[i] = r.i32()
			}
			f.Days[0], f.Days[1], f.Sep = r.str(), r.str(), r.str()
			for range r.count(4) {
				f.Strings = append(f.Strings, r.str())
			}
		case PieceGroups:
			p.Body = decodePieces(r, depth+1)
			p.Between = decodePieces(r, depth+1)
		}
		if r.err != nil {
			return nil
		}
		ps = append(ps, p)
	}
	return ps
}
