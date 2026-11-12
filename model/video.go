package model

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"image"
	"math"
	"slices"
	"strconv"
	"unsafe"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// VideoSample is what a video's statement samples (SampleVideo): the video
// of N frames read at Rate frames a second, the frames sampled -- K of them,
// then the last repeated to fill a group -- and, one grid, the positions its
// groups stand apart.
type VideoSample struct {
	N      int
	Rate   float64
	Frames []int
	K      int
	Stride int
}

// SampleVideo evaluates v's count, indices and stride for a video of n
// frames at fps frames a second, 0 where the caller states none: each
// frame once where v says so, the last repeated to fill a group.
func SampleVideo(v *jlm.Video, n int, fps float64) (VideoSample, error) {
	if n < 1 {
		return VideoSample{}, fmt.Errorf("model: a video of %d frames", n)
	}
	s := VideoSample{N: n, Rate: fps}
	if !(s.Rate > 0) {
		s.Rate = float64(v.DefaultFps)
	}
	var in [jlm.VideoOperands]float64
	in[jlm.VideoN], in[jlm.VideoRate] = float64(n), s.Rate
	c, err := evalSize(&v.Count, in[:])
	if err != nil {
		return VideoSample{}, fmt.Errorf("model: the video's count: %w", err)
	}
	if !(c >= 1 && (v.MaxFrames == 0 || c <= float64(v.MaxFrames))) {
		return VideoSample{}, fmt.Errorf("model: a video of %d frames at %v a second samples %v, where it states 1 to %d",
			n, s.Rate, c, v.MaxFrames)
	}
	k := int(c)
	in[jlm.VideoK] = float64(k)
	for i := range k {
		in[jlm.VideoAt] = float64(i)
		x, err := evalSize(&v.Index, in[:])
		if err != nil {
			return VideoSample{}, fmt.Errorf("model: the video's index: %w", err)
		}
		if math.IsNaN(x) {
			return VideoSample{}, fmt.Errorf("model: the video's frame %d of %d is at %v", i, k, x)
		}
		f := int(min(max(math.Trunc(x), 0), float64(n-1)))
		if v.Once && slices.Contains(s.Frames, f) {
			continue
		}
		s.Frames = append(s.Frames, f)
	}
	s.K = len(s.Frames)
	for len(s.Frames)%int(v.Group) != 0 {
		s.Frames = append(s.Frames, s.Frames[len(s.Frames)-1])
	}
	if v.Stride != nil {
		x, err := evalSize(v.Stride, in[:])
		if err != nil {
			return VideoSample{}, fmt.Errorf("model: the video's stride: %w", err)
		}
		s.Stride = int(x)
	}
	return s, nil
}

// VideoFrameSize is the size v's own resize gives sample s's frames of
// w x h pixels, its count the one v states the resize reads.
func VideoFrameSize(v *jlm.Video, s VideoSample, w, h int) (int, int, error) {
	if v.AsPictures() {
		return 0, 0, fmt.Errorf("model: a video read as pictures is sized as a picture is")
	}
	sized, err := evalSize(&v.Sized, []float64{float64(s.K), float64(len(s.Frames))})
	if err != nil {
		return 0, 0, fmt.Errorf("model: the video's sized count: %w", err)
	}
	r := v.Pre[0].Ints
	if per := int(r[5]); !(len(r) > 6 && r[6] == 1) && int(sized) < per {
		return 0, 0, fmt.Errorf("model: a video of %d frames samples %d, fewer than the %d a picture of the tower takes", s.N, int(sized), per)
	}
	return videoSize(int(sized), w, h, r)
}

// VideoPlan is how the tower reads a video (jlm.Video): what it samples, and
// the square a group's frames are resized to, or a frame's last square
// where it is read as a picture.
type VideoPlan struct {
	VideoSample
	Square Square
}

// VideoPlan is the plan for a video of n frames of w x h pixels at fps
// frames a second, 0 where the caller states none.
func (tg *TowerGraphs) VideoPlan(n, w, h int, fps float64) (VideoPlan, error) {
	v := tg.Video
	if v == nil {
		return VideoPlan{}, fmt.Errorf("model: the tower reads no video")
	}
	if n < 1 || w < 1 || h < 1 {
		return VideoPlan{}, fmt.Errorf("model: a video of %d frames of %dx%d", n, w, h)
	}
	s, err := SampleVideo(v, n, fps)
	if err != nil {
		return VideoPlan{}, err
	}
	if v.AsPictures() {
		sq, err := tg.FrameSquare(w, h)
		if err != nil {
			return VideoPlan{}, err
		}
		return VideoPlan{VideoSample: s, Square: sq}, nil
	}
	ow, oh, err := VideoFrameSize(v, s, w, h)
	if err != nil {
		return VideoPlan{}, err
	}
	return VideoPlan{VideoSample: s, Square: tg.VideoSquare(ow, oh)}, nil
}

// evalSize is a stated expression of a video's or a picture's sizes
// (jlm.Video's count, indices, stride and prompt values, a picture's
// geometry) at operands in, in float64 as the processors compute them in
// numpy, torch and Python: geometry from the frames' and pictures' sizes,
// once a video or a picture, not arithmetic a token runs.
func evalSize(e *jlm.Expr, in []float64) (float64, error) {
	v, _, err := evalInto(e, in, nil)
	return v, err
}

// evalInto is evalSize with vals the terms' scratch, grown and returned.
func evalInto(e *jlm.Expr, in, vals []float64) (float64, []float64, error) {
	if cap(vals) < len(e.Terms) {
		vals = make([]float64, len(e.Terms))
	}
	vals = vals[:len(e.Terms)]
	v, err := evalTerms(e, in, vals)
	return v, vals, err
}

// arg is operand a's value, the terms' values so far vals.
func arg(a jlm.Arg, in, vals []float64) float64 {
	switch a.Kind {
	case jlm.ArgIn:
		return in[a.Index]
	case jlm.ArgTerm:
		return vals[a.Index]
	case jlm.ArgExact:
		return a.Exact
	}
	return float64(a.Value)
}

func evalTerms(e *jlm.Expr, in, vals []float64) (float64, error) {
	for k, t := range e.Terms {
		if t.Reduce {
			return 0, fmt.Errorf("model: term %d of a size reduces a row", k)
		}
		a, b, c := arg(t.Args[0], in, vals), arg(t.Args[1], in, vals), arg(t.Args[2], in, vals)
		var v float64
		switch t.Fn {
		case jlm.FnAdd:
			v = a + b
		case jlm.FnSub:
			v = a - b
		case jlm.FnMul:
			v = a * b
		case jlm.FnDiv:
			v = a / b
		case jlm.FnMax:
			v = max(a, b)
		case jlm.FnMin:
			v = min(a, b)
		case jlm.FnSqrt:
			v = math.Sqrt(a)
		case jlm.FnWhere:
			v = c
			if a > 0 {
				v = b
			}
		case jlm.FnLess:
			if a < b {
				v = 1
			}
		case jlm.FnEqual:
			if a == b {
				v = 1
			}
		case jlm.FnFma:
			v = math.FMA(a, b, c)
		case jlm.FnAbs:
			v = math.Abs(a)
		case jlm.FnRound:
			v = math.Round(a)
		case jlm.FnFloor:
			v = math.Floor(a)
		case jlm.FnCeil:
			v = math.Ceil(a)
		case jlm.FnTrunc:
			v = math.Trunc(a)
		case jlm.FnRoundEven:
			v = math.RoundToEven(a)
		case jlm.FnF32:
			v = float64(float32(a))
		case jlm.FnRecip:
			v = 1 / a
		case jlm.FnLog:
			v = math.Log(a)
		case jlm.FnExp:
			v = math.Exp(a)
		default:
			return 0, fmt.Errorf("model: term %d of a size is %v, not a function of sizes", k, t.Fn)
		}
		vals[k] = v
	}
	return vals[len(vals)-1], nil
}

// formatValue is v as f writes it in a video's prompt.
func formatValue(f *jlm.ValueFormat, v float64) (string, error) {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return "", fmt.Errorf("model: a video's prompt writes a value of %v", v)
	}
	switch f.Kind {
	case jlm.FormatDecimal:
		if f.Trunc {
			u := math.Pow(10, float64(f.Digits))
			v = math.Trunc(v*u) / u
		}
		return strconv.FormatFloat(v, 'f', int(f.Digits), 64), nil
	case jlm.FormatClock:
		if v < 0 {
			return "", fmt.Errorf("model: a clock of %v seconds", v)
		}
		rem, out := int64(v), ""
		if f.Days[0] != "" {
			if d := rem / 86400; d > 0 {
				out = strconv.FormatInt(d, 10) + f.Days[min(d-1, 1)]
			}
			rem %= 86400
		}
		sep := ""
		for i, unit := range [3]int64{3600, 60, 1} {
			w := int(f.Clock[i])
			if w == 0 {
				continue
			}
			n := rem / unit
			rem -= n * unit
			out += sep + fmt.Sprintf("%0*d", w, n)
			sep = f.Sep
		}
		return out, nil
	case jlm.FormatTable:
		i := int(math.Trunc(v))
		if !(v >= 0) || i >= len(f.Strings) {
			return "", fmt.Errorf("model: a value of %v past its table of %d", v, len(f.Strings))
		}
		return f.Strings[i], nil
	}
	return "", fmt.Errorf("model: a value written by format %d", f.Kind)
}

// VideoSquare is a video pair's square, its frames resized to ow x oh: the
// smallest video class whose frame holds it, else the largest.
func (tg *TowerGraphs) VideoSquare(ow, oh int) Square {
	c := 0
	for c < len(tg.Videos)-1 && !tg.Videos[c].Pic.Holds(ow, oh) {
		c++
	}
	return Square{Class: c, Ow: ow, Oh: oh, Pw: ow, Ph: oh, Gw: ow / tg.Patch, Gh: oh / tg.Patch}
}

// videoSize is the size jlm.PreVideoResize resizes k sampled frames of
// w x h to: each frame alone where the step says so (Ints[6]), the count
// then read as one and no side lifted.
func videoSize(k, w, h int, r []int32) (int, int, error) {
	f, minPix, maxPix, t := float64(r[0]), float64(r[1]), float64(r[2]), float64(r[5])
	alone := len(r) > 6 && r[6] == 1
	if alone {
		k, t = 1, 1
	}
	fh, fw := float64(h), float64(w)
	if !alone && (h < int(r[0]) || w < int(r[0])) {
		s := max(f/fh, f/fw)
		fh, fw = math.Trunc(fh*s), math.Trunc(fw*s)
	}
	if max(fh, fw)/min(fh, fw) > 200 {
		return 0, 0, fmt.Errorf("model: a video of %dx%d frames is past the aspect of 200 its resize reads", w, h)
	}
	hb, wb := math.RoundToEven(fh/f)*f, math.RoundToEven(fw/f)*f
	tb := math.RoundToEven(float64(k)/t) * t
	switch {
	case tb*hb*wb > maxPix:
		beta := math.Sqrt(float64(k) * fh * fw / maxPix)
		hb, wb = max(f, math.Floor(fh/beta/f)*f), max(f, math.Floor(fw/beta/f)*f)
	case tb*hb*wb < minPix:
		beta := math.Sqrt(minPix / (float64(k) * fh * fw))
		hb, wb = math.Ceil(fh*beta/f)*f, math.Ceil(fw*beta/f)*f
	}
	return int(wb), int(hb), nil
}

// videoEntry is the video's entry for frames of sample format px: each
// frame through the resize the video states (graph.OpPicture, the second
// frame's from the request's frame word), then the entry program over both
// frames' rows, grid patch rows and extra rows beside them, for frames at
// most side pixels a side. The taps reach the deepest shrink the video's
// bounds allow: a frame of the most frames' share of the area.
func videoEntry(t *jlm.Tower, px0, px1 uint8, grid, extra, side int, f graph.Facts) (*graph.Graph, error) {
	v := t.Video
	r := v.Pre[0].Ints
	a := graph.Attrs{Pixels: px0, PicW: composer.PictureMaxSide, PicH: composer.PictureMaxSide, PicArea: side * side, PicLive: true}
	if err := resizeChain(&a, jlm.Filter(r[3])); err != nil {
		return nil, err
	}
	a.PicTorch = jlm.Precision(r[4]) == jlm.PrecTorch
	n := v.Pre[1]
	for c := range 3 {
		a.Mean[c], a.Std[c] = float64(n.Floats[c]), float64(n.Floats[3+c])
	}
	a.PicBF16 = n.Kind == jlm.PreNormalizeBF16
	if a.Chain == graph.ChainPIL {
		// A frame sized alone is a picture of its own size (its area's
		// bound its own); else the most frames share the area.
		root := int(math.Sqrt(float64(r[2]) / float64(v.MaxFrames)))
		if len(r) > 6 && r[6] == 1 {
			root = int(math.Sqrt(float64(r[2])))
		}
		a.Taps = composer.LiveTaps(a.Filter, root)
		// Its frames run in bands (towerClass.Band); the entry finishes
		// their sums.
		a.PicPass = graph.PicFinish
	}
	P := int(v.Entry.Ops[0].Span)
	b := graph.NewBlock(-1)
	f0 := b.Emit(graph.OpPicture, nil, []graph.Result{{Name: "frame", Width: 3 * P * P}}, nil, a, nil, nil)[0]
	a.Pixels, a.PicFrame = px1, true
	f1 := b.Emit(graph.OpPicture, nil, []graph.Result{{Name: "frame.next", Width: 3 * P * P}}, nil, a, nil, nil)[0]
	out, err := LowerProgram(b, t, &v.Entry, []graph.ValueID{f0, f1}, grid+extra, f)
	if err != nil {
		return nil, err
	}
	b.Out(out)
	return b.G, graph.Verify(b.G, f)
}

// VideoParts is a video as a prompt places it: its spans, and the text the
// video's prompt writes before its first ids or rows and after its last,
// apart, for the caller to encode with the prompt's text beside them, as a
// processor expands its placeholder into a string before tokenizing it.
type VideoParts struct {
	Lead  string
	Spans []Span
	Tail  string
}

// VideoSpans is the spans a video of frames at fps frames a second (0 where
// the caller states none) stands in a prompt as (jlm.Video), its text at
// either end encoded alone (VideoParts).
func (m *Model) VideoSpans(frames []image.Image, fps float64) ([]Span, error) {
	vp, err := m.VideoParts(frames, fps)
	if err != nil {
		return nil, err
	}
	out := vp.Spans
	if lead := m.encodeText(vp.Lead); len(lead) > 0 {
		out = append([]Span{{Tokens: lead}}, out...)
	}
	if tail := m.encodeText(vp.Tail); len(tail) > 0 {
		out = append(out, Span{Tokens: tail})
	}
	return out, nil
}

// VideoParts is the video of frames at fps frames a second as its prompt
// writes it (jlm.Video.Prompt): its pieces in order, the text between two
// ids or rows encoded as one, as the reference tokenizes it, each group's
// rows a picture span. The frames are one size, as the reference stacks
// them.
func (m *Model) VideoParts(frames []image.Image, fps float64) (VideoParts, error) {
	tg := m.Tower
	if tg == nil || tg.Video == nil {
		return VideoParts{}, fmt.Errorf("model: the model reads no video")
	}
	if len(frames) == 0 {
		return VideoParts{}, fmt.Errorf("model: a video of no frames")
	}
	sz := frames[0].Bounds().Size()
	for i, f := range frames {
		if f.Bounds().Size() != sz {
			return VideoParts{}, fmt.Errorf("model: frame %d of a video is %v and frame 0 %v", i, f.Bounds().Size(), sz)
		}
	}
	p, err := tg.VideoPlan(len(frames), sz.X, sz.Y, fps)
	if err != nil {
		return VideoParts{}, err
	}
	items, err := VideoLayout(tg.Video, p.VideoSample)
	if err != nil {
		return VideoParts{}, err
	}
	rows := tg.SquareOut(p.Square)
	var grid ImageGrid
	if tg.GridDiv > 0 {
		grid = ImageGrid{T: 1, H: p.Square.Gh / tg.GridDiv, W: p.Square.Gw / tg.GridDiv, Line: tg.Line, Ends: tg.Ends}
	}
	var out VideoParts
	var toks []int32
	for i, it := range items {
		switch {
		case it.Group >= 0:
			if len(toks) > 0 {
				out.Spans = append(out.Spans, Span{Tokens: toks})
			}
			toks = nil
			sp, err := videoSpan(tg.Video, p, frames, it.Group, rows, grid)
			if err != nil {
				return VideoParts{}, err
			}
			// A group's rows see each other as a picture's do.
			sp.Bidir = m.Container.Vision().Tower.Span.Bidir
			out.Spans = append(out.Spans, sp)
		case it.IDs != nil:
			toks = append(toks, it.IDs...)
		case i == 0:
			out.Lead = it.Text
		case i == len(items)-1:
			out.Tail = it.Text
		default:
			toks = append(toks, m.encodeText(it.Text)...)
		}
	}
	if len(toks) > 0 {
		out.Spans = append(out.Spans, Span{Tokens: toks})
	}
	return out, nil
}

// VideoItem is one item of a video's prompt laid out (VideoLayout): text,
// ids, or the rows of group Group (-1 for text or ids).
type VideoItem struct {
	Text  string
	IDs   []int32
	Group int
}

// VideoLayout is v's prompt for sample s, its pieces in order: each value
// written, the groups' bodies repeated with their operands, and the text
// between two ids or rows joined into one item, as the reference tokenizes
// it.
func VideoLayout(v *jlm.Video, s VideoSample) ([]VideoItem, error) {
	l := videoLayout{v: v, s: s}
	l.in[jlm.VideoN], l.in[jlm.VideoRate] = float64(s.N), s.Rate
	l.in[jlm.VideoK], l.in[jlm.VideoSeconds] = float64(s.K), float64(s.N)/s.Rate
	if err := l.pieces(v.Prompt); err != nil {
		return nil, err
	}
	l.text()
	return l.out, nil
}

// videoLayout lays a video's prompt out (VideoLayout): the text pending
// until the next ids or rows, and the operands its values read.
type videoLayout struct {
	v       *jlm.Video
	s       VideoSample
	in      [jlm.VideoOperands]float64
	pending string
	out     []VideoItem
}

// text ends the text pending before ids or rows, an item of its own.
func (l *videoLayout) text() {
	if l.pending != "" {
		l.out = append(l.out, VideoItem{Text: l.pending, Group: -1})
	}
	l.pending = ""
}

func (l *videoLayout) pieces(ps []jlm.VideoPiece) error {
	for _, pc := range ps {
		switch pc.Kind {
		case jlm.PieceIDs:
			l.text()
			l.out = append(l.out, VideoItem{IDs: pc.IDs, Group: -1})
		case jlm.PieceText:
			l.pending += pc.Text
		case jlm.PieceValue:
			x, err := evalSize(&pc.Value, l.in[:])
			if err != nil {
				return err
			}
			s, err := formatValue(&pc.Format, x)
			if err != nil {
				return err
			}
			l.pending += s
		case jlm.PieceGroups:
			if err := l.groups(pc); err != nil {
				return err
			}
		case jlm.PieceSpan:
			l.text()
			l.out = append(l.out, VideoItem{Group: int(l.in[jlm.VideoGroup])})
		}
	}
	return nil
}

// groups lays piece pc's body out once per group, its Between before every
// one but the first, each with its group's operands.
func (l *videoLayout) groups(pc jlm.VideoPiece) error {
	g := int(l.v.Group)
	for j := range len(l.s.Frames) / g {
		if j > 0 {
			if err := l.pieces(pc.Between); err != nil {
				return err
			}
		}
		f0, f1 := l.s.Frames[j*g], l.s.Frames[j*g+g-1]
		l.in[jlm.VideoGroup] = float64(j)
		l.in[jlm.VideoF0], l.in[jlm.VideoT0] = float64(f0), float64(f0)/l.s.Rate
		l.in[jlm.VideoF1], l.in[jlm.VideoT1] = float64(f1), float64(f1)/l.s.Rate
		if err := l.pieces(pc.Body); err != nil {
			return err
		}
	}
	for _, k := range []int{jlm.VideoGroup, jlm.VideoF0, jlm.VideoT0, jlm.VideoF1, jlm.VideoT1} {
		l.in[k] = 0
	}
	return nil
}

// videoSpan is group j's rows: a frame read as a picture, or a group read
// by the video's entry; one frame of a grid of every group where the video
// states a stride.
func videoSpan(v *jlm.Video, p VideoPlan, frames []image.Image, j, rows int, grid ImageGrid) (Span, error) {
	g := int(v.Group)
	if v.AsPictures() {
		f := frames[p.Frames[j]]
		k, err := videoKey(p.Square, f)
		if err != nil {
			return Span{}, err
		}
		return Span{Picture: f, Whole: true, PicRows: rows, Grid: grid, Key: k}, nil
	}
	f0, f1 := frames[p.Frames[j*g]], frames[p.Frames[j*g+g-1]]
	k, err := videoKey(p.Square, f0, f1)
	if err != nil {
		return Span{}, err
	}
	if v.Stride != nil && !grid.Zero() {
		grid.T, grid.Frame, grid.Frames, grid.Stride = len(p.Frames)/g, j, 1, p.Stride
	}
	return Span{Picture: f0, Frame: f1, FrameW: p.Square.Ow, FrameH: p.Square.Oh, PicRows: rows, Grid: grid, Key: k}, nil
}

// encodeText is text's tokens with no specials, none for no text (a
// tokenizer's dummy prefix would otherwise make one of nothing).
func (m *Model) encodeText(text string) []int32 {
	if text == "" {
		return nil
	}
	return m.Vocab.Encode(text, false)
}

// videoKey names a video's picture -- a pair, or a frame read whole -- by
// everything the tower reads of it: its frames, as pictureKey names one,
// and the square they are resized to.
func videoKey(sq Square, frames ...image.Image) (*RowsKey, error) {
	hs := sha256.New()
	hs.Write([]byte("video"))
	var sz [8]byte
	binary.LittleEndian.PutUint32(sz[:], uint32(sq.Ow))
	binary.LittleEndian.PutUint32(sz[4:], uint32(sq.Oh))
	hs.Write(sz[:])
	for _, f := range frames {
		k, err := pictureKey(f)
		if err != nil {
			return nil, err
		}
		hs.Write(k[:])
	}
	var k RowsKey
	hs.Sum(k[:0])
	return &k, nil
}

// BandKey is the band graph a frame's picture op in sample format px takes
// (towerClass.Band): a video pair's second frame (frame 1) its own.
func BandKey(frame int, px uint8) uint8 { return uint8(frame)<<4 | px }

// ColsKey is the band graph a band key's column bands run (graph.PicCols),
// and ColBandKey the band graph that then reads their sums
// (graph.Attrs.PicColumns): a picture whose windows outrun its taps runs
// these in place of the band graph.
func ColsKey(band uint8) uint8 { return band | 1<<6 }

// ColBandKey is ColsKey's row band.
func ColBandKey(band uint8) uint8 { return band | 1<<7 }

// ColumnPic is picture op a as its column graphs run it (ColsKey,
// ColBandKey): their row bands are of at most Taps rows, so their
// horizontal sums hold that many.
func ColumnPic(a graph.Attrs) graph.Attrs {
	a.PicSide = min(a.PicSide, a.Taps)
	return a
}

// columnBands adds band graph key's column graphs to bands (ColsKey,
// ColBandKey).
func columnBands(bands map[uint8]*graph.Graph, key uint8, f graph.Facts) error {
	g := bands[key]
	o := g.Ops[0]
	for _, c := range []struct {
		key     uint8
		pass    uint8
		columns bool
	}{{ColsKey(key), graph.PicCols, false}, {ColBandKey(key), graph.PicBand, true}} {
		a := ColumnPic(o.Attr)
		a.PicPass, a.PicColumns = c.pass, c.columns
		b := graph.NewBlock(-1)
		b.Out(b.Emit(graph.OpPicture, nil, []graph.Result{{Name: g.Values[o.Out[0]].Name, Width: 3}}, nil, a, nil, nil)[0])
		if err := graph.Verify(b.G, f); err != nil {
			return err
		}
		bands[c.key] = b.G
	}
	return nil
}

// PairKey is the entry a video pair of frames in sample formats px0 and px1
// takes (TowerGraphs.Videos): every ordered pair of formats is an entry of
// its own, each frame's picture op reading its own layout.
func PairKey(px0, px1 uint8) uint8 { return px0<<4 | px1 }

// PairWords is a video pair's frames as the step's picture table reads
// them -- the first frame's planes, then the second's from the word after
// its last -- each frame's layout and the byte the second starts at. The
// frames are one size, as the reference stacks them, and may decode to
// different layouts. The frames' palettes are written into ent0 and ent1,
// the table into pad, which grows to hold it.
func PairWords(f0, f1 image.Image, pad *[]uint32, ent0, ent1 *[4 * 256]byte) ([]uint32, PictureLayout, PictureLayout, int, error) {
	l, p0, err := picturePlanes(f0, ent0)
	if err != nil {
		return nil, l, l, 0, err
	}
	l1, p1, err := picturePlanes(f1, ent1)
	if err != nil {
		return nil, l, l1, 0, err
	}
	if l.W != l1.W || l.H != l1.H {
		return nil, l, l1, 0, fmt.Errorf("model: a video's frames are %dx%d and %dx%d; a pair is one size", l.W, l.H, l1.W, l1.H)
	}
	end := func(l PictureLayout, p [3][]byte) int {
		switch {
		case p[2] != nil:
			return l.p3 + len(p[2])
		case p[1] != nil:
			return l.p2 + len(p[1])
		}
		return len(p[0])
	}
	off := (end(l, p0) + 3) &^ 3
	n := (off + end(l1, p1) + 3) / 4
	if n > composer.PictureWords {
		return nil, l, l1, 0, fmt.Errorf("model: a video pair of %d bytes is past the %d the picture table holds", 4*n, 4*composer.PictureWords)
	}
	if cap(*pad) < n {
		*pad = make([]uint32, n)
	}
	words := (*pad)[:n]
	b := unsafe.Slice((*byte)(unsafe.Pointer(&words[0])), 4*n)
	for i, f := range [2]struct {
		l PictureLayout
		p [3][]byte
	}{{l, p0}, {l1, p1}} {
		at := b[i*off:]
		copy(at, f.p[0])
		copy(at[f.l.p2:], f.p[1])
		copy(at[f.l.p3:], f.p[2])
	}
	return words, l, l1, off, nil
}
