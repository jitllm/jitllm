package model

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"image"
	"math"
	"strconv"
	"unsafe"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// VideoPlan is how the tower reads a video (jlm.Video): the frames it
// samples, an odd count's last repeated, two to a picture of the tower; the
// square both frames of a pair are resized to; and each pair's time as the
// prompt writes it.
type VideoPlan struct {
	Frames []int
	Square Square
	Stamps []string
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
	r := v.Pre[0].Ints
	pair := int(v.Entry.NIn)
	rate := fps
	if !(rate > 0) {
		rate = float64(v.DefaultFps)
	}
	k := min(max(int(float64(n)/rate*float64(v.Fps)), int(v.MinFrames)), int(v.MaxFrames), n)
	if k < int(r[5]) {
		return VideoPlan{}, fmt.Errorf("model: a video of %d frames samples %d, fewer than the %d a picture of the tower takes", n, k, r[5])
	}
	var p VideoPlan
	// numpy's linspace(0, n-1, k), rounded half to even.
	for i := range k {
		at := float64(n-1) * float64(i) / float64(k-1)
		if k > 1 {
			at = float64(i) * (float64(n-1) / float64(k-1))
		}
		if i == k-1 {
			at = float64(n - 1)
		}
		p.Frames = append(p.Frames, int(math.RoundToEven(at)))
	}
	for len(p.Frames)%pair != 0 {
		p.Frames = append(p.Frames, p.Frames[len(p.Frames)-1])
	}
	for i := 0; i < len(p.Frames); i += pair {
		t := (float64(p.Frames[i])/rate + float64(p.Frames[i+pair-1])/rate) / 2
		p.Stamps = append(p.Stamps, strconv.FormatFloat(t, 'f', int(v.StampDigits), 64))
	}
	ow, oh, err := videoSize(k, w, h, r)
	if err != nil {
		return VideoPlan{}, err
	}
	p.Square = tg.VideoSquare(ow, oh)
	return p, nil
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
// w x h to.
func videoSize(k, w, h int, r []int32) (int, int, error) {
	f, minPix, maxPix, t := float64(r[0]), float64(r[1]), float64(r[2]), float64(r[5])
	fh, fw := float64(h), float64(w)
	if h < int(r[0]) || w < int(r[0]) {
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
		root := int(math.Sqrt(float64(r[2]) / float64(v.MaxFrames)))
		a.Taps = composer.PILTapCount(a.Filter, 2*max(composer.PictureMaxSide, root), root)
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

// VideoSpans is the spans a video of frames at fps frames a second (0 where
// the caller states none) stands in a prompt as (jlm.Video): Before, then
// for each pair of the frames it samples the pair's time and the tower's
// span around the pair's rows, then After. The frames are one size, as the
// reference stacks them.
func (m *Model) VideoSpans(frames []image.Image, fps float64) ([]Span, error) {
	tg := m.Tower
	if tg == nil || tg.Video == nil {
		return nil, fmt.Errorf("model: the model reads no video")
	}
	if len(frames) == 0 {
		return nil, fmt.Errorf("model: a video of no frames")
	}
	sz := frames[0].Bounds().Size()
	for i, f := range frames {
		if f.Bounds().Size() != sz {
			return nil, fmt.Errorf("model: frame %d of a video is %v and frame 0 %v", i, f.Bounds().Size(), sz)
		}
	}
	p, err := tg.VideoPlan(len(frames), sz.X, sz.Y, fps)
	if err != nil {
		return nil, err
	}
	v, t := tg.Video, m.Container.Vision().Tower
	rows := tg.SquareOut(p.Square)
	var grid ImageGrid
	if tg.GridDiv > 0 {
		grid = ImageGrid{T: 1, H: p.Square.Gh / tg.GridDiv, W: p.Square.Gw / tg.GridDiv, Line: tg.Line, Ends: tg.Ends}
	}
	toks := append([]int32(nil), v.Before...)
	var out []Span
	for i, st := range p.Stamps {
		f0, f1 := frames[p.Frames[2*i]], frames[p.Frames[2*i+1]]
		k, err := pairKey(f0, f1, p.Square)
		if err != nil {
			return nil, err
		}
		toks = append(toks, m.Vocab.Encode(v.StampBefore+st+v.StampAfter, false)...)
		toks = append(toks, t.Span.Before...)
		out = append(out, Span{Tokens: toks},
			Span{Picture: f0, Frame: f1, FrameW: p.Square.Ow, FrameH: p.Square.Oh, PicRows: rows, Grid: grid, Key: k})
		toks = append(append([]int32(nil), t.Span.After...), nil...)
	}
	return append(out, Span{Tokens: append(toks, v.After...)}), nil
}

// pairKey names a video pair by everything the tower reads of it: both
// frames, as pictureKey names one, and the square they are resized to.
func pairKey(f0, f1 image.Image, sq Square) (*RowsKey, error) {
	hs := sha256.New()
	hs.Write([]byte("video"))
	var sz [8]byte
	binary.LittleEndian.PutUint32(sz[:], uint32(sq.Ow))
	binary.LittleEndian.PutUint32(sz[4:], uint32(sq.Oh))
	hs.Write(sz[:])
	for _, f := range []image.Image{f0, f1} {
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
