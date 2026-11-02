package model

import (
	"fmt"
	"image"
	"math"
	"strconv"

	"github.com/jitllm/jitllm/format/jlm"
)

// A picture as a tower that reads any grid takes it: an overview and, when the
// picture is big enough, a grid of slices (llava-uhd, as MiniCPM-V's processor
// does it). Every piece is its own encode and its own run of Tokens()
// embeddings, and the pieces enter the prompt wrapped in the model's own
// markers. It is still one model: the pieces are rows through the same tower
// blocks, and their embeddings rows of the same prefill.

// Layout is how one picture is cut, decided from its size alone, so the
// prompt (and the instance it sizes) can be built before a pixel is resampled.
type Layout struct {
	W, H   int   // the picture's own size
	Pieces []Box // the overview first, then the slices row by row
	// Rows and Cols are the slice grid, zero when the overview is all.
	Rows, Cols int
	// RefW and RefH are the size the picture is resampled to before it is
	// cut into slices; zero when it is not cut.
	RefW, RefH int
}

// Box is one piece of a Layout: Px x Py pixels of the overview, or of the
// refined picture at (X, Y) when the piece is a slice.
type Box struct {
	X, Y, W, H int
	Slice      bool
}

// Grid is a piece's patch grid.
func (b Box) Grid(patch int) (gh, gw int) { return b.H / patch, b.W / patch }

// sliceMax is the processor's max_slice_nums (MiniCPM-V's preprocessor_config
// and llama.cpp's mtmd-image.cpp both say 9).
const sliceMax = 9

// Layout plans a w x h picture the way MiniCPM-V's own image processor does
// (image_processing_minicpmv.py), which is what the weights were trained on.
//
// Two places where llama.cpp's port (mtmd-image.cpp, llava_uhd) differs, by
// choice (RULE 7m): a picture with neither side over ImageSz is squashed to an
// ImageSz square there and kept at its aspect here; and a picture whose area
// is under ImageSz^2 but has one long side is sliced there and not here. For
// a square picture no bigger than ImageSz, and for one whose sides are whole
// slices, the two agree -- which is what the llama.cpp comparisons use.
func (t *Tower) Layout(w, h int) (Layout, error) {
	c := t.Cfg
	if c.PosBuckets == 0 {
		return Layout{}, fmt.Errorf("model: this tower reads one %dx%d grid and is not sliced", c.ImageSz, c.ImageSz)
	}
	if w <= 0 || h <= 0 {
		return Layout{}, fmt.Errorf("model: a %dx%d picture", w, h)
	}
	if c.Mean[0] != c.Mean[1] || c.Mean[1] != c.Mean[2] || c.Std[0] != c.Std[1] || c.Std[1] != c.Std[2] {
		return Layout{}, fmt.Errorf("model: a per-channel normalisation (%v, %v) is not implemented "+
			"for a sliced tower", c.Mean, c.Std)
	}
	res, patch := c.ImageSz, c.PatchSz
	lay := Layout{W: w, H: h}
	grid := slicedGrid(w, h, res)
	add := func(b Box) error {
		if gh, gw := b.Grid(patch); gh*gw > c.MaxSeq() || gh < 1 || gw < 1 {
			return fmt.Errorf("model: a %dx%d piece is %dx%d patches, past this tower's %d",
				b.W, b.H, gw, gh, c.MaxSeq())
		}
		lay.Pieces = append(lay.Pieces, b)
		return nil
	}
	if grid == nil {
		ow, oh := bestResize(w, h, res, patch, true)
		return lay, add(Box{W: ow, H: oh})
	}
	ow, oh := bestResize(w, h, res, patch, false)
	if err := add(Box{W: ow, H: oh}); err != nil {
		return lay, err
	}
	cols, rows := grid[0], grid[1]
	rw, rh := refineSize(w, h, cols, rows, res, patch)
	sw, sh := rw/cols, rh/rows
	lay.Rows, lay.Cols, lay.RefW, lay.RefH = rows, cols, rw, rh
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if err := add(Box{X: x * sw, Y: y * sh, W: sw, H: sh, Slice: true}); err != nil {
				return lay, err
			}
		}
	}
	return lay, nil
}

// ensureDivide is the processor's: the nearest whole multiple of p, at least p.
// Python's round is to even, which is what decides a .5.
func ensureDivide(length float64, p int) int {
	return max(int(math.RoundToEven(length/float64(p)))*p, p)
}

// bestResize aims at res^2 pixels at the picture's aspect, each side a whole
// number of patches (find_best_resize).
func bestResize(w, h, res, patch int, up bool) (int, int) {
	fw, fh := float64(w), float64(h)
	if w*h > res*res || up {
		r := fw / fh
		fh = math.Trunc(float64(res) / math.Sqrt(r))
		fw = math.Trunc(fh * r)
	}
	return ensureDivide(fw, patch), ensureDivide(fh, patch)
}

// refineSize is the size the picture is resampled to before slicing, a whole
// number of best-sized slices (get_refine_size, with upscaling allowed).
func refineSize(w, h, cols, rows, res, patch int) (int, int) {
	rw := ensureDivide(float64(w), cols)
	rh := ensureDivide(float64(h), rows)
	gw, gh := bestResize2(float64(rw)/float64(cols), float64(rh)/float64(rows), res, patch)
	return gw * cols, gh * rows
}

// bestResize2 is bestResize on a fractional size, which refineSize hands it:
// the processor divides without rounding first.
func bestResize2(w, h float64, res, patch int) (int, int) {
	r := w / h
	h = math.Trunc(float64(res) / math.Sqrt(r))
	w = math.Trunc(h * r)
	return ensureDivide(w, patch), ensureDivide(h, patch)
}

// slicedGrid is get_sliced_grid: {cols, rows}, or nil when the picture is not
// cut at all (its area at most res^2).
func slicedGrid(w, h, res int) []int {
	logRatio := math.Log(float64(w) / float64(h))
	ratio := float64(w) * float64(h) / float64(res*res)
	multiple := min(int(math.Ceil(ratio)), sliceMax)
	if multiple <= 1 {
		return nil
	}
	var cands [][2]int
	for _, n := range []int{multiple - 1, multiple, multiple + 1} {
		if n == 1 || n > sliceMax {
			continue
		}
		for m := 1; m <= n; m++ {
			if n%m == 0 {
				cands = append(cands, [2]int{m, n / m})
			}
		}
	}
	best, minErr := []int{1, 1}, math.Inf(1)
	for _, g := range cands {
		if e := math.Abs(logRatio - math.Log(float64(g[0])/float64(g[1]))); e < minErr {
			best, minErr = []int{g[0], g[1]}, e
		}
	}
	return best
}

// PictureSpans lays out one picture's run of the prompt in MiniCPM-V's own
// markers, each piece of lay a picture span (Tower.PicturePieces), which the
// prompt's prefill encodes:
//
//	<image_id>idx</image_id><image> OVERVIEW </image>
//	<slice> S </slice><slice> S </slice>\n<slice> S </slice>...
//
// The image id is the processor's (use_image_id in its config); llama.cpp's
// mtmd does not emit it, so the comparisons against llama-mtmd-cli build
// their prompt without it (withID false). The markers are looked up as pieces,
// never encoded, as mtmd's lookup_token does; the id's digits are text.
func (m *Model) PictureSpans(lay Layout, idx int, withID bool, pieces []*Picture) ([]Span, error) {
	tw := m.Tower()
	if tw == nil {
		return nil, fmt.Errorf("model: this model has no vision tower")
	}
	if tw.Cfg.Projector != jlm.ProjResampler.String() {
		return nil, fmt.Errorf("model: projector %s does not slice", tw.Cfg.Projector)
	}
	id := func(s string) ([]int32, error) {
		v, ok := m.Vocab.ID(s)
		if !ok {
			return nil, fmt.Errorf("model: the vocabulary has no %q, so the image markers cannot be built", s)
		}
		return []int32{v}, nil
	}
	var out []Span
	tok := func(s string) error {
		ids, err := id(s)
		if err == nil {
			out = append(out, Span{Tokens: ids})
		}
		return err
	}
	if len(pieces) != len(lay.Pieces) {
		return nil, fmt.Errorf("model: %d pictures for a layout of %d pieces", len(pieces), len(lay.Pieces))
	}
	piece := func(i int) { out = append(out, Span{Picture: pieces[i]}) }
	if withID {
		if err := tok("<image_id>"); err != nil {
			return nil, err
		}
		out = append(out, Span{Tokens: m.Vocab.Encode(strconv.Itoa(idx), false)})
		if err := tok("</image_id>"); err != nil {
			return nil, err
		}
	}
	if err := tok("<image>"); err != nil {
		return nil, err
	}
	piece(0)
	if err := tok("</image>"); err != nil {
		return nil, err
	}
	for r := 0; r < lay.Rows; r++ {
		if r > 0 {
			if err := tok("\n"); err != nil {
				return nil, err
			}
		}
		for c := 0; c < lay.Cols; c++ {
			if err := tok("<slice>"); err != nil {
				return nil, err
			}
			piece(1 + r*lay.Cols + c)
			if err := tok("</slice>"); err != nil {
				return nil, err
			}
		}
	}
	return mergeTokenSpans(out), nil
}

// mergeTokenSpans joins neighbouring token spans, which PrefillMixed would
// treat the same and which a reader of the spans finds easier to check.
func mergeTokenSpans(in []Span) []Span {
	var out []Span
	for _, sp := range in {
		if n := len(out); n > 0 && !sp.rows() && !out[n-1].rows() {
			out[n-1].Tokens = append(out[n-1].Tokens, sp.Tokens...)
			continue
		}
		out = append(out, sp)
	}
	return out
}

// Pieces resamples img into every piece of lay and normalises each: the
// overview from the picture itself, the slices cut from the picture resampled
// once to the refined size, as the processor does -- Pillow's BICUBIC, which is
// imageproc.go's resampler at Pillow's precision. Each is (Grid) pixels
// [y][x][channel].
func (t *Tower) Pieces(img image.Image, lay Layout) ([][]float32, error) {
	c := t.Cfg
	src := toRGB8(img)
	if src.w != lay.W || src.h != lay.H {
		return nil, fmt.Errorf("model: a %dx%d picture for a %dx%d layout", src.w, src.h, lay.W, lay.H)
	}
	mean, std := c.Mean, c.Std
	if c.fault == faultNoNorm {
		mean, std = [3]float64{}, [3]float64{1, 1, 1}
	}
	out := make([][]float32, len(lay.Pieces))
	var ref rgb8
	if lay.Rows > 0 {
		ref = src.resizePIL(lay.RefW, lay.RefH, aaBicubic)
	}
	for i, b := range lay.Pieces {
		var pc rgb8
		if !b.Slice {
			pc = src.resizePIL(b.W, b.H, aaBicubic)
		} else {
			pc = ref.crop(b.X, b.Y, b.W, b.H)
		}
		out[i] = make([]float32, 3*b.W*b.H)
		pc.normalizeInto(out[i], mean, std)
	}
	return out, nil
}
