package model

import (
	"fmt"
	"image"
	"image/draw"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// The reference processors' resize, for the projectors whose processor is a
// transformers TorchvisionBackend: gemma3 (Gemma3ImageProcessor, bilinear) and
// internvl (GotOcr2ImageProcessor, bicubic, with tiling).
//
// Both resize a uint8 tensor with torchvision's antialiased resampler, which is
// PIL's: a separable filter whose support widens with the downscale factor,
// weights normalised to one and converted to int16 fixed point at the widest
// precision the largest weight allows, a horizontal pass to uint8 and then a
// vertical one. Only then are the values rescaled and normalised in float32.
// Every step here is that algorithm, so the result is the reference's to the
// bit -- held so by TestPreprocessMatchesTransformers -- rather than an
// interpolation that "looks right". Every pixel passes through generated code
// (engine/nn/image.go): the decoded rows' conversion, both passes and the
// normalisation's lookup. What is Go here is geometry -- the filter's taps
// and windows from the picture's sizes -- and the tables of the tower's
// constants.

// rgb8 is a decoded picture, [y][x][channel] bytes.
type rgb8 struct {
	w, h int
	px   []uint8
}

// pixelSource is a decoded picture as nn.PixelRowJIT reads it: the format,
// the chroma's horizontal shift, and each row's bytes.
type pixelSource struct {
	f   nn.PixFmt
	sub int
	// row is row y's bytes (a YCbCr picture's luma), its chroma rows from
	// the first x's sample, and that x.
	row func(y int) (src, cb, cr []uint8, x0 int)
}

// pixelSourceOf reads img in its decoder's own layout -- *image.RGBA, NRGBA,
// Gray, RGBA64 and YCbCr, which are what image/png and image/jpeg return for
// all but palette, 16-bit-gray, 64-bit-NRGBA and CMYK pictures. Those, and a
// YCbCr picture at a negative origin (whose chroma rounding is not a shift),
// are drawn into an RGBA64 first: image/draw's conversion, the same
// At().RGBA() the pixel loop read, and file decoding rather than compute.
func pixelSourceOf(img image.Image) pixelSource {
	b := img.Bounds()
	w := b.Dx()
	rows := func(pix []uint8, stride, off, per int) func(int) ([]uint8, []uint8, []uint8, int) {
		return func(y int) ([]uint8, []uint8, []uint8, int) {
			o := off + y*stride
			return pix[o : o+per*w], nil, nil, 0
		}
	}
	switch m := img.(type) {
	case *image.RGBA:
		return pixelSource{nn.PixRGBA, 0, rows(m.Pix, m.Stride, m.PixOffset(b.Min.X, b.Min.Y), 4)}
	case *image.NRGBA:
		return pixelSource{nn.PixNRGBA, 0, rows(m.Pix, m.Stride, m.PixOffset(b.Min.X, b.Min.Y), 4)}
	case *image.Gray:
		return pixelSource{nn.PixGray, 0, rows(m.Pix, m.Stride, m.PixOffset(b.Min.X, b.Min.Y), 1)}
	case *image.RGBA64:
		return pixelSource{nn.PixRGBA64, 0, rows(m.Pix, m.Stride, m.PixOffset(b.Min.X, b.Min.Y), 8)}
	case *image.YCbCr:
		if sx, sy, ok := chromaShifts(m.SubsampleRatio); ok && b.Min.X >= 0 && b.Min.Y >= 0 {
			n := (b.Max.X-1)>>sx - b.Min.X>>sx + 1
			return pixelSource{nn.PixYCbCr, sx, func(y int) ([]uint8, []uint8, []uint8, int) {
				ay := b.Min.Y + y
				yo := m.YOffset(b.Min.X, ay)
				co := (ay>>sy - b.Min.Y>>sy) * m.CStride
				return m.Y[yo : yo+w], m.Cb[co : co+n], m.Cr[co : co+n], b.Min.X
			}}
		}
	}
	d := image.NewRGBA64(image.Rect(0, 0, w, b.Dy()))
	draw.Draw(d, d.Bounds(), img, b.Min, draw.Src)
	return pixelSource{nn.PixRGBA64, 0, rows(d.Pix, d.Stride, 0, 8)}
}

// chromaShifts is a subsampling ratio's horizontal and vertical chroma
// shifts: image.YCbCr.COffset's divisions, for a non-negative origin.
func chromaShifts(r image.YCbCrSubsampleRatio) (sx, sy int, ok bool) {
	switch r {
	case image.YCbCrSubsampleRatio444:
		return 0, 0, true
	case image.YCbCrSubsampleRatio422:
		return 1, 0, true
	case image.YCbCrSubsampleRatio420:
		return 1, 1, true
	case image.YCbCrSubsampleRatio440:
		return 0, 1, true
	case image.YCbCrSubsampleRatio411:
		return 2, 0, true
	case image.YCbCrSubsampleRatio410:
		return 2, 1, true
	}
	return 0, 0, false
}

// toRGB8 decodes to 8-bit RGB. An opaque picture is its samples' high bytes,
// which is what the reference sees. A translucent one is composited over white
// as transformers' convert_to_rgb does; its rounding is not held to PIL's.
// A JPEG's samples are Go's decoder's, whose IDCT is not libjpeg-turbo's, so
// a JPEG differs from the reference by a few levels before any of this runs.
// Each row is one generated kernel (nn.PixelRowJIT), bit for bit what
// image.Image's At(x, y).RGBA() gave pixel by pixel.
func toRGB8(img image.Image) rgb8 {
	b := img.Bounds()
	p := rgb8{w: b.Dx(), h: b.Dy(), px: make([]uint8, 3*b.Dx()*b.Dy())}
	if p.w <= 0 || p.h <= 0 {
		return p
	}
	src := pixelSourceOf(img)
	for y := 0; y < p.h; y++ {
		row, cb, cr, x0 := src.row(y)
		nn.PixelRowJIT(src.f, src.sub, nn.PixOver8, p.px[3*y*p.w:3*(y+1)*p.w], nil, row, cb, cr, x0, p.w)
	}
	return p
}

// aaFilter is the resampling kernel and its support at scale one.
type aaFilter int

const (
	aaBilinear aaFilter = iota // triangle, support 1
	aaBicubic                  // Keys with a = -0.5, support 2: PIL's and torch's antialiased bicubic
)

func (f aaFilter) support() float64 {
	if f == aaBicubic {
		return 2
	}
	return 1
}

func (f aaFilter) at(x float64) float64 {
	if x < 0 {
		x = -x
	}
	if f == aaBicubic {
		const a = -0.5
		switch {
		case x < 1:
			return ((a+2)*x-(a+3))*x*x + 1
		case x < 2:
			return (((x-5)*x+8)*x - 4) * a
		}
		return 0
	}
	if x < 1 {
		return 1 - x
	}
	return 0
}

// aaTaps is one output sample's window: its first input index and its
// fixed-point weights, which sum to 1<<prec.
type aaTaps struct {
	lo int
	w  []int32
}

// pilPrec is Pillow's fixed-point precision, 22 bits whatever the weights
// (PRECISION_BITS = 32 - 8 - 2), where torchvision picks the widest an int16
// holds: the one difference between the two resamplers, and enough to move a
// sample by one step.
const pilPrec = 22

// aaWeights is the window of every output sample along an axis of in samples
// resampled to out, and the fixed-point precision they share: torchvision's
// choice, or Pillow's when pil is set.
func aaWeights(in, out int, f aaFilter, pil bool) (prec uint, taps []aaTaps) {
	los, fw := aaFloats(in, out, f)
	maxW := 0.0
	for _, ws := range fw {
		for _, w := range ws {
			maxW = max(maxW, w)
		}
	}
	// The widest precision at which the largest weight still fits an int16,
	// capped at 22 bits: torchvision's choice, so the rounding is its too.
	for prec < 21 && !pil {
		if int(0.5+maxW*float64(int(1)<<(prec+1))) >= 1<<15 {
			break
		}
		prec++
	}
	if pil {
		prec = pilPrec
	}
	taps = make([]aaTaps, out)
	for i, ws := range fw {
		t := aaTaps{lo: los[i], w: make([]int32, len(ws))}
		for j, w := range ws {
			v := w * float64(int(1)<<prec)
			if w < 0 {
				t.w[j] = int32(-0.5 + v)
			} else {
				t.w[j] = int32(0.5 + v)
			}
		}
		taps[i] = t
	}
	return prec, taps
}

// aaFloats is aaWeights before the fixed point: each output sample's first
// input index and its weights, normalised to one. PyTorch's antialiased
// interpolate on a float tensor uses them as they are.
func aaFloats(in, out int, f aaFilter) (los []int, fw [][]float64) {
	scale := float64(in) / float64(out)
	fs := max(scale, 1)
	sup, inv := f.support()*fs, 1/fs
	fw = make([][]float64, out)
	los = make([]int, out)
	for i := 0; i < out; i++ {
		center := scale * (float64(i) + 0.5)
		lo := max(int(center-sup+0.5), 0)
		hi := min(int(center+sup+0.5), in)
		ws := make([]float64, hi-lo)
		tot := 0.0
		for j := range ws {
			ws[j] = f.at((float64(j+lo) - center + 0.5) * inv)
			tot += ws[j]
		}
		if tot != 0 {
			for j := range ws {
				ws[j] /= tot
			}
		}
		fw[i], los[i] = ws, lo
	}
	return los, fw
}

// resize is the antialiased resample to ow x oh: the horizontal pass, then the
// vertical one, each rounding to uint8. An axis whose size does not change is
// not resampled at all, as in the reference.
func (p rgb8) resize(ow, oh int, f aaFilter) rgb8 { return p.resizeAt(ow, oh, f, false) }

// resizePIL is resize at Pillow's precision: Image.resize, which the PIL
// processors (MiniCPM-V's, Janus-Pro's, Phi-4-reasoning-vision's) call.
func (p rgb8) resizePIL(ow, oh int, f aaFilter) rgb8 { return p.resizeAt(ow, oh, f, true) }

func (p rgb8) resizeAt(ow, oh int, f aaFilter, pil bool) rgb8 {
	if ow != p.w {
		// The horizontal pass, a pixel at a time over every row as Pillow's
		// own loop runs: each output pixel's window of source pixels, its
		// three channels together, the bytes read and written in place.
		prec, taps := aaWeights(p.w, ow, f, pil)
		w, lo, cols := hWindows(taps, p.w)
		q := rgb8{w: ow, h: p.h, px: make([]uint8, 3*ow*p.h)}
		nn.ResampleHJIT(q.px, 3*ow, p.px, 3*p.w, p.w, p.h, w, lo, cols, prec)
		p = q
	}
	if oh != p.h {
		// The vertical pass reads a row per tap: one kernel call per output
		// row over its window's source rows.
		prec, taps := aaWeights(p.h, oh, f, pil)
		row := 3 * p.w
		q := rgb8{w: p.w, h: oh, px: make([]uint8, oh*row)}
		for y, t := range taps {
			nn.ResampleRowJIT(q.px[y*row:(y+1)*row], p.px[t.lo*row:], row, t.w, prec)
		}
		p = q
	}
	return p
}

// hWindows lays the horizontal pass's windows out as its kernel reads them:
// every window cols taps long, the longest's, its weights zero-padded and the
// window moved left where it would run past the row. The sums are integers,
// so neither changes one; lo is each window's first byte.
func hWindows(taps []aaTaps, inW int) (w []int32, lo []int64, cols int) {
	for _, t := range taps {
		cols = max(cols, len(t.w))
	}
	w, lo = make([]int32, len(taps)*cols), make([]int64, len(taps))
	for x, t := range taps {
		start := min(t.lo, inW-cols)
		copy(w[x*cols+t.lo-start:], t.w)
		lo[x] = int64(3 * start)
	}
	return w, lo, cols
}

// crop is the w x h window at (x0, y0).
func (p rgb8) crop(x0, y0, w, h int) rgb8 {
	q := rgb8{w: w, h: h, px: make([]uint8, 3*w*h)}
	for y := 0; y < h; y++ {
		copy(q.px[3*y*w:3*(y+1)*w], p.px[3*((y0+y)*p.w+x0):])
	}
	return q
}

// normalizeInto writes each sample's normalised value from normLUT's table:
// with 8-bit samples the normalisation is a function of 256 inputs per
// channel, so the per-pixel work is a lookup (nn.PixLUTJIT) and the
// arithmetic is the table.
func (p rgb8) normalizeInto(dst []float32, mean, std [3]float64) {
	lut := normLUT(mean, std)
	nn.PixLUTJIT(dst, p.px, &lut)
}

// normLUT is (v - 255*mean) / (255*std) for every 8-bit v, in float32 and in
// that order: torchvision's fused rescale-and-normalise scales mean and std by
// 1/rescale_factor in float32 and then subtracts and divides.
func normLUT(mean, std [3]float64) (lut [3][256]float32) {
	for ch := 0; ch < 3; ch++ {
		// The explicit conversions round each product before it is used: Go
		// may otherwise fuse v - mean*255 into one multiply-subtract (arm64
		// does), which is not the reference's two roundings.
		m, sd := float32(float32(mean[ch])*255), float32(float32(std[ch])*255)
		for v := range lut[ch] {
			d := float32(v) - m
			lut[ch][v] = d / sd
		}
	}
	return lut
}

// gridOf is InternVL's tiling: the (columns, rows) grid with between minT and
// maxT tiles whose aspect ratio is closest to the picture's. A tie goes to the
// later (larger) grid while the picture's area is more than half the grid's,
// in the reference's own candidate order -- every (columns, rows) pair with
// columns outer, stably sorted by tile count.
func gridOf(w, h, tile, minT, maxT int) (cols, rows int) {
	type grid struct{ c, r int }
	var gs []grid
	for c := 1; c <= maxT; c++ {
		for r := 1; r <= maxT; r++ {
			if c*r <= maxT && c*r >= minT {
				gs = append(gs, grid{c, r})
			}
		}
	}
	// A stable sort by tile count, written out: insertion keeps equal counts in
	// generation order.
	for i := 1; i < len(gs); i++ {
		for j := i; j > 0 && gs[j-1].c*gs[j-1].r > gs[j].c*gs[j].r; j-- {
			gs[j-1], gs[j] = gs[j], gs[j-1]
		}
	}
	aspect := float64(w) / float64(h)
	area := float64(w * h)
	best, bestDiff := grid{1, 1}, -1.0
	for _, g := range gs {
		d := aspect - float64(g.c)/float64(g.r)
		if d < 0 {
			d = -d
		}
		switch {
		case bestDiff < 0 || d < bestDiff:
			best, bestDiff = g, d
		case d == bestDiff && area > 0.5*float64(tile*tile*g.c*g.r):
			best = g
		}
	}
	return best.c, best.r
}

// tilesOf is how many squares a w x h picture becomes: the grid, plus a
// thumbnail of the whole picture when the grid is more than one tile. One for
// a tower that does not tile.
func (c TowerConfig) tilesOf(w, h int) int {
	if c.MaxTiles == 0 {
		return 1
	}
	if c.Kind == jlm.ProjLlama4 {
		return c.llama4Tiles(w, h)
	}
	cols, rows := gridOf(w, h, c.ImageSz, c.MinTiles, c.MaxTiles)
	if cols*rows == 1 {
		return 1
	}
	return cols*rows + 1
}

// TokensFor is how many embeddings a w x h picture becomes: Tokens per tile.
// A caller sizing a prompt before the picture is encoded asks this rather than
// Tokens, which is one tile's.
func (c TowerConfig) TokensFor(w, h int) int {
	if c.Kind == jlm.ProjLFM2 {
		return c.lfm2Tokens(w, h)
	}
	return c.tilesOf(w, h) * c.Tokens()
}

// preprocessReference is the reference processor's pixels for a projector
// that has one here: every tile, ImageSz square, [tile][y][x][channel].
func (c TowerConfig) preprocessReference(img image.Image) ([]float32, error) {
	p := toRGB8(img)
	if p.w == 0 || p.h == 0 {
		return nil, fmt.Errorf("model: Preprocess: empty image")
	}
	n := c.ImageSz
	one := n * n * 3
	switch c.Kind {
	case jlm.ProjGemma3, jlm.ProjGemma3nV:
		// Squash to the square, no crop and no pan-and-scan (the processor's
		// default, and llama.cpp's only mode). Gemma 3n's SiglipImageProcessor
		// is the same resize, its pixels [0, 1].
		out := make([]float32, one)
		p.resize(n, n, aaBilinear).normalizeInto(out, c.Mean, c.Std)
		return out, nil
	case jlm.ProjInternVL:
		cols, rows := gridOf(p.w, p.h, n, c.MinTiles, c.MaxTiles)
		tiles := c.tilesOf(p.w, p.h)
		out := make([]float32, tiles*one)
		big := p.resize(cols*n, rows*n, aaBicubic)
		// Row-major tiles, then the thumbnail: the processor's order, and the
		// order the embeddings must reach the text model in.
		for i := 0; i < cols*rows; i++ {
			big.crop((i%cols)*n, (i/cols)*n, n, n).normalizeInto(out[i*one:(i+1)*one], c.Mean, c.Std)
		}
		if tiles > cols*rows {
			p.resize(n, n, aaBicubic).normalizeInto(out[(tiles-1)*one:], c.Mean, c.Std)
		}
		if c.fault == towerFaultTilesReversed {
			for i, j := 0, tiles-1; i < j; i, j = i+1, j-1 {
				for k := 0; k < one; k++ {
					out[i*one+k], out[j*one+k] = out[j*one+k], out[i*one+k]
				}
			}
		}
		return out, nil
	}
	return nil, fmt.Errorf("model: projector %s has no reference processor here", c.Projector)
}
