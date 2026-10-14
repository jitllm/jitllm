package model

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"os"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// Image preprocessing: a decoded picture to the float vector Encode wants.
//
// image_mean and image_std come from the file; the resize filter does not, so
// it is a convention here and can only be validated against the reference
// (libmtmd's PIL-compatible resampler) on a real image. A mismatch shifts
// every patch embedding slightly without faulting. Bilinear is what most of
// these pipelines use.

// Preprocess decodes an image and produces its pixels laid out [y][x][channel],
// normalised by the tower's own mean and std: ImageSz x ImageSz for a
// fixed-size tower, and for a dynamic-resolution one (Qwen-VL) the size
// TowerConfig.ResizeTo picks, which also sets the grid Encode runs (Grid).
func (s *State) Preprocess(r io.Reader) ([]float32, error) {
	img, _, err := image.Decode(r)
	if err != nil {
		return nil, fmt.Errorf("model: Preprocess: %w", err)
	}
	return s.PreprocessImage(img)
}

// ImageSize reads a picture's width and height from its header, which is all
// TowerConfig.GridFor needs to size a prompt before the picture is decoded.
func ImageSize(path string) (w, h int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, 0, err
	}
	defer f.Close()
	cfg, _, err := image.DecodeConfig(f)
	if err != nil {
		return 0, 0, fmt.Errorf("model: %s: %w", path, err)
	}
	return cfg.Width, cfg.Height, nil
}

// PreprocessImage is Preprocess on an already-decoded image.
//
// The resize is two passes of the generated f32 matvec. Bilinear resampling
// is separable and linear: Rx (n x sw) applied to every input row, then
// Ry (n x sh) to every column of the result. Rx and Ry are geometry built once
// per image; the pixels only pass through generated code. The normalisation
// folds in: 1/(65535*std) scales Rx, and -mean/std rides an extra column of Rx
// against a constant 1 appended to every input row (Ry's rows sum to one).
// A dense matvec over a two-tap matrix is sw times a gather's work, which is
// milliseconds once per image.
func (s *State) PreprocessImage(img image.Image) ([]float32, error) {
	// The resampling filter is the reference's. transformers' Qwen-VL processor
	// resizes with Pillow's BICUBIC, and so do MiniCPM-V's and Janus-Pro's:
	// one resampler serves all of them (rgb8.resizePIL, Pillow's own fixed
	// point, 8-bit after each pass). SmolVLM's and llava's pipelines are
	// bilinear, on the generated matvec below.
	if s.vis.t.Cfg.Kind == jlm.ProjKimiVL {
		return s.preprocessKimi(toRGB8(img))
	}
	if s.vis.t.Cfg.Kind == jlm.ProjLFM2 {
		return nil, fmt.Errorf("model: an lfm2 tower takes a picture as its pieces (Tower.LFM2Pieces), not one grid")
	}
	if s.vis.t.Cfg.Dynamic() {
		return s.preprocessPIL(img)
	}
	return s.preprocessWith(img, bilinearTaps)
}

// preprocessPIL is a dynamic-resolution tower's processor: the picture resized
// to ResizeTo's grid with Pillow's BICUBIC, then rescaled and normalised. It
// sets the grid Encode runs.
func (s *State) preprocessPIL(img image.Image) ([]float32, error) {
	c := s.vis.t.Cfg
	src := toRGB8(img)
	if src.w <= 0 || src.h <= 0 {
		return nil, fmt.Errorf("model: Preprocess: empty image")
	}
	nw, nh, err := c.ResizeTo(src.w, src.h)
	if err != nil {
		return nil, err
	}
	s.vis.gh, s.vis.gw = nh/c.PatchSz, nw/c.PatchSz
	out := make([]float32, nw*nh*3)
	// Phi-4-reasoning-vision's processor resamples BILINEAR.
	f := aaBicubic
	if c.Kind == jlm.ProjPhi4 && !phi4Bicubic {
		f = aaBilinear
	}
	src.resizePIL(nw, nh, f).normalizeInto(out, c.Mean, c.Std)
	return out, nil
}

// tapsFunc builds an n x k resampling matrix from src samples (bilinearTaps).
type tapsFunc func(n, src, k int, scale, shift float32) []float32

// preprocessWith is PreprocessImage with the resampling filter named, which
// is how a gate runs the wrong one.
func (s *State) preprocessWith(img image.Image, taps tapsFunc) ([]float32, error) {
	c := s.vis.t.Cfg
	if c.Projector == jlm.ProjJanus.String() {
		return s.letterbox(img)
	}
	// gemma3 and internvl are held to their transformers processor to the bit
	// (imageproc.go); internvl's result is every tile, one after another.
	if c.Kind == jlm.ProjGemma3 || c.Kind == jlm.ProjInternVL || c.Kind == jlm.ProjGemma3nV {
		return c.preprocessReference(img)
	}
	if c.Kind == jlm.ProjLlama4 {
		px, _, _ := c.llama4Preprocess(toRGB8(img))
		return px, nil
	}
	if c.ImageSz <= 0 {
		return nil, fmt.Errorf("model: Preprocess: tower image size is %d", c.ImageSz)
	}
	b := img.Bounds()
	sw, sh := b.Dx(), b.Dy()
	if sw <= 0 || sh <= 0 {
		return nil, fmt.Errorf("model: Preprocess: empty image")
	}
	nw, nh, err := c.ResizeTo(sw, sh)
	if err != nil {
		return nil, err
	}
	if c.Dynamic() {
		s.vis.gh, s.vis.gw = nh/c.PatchSz, nw/c.PatchSz
	}

	// The pixels, raw 16-bit, as three planes of [sh][sw+1]; the last column is
	// the constant the shift multiplies. Each row is one generated kernel
	// (nn.PixelRowJIT), the 16-bit samples At(x, y).RGBA() gave.
	kx := sw + 1
	plane := make([]float32, 3*sh*kx)
	src := pixelSourceOf(img)
	for y := 0; y < sh; y++ {
		row, cb, cr, x0 := src.row(y)
		planes := [3][]float32{plane[y*kx : y*kx+sw], plane[(sh+y)*kx : (sh+y)*kx+sw], plane[(2*sh+y)*kx : (2*sh+y)*kx+sw]}
		nn.PixelRowJIT(src.f, src.sub, nn.PixPlanes, nil, &planes, row, cb, cr, x0, sw)
		for ch := 0; ch < 3; ch++ {
			plane[ch*sh*kx+y*kx+sw] = 1
		}
	}

	// Squash to the target, do not crop: SmolVLM's preprocessor resizes to a
	// square target with no centre crop. Bilinear's taps are convex and cannot
	// overshoot, so the normalisation folds into the first pass. (A Qwen-VL
	// tower's Pillow BICUBIC is preprocessPIL; it reaches this function only
	// as its gate's violation.)
	ry := taps(nh, sh, sh, 1, 0)
	mid := make([]float32, 3*sh*nw)  // [ch][y][x_out]
	midT := make([]float32, 3*nw*sh) // [ch][x_out][y]
	col := make([]float32, nh)
	out := make([]float32, nw*nh*3)
	scale, shift := channelNorms(c)
	for ch := 0; ch < 3; ch++ {
		rx := taps(nw, sw, kx, scale[ch], shift[ch])
		if err := s.resizeRows(mid[ch*sh*nw:(ch+1)*sh*nw], rx, plane[ch*sh*kx:(ch+1)*sh*kx], nw, kx, sh); err != nil {
			return nil, err
		}
		// The vertical pass reads a column as a vector: mid transposed, and
		// each column's result scattered into the interleaved output, both
		// strided copies on generated code.
		nn.Copy32JIT(midT[ch*nw*sh:(ch+1)*nw*sh], 1, sh, mid[ch*sh*nw:(ch+1)*sh*nw], nw, 1, sh, nw)
		for x := 0; x < nw; x++ {
			if !s.jit.MatVec(col, quant.F32, f32Bytes(ry), midT[(ch*nw+x)*sh:(ch*nw+x+1)*sh], nh, sh) {
				return nil, fmt.Errorf("model: Preprocess: no generated F32 matvec at k=%d", sh)
			}
			nn.Copy32JIT(out[x*3+ch:], 0, 3*nw, col, 0, 1, 1, nh)
		}
	}
	return out, nil
}

// resizeRows runs the horizontal pass, one generated matvec per source row.
func (s *State) resizeRows(dst, rx, src []float32, nw, kx, sh int) error {
	for y := 0; y < sh; y++ {
		if !s.jit.MatVec(dst[y*nw:(y+1)*nw], quant.F32, f32Bytes(rx), src[y*kx:(y+1)*kx], nw, kx) {
			return fmt.Errorf("model: Preprocess: no generated F32 matvec at k=%d", kx)
		}
	}
	return nil
}

// channelNorms is each channel's normalisation as the resize's weights carry
// it: raw 16-bit samples times scale, plus shift, is (v/65535 - mean) / std.
// Three pairs of constants per picture.
func channelNorms(c TowerConfig) (scale, shift [3]float32) {
	norm := func(m, s float64) (float32, float32) { return float32(1 / (65535 * s)), float32(-m / s) }
	scale[0], shift[0] = norm(c.Mean[0], c.Std[0])
	scale[1], shift[1] = norm(c.Mean[1], c.Std[1])
	scale[2], shift[2] = norm(c.Mean[2], c.Std[2])
	return scale, shift
}

// bilinearTaps is the n x k resampling matrix from src samples to n: row i
// holds scale*(1-w) and scale*w at the two source samples around output i's
// centre, and, when k > src, shift in column src. Pixel centres, not corners:
// sampling at (i+0.5)*src/n-0.5 is what every reference resampler does, and
// being off by half a pixel shifts the whole image.
func bilinearTaps(n, src, k int, scale, shift float32) []float32 {
	m := make([]float32, n*k)
	step := float64(src) / float64(n)
	for i := 0; i < n; i++ {
		f := (float64(i)+0.5)*step - 0.5
		i0 := int(math.Floor(f))
		w := float32(f - float64(i0))
		i1 := clampi(i0+1, 0, src-1)
		i0 = clampi(i0, 0, src-1)
		w0, w1 := scale*(1-w), scale*w
		if i0 == i1 {
			w0 = scale
		}
		m[i*k+i0] = w0
		if i1 != i0 {
			m[i*k+i1] = w1
		}
		if k > src {
			m[i*k+src] = shift
		}
	}
	return m
}

func clampi(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// letterbox is Janus-Pro's processor (transformers' JanusImageProcessorPil):
// the longest side to ImageSz at the picture's aspect, each side round()ed
// (to even on a half, as Python's round) and at least 14, Pillow BICUBIC, then
// centred on a square of the mean's colour -- int(mean*255), 127 for 0.5 --
// and normalised. A square picture is never padded.
//
// llama.cpp's fixed-size port takes ceil() of each side where the processor
// rounds; the two agree whenever the fraction is under a half, and for every
// square picture.
func (s *State) letterbox(img image.Image) ([]float32, error) {
	c := s.vis.t.Cfg
	src := toRGB8(img)
	n := c.ImageSz
	delta := float64(n) / float64(max(src.w, src.h))
	nw := max(int(math.RoundToEven(float64(src.w)*delta)), 14)
	nh := max(int(math.RoundToEven(float64(src.h)*delta)), 14)
	if nw > n || nh > n {
		return nil, fmt.Errorf("model: a %dx%d picture letterboxes to %dx%d, past %d", src.w, src.h, nw, nh, n)
	}
	res := src.resizePIL(nw, nh, aaBicubic)
	sq := rgb8{w: n, h: n, px: make([]uint8, 3*n*n)}
	// The pad colour is three constants of the tower's mean, int(mean*255) as
	// the processor takes them; the fill copies them.
	bg := [3]uint8{uint8(c.Mean[0] * 255), uint8(c.Mean[1] * 255), uint8(c.Mean[2] * 255)}
	for i := range sq.px {
		sq.px[i] = bg[i%3]
	}
	x0, y0 := (n-nw)/2, (n-nh)/2
	for y := 0; y < nh; y++ {
		copy(sq.px[3*((y0+y)*n+x0):3*((y0+y)*n+x0+nw)], res.px[3*y*nw:3*(y+1)*nw])
	}
	out := make([]float32, 3*n*n)
	sq.normalizeInto(out, c.Mean, c.Std)
	return out, nil
}
