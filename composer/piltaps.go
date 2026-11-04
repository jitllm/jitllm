package composer

import (
	"math"

	"github.com/jitllm/jitllm/graph"
)

// PictureSide is the widest side a picture resized by Pillow's filters
// (graph.ChainPIL) has: the taps a form unrolls reach a reduction from it.
const PictureSide = 4096

// pilPrecision is Pillow's fixed point for an 8-bit resample (Resample.c's
// PRECISION_BITS): 32 bits less the sample's 8 and 2 of headroom.
const pilPrecision = 32 - 8 - 2

// pilSupport is a Pillow filter's support at unit scale.
func pilSupport(filter uint8) float64 {
	if filter == graph.FilterCubic {
		return 2
	}
	return 1
}

// pilFilter is the filter's weight at x (Resample.c's bilinear_filter and
// bicubic_filter, a = -0.5).
func pilFilter(filter uint8, x float64) float64 {
	if x < 0 {
		x = -x
	}
	if filter != graph.FilterCubic {
		if x < 1 {
			return 1 - x
		}
		return 0
	}
	const a = -0.5
	switch {
	case x < 1:
		return ((a+2)*x-(a+3))*x*x + 1
	case x < 2:
		return (((x-5)*x+8)*x - 4) * a
	}
	return 0
}

// PILTapCount is the taps an output reaches resizing a side of at most in
// samples to out by filter: Pillow's ksize, its support widened by the
// reduction.
func PILTapCount(filter uint8, in, out int) int {
	scale := max(float64(in)/float64(out), 1)
	return int(math.Ceil(pilSupport(filter)*scale))*2 + 1
}

// PILTaps writes the taps of outputs from to from+n-1 of a side of in
// samples resized to out by filter into dst, n entries of k+1 words: the
// first sample an output reads, then its k weights in Pillow's fixed point
// (int32, zero past its window) -- Resample.c's precompute_coeffs and
// normalize_coeffs_8bpc, in float64 as Pillow computes them, so the passes
// the picture op runs on them are Pillow's to the bit. A window short of the
// whole side is a crop of the resize: each output's taps are its own, and
// so is the 8-bit sample the first pass leaves for it.
func PILTaps(filter uint8, in, out, from, n, k int, dst []uint32) {
	scale := float64(in) / float64(out)
	fscale := max(scale, 1)
	support := pilSupport(filter) * fscale
	ss := 1 / fscale
	for i := range n {
		o := from + i
		center := (float64(o) + 0.5) * scale
		xmin := max(int(center-support+0.5), 0)
		xmax := min(int(center+support+0.5), in) - xmin
		xmax = min(xmax, k)
		// The window's weights twice, summed and then each divided by the
		// sum, the values Pillow's one pass stores: nothing is held between.
		w := func(x int) float64 { return pilFilter(filter, (float64(x+xmin)-center+0.5)*ss) }
		ww := 0.0
		for x := range xmax {
			ww += w(x)
		}
		e := dst[i*(k+1) : (i+1)*(k+1)]
		e[0] = uint32(xmin)
		for x := range k {
			v := 0.0
			if x < xmax && ww != 0 {
				v = w(x) / ww
			}
			if v < 0 {
				e[1+x] = uint32(int32(-0.5 + v*(1<<pilPrecision)))
			} else {
				e[1+x] = uint32(int32(0.5 + v*(1<<pilPrecision)))
			}
		}
	}
}
