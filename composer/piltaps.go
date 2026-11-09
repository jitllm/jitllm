package composer

import (
	"math"

	"github.com/jitllm/jitllm/graph"
)

// PictureSide is the widest side a picture resized by Pillow's filters
// (graph.ChainPIL) has: the taps a form unrolls reach a reduction from it.
const PictureSide = 4096

// PictureMaxSide is the widest side a picture resized by Pillow's filters
// may have: JPEG's, its sizes 16-bit. The taps a resize reads reach a
// shrink from it; a taller picture than PictureSide is its bands
// (graph.Attrs.PicPass).
const PictureMaxSide = 65535

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

// PicHCap is the pixels a Pillow resize's horizontal sums hold: PicSide
// source rows of the op's width, or for a picture of its own size of the
// width of its area's square -- a wider output's band holds fewer rows
// (PicBandRows).
func PicHCap(a graph.Attrs) int {
	if a.PicArea == 0 {
		return a.PicSide * a.PicW
	}
	s := int(math.Sqrt(float64(a.PicArea)))
	for s*s > a.PicArea {
		s--
	}
	for (s+1)*(s+1) <= a.PicArea {
		s++
	}
	return a.PicSide * min(a.PicW, s)
}

// PicBandRows is the source rows a band of a Pillow resize to an output ow
// wide holds: PicSide, or as many as its horizontal sums hold (PicHCap).
func PicBandRows(a graph.Attrs, ow int) int {
	ow = min(max(ow, 1), max(a.PicW, 1))
	return max(min(a.PicSide, PicHCap(a)/ow), 1)
}

// PicEntries is the most tap entries a Pillow resize's request holds, its
// output's columns' and rows' together (PILTaps): PicW + PicH, or for a
// picture of its own size the most an output of its area reaches.
func PicEntries(a graph.Attrs) int {
	if a.PicArea == 0 {
		return a.PicW + a.PicH
	}
	ceil := func(n, d int) int { return (n + d - 1) / d }
	return max(a.PicW+min(a.PicH, ceil(a.PicArea, a.PicW)), a.PicH+min(a.PicW, ceil(a.PicArea, a.PicH)))
}

// PILEntryWords is the words a tap entry of at most k taps takes in the
// request (PILTaps).
func PILEntryWords(k int) int { return k + 2 }

// PILTaps writes the taps of outputs from to from+n-1 of a side of in
// samples resized to out by filter into dst, n entries of PILEntryWords(k)
// words: the first sample an output reads, how many it reads, then its k
// weights in Pillow's fixed point (int32, zero past its window) -- Resample.c's precompute_coeffs and
// normalize_coeffs_8bpc, in float64 as Pillow computes them, so the passes
// the picture op runs on them are Pillow's to the bit. A window short of the
// whole side is a crop of the resize: each output's taps are its own, and
// so is the 8-bit sample the first pass leaves for it.
func PILTaps(filter uint8, in, out, from, n, k int, dst []uint32) {
	PILTapsAt(filter, false, in, out, from, n, k, dst)
}

// PILTapsAt is PILTaps, at torchvision's precision where torch says: each
// weight rounded at the widest precision p (to 21 bits) at which the
// axis's largest holds in an int16, then shifted up to Pillow's -- the same
// sums, (s + 2^(p-1)) >> p, through the picture op's one finish.
func PILTapsAt(filter uint8, torch bool, in, out, from, n, k int, dst []uint32) {
	scale := float64(in) / float64(out)
	fscale := max(scale, 1)
	support := pilSupport(filter) * fscale
	ss := 1 / fscale
	prec := pilPrec(filter, torch, in, out)
	for i := range n {
		o := from + i
		center := (float64(o) + 0.5) * scale
		xmin := max(int(center-support+0.5), 0)
		xmax := min(int(center+support+0.5), in) - xmin
		xmax = min(xmax, k)
		if o >= out {
			// Past the resize's edge (a canvas's pad): no sample, black.
			xmax = 0
		}
		// The window's weights twice, summed and then each divided by the
		// sum, the values Pillow's one pass stores: nothing is held between.
		w := func(x int) float64 { return pilFilter(filter, (float64(x+xmin)-center+0.5)*ss) }
		ww := 0.0
		for x := range xmax {
			ww += w(x)
		}
		e := dst[i*PILEntryWords(k) : (i+1)*PILEntryWords(k)]
		e[0], e[1] = uint32(xmin), uint32(xmax)
		for x := range k {
			v := 0.0
			if x < xmax && ww != 0 {
				v = w(x) / ww
			}
			e[2+x] = pilFixed(v, prec)
		}
	}
}

// PILChunk is PILTapsAt clipped to the source samples lo to hi-1, a band
// of at most k: each entry's first sample is the later of its window's and
// lo, its count the window's samples from there below hi (none where the
// window misses the band), its weights the whole window's, normalised over
// all of it -- so the bands' sums of an output add to the one pass's, and
// the 8-bit sample is rounded once, after the last band. Only an entry's
// counted weights are written: the passes read no other, and a band
// touches only the outputs whose windows meet it. prec is the side's
// precision (PILPrec).
func PILChunk(filter uint8, prec, in, out, from, n, k, lo, hi int, dst []uint32) {
	scale := float64(in) / float64(out)
	fscale := max(scale, 1)
	support := pilSupport(filter) * fscale
	ss := 1 / fscale
	for i := range n {
		o := from + i
		center := (float64(o) + 0.5) * scale
		xmin := max(int(center-support+0.5), 0)
		xmax := min(int(center+support+0.5), in) - xmin
		if o >= out {
			xmax = 0
		}
		first := max(xmin, lo)
		count := min(max(min(xmin+xmax, hi)-first, 0), k)
		e := dst[i*PILEntryWords(k) : (i+1)*PILEntryWords(k)]
		e[0], e[1] = uint32(first), uint32(count)
		if count == 0 {
			continue
		}
		w := func(x int) float64 { return pilFilter(filter, (float64(x+xmin)-center+0.5)*ss) }
		ww := 0.0
		for x := range xmax {
			ww += w(x)
		}
		for x := range count {
			v := 0.0
			if ww != 0 {
				v = w(first-xmin+x) / ww
			}
			e[2+x] = pilFixed(v, prec)
		}
	}
}

// PILPrec is the precision a side of in samples resized to out by filter
// rounds its weights at: Pillow's, or torchvision's where torch says
// (PILTapsAt). PILChunk takes it once a side rather than each band.
func PILPrec(filter uint8, torch bool, in, out int) int { return pilPrec(filter, torch, in, out) }

// pilFixed is weight v in Pillow's fixed point at precision prec, shifted up
// to Pillow's own.
func pilFixed(v float64, prec int) uint32 {
	var q int32
	if v < 0 {
		q = int32(-0.5 + v*float64(int(1)<<prec))
	} else {
		q = int32(0.5 + v*float64(int(1)<<prec))
	}
	return uint32(q << (pilPrecision - prec))
}

// pilPrec is the precision a side of in samples resized to out rounds its
// weights at: Pillow's, or torchvision's where torch says (PILTapsAt).
func pilPrec(filter uint8, torch bool, in, out int) int {
	scale := float64(in) / float64(out)
	fscale := max(scale, 1)
	support := pilSupport(filter) * fscale
	ss := 1 / fscale
	prec := pilPrecision
	if torch {
		// The largest weight over every output of the axis, as torchvision
		// computes it before it rounds any.
		most := 0.0
		for o := range out {
			center := (float64(o) + 0.5) * scale
			xmin := max(int(center-support+0.5), 0)
			xmax := min(int(center+support+0.5), in) - xmin
			ww := 0.0
			for x := range xmax {
				ww += pilFilter(filter, (float64(x+xmin)-center+0.5)*ss)
			}
			for x := range xmax {
				if ww != 0 {
					most = max(most, pilFilter(filter, (float64(x+xmin)-center+0.5)*ss)/ww)
				}
			}
		}
		prec = 0
		for prec < 21 && int(0.5+most*float64(int(1)<<(prec+1))) < 1<<15 {
			prec++
		}
	}
	return prec
}
