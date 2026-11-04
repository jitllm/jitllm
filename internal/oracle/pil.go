package oracle

import "math"

// PILResize is Pillow's Image.resize of an RGB picture, three bytes a pixel,
// w x h, to ow x oh by its BILINEAR filter, or BICUBIC where cubic: Resample.c's
// ImagingResample for 8-bit images -- each output's window and weights from
// the reduction's scale in float64, the weights in 22-bit fixed point, a
// horizontal pass to bytes where the width changes, then a vertical one where
// the height does.
func PILResize(rgb []byte, w, h, ow, oh int, cubic bool) []byte {
	support, filter := 1.0, func(x float64) float64 {
		x = math.Abs(x)
		if x < 1 {
			return 1 - x
		}
		return 0
	}
	if cubic {
		support, filter = 2, func(x float64) float64 {
			const a = -0.5
			x = math.Abs(x)
			if x < 1 {
				return ((a+2)*x-(a+3))*x*x + 1
			}
			if x < 2 {
				return (((x-5)*x+8)*x - 4) * a
			}
			return 0
		}
	}
	const bits = 22
	coeffs := func(in, out int) (bounds []int, kk []int, ksize int) {
		scale := float64(in) / float64(out)
		fs := math.Max(scale, 1)
		sup := support * fs
		ksize = int(math.Ceil(sup))*2 + 1
		bounds, kk = make([]int, 2*out), make([]int, out*ksize)
		k := make([]float64, ksize)
		for xx := range out {
			center := (float64(xx) + 0.5) * scale
			ss := 1 / fs
			xmin := int(center - sup + 0.5)
			if xmin < 0 {
				xmin = 0
			}
			xmax := int(center + sup + 0.5)
			if xmax > in {
				xmax = in
			}
			xmax -= xmin
			ww := 0.0
			for x := range xmax {
				k[x] = filter((float64(x+xmin) - center + 0.5) * ss)
				ww += k[x]
			}
			for x := range ksize {
				v := 0.0
				if x < xmax && ww != 0 {
					v = k[x] / ww
				}
				if v < 0 {
					kk[xx*ksize+x] = int(-0.5 + v*(1<<bits))
				} else {
					kk[xx*ksize+x] = int(0.5 + v*(1<<bits))
				}
			}
			bounds[2*xx], bounds[2*xx+1] = xmin, xmax
		}
		return bounds, kk, ksize
	}
	clip := func(v int) byte {
		v >>= bits
		if v < 0 {
			return 0
		}
		if v > 255 {
			return 255
		}
		return byte(v)
	}
	img, iw := rgb, w
	if ow != w {
		b, kk, ks := coeffs(w, ow)
		next := make([]byte, ow*h*3)
		for y := range h {
			for x := range ow {
				for c := range 3 {
					ss := 1 << (bits - 1)
					for t := range b[2*x+1] {
						ss += int(img[(y*iw+b[2*x]+t)*3+c]) * kk[x*ks+t]
					}
					next[(y*ow+x)*3+c] = clip(ss)
				}
			}
		}
		img, iw = next, ow
	}
	if oh != h {
		b, kk, ks := coeffs(h, oh)
		next := make([]byte, iw*oh*3)
		for y := range oh {
			for x := range iw {
				for c := range 3 {
					ss := 1 << (bits - 1)
					for t := range b[2*y+1] {
						ss += int(img[((b[2*y]+t)*iw+x)*3+c]) * kk[y*ks+t]
					}
					next[(y*iw+x)*3+c] = clip(ss)
				}
			}
		}
		img = next
	}
	return img
}

// PILPattern is scripts/pilgold.py's pattern picture, RGB: red (7x + 13y)
// mod 256, green (3xy + 5) mod 256, blue 11(x + 2y) mod 256.
func PILPattern(w, h int) []byte {
	b := make([]byte, w*h*3)
	for y := range h {
		for x := range w {
			i := (y*w + x) * 3
			b[i], b[i+1], b[i+2] = byte(7*x+13*y), byte(3*x*y+5), byte(11*(x+2*y))
		}
	}
	return b
}
