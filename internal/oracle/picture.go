package oracle

import "math"

// Picture is a tower's pixels from a decoded picture (graph.OpPicture): its
// samples pix, w x h pixels in rows of stride bytes, four bytes a pixel (red,
// green, blue, an alpha dropped) or one where gray; resampled as PictureOf
// resamples.
func Picture(pix []byte, gray bool, w, h, stride, ow, oh int, mean, std [3]float64) []float32 {
	return PictureOf(func(y, x int) [3]byte {
		if gray {
			v := pix[y*stride+x]
			return [3]byte{v, v, v}
		}
		p := pix[y*stride+4*x:]
		return [3]byte{p[0], p[1], p[2]}
	}, w, h, ow, oh, mean, std)
}

// PictureOf is a w x h picture whose pixel (y, x) has the red, green and
// blue bytes rgb gives, resampled to ow x oh by the two samples around each
// output's centre on each axis ((i+0.5)*n/size - 0.5, the indices clamped to
// the picture), weighted 1-t and t, horizontal first; each sample read as
// [0, 1] and channel c normalised as (v - mean[c]) / std[c]. The result is
// [y][x][channel], in float64.
func PictureOf(rgb func(y, x int) [3]byte, w, h, ow, oh int, mean, std [3]float64) []float32 {
	out := make([]float32, ow*oh*3)
	taps := func(i, n, size int) (int, int, float64) {
		at := (float64(i)+0.5)*float64(n)/float64(size) - 0.5
		f := math.Floor(at)
		clamp := func(v int) int { return min(max(v, 0), n-1) }
		return clamp(int(f)), clamp(int(f) + 1), at - f
	}
	for oy := range oh {
		y0, y1, ty := taps(oy, h, oh)
		for ox := range ow {
			x0, x1, tx := taps(ox, w, ow)
			a, b, c, d := rgb(y0, x0), rgb(y0, x1), rgb(y1, x0), rgb(y1, x1)
			for ch := range 3 {
				top := float64(a[ch])*(1-tx) + float64(b[ch])*tx
				bot := float64(c[ch])*(1-tx) + float64(d[ch])*tx
				v := (top*(1-ty) + bot*ty) / 255
				out[(oy*ow+ox)*3+ch] = float32((v - mean[ch]) / std[ch])
			}
		}
	}
	return out
}
