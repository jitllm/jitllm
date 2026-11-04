package oracle

import "math"

// Picture is a tower's pixels from a decoded picture (graph.OpPicture): its
// samples pix, w x h pixels in rows of stride bytes, four bytes a pixel (red,
// green, blue, an alpha dropped) or one where gray; resampled to ow x oh by
// the two samples around each output's centre on each axis ((i+0.5)*n/size
// - 0.5, the indices clamped to the picture), weighted 1-t and t, horizontal
// first; each sample read as [0, 1] and channel c normalised as
// (v - mean[c]) / std[c]. The result is [y][x][channel], in float64.
func Picture(pix []byte, gray bool, w, h, stride, ow, oh int, mean, std [3]float64) []float32 {
	out := make([]float32, ow*oh*3)
	taps := func(i, n, size int) (int, int, float64) {
		at := (float64(i)+0.5)*float64(n)/float64(size) - 0.5
		f := math.Floor(at)
		clamp := func(v int) int { return min(max(v, 0), n-1) }
		return clamp(int(f)), clamp(int(f) + 1), at - f
	}
	sample := func(y, x, c int) float64 {
		if gray {
			return float64(pix[y*stride+x])
		}
		return float64(pix[y*stride+4*x+c])
	}
	for oy := range oh {
		y0, y1, ty := taps(oy, h, oh)
		for ox := range ow {
			x0, x1, tx := taps(ox, w, ow)
			for c := range 3 {
				top := sample(y0, x0, c)*(1-tx) + sample(y0, x1, c)*tx
				bot := sample(y1, x0, c)*(1-tx) + sample(y1, x1, c)*tx
				v := (top*(1-ty) + bot*ty) / 255
				out[(oy*ow+ox)*3+c] = float32((v - mean[c]) / std[c])
			}
		}
	}
	return out
}
