package model

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/format/jlm"
)

// Kimi-VL's MoonViT (jlm.ProjKimiVL): CLIP's LayerNorm, GELU-tanh block with
// biased q/k/v, and three pieces of its own.
//
//	the picture     resized only when its patches pass the processor's
//	                in_token_limit, then padded black on the right and the
//	                bottom to whole merge units (KimiVLImageProcessor,
//	                pad_input); the tower runs the grid that makes
//	the position    a learned table resampled BICUBIC to the grid
//	                (F.interpolate, align_corners False) and added; and the
//	                2-D rotary in complex pairs: pair 2j turns by the patch's
//	                column, 2j+1 by its row, both at base^(-4j/HeadDim)
//	the projector   a LayerNorm over each patch (RoleVProjInNorm) BEFORE the
//	                2x2 shuffle, then linear, GELU, linear
//
// The resize is Pillow's BICUBIC (rgb8.resizePIL), the table's taps and
// weights generated code per grid (posLerp: interpolate's coordinate, (i+0.5)
// times side/size minus 0.5, which at the table's own size is exactly i with
// weights (0, 1, 0, 0)), and every matrix the packed matmul.

// kimiTokenLimit is Kimi-VL's processor's in_token_limit: a picture of more
// whole patches is scaled down to about that many.
const kimiTokenLimit = 4096

// kimiCapacity is the most patches a picture under limit pads to: a side of
// a whole patches pads to at most the even number at or under a+2, and the
// processor refuses a padded side of 512 patches or more.
func kimiCapacity(limit int) int {
	pad := func(a int) int { return (a + 2) / 2 * 2 }
	best := 0
	for a := 1; a <= min(limit, 508); a++ {
		best = max(best, pad(a)*pad(min(limit/a, 508)))
	}
	return best
}

// loadKimiVL reads the table's side and the projector's norm over each patch;
// buildTower reads the two matrices as it does every MLP projector's.
func (t *Tower) loadKimiVL(vec func(jlm.Role, int32) ([]float32, error)) error {
	c := &t.Cfg
	rows := len(t.posW) / c.NEmbd
	side := int(math.Sqrt(float64(rows)))
	if side < 4 || side*side != rows || rows*c.NEmbd != len(t.posW) {
		return fmt.Errorf("model: OpenTower: the position table is %d floats, not a square of %d-wide rows",
			len(t.posW), c.NEmbd)
	}
	c.PosSide = side
	if c.HeadDim%4 != 0 {
		return fmt.Errorf("model: OpenTower: a kimivl head of %d does not split into (column, row) pairs", c.HeadDim)
	}
	if t.postLnW == nil || t.postLnB == nil {
		return fmt.Errorf("model: OpenTower: a kimivl tower needs its final LayerNorm (v.post_ln)")
	}
	var err error
	if t.projNorm, err = vec(jlm.RoleVProjInNorm, jlm.DenseBlock); err != nil {
		return err
	}
	if t.projNormB, err = vec(jlm.RoleVProjInNormBias, jlm.DenseBlock); err != nil {
		return err
	}
	if len(t.projNorm) != c.NEmbd || len(t.projNormB) != c.NEmbd {
		return fmt.Errorf("model: OpenTower: the projector's LayerNorm is %d and %d floats, want %d",
			len(t.projNorm), len(t.projNormB), c.NEmbd)
	}
	return nil
}

// kimiResize is the size a w x h picture is resampled to before its pad:
// itself, or int(side*scale) past the patch limit.
func (c TowerConfig) kimiResize(w, h int) (rw, rh int) {
	p := c.PatchSz
	if n := (w / p) * (h / p); n > c.TokenLimit {
		s := math.Sqrt(float64(c.TokenLimit) / float64(n))
		return int(float64(w) * s), int(float64(h) * s)
	}
	return w, h
}

// kimiPadded is the picture the tower runs: kimiResize's, padded to whole
// merge units.
func (c TowerConfig) kimiPadded(w, h int) (pw, ph int, err error) {
	if w <= 0 || h <= 0 {
		return 0, 0, fmt.Errorf("model: a %dx%d picture", w, h)
	}
	rw, rh := c.kimiResize(w, h)
	u := c.PatchSz * c.Scale
	pw, ph = (rw+u-1)/u*u, (rh+u-1)/u*u
	if pw/c.PatchSz >= 512 || ph/c.PatchSz >= 512 {
		return 0, 0, fmt.Errorf("model: a %dx%d picture is %dx%d patches, past the 512 the reference allows",
			w, h, pw/c.PatchSz, ph/c.PatchSz)
	}
	if (pw/c.PatchSz)*(ph/c.PatchSz) > c.Patches() {
		return 0, 0, fmt.Errorf("model: a %dx%d picture pads to %dx%d, past the %d patches this tower holds",
			w, h, pw, ph, c.Patches())
	}
	return pw, ph, nil
}

// preprocessKimi is Kimi-VL's processor: the picture resampled only past the
// patch limit, padded black, then normalised. It sets the grid Encode runs.
func (s *State) preprocessKimi(src rgb8) ([]float32, error) {
	c := s.vis.t.Cfg
	pw, ph, err := c.kimiPadded(src.w, src.h)
	if err != nil {
		return nil, err
	}
	if nw, nh := c.kimiResize(src.w, src.h); nw != src.w || nh != src.h {
		src = src.resizePIL(nw, nh, aaBicubic)
	}
	pad := rgb8{w: pw, h: ph, px: make([]uint8, 3*pw*ph)}
	for y := 0; y < src.h; y++ {
		copy(pad.px[3*y*pw:], src.px[3*y*src.w:3*(y+1)*src.w])
	}
	s.vis.gh, s.vis.gw = ph/c.PatchSz, pw/c.PatchSz
	out := make([]float32, pw*ph*3)
	pad.normalizeInto(out, c.Mean, c.Std)
	return out, nil
}
