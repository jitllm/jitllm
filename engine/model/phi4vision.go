package model

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/nn"
)

// Phi-4-reasoning-vision: SigLIP2's NaFlex tower at the picture's own size,
// then llava's two matrices and a GELU over every patch.
//
// What is the family's own, each read off the reference (the checkpoint's
// processing_phi4_visionr.py and transformers' Siglip2VisionEmbeddings):
//
//	the resize     Siglip2ImageProcessorNoUpscale: the picture's own patch
//	               count clamped to the file's [MinPixels, MaxPixels] in
//	               patches, through get_image_size_for_max_num_patches --
//	               which even an in-range picture goes through, so its sides
//	               become whole patches -- and Pillow BILINEAR
//	the positions  the tower's square table resized to each grid:
//	               F.interpolate, bilinear, align_corners=False,
//	               antialias=True, in float
//	the depth      hidden_states[-2]: llama.cpp's converter writes one block
//	               fewer and no post-LayerNorm, so the container's tower is
//	               already that depth

// phi4Pixels gives the tower the bounds the file states.
func phi4Pixels(c *TowerConfig, lo, hi int) error {
	p2 := c.PatchSz * c.PatchSz
	if lo < p2 || hi < lo {
		return fmt.Errorf("model: OpenTower: phi4 reads %d to %d pixels, under one %d-pixel patch or inverted",
			lo, hi, c.PatchSz)
	}
	c.MinPixels, c.MaxPixels = lo, hi
	return nil
}

// phi4Resize is ResizeTo for Phi-4-reasoning-vision (naflexSize).
func (c TowerConfig) phi4Resize(w, h int) (rw, rh int, err error) {
	if w <= 0 || h <= 0 {
		return 0, 0, fmt.Errorf("model: a %dx%d picture", w, h)
	}
	p2 := c.PatchSz * c.PatchSz
	rw, rh = naflexSize(w, h, c.PatchSz, c.MinPixels/p2, c.MaxPixels/p2)
	return rw, rh, nil
}

// naflexSize is the size Phi-4-reasoning-vision's processor resamples a w x h
// picture to: the picture at its own patch count, clamped to [lo, hi]
// patches, through transformers' get_image_size_for_max_num_patches -- a
// binary search for the largest scale whose ceil()ed, patch-aligned size
// holds the count.
func naflexSize(w, h, patch, lo, hi int) (int, int) {
	n := max((w/patch)*(h/patch), 1)
	target := min(max(n, lo), hi)
	scaled := func(scale float64, size int) int {
		v := math.Ceil(float64(size)*scale/float64(patch)) * float64(patch)
		return int(math.Max(float64(patch), v))
	}
	const eps = 1e-5
	smin, smax := eps/10, 100.0
	for smax-smin >= eps {
		sc := (smin + smax) / 2
		th, tw := scaled(sc, h), scaled(sc, w)
		if float64(th)/float64(patch)*(float64(tw)/float64(patch)) <= float64(target) {
			smin = sc
		} else {
			smax = sc
		}
	}
	return scaled(smin, w), scaled(smin, h)
}

// phi4Table is the tower's square position table resized to a gh x gw grid,
// as Siglip2VisionEmbeddings.resize_positional_embeddings resizes it:
// separable, PyTorch's antialiasing taps in float, each sum a generated axpy
// over a whole row of channels. Kept for the last grid, so a second picture
// of the same size allocates nothing.
func (s *State) phi4Table(gh, gw int) []float32 {
	v := s.vis
	if s.tfault() == faultPhi4NoResize {
		return v.t.posW
	}
	if s.tfault() == faultPhi4PosTransposed {
		gh, gw = gw, gh
	}
	if v.posTab != nil && v.posAt == [2]int{gh, gw} {
		return v.posTab
	}
	c := v.t.Cfg
	side := c.ImageSz / c.PatchSz
	E := c.NEmbd
	src := v.t.posW
	// Across: [side][gw][E] from [side][side][E].
	fx, wx := aaFloats(side, gw, aaBilinear)
	mid := make([]float32, side*gw*E)
	for y := 0; y < side; y++ {
		for xx := 0; xx < gw; xx++ {
			dst := mid[(y*gw+xx)*E : (y*gw+xx+1)*E]
			for k, w := range wx[xx] {
				x := fx[xx] + k
				nn.Axpy32JIT(dst, src[(y*side+x)*E:(y*side+x+1)*E], float32(w))
			}
		}
	}
	// Down: [gh][gw][E], each output row a sum of whole rows of mid.
	fy, wy := aaFloats(side, gh, aaBilinear)
	row := gw * E
	out := make([]float32, gh*row)
	for yy := 0; yy < gh; yy++ {
		dst := out[yy*row : (yy+1)*row]
		for k, w := range wy[yy] {
			y := fy[yy] + k
			nn.Axpy32JIT(dst, mid[y*row:(y+1)*row], float32(w))
		}
	}
	v.posTab, v.posAt = out, [2]int{gh, gw}
	return out
}
