package model

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// Pixtral / Mistral 3's vision: a tower that reads the picture at its own
// size, and a head that interleaves a text row into its output.
//
// What is the family's own, each read off the reference rather than derived:
//
//	the resize          transformers' PixtralImageProcessor: the longest side
//	                    brought to at most ImageSz (floor), each side then
//	                    rounded UP to a whole merge unit, Pillow BICUBIC
//	the rotary          PixtralVisionRotaryEmbedding: head_dim/4 frequencies
//	                    for the grid row, taking the even indices of one
//	                    base^(-2i/head_dim) sequence, and head_dim/4 for the
//	                    column taking the odd ones (build_rope_2d's
//	                    freq_scale_odd); the GGUF's q and k are permuted for
//	                    interleaved pairs, so the pairs are (2j, 2j+1)
//	the merger          Mistral3PatchMerger: an RMSNorm over each patch (the
//	                    tower's post-norm position), Scale x Scale patches
//	                    concatenated, one matrix; the converter puts its
//	                    columns in the shuffle's order (convert.permuteMerger)
//	the break rows      PixtralProcessor writes [IMG_BREAK] after each row of
//	                    image tokens and [IMG_END] in place of the last; the
//	                    head writes the break's embedding (RoleVImgBreak) and
//	                    the prompt's markers write [IMG_END]
//
// Everything else -- the RMSNorm block, the gated SiLU MLP, bidirectional
// attention over the picture, llava's linear, GELU, linear -- is the runner's
// and the head's as every other family runs it.

// pixtralPixels gives a Pixtral tower its bounds: a picture is at most
// ImageSz on its longest side, rounded up to a whole merge unit, so its grid
// is at most that square of patches. There is no floor: a picture of one
// pixel is one merge unit.
func pixtralPixels(c *TowerConfig) {
	u := c.PatchSz * max(c.Scale, 1)
	side := (c.ImageSz + u - 1) / u * u
	c.MinPixels, c.MaxPixels = 0, side*side
}

// maxSide is the most patches on one side of a Pixtral picture.
func (c TowerConfig) maxSide() int {
	return int(math.Sqrt(float64(c.MaxPixels))) / c.PatchSz
}

// pixtralResize is PixtralImageProcessor's output size for a w x h picture:
// ratio = max(h, w) / ImageSz, both sides floored by it when it is over one,
// then each rounded up to a whole unit of PatchSz*Scale pixels.
func (c TowerConfig) pixtralResize(w, h int) (rw, rh int, err error) {
	if w <= 0 || h <= 0 {
		return 0, 0, fmt.Errorf("model: a %dx%d picture", w, h)
	}
	L := float64(c.ImageSz)
	ratio := math.Max(float64(h)/L, float64(w)/L)
	if ratio > 1 {
		h = int(math.Floor(float64(h) / ratio))
		w = int(math.Floor(float64(w) / ratio))
	}
	u := c.PatchSz * max(c.Scale, 1)
	rh = ((max(h, 1)-1)/u + 1) * u
	rw = ((max(w, 1)-1)/u + 1) * u
	if (rh/c.PatchSz)*(rw/c.PatchSz) > c.Patches() {
		return 0, 0, fmt.Errorf("model: a %dx%d picture resizes to %dx%d, over the %d patches this tower holds",
			w, h, rw, rh, c.Patches())
	}
	return rw, rh, nil
}

// ropeNeox reports whether the tower's 2-D rotary pairs a head's dimensions
// (i, i+HeadDim/2) rather than (2i, 2i+1), as its container states
// (jlm.VisionRope.Neox).
func (c TowerConfig) ropeNeox() bool {
	return c.conv.Rope == nil || c.conv.Rope.Neox
}

// The gates' violations (towerFault values faultPixtralFreq and faultPixtralMerge): Qwen-VL's frequencies
// (each half from the first) where Pixtral's take the even and the odd, and
// the merger handed the unfold's channel-major order its columns no longer
// are in.

// pixtralRows is how many rows a gh x gw patch grid becomes: the merged grid
// and a break row after each merged row but the last.
func (c TowerConfig) pixtralRows(gh, gw int) int {
	sc := max(c.Scale, 1)
	return gh/sc*(gw/sc) + gh/sc - 1
}

// loadPixtral reads the head: the merger when the tower merges, the MLP and
// the break row.
func (t *Tower) loadPixtral(f *jlm.File, get func(jlm.Role, int32) (tensor, error),
	vec func(jlm.Role, int32) ([]float32, error)) error {
	c := &t.Cfg
	var err error
	if t.projW, err = get(jlm.RoleVProj, jlm.DenseBlock); err != nil {
		return err
	}
	if t.projB, err = optVecOf(f, vec, jlm.RoleVProjBias); err != nil {
		return err
	}
	if t.projW2, err = get(jlm.RoleVProj2, jlm.DenseBlock); err != nil {
		return err
	}
	if t.projB2, err = optVecOf(f, vec, jlm.RoleVProj2Bias); err != nil {
		return err
	}
	in := c.NEmbd
	if c.Scale > 1 {
		if t.mergeW, err = get(jlm.RoleVMerge, jlm.DenseBlock); err != nil {
			return fmt.Errorf("model: OpenTower: a pixtral tower that merges %dx%d patches needs its merger: %w",
				c.Scale, c.Scale, err)
		}
		if t.mergeW.k != c.NEmbd*c.Scale*c.Scale || t.mergeW.rows != c.NEmbd {
			return fmt.Errorf("model: OpenTower: the patch merger is %dx%d, want %d rows of %d",
				t.mergeW.k, t.mergeW.rows, c.NEmbd, c.NEmbd*c.Scale*c.Scale)
		}
	} else if f.Has(jlm.RoleVMerge, jlm.DenseBlock, -1) {
		return fmt.Errorf("model: OpenTower: a patch merger on a pixtral tower that merges nothing")
	}
	if t.projW.k != in || t.projW2.k != t.projW.rows || t.projW2.rows != c.ProjDim {
		return fmt.Errorf("model: OpenTower: the projector is %dx%d then %dx%d, want %d in and %d out",
			t.projW.k, t.projW.rows, t.projW2.k, t.projW2.rows, in, c.ProjDim)
	}
	if t.imgBreak, err = vec(jlm.RoleVImgBreak, jlm.DenseBlock); err != nil {
		return err
	}
	if len(t.imgBreak) != c.ProjDim {
		return fmt.Errorf("model: OpenTower: the [IMG_BREAK] row is %d floats, want %d", len(t.imgBreak), c.ProjDim)
	}
	return nil
}

// optVecOf is vec for a role the container may not carry.
func optVecOf(f *jlm.File, vec func(jlm.Role, int32) ([]float32, error), r jlm.Role) ([]float32, error) {
	if !f.Has(r, jlm.DenseBlock, -1) {
		return nil, nil
	}
	return vec(r, jlm.DenseBlock)
}

// projectPixtral is the head over a picture's patch rows (the post-norm has
// run): the shuffle and the merger when the tower merges, llava's MLP, and the
// merged grid's rows written out with a break row after each grid row but the
// last.
func (s *State) projectPixtral(patches []float32) ([]float32, error) {
	v := s.vis
	t, c := v.t, v.t.Cfg
	gh, gw := s.patchGrid()
	sc := max(c.Scale, 1)
	mh, mw := gh/sc, gw/sc
	nt := mh * mw
	src := patches
	if sc > 1 {
		w := c.NEmbd * sc * sc
		for gy := 0; gy < mh; gy++ {
			for gx := 0; gx < mw; gx++ {
				dst := v.grp[(gy*mw+gx)*w : (gy*mw+gx+1)*w]
				for dy := 0; dy < sc; dy++ {
					for dx := 0; dx < sc; dx++ {
						p := (gy*sc+dy)*gw + gx*sc + dx
						slot := dy*sc + dx
						if s.tfault() == faultPixtralMerge {
							for ch := 0; ch < c.NEmbd; ch++ {
								dst[ch*sc*sc+slot] = patches[p*c.NEmbd+ch]
							}
							continue
						}
						copy(dst[slot*c.NEmbd:], patches[p*c.NEmbd:(p+1)*c.NEmbd])
					}
				}
			}
		}
		v.stage("proj_in", v.grp[:nt*w])
		s.jit.NewInput()
		if err := s.mm(v.mrg, t.mergeW, v.grp, nt); err != nil {
			return nil, err
		}
		src = v.mrg
	}
	s.jit.NewInput()
	if err := s.mm(v.hid, t.projW, src, nt); err != nil {
		return nil, err
	}
	s.visBiasRows(v.hid, t.projB, nt)
	// GELU-tanh, as every projector here runs it (see project).
	s.actAll(v.hid[:nt*t.projW.rows], nn.ActGELU)
	s.jit.NewInput()
	if err := s.mm(v.grp, t.projW2, v.hid, nt); err != nil {
		return nil, err
	}
	s.visBiasRows(v.grp, t.projB2, nt)
	d := c.ProjDim
	if s.tfault() == faultPixtralNoBreaks {
		copy(v.out, v.grp[:nt*d])
		v.stage("out", v.out[:nt*d])
		return v.out[:nt*d], nil
	}
	o := 0
	for y := 0; y < mh; y++ {
		copy(v.out[o*d:], v.grp[y*mw*d:(y+1)*mw*d])
		o += mw
		if y+1 < mh {
			copy(v.out[o*d:(o+1)*d], t.imgBreak)
			o++
		}
	}
	v.stage("out", v.out[:o*d])
	return v.out[:o*d], nil
}

// GridRows is how many positions a picture whose merged grid is g takes in a
// prompt: the grid's rows, and for Pixtral a break row after each grid row
// but the last.
func (c TowerConfig) GridRows(g ImageGrid) int {
	if c.Kind == jlm.ProjPixtral && g.H > 0 {
		return g.Rows() + g.H - 1
	}
	return g.Rows()
}
