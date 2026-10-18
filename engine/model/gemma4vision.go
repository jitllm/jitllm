package model

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// Gemma 4's vision: an RMSNorm ViT at the picture's own size within a soft
// token budget, and a pooling head.
//
// What is the family's own, each read off the reference
// (Gemma4VisionModel, Gemma4ImageProcessor):
//
//	the resize          the largest size that keeps the picture's aspect,
//	                    has whole 3x3 groups of 16-pixel patches on both
//	                    sides and at most 280 x 9 patches (the budget the
//	                    processor states; a picture is resized UP to it),
//	                    Pillow BICUBIC, pixels to 2x - 1 (mean and std 0.5)
//	the positions       two learned tables, the column's row of the first
//	                    and the row's of the second, added to each patch
//	the block           sandwich RMSNorms, a per-head q and k norm and a
//	                    weightless v norm, scores at scale one, a gated GELU
//	                    MLP, and every matrix's input and output clamped to
//	                    bounds the checkpoint carries (Gemma4ClippableLinear)
//	the rotary          each half of a head turned NEOX-paired on its own, the
//	                    column the first half and the row the second; the
//	                    converter swaps each head's middle quarters of q and k
//	                    (convert.halfNeoxToNeox), which makes it Qwen-VL's
//	                    table over the whole head
//	the head            the 3x3 average pool over the patch grid, times
//	                    sqrt(NEmbd), (x - std_bias) * std_scale where the
//	                    checkpoint standardises (the 26B and 31B), an
//	                    unweighted RMSNorm, and one matrix

// gemma4SoftTokens is the processor's soft-token budget (max_soft_tokens):
// the mmproj does not state it, every published processor_config does.
const gemma4SoftTokens = 280

// gemma4Pixels bounds a Gemma 4 picture: at most gemma4SoftTokens pooled
// groups of Scale x Scale patches.
func gemma4Pixels(c *TowerConfig) {
	c.MinPixels = 0
	c.MaxPixels = gemma4SoftTokens * c.Scale * c.Scale * c.PatchSz * c.PatchSz
}

// gemma4Resize is get_aspect_ratio_preserving_size: the picture scaled to the
// patch budget's area at its own aspect, each side floored to a whole group
// of Scale patches, a side that floors to nothing given one group and the
// other the aspect's whole multiple, capped.
func (c TowerConfig) gemma4Resize(w, h int) (rw, rh int, err error) {
	if w <= 0 || h <= 0 {
		return 0, 0, fmt.Errorf("model: a %dx%d picture", w, h)
	}
	maxPatches := c.MaxPixels / (c.PatchSz * c.PatchSz)
	target := float64(maxPatches * c.PatchSz * c.PatchSz)
	f := math.Sqrt(target / (float64(h) * float64(w)))
	side := c.Scale * c.PatchSz
	th := int(math.Floor(f*float64(h)/float64(side))) * side
	tw := int(math.Floor(f*float64(w)/float64(side))) * side
	if th == 0 && tw == 0 {
		return 0, 0, fmt.Errorf("model: a %dx%d picture resizes to nothing", w, h)
	}
	maxSide := (maxPatches / (c.Scale * c.Scale)) * side
	switch {
	case th == 0:
		th, tw = side, min(int(math.Floor(float64(w)/float64(h)))*side, maxSide)
	case tw == 0:
		tw, th = side, min(int(math.Floor(float64(h)/float64(w)))*side, maxSide)
	}
	if float64(th*tw) > target {
		return 0, 0, fmt.Errorf("model: a %dx%d picture resizes to %dx%d, over %d patches", w, h, tw, th, maxPatches)
	}
	return tw, th, nil
}

// gemma4Rope is the tower's 2-D rotary over the head the converter laid out:
// NEOX pairs over the whole head, the first HeadDim/4 turned by the column
// and the next by the row, each from base^0 (the reference's per-half
// frequencies, base^(-2i/(HeadDim/2))). The base is 100, the loader's
// constant as llama.cpp's is.
func (c TowerConfig) gemma4Rope() nn.Rope {
	q := c.HeadDim / 4
	return nn.Rope{NRot: c.HeadDim / 2, Base: 100, Neox: true,
		Runs: []nn.RopeRun{{Pairs: q, Axis: 0}, {Pairs: q, Axis: 1}}}
}

// gemma4Head is what Gemma 4's head reads beyond the projection: the
// standardisation, folded with the sqrt(NEmbd) before it, and the weightless
// norms' ones.
type gemma4Head struct {
	// scale multiplies every pooled row, sqrt(NEmbd).
	scale float32
	// bias and std are the standardisation, (x - bias) * std, nil when the
	// checkpoint does not standardise.
	bias, std []float32
	// ones is an NEmbd-wide vector of ones: the unweighted RMSNorm's weight
	// and, cut to HeadDim, the v norm's.
	ones []float32
}

// loadGemma4V reads the head: the projection and the standardisation.
func (t *Tower) loadGemma4V(f *jlm.File, get func(jlm.Role, int32) (tensor, error),
	vec func(jlm.Role, int32) ([]float32, error)) error {
	c := &t.Cfg
	var err error
	if t.projW, err = get(jlm.RoleVProj, jlm.DenseBlock); err != nil {
		return err
	}
	if t.projW.k != c.NEmbd || t.projW.rows != c.ProjDim {
		return fmt.Errorf("model: OpenTower: gemma4v's projection is %dx%d, want %d to %d",
			t.projW.k, t.projW.rows, c.NEmbd, c.ProjDim)
	}
	h := &gemma4Head{scale: float32(math.Sqrt(float64(c.NEmbd))), ones: make([]float32, c.NEmbd)}
	for i := range h.ones {
		h.ones[i] = 1
	}
	if h.bias, err = optVecOf(f, vec, jlm.RoleVStdBias); err != nil {
		return err
	}
	if h.std, err = optVecOf(f, vec, jlm.RoleVStdScale); err != nil {
		return err
	}
	if (h.bias == nil) != (h.std == nil) || h.bias != nil && (len(h.bias) != c.NEmbd || len(h.std) != c.NEmbd) {
		return fmt.Errorf("model: OpenTower: gemma4v's standardisation is %d and %d floats, want both %d or neither",
			len(h.bias), len(h.std), c.NEmbd)
	}
	for i := range t.pending {
		b := &t.pending[i]
		if b.qNorm == nil || b.kNorm == nil || len(b.qNorm) != c.HeadDim || len(b.kNorm) != c.HeadDim {
			return fmt.Errorf("model: OpenTower: gemma4v block %d's q/k norms are %d and %d floats, want %d",
				i, len(b.qNorm), len(b.kNorm), c.HeadDim)
		}
		if b.clamp != nil && len(b.clamp) != 28 {
			return fmt.Errorf("model: OpenTower: gemma4v block %d's clamps are %d floats, want 28", i, len(b.clamp))
		}
	}
	t.g4 = h
	return nil
}

// clampIn is matrix slot's input under its block's clipped-linear bounds:
// src itself when the block has none or the bounds clamp nothing, otherwise
// a clamped copy in the State's scratch, the activation marked new.
func (s *State) clampIn(l *layer, slot int, src []float32) []float32 {
	if l.clamp == nil || s.tfault() == faultGemma4NoClamp {
		return src
	}
	lo, hi := l.clamp[4*slot], l.clamp[4*slot+1]
	if math.IsInf(float64(lo), -1) && math.IsInf(float64(hi), 1) {
		return src
	}
	if cap(s.bclamp) < len(src) {
		s.bclamp = make([]float32, len(src))
	}
	dst := s.bclamp[:len(src)]
	copy(dst, src)
	nn.ClampRange32JIT(dst, lo, hi)
	s.jit.NewInput()
	return dst
}

// clampOut clamps matrix slot's output in place under its block's bounds.
func (s *State) clampOut(l *layer, slot int, out []float32) {
	if l.clamp == nil || s.tfault() == faultGemma4NoClamp {
		return
	}
	lo, hi := l.clamp[4*slot+2], l.clamp[4*slot+3]
	if math.IsInf(float64(lo), -1) && math.IsInf(float64(hi), 1) {
		return
	}
	nn.ClampRange32JIT(out, lo, hi)
}

// The clamped matrices' slots in a block's bounds.
const (
	clampQ = iota
	clampK
	clampV
	clampO
	clampGate
	clampUp
	clampDown
)

// projectGemma4V is the head over a picture's patch rows: the Scale x Scale
// average pool over the grid, sqrt(NEmbd) and the standardisation, an
// unweighted RMSNorm, and the projection. Every arithmetic step is a
// generated kernel; the standardisation's per-channel scale is each pooled
// row times the scale vector, the gated kernel's identity product.
func (s *State) projectGemma4V(patches []float32) ([]float32, error) {
	v := s.vis
	t, c := v.t, v.t.Cfg
	h := t.g4
	gh, gw := s.patchGrid()
	sc := c.Scale
	mh, mw := gh/sc, gw/sc
	nt := mh * mw
	E := c.NEmbd
	inv := 1 / float32(sc*sc)
	for gy := 0; gy < mh; gy++ {
		for gx := 0; gx < mw; gx++ {
			g := gy*mw + gx
			dst := v.grp[g*E : (g+1)*E]
			clear(dst)
			for dy := 0; dy < sc; dy++ {
				for dx := 0; dx < sc; dx++ {
					if s.tfault() == faultGemma4NoPool && (dy|dx) != 0 {
						continue
					}
					p := (gy*sc+dy)*gw + gx*sc + dx
					nn.Axpy32JIT(dst, patches[p*E:(p+1)*E], 1)
				}
			}
			if s.tfault() != faultGemma4NoPool {
				nn.Scale32JIT(dst, inv)
			}
			nn.Scale32JIT(dst, h.scale)
			if h.bias != nil {
				nn.Axpy32JIT(dst, h.bias, -1)
			}
		}
	}
	rows := v.grp[:nt*E]
	if h.std != nil {
		// The per-channel scale is an elementwise product with the row.
		for g := 0; g < nt; g++ {
			nn.ActMul32JIT(rows[g*E:(g+1)*E], h.std, nn.ActIdentity)
		}
	}
	s.normRows(rows, rows, h.ones, nil, E, nt, false, c.Eps, s.rowChunk(nt, E))
	v.stage("proj_in", rows)
	s.jit.NewInput()
	if err := s.mm(v.out, t.projW, rows, nt); err != nil {
		return nil, err
	}
	v.stage("out", v.out[:nt*c.ProjDim])
	return v.out[:nt*c.ProjDim], nil
}
