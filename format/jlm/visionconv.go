package jlm

import "fmt"

// ConventionsOf is what tower v's projector means for the conventions its
// tensors do not show: the facts each projector's reference decides by its
// name, stated once here so a converter writes them (v31) and a migration
// from v30 writes the same, and every reader after reads the facts.
func ConventionsOf(v *Vision) (VisionConv, error) {
	c := VisionConv{Norm: VisionLayerNorm, ProjAct: VisionGELUTanh, Class: ClassFirst}
	switch v.Projector {
	case ProjQwen25VL, ProjPixtral, ProjGemma4V, ProjGLM4V:
		c.Norm = VisionRMSNorm
	}
	if v.Projector == ProjLFM2 {
		// Lfm2VlMultiModalProjector's ACT2FN["gelu"]; every other two-matrix
		// projector runs llama.cpp's tanh form.
		c.ProjAct = VisionGELUErf
	}
	if v.Projector == ProjLlama4 {
		c.Class = ClassLast
	}
	switch v.Projector {
	case ProjQwen3VL:
		c.PosChain, c.PosFilter = ChainAligned, FilterBilinear
	case ProjGLM4V:
		c.PosChain, c.PosFilter = ChainGridSample, FilterCubic
	case ProjKimiVL:
		c.PosChain, c.PosFilter = ChainHalfPixel, FilterCubic
	case ProjHunyuanVL:
		c.PosChain, c.PosFilter = ChainHalfPixel, FilterLinear
	}
	if !v.Flags.Has(FlagVisionRope) {
		return c, nil
	}
	if v.NHead == 0 || v.NEmbd%v.NHead != 0 {
		return c, fmt.Errorf("jlm: a %d-wide tower of %d heads", v.NEmbd, v.NHead)
	}
	hd := v.NEmbd / v.NHead
	q := hd / 4
	// Each reference's 2-D rotary: Qwen-VL's halves of a head's pairs, the
	// row then the column, each from the first frequency of
	// base^(-2i/(HeadDim/2)), NEOX's halves.
	r := &VisionRope{Base: 10000, FreqDim: hd / 2, Neox: true,
		Runs: []VisionRun{{Pairs: q, Axis: 0, Step: 1}, {Pairs: q, Axis: 1, Step: 1}}}
	switch v.Projector {
	case ProjPixtral:
		// The row on the even frequencies of base^(-2i/HeadDim), the
		// column on the odd, adjacent pairs.
		r = &VisionRope{Base: 10000, FreqDim: hd,
			Runs: []VisionRun{{Pairs: q, Axis: 0, Freq: 0, Step: 2}, {Pairs: q, Axis: 1, Freq: 1, Step: 2}}}
	case ProjLlama4:
		// The column then the row, each from one, adjacent pairs.
		r = &VisionRope{Base: 10000, FreqDim: hd / 2, Offset: 1,
			Runs: []VisionRun{{Pairs: q, Axis: 1, Step: 1}, {Pairs: q, Axis: 0, Step: 1}}}
	case ProjGemma4V:
		// The column then the row, at base 100.
		r = &VisionRope{Base: 100, FreqDim: hd / 2, Neox: true,
			Runs: []VisionRun{{Pairs: q, Axis: 1, Step: 1}, {Pairs: q, Axis: 0, Step: 1}}}
	case ProjKimiVL:
		// Complex pairs, the column on the even and the row on the odd, a
		// pair of each at one frequency of base^(-2i/HeadDim).
		r = &VisionRope{Base: 10000, FreqDim: hd}
		for p := range hd / 2 {
			r.Runs = append(r.Runs, VisionRun{Pairs: 1, Axis: 1 - p%2, Freq: p / 2 * 2, Step: 1})
		}
	}
	c.Rope = r
	return c, nil
}
