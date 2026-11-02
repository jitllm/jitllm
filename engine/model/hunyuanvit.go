package model

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// HunyuanVL's tower (jlm.ProjHunyuanVL): a LayerNorm, GELU ViT with biased
// q/k/v and no rotary over the patches in raster order (its processor
// flattens them row-major, unlike Qwen-VL's merge groups), and two pieces of
// its own.
//
//	the table     a learned square table resampled BILINEAR with
//	              align_corners False (F.interpolate) to the grid
//	the merger    an RMSNorm, a 2x2 convolution (a matrix over the (dy, dx,
//	              channel) row, as GLM-4.xV's), GELU, a 1x1 convolution, a
//	              newline row after every merged row, a linear, the begin and
//	              end rows around the picture, and an RMSNorm: H*(W+1)+2 rows
//
// GELU is the engine's tanh form where the reference takes erf, in the tower
// and the merger alike (a divergence every projector here makes).

// hyProj is HunyuanVL's merger beyond the Tower's own projW/projB (the 1x1
// convolution) and projW2/projB2 (the linear).
type hyProj struct {
	pre, post         []float32 // the RMSNorms' weights, before and after
	merge             tensor    // the 2x2 convolution as a matrix
	mergeB            []float32
	newline           []float32 // the 1x1 convolution's width
	begin, end        []float32 // the text model's width
	eps               float64
	mergeW, lineWidth int
}

// loadHunyuanVL reads the table's side and the merger.
func (t *Tower) loadHunyuanVL(get func(jlm.Role, int32) (tensor, error),
	vec, optVec func(jlm.Role, int32) ([]float32, error)) error {
	c := &t.Cfg
	E, P := c.NEmbd, c.ProjDim
	rows := len(t.posW) / E
	side := int(math.Sqrt(float64(rows)))
	if side < 2 || side*side != rows || rows*E != len(t.posW) {
		return fmt.Errorf("model: OpenTower: the position table is %d floats, not a square of %d-wide rows",
			len(t.posW), E)
	}
	c.PosSide = side
	h := &hyProj{eps: c.Eps}
	var err error
	if h.pre, err = vec(jlm.RoleVProjInNorm, jlm.DenseBlock); err != nil {
		return err
	}
	if h.post, err = vec(jlm.RoleVProjNorm, jlm.DenseBlock); err != nil {
		return err
	}
	if h.merge, err = get(jlm.RoleVMergeConv, jlm.DenseBlock); err != nil {
		return err
	}
	if h.mergeB, err = optVec(jlm.RoleVMergeConvBias, jlm.DenseBlock); err != nil {
		return err
	}
	if h.newline, err = vec(jlm.RoleVImgNewline, jlm.DenseBlock); err != nil {
		return err
	}
	if h.begin, err = vec(jlm.RoleVImgBegin, jlm.DenseBlock); err != nil {
		return err
	}
	if h.end, err = vec(jlm.RoleVImgEnd, jlm.DenseBlock); err != nil {
		return err
	}
	if t.projW, err = get(jlm.RoleVProj, jlm.DenseBlock); err != nil {
		return err
	}
	if t.projB, err = optVec(jlm.RoleVProjBias, jlm.DenseBlock); err != nil {
		return err
	}
	if t.projW2, err = get(jlm.RoleVProj2, jlm.DenseBlock); err != nil {
		return err
	}
	if t.projB2, err = optVec(jlm.RoleVProj2Bias, jlm.DenseBlock); err != nil {
		return err
	}
	h.mergeW, h.lineWidth = h.merge.rows, t.projW.rows
	switch {
	case len(h.pre) != E || len(h.post) != P:
		return fmt.Errorf("model: OpenTower: the merger's norms are %d and %d floats, want %d and %d",
			len(h.pre), len(h.post), E, P)
	case h.merge.k != E*c.Scale*c.Scale || t.projW.k != h.merge.rows:
		return fmt.Errorf("model: OpenTower: the merger's convolutions are %dx%d then %dx%d from a %d-wide row",
			h.merge.k, h.merge.rows, t.projW.k, t.projW.rows, E*c.Scale*c.Scale)
	case len(h.newline) != t.projW.rows || t.projW2.k != t.projW.rows || t.projW2.rows != P:
		return fmt.Errorf("model: OpenTower: the merger's newline is %d floats and its linear %dx%d, around %d to %d",
			len(h.newline), t.projW2.k, t.projW2.rows, t.projW.rows, P)
	case len(h.begin) != P || len(h.end) != P:
		return fmt.Errorf("model: OpenTower: the begin and end rows are %d and %d floats, want %d",
			len(h.begin), len(h.end), P)
	}
	t.hy = h
	return nil
}

// hyMaxTokens is the most rows one picture makes: its merged units, a
// newline per merged row (at most one per unit), and the begin and end.
func (c TowerConfig) hyMaxTokens() int { return 2*c.Tokens() + 2 }

// gridOf is the ImageGrid of an H x W merged grid: HunyuanVL's carries its
// newline column and its two end rows (ImageGrid.Line, Ends).
func (c TowerConfig) gridOf(h, w int) ImageGrid {
	if c.Kind == jlm.ProjHunyuanVL {
		return ImageGrid{T: 1, H: h, W: w, Line: 1, Ends: 1}
	}
	return ImageGrid{T: 1, H: h, W: w}
}

// projectHunyuan is HunyuanVL's merger over a picture's raster of patch rows
// into v.out: H*(W+1)+2 rows of ProjDim.
func (s *Instance) projectHunyuan(patches []float32) ([]float32, error) {
	v := s.vis
	t, c := v.t, v.t.Cfg
	h := t.hy
	gh, gw := s.patchGrid()
	E, sc, P := c.NEmbd, c.Scale, c.ProjDim
	n := gh * gw
	s.normRows(patches, patches, h.pre, nil, E, n, false, h.eps, s.rowChunk(n, E))
	gsh, gsw, w := gh/sc, gw/sc, E*sc*sc
	nt := gsh * gsw
	for gy := 0; gy < gsh; gy++ {
		for gx := 0; gx < gsw; gx++ {
			dst := v.grp[(gy*gsw+gx)*w : (gy*gsw+gx+1)*w]
			for dy := 0; dy < sc; dy++ {
				for dx := 0; dx < sc; dx++ {
					q := (gy*sc+dy)*gw + gx*sc + dx
					slot := dy*sc + dx
					if c.fault == towerFaultShuffleSwap {
						slot = dx*sc + dy
					}
					copy(dst[slot*E:], patches[q*E:(q+1)*E])
				}
			}
		}
	}
	v.stage("proj_in", v.grp[:nt*w])
	s.jit.NewInput()
	conv := v.hid[:nt*h.mergeW]
	if err := s.mm(conv, h.merge, v.grp, nt); err != nil {
		return nil, err
	}
	s.visBiasRows(conv, h.mergeB, nt)
	s.actAll(conv, nn.ActGELU)
	// The 1x1 convolution, each merged row then its newline: the rows the
	// linear reads are the grid's raster with a column appended.
	s.jit.NewInput()
	L := h.lineWidth
	lines := v.hyLines[:gsh*(gsw+1)*L]
	for gy := 0; gy < gsh; gy++ {
		row := lines[gy*(gsw+1)*L : (gy+1)*(gsw+1)*L]
		if err := s.mm(row[:gsw*L], t.projW, conv[gy*gsw*h.mergeW:(gy+1)*gsw*h.mergeW], gsw); err != nil {
			return nil, err
		}
		s.visBiasRows(row[:gsw*L], t.projB, gsw)
		nl := h.newline
		if c.fault == faultHyNoNewline {
			nl = row[(gsw-1)*L : gsw*L]
		}
		copy(row[gsw*L:], nl)
	}
	rows := gsh*(gsw+1) + 2
	out := v.out[:rows*P]
	s.jit.NewInput()
	if err := s.mm(out[P:(rows-1)*P], t.projW2, lines, gsh*(gsw+1)); err != nil {
		return nil, err
	}
	s.visBiasRows(out[P:(rows-1)*P], t.projB2, gsh*(gsw+1))
	copy(out[:P], h.begin)
	copy(out[(rows-1)*P:], h.end)
	s.normRows(out, out, h.post, nil, P, rows, false, h.eps, s.rowChunk(rows, P))
	v.stage("out", out)
	return out, nil
}
