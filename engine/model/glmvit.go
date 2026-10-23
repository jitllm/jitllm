package model

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// GLM-4.xV's tower (jlm.ProjGLM4V): Qwen2.5-VL's RMSNorm, gated-SiLU block
// with the 2-D rotary and no windows, and three pieces of its own.
//
//	the entry       the patch embedding RMSNormed (RoleVEmbdNorm), then a
//	                learned table resampled BICUBIC to the picture's grid
//	                and added (transformers' F.grid_sample with
//	                align_corners False and border padding; llama.cpp's
//	                GGML_SCALE_MODE_BICUBIC)
//	the merge       the post-norm, then a 2x2 convolution over each merge
//	                group -- a matrix over the shuffled row, laid out
//	                (dy, dx, channel) by the converter
//	the projector   linear, LayerNorm, GELU, then a gated SiLU MLP
//
// The table's taps and weights are setup per grid, the rows generated axpys
// (posLerp, 16 taps), and every matrix the packed matmul.

// glmMinPixels is GLM-4.xV's processor floor per frame: preprocessor_config's
// size.shortest_edge (112x112) bounds a still image's two temporal frames
// together (glmResize).
const glmMinPixels = 112 * 112 / 2

// glmProj is GLM-4.xV's projector: the merging convolution and the MLP after
// it. projW (the linear), projNorm and projNormB (the LayerNorm) are the
// Tower's own fields.
type glmProj struct {
	merge             tensor
	mergeB            []float32
	gate, up, down    tensor
	gateB, upB, downB []float32
}

// loadGLM4V reads the entry's norm and the projector.
func (t *Tower) loadGLM4V(c0 *jlm.File, get func(jlm.Role, int32) (tensor, error),
	vec, optVec func(jlm.Role, int32) ([]float32, error)) error {
	c := &t.Cfg
	E, w := c.NEmbd, c.NEmbd*c.Scale*c.Scale
	var err error
	if t.embdNorm, err = vec(jlm.RoleVEmbdNorm, jlm.DenseBlock); err != nil {
		return err
	}
	if len(t.embdNorm) != E {
		return fmt.Errorf("model: OpenTower: the patch embedding's norm is %d floats, want %d", len(t.embdNorm), E)
	}
	rows := len(t.posW) / E
	side := int(math.Sqrt(float64(rows)))
	if side < 4 || side*side != rows || rows*E != len(t.posW) {
		return fmt.Errorf("model: OpenTower: the position table is %d floats, not a square of %d-wide rows",
			len(t.posW), E)
	}
	c.PosSide = side
	if t.postLnW == nil || t.postLnB != nil {
		return fmt.Errorf("model: OpenTower: a glm4v tower needs its post RMSNorm (v.post_ln, weight only)")
	}
	g := &glmProj{}
	t.glm = g
	if g.merge, err = get(jlm.RoleVMergeConv, jlm.DenseBlock); err != nil {
		return err
	}
	if g.mergeB, err = optVec(jlm.RoleVMergeConvBias, jlm.DenseBlock); err != nil {
		return err
	}
	if t.projW, err = get(jlm.RoleVProj, jlm.DenseBlock); err != nil {
		return err
	}
	if t.projNorm, err = vec(jlm.RoleVProjNorm, jlm.DenseBlock); err != nil {
		return err
	}
	if t.projNormB, err = vec(jlm.RoleVProjNormBias, jlm.DenseBlock); err != nil {
		return err
	}
	for _, m := range []struct {
		role, biasRole jlm.Role
		w              *tensor
		b              *[]float32
	}{{jlm.RoleVProjGate, jlm.RoleVProjGateBias, &g.gate, &g.gateB},
		{jlm.RoleVProjUp, jlm.RoleVProjUpBias, &g.up, &g.upB},
		{jlm.RoleVProjDown, jlm.RoleVProjDownBias, &g.down, &g.downB}} {
		if *m.w, err = get(m.role, jlm.DenseBlock); err != nil {
			return err
		}
		if *m.b, err = optVec(m.biasRole, jlm.DenseBlock); err != nil {
			return err
		}
	}
	P, ctx := c.ProjDim, g.gate.rows
	switch {
	case g.merge.k != w || g.merge.rows != P:
		return fmt.Errorf("model: OpenTower: the merging convolution is %dx%d, want %d to %d", g.merge.k,
			g.merge.rows, w, P)
	case t.projW.k != P || t.projW.rows != P || len(t.projNorm) != P || len(t.projNormB) != P:
		return fmt.Errorf("model: OpenTower: the projector's linear is %dx%d, want %d square", t.projW.k,
			t.projW.rows, P)
	case g.gate.k != P || g.up.k != P || g.up.rows != ctx || g.down.k != ctx || g.down.rows != P:
		return fmt.Errorf("model: OpenTower: the projector's MLP is %dx%d, %dx%d and %dx%d around %d",
			g.gate.k, g.gate.rows, g.up.k, g.up.rows, g.down.k, g.down.rows, P)
	case g.mergeB != nil && len(g.mergeB) != P:
		return fmt.Errorf("model: OpenTower: the merging convolution's bias is %d floats, want %d", len(g.mergeB), P)
	}
	return nil
}

// projectGLM is GLM-4.xV's projector over one picture's patch rows (the
// post-norm already applied): the 2x2 merge, linear, LayerNorm, GELU and the
// gated MLP, into v.out.
func (s *State) projectGLM(patches []float32) ([]float32, error) {
	v := s.vis
	t, c := v.t, v.t.Cfg
	g := t.glm
	gh, gw := s.patchGrid()
	E, sc, P := c.NEmbd, c.Scale, c.ProjDim
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
	if err := s.mm(v.hid, g.merge, v.grp, nt); err != nil {
		return nil, err
	}
	s.visBiasRows(v.hid, g.mergeB, nt)
	s.jit.NewInput()
	x := v.out[:nt*P]
	if err := s.mm(x, t.projW, v.hid, nt); err != nil {
		return nil, err
	}
	s.visBiasRows(x, t.projB, nt)
	s.normRows(x, x, t.projNorm, t.projNormB, P, nt, true, 1e-5, s.rowChunk(nt, P))
	// GELU: GELU-tanh where the references take erf, as every projector
	// here does (a deliberate divergence under 1e-3, project's note).
	s.actAll(x, nn.ActGELU)
	s.jit.NewInput()
	ctx := g.gate.rows
	gate, up := v.glmGate[:nt*ctx], v.glmUp[:nt*ctx]
	if err := s.mm(gate, g.gate, x, nt); err != nil {
		return nil, err
	}
	s.visBiasRows(gate, g.gateB, nt)
	if err := s.mm(up, g.up, x, nt); err != nil {
		return nil, err
	}
	s.visBiasRows(up, g.upB, nt)
	s.actmulAll(gate, up, nn.ActSiLU)
	s.jit.NewInput()
	out := v.grp[:nt*P]
	if err := s.mm(out, g.down, gate, nt); err != nil {
		return nil, err
	}
	s.visBiasRows(out, g.downB, nt)
	copy(v.out[:nt*P], out)
	v.stage("out", v.out[:nt*P])
	return v.out[:nt*P], nil
}

// glmResize is GLM-4.xV's smart_resize. Its processor feeds a still image as
// its two temporal frames and bounds frames*h*w, so with the tower's bounds
// per frame (MinPixels, MaxPixels: half the processor's) it is Qwen's, after a
// side shorter than one merge unit is scaled up.
func (c TowerConfig) glmResize(w, h int) (rw, rh int, err error) {
	f := float64(c.PatchSz * c.Scale)
	hf, wf := float64(h), float64(w)
	if hf < f || wf < f {
		s := math.Max(f/hf, f/wf)
		hf, wf = math.Trunc(hf*s), math.Trunc(wf*s)
	}
	if math.Max(hf, wf)/math.Min(hf, wf) > 200 {
		return 0, 0, fmt.Errorf("model: a %dx%d picture's aspect is over 200, which the reference refuses", w, h)
	}
	hb := math.RoundToEven(hf/f) * f
	wb := math.RoundToEven(wf/f) * f
	switch {
	case hb*wb > float64(c.MaxPixels):
		beta := math.Sqrt(hf * wf / float64(c.MaxPixels))
		hb = math.Max(f, math.Floor(hf/beta/f)*f)
		wb = math.Max(f, math.Floor(wf/beta/f)*f)
	case hb*wb < float64(c.MinPixels):
		beta := math.Sqrt(float64(c.MinPixels) / (hf * wf))
		hb = math.Ceil(hf*beta/f) * f
		wb = math.Ceil(wf*beta/f) * f
	}
	return int(wb), int(hb), nil
}
