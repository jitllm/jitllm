package model

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// Qwen3-VL's tower (jlm.ProjQwen3VL): Qwen2-VL's 2-D rotary block in
// LayerNorm and GELU-tanh, with two pieces of its own.
//
//	the position table   a learned PosSide x PosSide table, resampled to each
//	                     picture's patch grid by bilinear interpolation with
//	                     aligned corners and added after the patch embedding
//	                     (transformers' get_vision_interpolation_indices_and_
//	                     weights; llama.cpp's resize_position_embeddings)
//	deepstack            after each tapped block, a merger of its own -- the
//	                     2x2 shuffle FIRST, then a LayerNorm over the shuffled
//	                     row, linear, GELU, linear -- whose rows the text
//	                     model adds into the picture's rows after its block k
//	                     for tap k (Span.Deep)
//
// Both are the tower's own arithmetic on generated kernels: the table's
// weights and indices are setup per grid, the rows are axpys and the merger is
// the main merger's matmuls. The deepstack rows leave the tower beside the
// merged ones, [taps][rows][ProjDim] after them in one buffer, so the image
// cache keeps a picture whole.

// qwen3MinPixels is Qwen3-VL's processor floor (preprocessor_config's
// size.shortest_edge, 256x256). llama.cpp's is 8 tokens; the processor is
// what the weights saw (RULE 7m).
const qwen3MinPixels = 65536

// deepstack is one tap's merger: a LayerNorm over the 2x2-shuffled row, then
// linear, GELU, linear.
type deepstack struct {
	block       int // the tower block (segment-local) whose output it reads
	norm, normB []float32
	fc1, fc2    tensor
	fc1B, fc2B  []float32
}

// loadQwen3VL reads the position table's side and the deepstack mergers.
// get and vec read a tensor by (role, block, index).
func (t *Tower) loadQwen3VL(c0 *jlm.File, get func(jlm.Role, int32, int32) (tensor, error),
	vec func(jlm.Role, int32, int32) ([]float32, error)) error {
	c := &t.Cfg
	rows := len(t.posW) / c.NEmbd
	side := int(math.Sqrt(float64(rows)))
	if side < 2 || side*side != rows || rows*c.NEmbd != len(t.posW) {
		return fmt.Errorf("model: OpenTower: the position table is %d floats, not a square of %d-wide rows",
			len(t.posW), c.NEmbd)
	}
	c.PosSide = side
	w := c.NEmbd * c.Scale * c.Scale
	for b := 0; b < c.NLayer; b++ {
		bi := int32(b)
		if !c0.Has(jlm.RoleVDsFC1, jlm.DenseBlock, bi) {
			continue
		}
		d := deepstack{block: b}
		var err error
		if d.fc1, err = get(jlm.RoleVDsFC1, jlm.DenseBlock, bi); err != nil {
			return err
		}
		if d.fc2, err = get(jlm.RoleVDsFC2, jlm.DenseBlock, bi); err != nil {
			return err
		}
		for _, v := range []struct {
			role jlm.Role
			dst  *[]float32
		}{{jlm.RoleVDsNorm, &d.norm}, {jlm.RoleVDsNormBias, &d.normB},
			{jlm.RoleVDsFC1Bias, &d.fc1B}, {jlm.RoleVDsFC2Bias, &d.fc2B}} {
			if *v.dst, err = vec(v.role, jlm.DenseBlock, bi); err != nil {
				return err
			}
		}
		if d.fc1.k != w || d.fc2.k != d.fc1.rows || d.fc2.rows != c.ProjDim || len(d.norm) != w ||
			len(d.normB) != w || len(d.fc1B) != d.fc1.rows || len(d.fc2B) != c.ProjDim {
			return fmt.Errorf("model: OpenTower: deepstack tap %d is %dx%d then %dx%d, want a %d-wide row to %d",
				b, d.fc1.k, d.fc1.rows, d.fc2.k, d.fc2.rows, w, c.ProjDim)
		}
		t.deep = append(t.deep, d)
		c.Deep = append(c.Deep, b)
	}
	return nil
}

// RowWidth is the floats one emitted row carries in the tower's output: the
// merged row and, for a deepstack tower, one row per tap after them.
func (c TowerConfig) RowWidth() int { return c.ProjDim * (1 + len(c.Deep)) }

// posLerp adds to the rows x the position rows of the gh x gw grid's n
// patches: per patch, the table rows its two axes' taps cross at, weighted by
// the product of the axes' weights, summed in the reference's order into
// v.lerpTab and then added, as it adds them. The taps and weights are each
// reference's own float32 arithmetic (axisChain) on generated code: AxisTaps
// per axis, LerpGrid for their outer product.
func (s *State) posLerp(x []float32, gh, gw, n int) {
	v := s.vis
	c := v.t.Cfg
	E, side := c.NEmbd, c.PosSide
	// The stated filter: bilinear reads 2x2 table rows per patch, the
	// bicubics 4x4. The fault reads a bicubic table bilinearly.
	f := nn.AxisBilinear
	switch {
	case c.fault == faultBilinearTable:
	case c.conv.PosFilter == jlm.FilterLinear:
		f = nn.AxisLinear
	case c.conv.PosFilter == jlm.FilterCubic:
		f = nn.AxisCubic
	}
	per := nn.AxisTapCount(f)
	taps, np := per*per, gh*gw
	if v.lerpAt != [3]int{gh, gw, int(f)} || c.fault == faultNoPosLerp {
		// Sized by the picture, not the tower's largest grid: a
		// dynamic tower's grid is far past any picture (65536 patches on
		// HunyuanOCR).
		if cap(v.lerpW) < 16*np {
			v.lerpW = make([]float32, 16*np)
			v.lerpI = make([]int32, 16*np)
		}
		if g := 4 * (gh + gw); cap(v.axW) < g {
			v.axW, v.axI = make([]float32, g), make([]int32, g)
		}
		v.lerpW, v.lerpI = v.lerpW[:taps*np], v.lerpI[:taps*np]
		// The axes' taps and weights, [tap][index], rows then columns.
		hw, ht := v.axW[:per*gh], v.axI[:per*gh]
		ww, wt := v.axW[per*gh:per*(gh+gw)], v.axI[per*gh:per*(gh+gw)]
		nn.AxisTapsJIT(f, hw, ht, gh, c.axisChain(gh, f), side-1)
		nn.AxisTapsJIT(f, ww, wt, gw, c.axisChain(gw, f), side-1)
		// Tap (a, b) of every grid patch, a plane in raster order.
		for a := 0; a < per; a++ {
			for b := 0; b < per; b++ {
				k := (a*per + b) * np
				nn.LerpGridJIT(v.lerpW[k:k+np], v.lerpI[k:k+np], hw[a*gh:(a+1)*gh], ht[a*gh:(a+1)*gh],
					ww[b*gw:(b+1)*gw], wt[b*gw:(b+1)*gw], side)
			}
		}
		// The violation: the table read at the patch's own row, unresampled.
		if c.fault == faultNoPosLerp {
			clear(v.lerpW)
			for p := 0; p < np; p++ {
				v.lerpI[p], v.lerpW[p] = int32(p%(side*side)), 1
			}
		}
		v.lerpAt = [3]int{gh, gw, int(f)}
	}
	if len(v.lerpTab) < n*E {
		v.lerpTab = make([]float32, n*E)
	}
	j := &s.rg.rows
	j.op, j.dst, j.src, j.b, j.w, j.idx, j.dim, j.side, j.gw = rowLerp, v.lerpTab, x, v.t.posW, v.lerpW, v.lerpI, E,
		taps, np
	s.rowRun(n, s.rowChunk(n, taps*E))
}

// axisChain is this tower's position-table resample on one axis of size
// outputs, as the nine float32 steps AxisTaps takes to its source coordinate:
//
//	bilinear (Qwen3-VL):          i*(side-1)/(size-1), aligned corners
//	grid_sample (GLM-4.xV):       (((i+0.5)/size*2 - 1) + 1)*side - 1)/2
//	interpolate (Kimi-VL bicubic,
//	HunyuanVL bilinear):          (i+0.5)*(side/size) - 0.5
//
// each in its reference's order, every step it does not take an identity.
func (c TowerConfig) axisChain(size int, f nn.AxisFilter) [9]float32 {
	side := float32(c.PosSide)
	chain := c.conv.PosChain
	if c.fault == faultBilinearTable {
		chain = jlm.ChainAligned
	}
	switch chain {
	case jlm.ChainAligned:
		return [9]float32{0, side - 1, float32(max(size-1, 1)), 1, 0, 0, 1, 0, 1}
	case jlm.ChainHalfPixel:
		return [9]float32{0.5, side / float32(size), 1, 1, -0.5, 0, 1, 0, 1}
	}
	return [9]float32{0.5, 1, float32(size), 2, -1, 1, side, -1, 2}
}

// lerpRows is rows lo..hi-1 of posLerp: each row's position row, the
// weighted sum of its patch's table rows (j.side of them, each a plane j.gw
// apart), added into it.
func (s *State) lerpRows(lo, hi int) {
	j := &s.rg.rows
	d := j.dim
	for i := lo; i < hi; i++ {
		pos := j.dst[i*d : (i+1)*d]
		clear(pos)
		p := s.patchAt(i)
		for k := 0; k < j.side; k++ {
			r := int(j.idx[k*j.gw+p])
			nn.Axpy32JIT(pos, j.b[r*d:(r+1)*d], j.w[k*j.gw+p])
		}
		nn.Axpy32JIT(j.src[i*d:(i+1)*d], pos, 1)
	}
}

// tapAfter is the first deepstack tap at or after segment-local block b, or -1.
func (c TowerConfig) tapAfter(b int) int {
	for _, d := range c.Deep {
		if d >= b {
			return d
		}
	}
	return -1
}

// deepstackAt runs tap k's merger over the residual x of the encode in
// progress, into its rows of v.out.
func (s *State) deepstackAt(k int, x []float32) error {
	v := s.vis
	t, c := v.t, v.t.Cfg
	d := &t.deep[k]
	p := &v.prog
	E, sc := c.NEmbd, c.Scale
	gsh, gsw, w := p.gh/sc, p.gw/sc, E*sc*sc
	nt := gsh * gsw
	// The shuffle: group (gy, gx) collects its sc x sc patches, dy outer and
	// dx inner, as the main merger's (project).
	for gy := 0; gy < gsh; gy++ {
		for gx := 0; gx < gsw; gx++ {
			dst := v.grp[(gy*gsw+gx)*w : (gy*gsw+gx+1)*w]
			for dy := 0; dy < sc; dy++ {
				for dx := 0; dx < sc; dx++ {
					q := (gy*sc+dy)*p.gw + gx*sc + dx
					copy(dst[(dy*sc+dx)*E:], x[(p.first+q)*E:(p.first+q+1)*E])
				}
			}
		}
	}
	grp := v.grp[:nt*w]
	if c.fault != faultDeepPreNorm {
		s.normRows(grp, grp, d.norm, d.normB, w, nt, true, c.Eps, s.rowChunk(nt, w))
	}
	s.jit.NewInput()
	if err := s.mm(v.hid, d.fc1, grp, nt); err != nil {
		return err
	}
	s.visBiasRows(v.hid, d.fc1B, nt)
	s.actAll(v.hid[:nt*d.fc1.rows], nn.ActGELU)
	s.jit.NewInput()
	out := v.out[(1+k)*nt*c.ProjDim : (2+k)*nt*c.ProjDim]
	if err := s.mm(out, d.fc2, v.hid, nt); err != nil {
		return err
	}
	s.visBiasRows(out, d.fc2B, nt)
	return nil
}
