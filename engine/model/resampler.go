package model

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// MiniCPM-V's projector: a perceiver resampler.
//
// It is one cross-attention layer. Queries learned rows attend over the
// tower's patches; the keys carry a 2-D sin/cos position computed for the
// patch grid in hand and the values carry none, so an image or slice of any
// shape becomes exactly Queries tokens. The graph is the reference's
// (resampler.py; llama.cpp's minicpmv.cpp):
//
//	v   = LN_kv(patches * W_kv)
//	k   = v + sincos2d(grid)
//	q   = LN_q(query)
//	out = proj * LN_post(MHA(q, k, v))
//
// with 128-wide heads, which no file states: resampler.py builds the attention
// with num_heads = embed_dim // 128, and llama.cpp hardcodes d_head = 128.

// rsHeadDim is the resampler's head width; see the comment above.
const rsHeadDim = 128

// rsPosBase is the sin/cos frequency base, the reference's 10000.
const rsPosBase = 10000.0

type resampler struct {
	d, heads, queries int
	query             []float32 // [queries][d], read once
	lnQW, lnQB        []float32
	lnKVW, lnKVB      []float32
	lnPostW, lnPostB  []float32
	bq, bk, bv, bo    []float32
	// kv is NEmbd -> d with no bias; the rest are d -> d. proj has no bias.
	kv, wq, wk, wv, wo, proj tensor
}

func (r *resampler) weights() []*tensor {
	return []*tensor{&r.kv, &r.wq, &r.wk, &r.wv, &r.wo, &r.proj}
}

// loadResampler reads the projector's tensors. get and vec are buildTower's
// loaders, so every refusal names the tensor the same way.
func loadResampler(c *TowerConfig, get func(jlm.Role, int32) (tensor, error),
	vec func(jlm.Role, int32) ([]float32, error)) (*resampler, error) {
	r := &resampler{}
	var err error
	db := int32(jlm.DenseBlock)
	for _, m := range []struct {
		role jlm.Role
		dst  *tensor
	}{
		{jlm.RoleVRsKV, &r.kv}, {jlm.RoleVRsQ, &r.wq}, {jlm.RoleVRsK, &r.wk},
		{jlm.RoleVRsV, &r.wv}, {jlm.RoleVRsOut, &r.wo}, {jlm.RoleVRsProj, &r.proj},
	} {
		if *m.dst, err = get(m.role, db); err != nil {
			return nil, err
		}
	}
	for _, v := range []struct {
		role jlm.Role
		dst  *[]float32
	}{
		{jlm.RoleVRsQuery, &r.query},
		{jlm.RoleVRsLnQ, &r.lnQW}, {jlm.RoleVRsLnQBias, &r.lnQB},
		{jlm.RoleVRsLnKV, &r.lnKVW}, {jlm.RoleVRsLnKVBias, &r.lnKVB},
		{jlm.RoleVRsLnPost, &r.lnPostW}, {jlm.RoleVRsLnPostBias, &r.lnPostB},
		{jlm.RoleVRsQBias, &r.bq}, {jlm.RoleVRsKBias, &r.bk},
		{jlm.RoleVRsVBias, &r.bv}, {jlm.RoleVRsOutBias, &r.bo},
	} {
		if *v.dst, err = vec(v.role, db); err != nil {
			return nil, err
		}
	}
	r.d = r.kv.rows
	if r.kv.k != c.NEmbd {
		return nil, fmt.Errorf("model: OpenTower: the resampler's kv projection reads %d, the tower emits %d",
			r.kv.k, c.NEmbd)
	}
	if r.d%rsHeadDim != 0 || r.d%4 != 0 {
		return nil, fmt.Errorf("model: OpenTower: the resampler is %d wide, not a whole number of %d-wide heads",
			r.d, rsHeadDim)
	}
	r.heads = r.d / rsHeadDim
	if len(r.query)%r.d != 0 || len(r.query) == 0 {
		return nil, fmt.Errorf("model: OpenTower: the resampler's query is %d floats, not rows of %d",
			len(r.query), r.d)
	}
	r.queries = len(r.query) / r.d
	for _, w := range []*tensor{&r.wq, &r.wk, &r.wv, &r.wo, &r.proj} {
		if w.k != r.d || w.rows != r.d {
			return nil, fmt.Errorf("model: OpenTower: a resampler matrix is %dx%d, want %dx%d",
				w.rows, w.k, r.d, r.d)
		}
	}
	for _, v := range [][]float32{r.lnQW, r.lnQB, r.lnKVW, r.lnKVB, r.lnPostW, r.lnPostB,
		r.bq, r.bk, r.bv, r.bo} {
		if len(v) != r.d {
			return nil, fmt.Errorf("model: OpenTower: a resampler vector is %d floats, want %d", len(v), r.d)
		}
	}
	if r.proj.rows != c.ProjDim {
		return nil, fmt.Errorf("model: OpenTower: the resampler emits %d, want %d", r.proj.rows, c.ProjDim)
	}
	return r, nil
}

// rsState is the resampler's working memory on a vision State.
type rsState struct {
	r *resampler
	// attn is the resampler's attention kernels: 128-wide heads at stride d,
	// a geometry of its own and a set of its own on the State's one JIT.
	attn *nn.AttnSet
	// qp is the projected queries, W_q * LN_q(query) + b_q: a property of the
	// weights alone, computed on the first encode (qReady).
	qp       []float32 // [queries][d]
	qReady   bool
	v, k, kp []float32 // [maxSeq][d]
	o, o2    []float32 // [queries][d]
	att      []float32 // [heads][attStride]
	acc      []float32 // [heads][rsHeadDim]
	stride   int
	// The sin/cos of each axis coordinate, [coordinate][d/4], rebuilt when the
	// grid's side grows. Geometry, built once per grid like visRope.
	sinW, cosW, sinH, cosH []float32
	tabW, tabH             int
}

func newRsState(r *resampler, jit *nn.JIT, maxSeq int) *rsState {
	st := &rsState{
		r:      r,
		attn:   jit.AttnSetFor(rsHeadDim, rsHeadDim, r.d, cpu.KVF32),
		qp:     make([]float32, r.queries*r.d),
		v:      make([]float32, maxSeq*r.d),
		k:      make([]float32, maxSeq*r.d),
		kp:     make([]float32, maxSeq*r.d),
		o:      make([]float32, r.queries*r.d),
		o2:     make([]float32, r.queries*r.d),
		stride: nn.SoftmaxPad(maxSeq),
	}
	st.att = make([]float32, r.heads*st.stride)
	st.acc = make([]float32, r.heads*rsHeadDim)
	return st
}

// axes builds the per-axis sin/cos for a grid of gh x gw, the reference's
// get_1d_sincos_pos_embed_from_grid_new: omega_i = base^(-i/(d/4)) for
// i < d/4, and a coordinate p gives [sin(p*omega), cos(p*omega)], omega and
// the product in float32 as the reference computes them -- on generated code
// (nn.SinCosTabJIT).
func (st *rsState) axes(gh, gw int) {
	q := st.r.d / 4
	build := func(n int, sin, cos *[]float32) {
		*sin, *cos = make([]float32, n*q), make([]float32, n*q)
		nn.SinCosTabJIT(*sin, *cos, q, n, rsPosBase)
	}
	if gw > st.tabW {
		build(gw, &st.sinW, &st.cosW)
		st.tabW = gw
	}
	if gh > st.tabH {
		build(gh, &st.sinH, &st.cosH)
		st.tabH = gh
	}
}

// resample turns n = gh*gw patch rows of the tower's output into Queries rows
// of the text model's width, written to out.
func (s *State) resample(out, patches []float32, gh, gw int) error {
	st, r := s.vis.rs, s.vis.t.rs
	n, d := gh*gw, r.d
	eps := s.vis.t.Cfg.Eps
	lnRows := func(x, w, b []float32, rows int) {
		s.normRows(x, x, w, b, d, rows, true, eps, s.rowChunk(rows, d))
	}
	// The queries depend on the weights alone.
	if !st.qReady {
		q := make([]float32, len(r.query))
		copy(q, r.query)
		if s.vis.fault != faultQueryRaw {
			lnRows(q, r.lnQW, r.lnQB, r.queries)
		}
		if err := s.mm(st.qp, r.wq, q, r.queries); err != nil {
			return err
		}
		s.addBiasRows(st.qp, r.bq, r.queries)
		st.qReady = true
	}
	// v = LN_kv(patches * W_kv); k = v + position.
	v, k := st.v[:n*d], st.k[:n*d]
	if err := s.mm(v, r.kv, patches, n); err != nil {
		return err
	}
	lnRows(v, r.lnKVW, r.lnKVB, n)
	st.axes(gh, gw)
	j := &s.rg.rows
	j.op, j.dst, j.src, j.dim, j.gw = rowKeyPos, k, v, d, gw
	s.rowRun(n, s.rowChunk(n, d))
	// K = W_k k + b_k into kp; V = W_v v + b_v into k, which is free now.
	kp := st.kp[:n*d]
	if err := s.mm(kp, r.wk, k, n); err != nil {
		return err
	}
	s.addBiasRows(kp, r.bk, n)
	vp := k
	if err := s.mm(vp, r.wv, v, n); err != nil {
		return err
	}
	s.addBiasRows(vp, r.bv, n)

	// Every query over every patch, head by head on the pool: no mask, since
	// a grid of any size is n real keys.
	j = &s.rg.rows
	j.op, j.src, j.b, j.dim, j.alpha = rowCross, kp, vp, n, float32(1/math.Sqrt(rsHeadDim))
	s.rowRun(r.heads, 1)
	if err := s.mm(st.o2, r.wo, st.o, r.queries); err != nil {
		return err
	}
	s.addBiasRows(st.o2, r.bo, r.queries)
	lnRows(st.o2, r.lnPostW, r.lnPostB, r.queries)
	return s.mm(out, r.proj, st.o2, r.queries)
}

// bucketPositions is the tower's position-table row for every patch of a
// gh x gw grid, the NaViT bucketing MiniCPM-V's SigLIP does
// (modeling_navit_siglip.py): a patch's fractional coordinate i/n on each axis
// is bucketed against the boundaries k/B, so a grid of any shape reads a
// B x B table. The bucket is floor(B*i/n), in exact integers.
//
// That is the rational the reference intends and what llama.cpp computes, and
// it is NOT bit-for-bit what the reference's float32 does: torch.arange builds
// both i/n and k/B in float32 -- on an AVX2 host, sixteen at a time from a
// base rounded to float32 -- so where i/n lands exactly on a boundary the two
// roundings pick the bucket: they part from the rational by one bucket at 1078
// coordinates of the axis lengths below 1500 (TestBucketPositionsMatchTorch
// counts them), though at none of a 448 slice's 32. The rounding depends on
// the vector width and the device, so a model trained on GPUs never saw the
// CPU's; the rational is the choice (RULE 7m).
func bucketPositions(dst []int32, gh, gw, B int) {
	for i := 0; i < gh; i++ {
		bh := int32(B * i / gh)
		for j := 0; j < gw; j++ {
			dst[i*gw+j] = bh*int32(B) + int32(B*j/gw)
		}
	}
}

// keyPosRows is rows lo..hi-1 of the resampler's keys (rowKeyPos): each
// patch row of v plus its 2-D position -- the w axis first, then the h axis,
// each sin then cos; resampler.py meshgrids (grid_w, grid_h), so its "emb_h"
// half is the COLUMN coordinate, and llama.cpp's concat(pos_w, pos_h) says the
// same. The four quarters are added in place: elementwise, so a quarter at a
// time is the bits of the whole row.
func (s *State) keyPosRows(lo, hi int) {
	j, st := &s.rg.rows, s.vis.rs
	d, gw := j.dim, j.gw
	q := d / 4
	for p := lo; p < hi; p++ {
		kr := j.dst[p*d : (p+1)*d]
		copy(kr, j.src[p*d:(p+1)*d])
		if s.vis.fault == faultNoKeyPos {
			continue
		}
		y, x := p/gw, p%gw
		nn.Axpy32JIT(kr[0:q], st.sinW[x*q:(x+1)*q], 1)
		nn.Axpy32JIT(kr[q:2*q], st.cosW[x*q:(x+1)*q], 1)
		nn.Axpy32JIT(kr[2*q:3*q], st.sinH[y*q:(y+1)*q], 1)
		nn.Axpy32JIT(kr[3*q:4*q], st.cosH[y*q:(y+1)*q], 1)
	}
}

// crossHeads is the resampler's cross-attention for heads lo..hi-1 (rowCross):
// every learned query over the n patch keys in src and values in b.
func (s *State) crossHeads(lo, hi int) {
	j, st, r := &s.rg.rows, s.vis.rs, s.vis.t.rs
	n, d := j.dim, r.d
	for h := lo; h < hi; h++ {
		off := h * rsHeadDim
		att := st.att[h*st.stride : h*st.stride+n]
		acc := st.acc[h*rsHeadDim : (h+1)*rsHeadDim]
		for qi := 0; qi < r.queries; qi++ {
			st.attn.AttnScores(att, j.src[off:], st.qp[qi*d+off:], n)
			nn.Scale32JIT(att, j.alpha)
			nn.Softmax32JIT(att, n)
			st.attn.AttnAcc(acc, j.b[off:], att, n)
			copy(st.o[qi*d+off:qi*d+off+rsHeadDim], acc)
		}
	}
}
