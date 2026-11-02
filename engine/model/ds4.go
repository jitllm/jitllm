package model

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/cpu"
)

// DeepSeek V4 (transformers' DeepseekV4ForCausalLM, llama.cpp's
// deepseek4.cpp; jlm.ArchDeepseek4).
//
// The residual is Config.HCMult streams, stream-major as Gemma 3n's are
// (Config.ResidW: stream k of row r at (k*rows + r)*n_embd). Each sublayer is
// wrapped in a hyper-connection: `pre` collapses the streams into its input,
// and `post` times its output plus the Sinkhorn-normalised mix of the streams
// (comb, applied transposed) are the next streams. The attention is one key
// head that is also the value, rotated on its last NRot dimensions and back
// after the softmax, with a sink per head and a window of SWAWindow on every
// block. A compressed block (Config.CompKinds) also attends to pooled
// ENTRIES, one per `rate` positions: HCA every visible entry, CSA the
// indexer's IdxTopK. A position's cached row holds its key and, on a
// compressed block, the compressor's two projections, so the window's pages
// hold every input the compressor still folds; the entries are a second
// paged history per compressed block (kvCache.ent; ds4kv.go). The FFN is the
// sqrt(softplus) mixture with a shared expert, a clamped SwiGLU, and
// NHashLayers hash-routed lead blocks.
//
// The graph, transcribed: docs/engineering-history/model-correctness.md,
// "engine/model/ds4.go".

// ds4Fault breaks one piece of DeepSeek V4's graph, set only by a gate
// (Config.ds4Fault) to show the fixture can see it. It is the device's type
// (nn.DS4Plan.Fault), so one violation reaches both tiers.
type ds4Fault = nn.DS4Fault

const (
	ds4FaultNone = nn.DS4FaultNone
	// ds4FaultCombSoftmax mixes the streams by the softmax alone, without the
	// Sinkhorn normalisation.
	ds4FaultCombSoftmax = nn.DS4FaultCombSoftmax
	// ds4FaultCombRows reads comb untransposed: x'_k = sum_j comb[k][j] x_j.
	ds4FaultCombRows = nn.DS4FaultCombRows
	// ds4FaultNoDerope leaves the attention output rotated.
	ds4FaultNoDerope = nn.DS4FaultNoDerope
	// ds4FaultWindowOnly attends to the window alone, no entries.
	ds4FaultWindowOnly = nn.DS4FaultWindowOnly
	// ds4FaultNoOverlap pools a CSA entry from its own window alone.
	ds4FaultNoOverlap = nn.DS4FaultNoOverlap
	// ds4FaultAllEntries attends to every visible entry on a CSA block too.
	ds4FaultAllEntries = nn.DS4FaultAllEntries
	// ds4FaultMainRope rotates a compressed block at the sliding blocks' base.
	ds4FaultMainRope = nn.DS4FaultMainRope
	// ds4FaultScoreRoute routes the hash blocks by their scores.
	ds4FaultScoreRoute = nn.DS4FaultScoreRoute
	// ds4FaultNoSinks drops the sinks.
	ds4FaultNoSinks = nn.DS4FaultNoSinks
	// ds4FaultHeadMean collapses the streams for the head by their mean.
	ds4FaultHeadMean = nn.DS4FaultHeadMean
)

// DSV4 reports that the model is DeepSeek V4's graph.
func (c *Config) DSV4() bool { return c.HCMult != 0 && c.HCLowRank == 0 }

// CompAt is block li's compression: CompNone on every block of every other
// model.
func (c *Config) CompAt(li int) jlm.CompKind {
	if li < 0 || li >= len(c.CompKinds) {
		return jlm.CompNone
	}
	return c.CompKinds[li]
}

// CompRateAt is how many positions one entry of block li folds, zero on a
// block with no compressor.
func (c *Config) CompRateAt(li int) int {
	switch c.CompAt(li) {
	case jlm.CompCSA:
		return c.CompRateCSA
	case jlm.CompHCA:
		return c.CompRateHCA
	}
	return 0
}

// ds4Row is where the parts of a DeepSeek V4 block's cached row are: the key,
// then (a compressed block) the compressor's kv and gate, then (CSA) the
// indexer compressor's, each coff heads wide.
type ds4Row struct {
	hd, ihd, coff, width int
	kv, gate, ikv, igate int
}

// ds4RowAt is block li's row.
func (c *Config) ds4RowAt(li int) ds4Row {
	r := ds4Row{hd: c.HeadDim, ihd: c.IdxHeadDim, width: c.HeadDim}
	switch c.CompAt(li) {
	case jlm.CompHCA:
		r.coff = 1
	case jlm.CompCSA:
		r.coff = 2
	default:
		return r
	}
	r.kv = r.hd
	r.gate = r.kv + r.coff*r.hd
	r.width = r.gate + r.coff*r.hd
	if c.CompAt(li) == jlm.CompCSA {
		r.ikv = r.width
		r.igate = r.ikv + 2*r.ihd
		r.width = r.igate + 2*r.ihd
	}
	return r
}

// EntRowAt is one entry of block li: the compressed key, then on a CSA block
// the indexer's. Zero on a block with no compressor.
func (c *Config) EntRowAt(li int) int {
	switch c.CompAt(li) {
	case jlm.CompHCA:
		return c.HeadDim
	case jlm.CompCSA:
		return c.HeadDim + c.IdxHeadDim
	}
	return 0
}

// ds4Config checks a DeepSeek V4 container against what the engine runs; the
// fields were copied by configFrom.
func ds4Config(out *Config, c *jlm.Config) error {
	if c.Arch != jlm.ArchDeepseek4 {
		if (out.HCMult != 0 && !out.Q4E()) || len(out.CompKinds) != 0 {
			return fmt.Errorf("model: %s carries DeepSeek V4's hyper-connections", out.Arch)
		}
		return nil
	}
	out.SWARopePlain = true
	out.ExpertSqrtSoftplus = c.Flags2.Has(jlm.Flag2ExpertSqrtSoftplus)
	switch {
	case out.HCMult != ds4Streams || out.HCIters < 1 || !(out.HCEps > 0):
		return fmt.Errorf("model: %s: %d hyper-connection streams over %d Sinkhorn rounds: the "+
			"generated mixer is built for %d streams", out.Arch, out.HCMult, out.HCIters, ds4Streams)
	case out.NKVHead != 1 || out.QLoraRank == 0 || out.MLA() || out.Hybrid() || out.PredStreams != 0:
		return fmt.Errorf("model: %s: %d kv heads, q_lora_rank %d", out.Arch, out.NKVHead, out.QLoraRank)
	case out.OGroups <= 0 || out.NHead%out.OGroups != 0 || out.OLoraRank <= 0:
		return fmt.Errorf("model: %s: %d output groups of rank %d over %d heads", out.Arch, out.OGroups,
			out.OLoraRank, out.NHead)
	case len(out.CompKinds) != out.NLayer || out.SWAWindow <= 0 || !out.SWA(0):
		return fmt.Errorf("model: %s: %d compression kinds for %d blocks, window %d", out.Arch,
			len(out.CompKinds), out.NLayer, out.SWAWindow)
	case out.SWAWindow < out.CompRateHCA || out.SWAWindow < 2*out.CompRateCSA:
		return fmt.Errorf("model: %s: a window of %d holds the compressor's inputs of HCA every %d "+
			"and CSA every %d only if it covers the first and two of the second", out.Arch, out.SWAWindow,
			out.CompRateHCA, out.CompRateCSA)
	case out.NRot <= 0 || out.NRot%2 != 0 || out.NRot > out.HeadDim || out.RopeNeox:
		return fmt.Errorf("model: %s: a %d-wide %s rotary on %d-wide heads", out.Arch, out.NRot,
			map[bool]string{true: "NEOX", false: "interleaved"}[out.RopeNeox], out.HeadDim)
	case !out.MoE() || out.NFFNShExp == 0 || out.NoExpertNorm || !out.ExpertSqrtSoftplus ||
		out.Act != nn.ActSwiGLUClamp || out.NHashLayers > out.NLayer:
		return fmt.Errorf("model: %s: a mixture of %d (%d shared, renormalised %v, sqrt-softplus %v, %v, "+
			"%d hash blocks)", out.Arch, out.NExpert, out.NFFNShExp, !out.NoExpertNorm,
			out.ExpertSqrtSoftplus, out.Act, out.NHashLayers)
	}
	for li, k := range out.CompKinds {
		switch k {
		case jlm.CompCSA:
			if out.CompRateCSA <= 0 || out.IdxHeads <= 0 || out.IdxTopK <= 0 || out.NRot > out.IdxHeadDim {
				return fmt.Errorf("model: %s: block %d is CSA every %d with an indexer of %d heads of %d "+
					"(top %d)", out.Arch, li, out.CompRateCSA, out.IdxHeads, out.IdxHeadDim, out.IdxTopK)
			}
		case jlm.CompHCA:
			if out.CompRateHCA <= 0 {
				return fmt.Errorf("model: %s: block %d is HCA with no rate", out.Arch, li)
			}
		}
	}
	return nil
}

// ds4Streams is the hyper-connections the generated mixer is built for:
// DeepSeek V4's own (llama.cpp asserts it too).
const ds4Streams = 4

// ds4Layer is one DeepSeek V4 block's weights beyond the common ones (the
// query latent wqa/qaNorm/wqb, the key wk/kvaNorm, the output's second half
// wo, the sinks and the mixture).
type ds4Layer struct {
	// The two hyper-connections: fn, and the scale expanded per mix (s0 on
	// the first H, s1 on the next H, s2 on the H*H) beside the base.
	attnFn, ffnFn       tensor
	attnScale, attnBase []float32
	ffnScale, ffnBase   []float32
	// woA is the grouped projection's bank, one sheet per group, and woAG
	// its views (cut as an expert bank is, re-cut when the bank moves).
	woA     tensor
	woAG    []tensor
	woAFrom bankSpans
	// The compressor (a compressed block) and the indexer's (CSA).
	compKV, compGate        tensor
	compAPE, compNorm       []float32
	idxCompKV, idxCompGate  tensor
	idxCompAPE, idxCompNorm []float32
	// hash is a hash-routed block's token table, NExpertUsed ids per token as
	// floats; nil on a block that routes by its scores.
	hash []float32
}

// tensors is every matrix the block holds beyond the common ones, for
// bindPacked and releaseLayer.
func (d *ds4Layer) tensors() []*tensor {
	return []*tensor{&d.attnFn, &d.ffnFn, &d.woA, &d.compKV, &d.compGate, &d.idxCompKV, &d.idxCompGate}
}

// ds4Head is the model-level stream collapse before the output norm.
type ds4Head struct {
	fn          tensor
	scale, base []float32
}

// loadDS4 reads block bi's DeepSeek V4 weights and checks their shapes.
func loadDS4(l *layer, bi int32, cfg *Config, get func(jlm.Role, int32, int32) (tensor, error),
	vec func(jlm.Role, int32) ([]float32, error)) error {
	d := &ds4Layer{}
	l.ds4 = d
	var err error
	h := cfg.HCMult
	mix, flat := (2+h)*h, h*cfg.NEmbd
	for _, x := range []struct {
		fn, base, scale jlm.Role
		dfn             *tensor
		dbase, dscale   *[]float32
	}{
		{jlm.RoleHCAttnFn, jlm.RoleHCAttnBase, jlm.RoleHCAttnScale, &d.attnFn, &d.attnBase, &d.attnScale},
		{jlm.RoleHCFFNFn, jlm.RoleHCFFNBase, jlm.RoleHCFFNScale, &d.ffnFn, &d.ffnBase, &d.ffnScale},
	} {
		if *x.dfn, err = get(x.fn, bi, -1); err != nil {
			return err
		}
		if *x.dbase, err = vec(x.base, bi); err != nil {
			return err
		}
		s, err := vec(x.scale, bi)
		if err != nil {
			return err
		}
		if x.dfn.k != flat || x.dfn.rows != mix || len(*x.dbase) != mix || len(s) != 3 {
			return fmt.Errorf("model: block %d's hyper-connection is %dx%d, base %d, scale %d; want %dx%d, "+
				"%d and 3", bi, x.dfn.rows, x.dfn.k, len(*x.dbase), len(s), mix, flat, mix)
		}
		*x.dscale = hcScales(s, h)
	}
	if l.qaNorm, err = vec(jlm.RoleAttnQANorm, bi); err != nil {
		return err
	}
	if l.kvaNorm, err = vec(jlm.RoleAttnKVANorm, bi); err != nil {
		return err
	}
	if d.woA, err = get(jlm.RoleAttnOutA, bi, -1); err != nil {
		return err
	}
	gw := cfg.NHead / cfg.OGroups * cfg.HeadDim
	for g := 0; g < cfg.OGroups; g++ {
		t, err := bankExpert(d.woA, jlm.RoleAttnOutA, cfg.OGroups, cfg.OLoraRank, gw, g)
		if err != nil {
			return err
		}
		d.woAG = append(d.woAG, t)
	}
	hd := cfg.HeadDim
	if l.wqa.k != cfg.NEmbd || l.wqa.rows != cfg.QLoraRank || len(l.qaNorm) != cfg.QLoraRank ||
		l.wqb.k != cfg.QLoraRank || l.wqb.rows != cfg.NHead*hd || l.wk.k != cfg.NEmbd || l.wk.rows != hd ||
		len(l.kvaNorm) != hd || l.wo.k != cfg.OGroups*cfg.OLoraRank || l.wo.rows != cfg.NEmbd ||
		len(l.sinks) != cfg.NHead {
		return fmt.Errorf("model: block %d's attention is q_a %dx%d (norm %d), q_b %dx%d, kv %dx%d (norm %d), "+
			"o_b %dx%d, %d sinks", bi, l.wqa.rows, l.wqa.k, len(l.qaNorm), l.wqb.rows, l.wqb.k, l.wk.rows,
			l.wk.k, len(l.kvaNorm), l.wo.rows, l.wo.k, len(l.sinks))
	}
	if r := cfg.CompRateAt(int(bi)); r != 0 {
		coff := cfg.ds4RowAt(int(bi)).coff
		if d.compKV, err = get(jlm.RoleCompKV, bi, -1); err != nil {
			return err
		}
		if d.compGate, err = get(jlm.RoleCompGate, bi, -1); err != nil {
			return err
		}
		if d.compAPE, err = vec(jlm.RoleCompAPE, bi); err != nil {
			return err
		}
		if d.compNorm, err = vec(jlm.RoleCompNorm, bi); err != nil {
			return err
		}
		if d.compKV.rows != coff*hd || d.compGate.rows != coff*hd || d.compKV.k != cfg.NEmbd ||
			len(d.compAPE) != r*coff*hd || len(d.compNorm) != hd {
			return fmt.Errorf("model: block %d's compressor is kv %dx%d, gate %d rows, bias %d, norm %d; want "+
				"%d rows, %d and %d", bi, d.compKV.rows, d.compKV.k, d.compGate.rows, len(d.compAPE),
				len(d.compNorm), coff*hd, r*coff*hd, hd)
		}
	}
	if cfg.CompAt(int(bi)) == jlm.CompCSA {
		ihd, r := cfg.IdxHeadDim, cfg.CompRateCSA
		if l.idxQB, err = get(jlm.RoleIdxQB, bi, -1); err != nil {
			return err
		}
		if l.idxProj, err = get(jlm.RoleIdxProj, bi, -1); err != nil {
			return err
		}
		if d.idxCompKV, err = get(jlm.RoleIdxCompKV, bi, -1); err != nil {
			return err
		}
		if d.idxCompGate, err = get(jlm.RoleIdxCompGate, bi, -1); err != nil {
			return err
		}
		if d.idxCompAPE, err = vec(jlm.RoleIdxCompAPE, bi); err != nil {
			return err
		}
		if d.idxCompNorm, err = vec(jlm.RoleIdxCompNorm, bi); err != nil {
			return err
		}
		if l.idxQB.k != cfg.QLoraRank || l.idxQB.rows != cfg.IdxHeads*ihd || l.idxProj.k != cfg.NEmbd ||
			l.idxProj.rows != cfg.IdxHeads || d.idxCompKV.rows != 2*ihd || d.idxCompGate.rows != 2*ihd ||
			len(d.idxCompAPE) != r*2*ihd || len(d.idxCompNorm) != ihd {
			return fmt.Errorf("model: block %d's indexer is q %dx%d, weights %dx%d, compressor %d/%d rows, "+
				"bias %d, norm %d", bi, l.idxQB.rows, l.idxQB.k, l.idxProj.rows, l.idxProj.k,
				d.idxCompKV.rows, d.idxCompGate.rows, len(d.idxCompAPE), len(d.idxCompNorm))
		}
	}
	if int(bi) < cfg.NHashLayers {
		if d.hash, err = vec(jlm.RoleHashExperts, bi); err != nil {
			return err
		}
		if len(d.hash) != cfg.NVocab*cfg.NExpertUsed {
			return fmt.Errorf("model: block %d's token table holds %d ids, want %d tokens of %d", bi,
				len(d.hash), cfg.NVocab, cfg.NExpertUsed)
		}
		if l.expProbsB != nil {
			return fmt.Errorf("model: block %d routes by token id and carries a selection bias", bi)
		}
	} else if l.expProbsB == nil {
		return fmt.Errorf("model: block %d routes by its scores and carries no selection bias", bi)
	}
	return nil
}

// hcScales is a hyper-connection's three scales, one per mix: s0 on the H
// collapse weights, s1 on the H placements, s2 on the H*H mixer. A load-time
// constant, built by copying.
func hcScales(s []float32, h int) []float32 {
	out := make([]float32, (2+h)*h)
	for i := range out {
		switch {
		case i < h:
			out[i] = s[0]
		case i < 2*h:
			out[i] = s[1]
		default:
			out[i] = s[2]
		}
	}
	return out
}

// loadDS4Head reads the model-level stream collapse.
func loadDS4Head(m *Model, cfg *Config, get func(jlm.Role, int32, int32) (tensor, error),
	vec func(jlm.Role, int32) ([]float32, error)) error {
	hd := &ds4Head{}
	var err error
	if hd.fn, err = get(jlm.RoleHCHeadFn, jlm.DenseBlock, -1); err != nil {
		return err
	}
	if hd.base, err = vec(jlm.RoleHCHeadBase, jlm.DenseBlock); err != nil {
		return err
	}
	s, err := vec(jlm.RoleHCHeadScale, jlm.DenseBlock)
	if err != nil {
		return err
	}
	h := cfg.HCMult
	if hd.fn.rows != h || hd.fn.k != h*cfg.NEmbd || len(hd.base) != h || len(s) != 1 {
		return fmt.Errorf("model: the head's hyper-connection is %dx%d, base %d, scale %d; want %dx%d, %d, 1",
			hd.fn.rows, hd.fn.k, len(hd.base), len(s), h, h*cfg.NEmbd, h)
	}
	// The mixer's head form reads eight floats of each, the last four
	// padding (nn.HCMix32JIT).
	hd.scale = make([]float32, 2*h)
	for i := range hd.scale[:h] {
		hd.scale[i] = s[0]
	}
	hd.base = append(hd.base[:h:h], make([]float32, h)...)
	m.ds4Head = hd
	return nil
}

// ds4Scratch is a instance's DeepSeek V4 buffers: per row of a call (grown to the
// widest call, so a warm decode token and a warm step allocate nothing) and
// per attention row.
type ds4Scratch struct {
	rows int
	// per row: the streams flattened and normed, the mixes, the mixer's
	// outputs (pre, post, comb), the collapsed and normed input, the query
	// latent and query (a head of slack, see ropeTail), the key row, the
	// compressor's projections, the indexer's query and weights, the heads'
	// output (slack too) and the grouped projection's, and the sublayer out.
	flat, flatn, mix, hc, coll, hn []float32
	qr, q, krow, ckv, cg, ikv, ig  []float32
	iq, iw, ao, oa, out            []float32
	// per row of the call: its sequence slot, position and token.
	rslot, rpos []int
	rid         []int32
	// one attention row at a time: the keys it reads, the indexer's keys and
	// scores, the scores (a sink and a softmax pad past each head's), the
	// compressor's gathered slots and the entry it writes, the rotary tables.
	keys, ikeys, isc, itmp, af []float32
	afStride                   int
	skv, sgate, ent, row       []float32
	// zrow is what a row of a hole in the history reads (rowAt).
	zrow           []float32
	cs, csInv, csE []float32
	ord            nn.SampleOrder
	// constants: ones for the unweighted norms, the conjugation signs, and
	// the hash block's selection mask.
	onesFlat, onesHD, signs, hashBias []float32
	// hmix is the head's mixes, padded to the mixer's eight.
	hmix          []float32
	attn, idxAttn *nn.AttnSet
}

// allocDS4 sizes a instance's DeepSeek V4 buffers for one row, and its streams.
func (s *Instance) allocDS4() {
	c := s.m.Cfg
	if !c.DSV4() {
		return
	}
	d := c.NEmbd
	s.xa = make([]float32, c.ResidW())
	s.x = s.xa[:d]
	g := &ds4Scratch{}
	s.ds = g
	g.onesFlat = make([]float32, c.HCMult*d)
	for i := range g.onesFlat {
		g.onesFlat[i] = 1
	}
	g.onesHD = make([]float32, max(c.HeadDim, c.IdxHeadDim))
	for i := range g.onesHD {
		g.onesHD[i] = 1
	}
	g.signs = make([]float32, c.NRot)
	for i := range g.signs {
		g.signs[i] = 1
		if i%2 == 1 {
			g.signs[i] = -1
		}
	}
	g.hashBias = make([]float32, c.NExpert)
	g.hmix = make([]float32, 2*c.HCMult)
	// The keys a row reads: its window, and every entry of the finest
	// compression (an HCA block reads them all; a CSA block reads its top-k).
	maxEnt := 0
	for li := 0; li < c.NLayer; li++ {
		if r := c.CompRateAt(li); r > 0 {
			maxEnt = max(maxEnt, s.maxSeq/r+1)
		}
	}
	nk := c.SWAWindow + maxEnt
	g.keys = make([]float32, nk*c.HeadDim)
	g.afStride = attStride(nk + 1)
	g.af = make([]float32, c.NHead*g.afStride)
	if c.IdxHeadDim > 0 {
		g.ikeys = make([]float32, maxEnt*c.IdxHeadDim)
		g.isc = make([]float32, maxEnt)
		g.itmp = make([]float32, maxEnt)
		g.ord.Reserve(maxEnt)
	}
	w := 2 * max(c.CompRateCSA, c.CompRateHCA) * c.HeadDim
	g.skv, g.sgate = make([]float32, w), make([]float32, w)
	g.ent = make([]float32, c.HeadDim+c.IdxHeadDim)
	g.row = make([]float32, c.MaxKVDim())
	g.zrow = make([]float32, c.MaxKVDim())
	g.cs, g.csInv, g.csE = make([]float32, c.NRot), make([]float32, c.NRot), make([]float32, c.NRot)
	g.attn = s.jit.AttnSetFor(c.HeadDim, c.HeadDim, c.HeadDim, cpu.KVF32)
	if c.IdxHeadDim > 0 {
		g.idxAttn = s.jit.AttnSetFor(c.IdxHeadDim, c.IdxHeadDim, c.IdxHeadDim, cpu.KVF32)
	}
	s.growDS4(1)
}

// growDS4 sizes the per-row buffers for n rows.
func (s *Instance) growDS4(n int) {
	c, g := s.m.Cfg, s.ds
	if g == nil || g.rows >= n {
		return
	}
	g.rows = n
	h, d, hd := c.HCMult, c.NEmbd, c.HeadDim
	mix := (2 + h) * h
	g.flat, g.flatn = make([]float32, n*h*d), make([]float32, n*h*d)
	g.mix, g.hc = make([]float32, n*mix), make([]float32, n*mix)
	g.coll, g.hn, g.out = make([]float32, n*d), make([]float32, n*d), make([]float32, n*d)
	g.qr = make([]float32, n*c.QLoraRank)
	g.q = make([]float32, n*c.NHead*hd+hd)
	g.ao = make([]float32, n*c.NHead*hd+hd)
	g.krow = make([]float32, n*hd)
	g.ckv, g.cg = make([]float32, n*2*hd), make([]float32, n*2*hd)
	g.ikv, g.ig = make([]float32, n*2*c.IdxHeadDim), make([]float32, n*2*c.IdxHeadDim)
	g.iq = make([]float32, n*c.IdxHeads*c.IdxHeadDim+c.IdxHeadDim)
	g.iw = make([]float32, n*c.IdxHeads)
	g.oa = make([]float32, n*c.OGroups*c.OLoraRank)
	g.rslot, g.rpos, g.rid = make([]int, n), make([]int, n), make([]int32, n)
}

// ds4Expand copies each row's embedding (stream 0) into the other streams.
func (s *Instance) ds4Expand(x []float32, rows int) {
	if s.cpuExpand(x, rows) == jit.DeclineServes {
		return
	}
	c := s.m.Cfg
	for r := 0; r < rows; r++ {
		x0 := c.stream(x, 0, rows, r)
		for k := 1; k < c.HCMult; k++ {
			copy(c.stream(x, k, rows, r), x0)
		}
	}
}

// hcCollapse writes sum_k pre[k] * stream k of flat into dst.
func (s *Instance) hcCollapse(dst, flat, pre []float32) {
	d := len(dst)
	clear(dst)
	for k := range pre {
		s.axpy(dst, flat[k*d:(k+1)*d], pre[k])
	}
}

// ds4Collapse is the head's stream collapse for n rows of x, written into
// stream 0 of each row, where the output norm reads it.
func (s *Instance) ds4Collapse(x []float32, n int) error {
	c, g, hd := s.m.Cfg, s.ds, s.m.ds4Head
	h, d := c.HCMult, c.NEmbd
	if c.ds4Fault != ds4FaultHeadMean && s.cpuCollapse(x, n) == jit.DeclineServes {
		return nil
	}
	for r := 0; r < n; r++ {
		f := g.flat[:h*d]
		for k := 0; k < h; k++ {
			copy(f[k*d:(k+1)*d], c.stream(x, k, n, r))
		}
		s.rmsnorm(g.flatn[:h*d], f, g.onesFlat, c.RMSEps)
		s.jit.NewInput()
		// The head's mixer reads eight floats, the last four padding.
		m8 := g.hmix[:2*h]
		if err := s.mv(m8[:h], hd.fn, g.flatn[:h*d]); err != nil {
			return err
		}
		pre := g.hc[:2*h]
		nn.HCMix32JIT(pre, m8, hd.scale, hd.base, float32(c.HCEps), 0, true)
		if c.ds4Fault == ds4FaultHeadMean {
			mean := 1 / float32(h)
			for k := range pre[:h] {
				pre[k] = mean
			}
		}
		pre = pre[:h]
		s.hcCollapse(c.stream(x, 0, n, r), f, pre)
	}
	s.jit.NewInput()
	return nil
}
