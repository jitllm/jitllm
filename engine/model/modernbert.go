package model

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// ModernBERT (jlm.ArchModernBERT) and the decision head Laya puts after it.
//
// The encoder is pre-norm: x += o(attn(LN(x))), then x += down(gelu(gate) *
// up)(LN(x)), with LayerNorms that carry no bias, no bias on any projection,
// NEOX rotary on q and k, and bidirectional attention -- global on every
// SWAPeriod-th layer from the first, at RopeBase, and within a symmetric
// window of SWAWindow/2 positions either side on the rest, at RopeBaseSWA.
// Layer 0 has no attention norm; a final norm ends the stack.
//
// Laya's head (rl_common.DecisionModel) adds the question type's row of the
// token-type table to every row, runs DecisionBlocks pre-norm blocks of
// torch's TransformerEncoderLayer (LayerNorms and projections with biases,
// DecisionHeads heads, no positions, a ReLU MLP), and scores each option's
// marker row: LayerNorm, a square projection, GELU, a projection to one value.
//
// It runs on the encoder's generated kernels, as BERT does: LayerNorm32JIT,
// the f32 attention scores and accumulate, the activation kernels and the
// matmuls. GELU is the erf form both in the GeGLU and in the scorer, as the
// reference computes it.

// mbBlock is one ModernBERT encoder block.
type mbBlock struct {
	wq, wk, wv, wo, gate, up, down tensor
	attnNorm, ffnNorm              []float32 // attnNorm is nil on block 0
	global                         bool
	// g is the block's verified graph (encgraph.go).
	g *graph.Graph
}

// headBlock is one of the decision head's pre-norm blocks.
type headBlock struct {
	wq, wk, wv, wo, up, down   tensor
	bq, bk, bv, bo, bUp, bDown []float32
	ln1W, ln1B, ln2W, ln2B     []float32
}

// modernBERT is the part of an encoder only ModernBERT has.
type modernBERT struct {
	embNorm, finalNorm []float32
	blocks             []mbBlock
	// trunk is the encoder's block count, the head's first block: len(blocks),
	// or an LFM2 encoder's (lfm2Enc) when the head sits on that trunk.
	trunk              int
	head               []headBlock
	headFFN, headHeads int
	types              []float32 // DecisionKind rows of NEmbd: choice, score, noul
	scNormW, scNormB   []float32
	sc, scOut          tensor
	scB, scOutB        []float32
	ropeG, ropeL       nn.Rope
	// local is the blocks that attend inside the window, at the local base:
	// all but every SWAPeriod'th from the first (nn.LayerPlan.SWALocal).
	local func(li int) bool
}

func (b *mbBlock) weights() []*tensor {
	return []*tensor{&b.wq, &b.wk, &b.wv, &b.wo, &b.gate, &b.up, &b.down}
}

func (b *headBlock) weights() []*tensor {
	return []*tensor{&b.wq, &b.wk, &b.wv, &b.wo, &b.up, &b.down}
}

// buildModernBERT loads a ModernBERT container, and its decision head when the
// config states one.
func buildModernBERT(c *jlm.File, m *Model, enc *encoder) error {
	cfg, jc := m.Cfg, c.Config()
	d := cfg.NEmbd
	get := func(role jlm.Role, block int32, rows, k int) (tensor, error) {
		e, ok := c.Find(role, block, -1)
		if !ok {
			return tensor{}, fmt.Errorf("model: missing %v (block %d)", role, block)
		}
		typ, ok := jlm.SourceType(e.Type)
		if !ok || !quant.Dequantable(typ) || e.NDim != 2 {
			return tensor{}, fmt.Errorf("model: %v (block %d) is %v in %d dimensions", role, block, e.Type, e.NDim)
		}
		qs, _, _ := c.Span(e)
		t := tensor{e: e, typ: typ, data: qs, rows: int(e.Dims[1]), k: int(e.Dims[0])}
		if jlm.Packed(e.Type) {
			t.packed = &nn.Packed{}
		}
		if (rows > 0 && t.rows != rows) || t.k != k {
			return tensor{}, fmt.Errorf("model: %v (block %d) is %dx%d, want %dx%d", role, block, t.rows, t.k, rows, k)
		}
		return t, nil
	}
	vec := func(role jlm.Role, block int32, want int) ([]float32, error) {
		e, ok := c.Find(role, block, -1)
		if !ok {
			return nil, fmt.Errorf("model: missing %v (block %d)", role, block)
		}
		typ, ok := jlm.SourceType(e.Type)
		if !ok {
			return nil, fmt.Errorf("model: %v is %v, which has no dequantizer", role, e.Type)
		}
		n := 1
		for i := 0; i < int(e.NDim); i++ {
			n *= int(e.Dims[i])
		}
		if n != want {
			return nil, fmt.Errorf("model: %v (block %d) has %d values, want %d", role, block, n, want)
		}
		qs, _, _ := c.Span(e)
		out := make([]float32, n)
		if err := quant.Dequant32(typ, qs, out); err != nil {
			return nil, fmt.Errorf("model: %v: %w", role, err)
		}
		return out, nil
	}
	mb := &modernBERT{}
	var err error
	if mb.embNorm, err = vec(jlm.RoleTokenEmbdNorm, jlm.DenseBlock, d); err != nil {
		return err
	}
	if mb.finalNorm, err = vec(jlm.RoleOutputNorm, jlm.DenseBlock, d); err != nil {
		return err
	}
	// A bias anywhere in the encoder is a different graph.
	for _, r := range []jlm.Role{jlm.RoleTokenEmbdNormBias, jlm.RoleOutputNormBias} {
		if c.Has(r, jlm.DenseBlock, -1) {
			return fmt.Errorf("model: ModernBERT carries %v; its norms have no bias", r)
		}
	}
	if cfg.NHead*cfg.HeadDim != d || cfg.NRot != cfg.HeadDim {
		return fmt.Errorf("model: %d heads of %d (rotary %d) do not make a %d-wide residual rotated whole",
			cfg.NHead, cfg.HeadDim, cfg.NRot, d)
	}
	mb.ropeG = nn.Rope{NRot: cfg.NRot, Base: cfg.RopeBase, Neox: true, Scale: 1}
	mb.ropeL = nn.Rope{NRot: cfg.NRot, Base: cfg.RopeBaseSWA, Neox: true, Scale: 1}
	mb.blocks = make([]mbBlock, cfg.NLayer)
	mb.trunk = len(mb.blocks)
	mb.local = func(li int) bool { return li < len(mb.blocks) && !mb.blocks[li].global }
	for i := range mb.blocks {
		b, bi := &mb.blocks[i], int32(i)
		b.global = cfg.SWAWindow == 0 || cfg.SWAPeriod <= 1 || i%cfg.SWAPeriod == 0
		for _, w := range []struct {
			role    jlm.Role
			dst     *tensor
			rows, k int
		}{{jlm.RoleAttnQ, &b.wq, d, d}, {jlm.RoleAttnK, &b.wk, d, d}, {jlm.RoleAttnV, &b.wv, d, d},
			{jlm.RoleAttnOut, &b.wo, d, d}, {jlm.RoleFFNGate, &b.gate, cfg.NFFN, d},
			{jlm.RoleFFNUp, &b.up, cfg.NFFN, d}, {jlm.RoleFFNDown, &b.down, d, cfg.NFFN}} {
			if *w.dst, err = get(w.role, bi, w.rows, w.k); err != nil {
				return err
			}
		}
		// Layer 0's attention norm is the identity, and its absence is the
		// fact: ModernBERT's first layer has none.
		if i > 0 || c.Has(jlm.RoleAttnNorm, bi, -1) {
			if b.attnNorm, err = vec(jlm.RoleAttnNorm, bi, d); err != nil {
				return err
			}
		}
		if b.ffnNorm, err = vec(jlm.RoleFFNNorm, bi, d); err != nil {
			return err
		}
	}
	if err := m.buildMBGraphs(c, mb); err != nil {
		return err
	}
	if jc.Decision == jlm.DecisionLaya {
		if err := buildLayaHead(c, m, mb, get, vec); err != nil {
			return err
		}
	}
	enc.mb = mb
	return nil
}

// buildLayaHead loads the decision head: its blocks after the encoder's, the
// type rows and the scorer.
func buildLayaHead(c *jlm.File, m *Model, mb *modernBERT,
	get func(jlm.Role, int32, int, int) (tensor, error), vec func(jlm.Role, int32, int) ([]float32, error)) error {
	cfg, jc := m.Cfg, c.Config()
	d := cfg.NEmbd
	mb.headHeads = int(jc.DecisionHeads)
	if mb.headHeads == 0 || d%mb.headHeads != 0 {
		return fmt.Errorf("model: %d decision heads do not divide a %d-wide residual", mb.headHeads, d)
	}
	tt, err := vec(jlm.RoleTokenTypes, jlm.DenseBlock, 3*d)
	if err != nil {
		return fmt.Errorf("%w (a decision head needs one type row per question type)", err)
	}
	mb.types = tt
	mb.head = make([]headBlock, jc.DecisionBlocks)
	for i := range mb.head {
		b, bi := &mb.head[i], int32(cfg.NLayer+i)
		up, ok := c.Find(jlm.RoleFFNUp, bi, -1)
		if !ok {
			return fmt.Errorf("model: decision block %d has no ffn_up", bi)
		}
		ff := int(up.Dims[1])
		if i == 0 {
			mb.headFFN = ff
		} else if ff != mb.headFFN {
			return fmt.Errorf("model: decision blocks %d and %d differ in MLP width", cfg.NLayer, bi)
		}
		for _, w := range []struct {
			role    jlm.Role
			dst     *tensor
			rows, k int
		}{{jlm.RoleAttnQ, &b.wq, d, d}, {jlm.RoleAttnK, &b.wk, d, d}, {jlm.RoleAttnV, &b.wv, d, d},
			{jlm.RoleAttnOut, &b.wo, d, d}, {jlm.RoleFFNUp, &b.up, ff, d}, {jlm.RoleFFNDown, &b.down, d, ff}} {
			if *w.dst, err = get(w.role, bi, w.rows, w.k); err != nil {
				return err
			}
		}
		for _, v := range []struct {
			role jlm.Role
			dst  *[]float32
			n    int
		}{{jlm.RoleAttnQBias, &b.bq, d}, {jlm.RoleAttnKBias, &b.bk, d}, {jlm.RoleAttnVBias, &b.bv, d},
			{jlm.RoleAttnOutBias, &b.bo, d}, {jlm.RoleFFNUpBias, &b.bUp, ff}, {jlm.RoleFFNDownBias, &b.bDown, d},
			{jlm.RoleAttnNorm, &b.ln1W, d}, {jlm.RoleAttnNormBias, &b.ln1B, d},
			{jlm.RoleFFNNorm, &b.ln2W, d}, {jlm.RoleFFNNormBias, &b.ln2B, d}} {
			if *v.dst, err = vec(v.role, bi, v.n); err != nil {
				return err
			}
		}
	}
	for _, v := range []struct {
		role jlm.Role
		dst  *[]float32
		n    int
	}{{jlm.RoleScorerNorm, &mb.scNormW, d}, {jlm.RoleScorerNormBias, &mb.scNormB, d},
		{jlm.RoleScorerBias, &mb.scB, d}, {jlm.RoleScorerOutBias, &mb.scOutB, 1}} {
		if *v.dst, err = vec(v.role, jlm.DenseBlock, v.n); err != nil {
			return err
		}
	}
	if mb.sc, err = get(jlm.RoleScorer, jlm.DenseBlock, d, d); err != nil {
		return err
	}
	if mb.scOut, err = get(jlm.RoleScorerOut, jlm.DenseBlock, 1, d); err != nil {
		return err
	}
	return nil
}

// mbBind binds the dense-region matrices the head reads (the scorer's), once.
func (m *Model) mbBind() {
	c := m.container
	if c == nil {
		return
	}
	m.bind.Lock()
	defer m.bind.Unlock()
	bindOne(c, &m.enc.mb.sc)
	bindOne(c, &m.enc.mb.scOut)
}

// mbEncode runs the ModernBERT encoder over one sequence and leaves the final
// norm's output in e.x, n rows of NEmbd.
func (e *Embedder) mbEncode(ids []int32) error {
	m, c, mb := e.m, e.m.Cfg, e.m.enc.mb
	n, d := len(ids), c.NEmbd
	e.grow(n)
	x := e.x[:n*d]
	for i, id := range ids {
		if err := m.embedRow(x[i*d:(i+1)*d], int(id)); err != nil {
			return err
		}
	}
	e.rowsLN(x, x, mb.embNorm, nil, n)
	qseq, qpos := e.packRows(n)
	for i := 0; i < n; i++ {
		pos := i
		if qseq != nil {
			pos = i - e.seqs[qseq[i]].Lo
		}
		e.jit.RopeTable(mb.ropeG, e.cs[i*c.NRot:(i+1)*c.NRot], pos)
		e.jit.RopeTable(mb.ropeL, e.csL[i*c.NRot:(i+1)*c.NRot], pos)
	}
	scale := float32(1 / math.Sqrt(float64(c.HeadDim)))
	if err := e.devPrepare(n, c.SWAWindow/2, 0); err != nil {
		return err
	}
	for li := 0; li < len(mb.blocks); li++ {
		// A run of placed blocks is one device call over the sequence; the
		// global table in cs, the local one in csSWA (nn.LayerPlan.LocalRope).
		if e.onDevice(li) {
			hi, err := e.devRun(li, len(mb.blocks), n, x, e.cs[:n*c.NRot], e.csL[:n*c.NRot])
			if err != nil {
				return err
			}
			li = hi - 1
			if e.afterBlock != nil {
				e.afterBlock(li, x)
			}
			continue
		}
		if err := m.encBlockPageIn(li); err != nil {
			return err
		}
		b := &mb.blocks[li]
		// 1. h = LN(x) (the identity on layer 0), q k v, the rotary.
		h := e.xb[:n*d]
		if b.attnNorm != nil {
			e.rowsLN(h, x, b.attnNorm, nil, n)
		} else {
			copy(h, x)
		}
		q, k, v := e.q[:n*d], e.k[:n*d], e.v[:n*d]
		for _, p := range []struct {
			w   tensor
			dst []float32
		}{{b.wq, q}, {b.wk, k}, {b.wv, v}} {
			if err := e.mm(p.dst, p.w, h, n); err != nil {
				return err
			}
		}
		cs := e.cs
		if !b.global && e.violation != "one-base" {
			cs = e.csL
		}
		for i := 0; i < n; i++ {
			t := cs[i*c.NRot : (i+1)*c.NRot]
			nn.RoPE32JIT(q[i*d:(i+1)*d], c.HeadDim, t, true)
			nn.RoPE32JIT(k[i*d:(i+1)*d], c.HeadDim, t, true)
		}
		// 2. Attention: every key on a global layer, the window's on a local
		// one -- a contiguous run of keys, so the window is a range, not a mask.
		w := n
		if !b.global && e.violation != "no-window" {
			w = c.SWAWindow / 2
		}
		e.attendSeqs(e.attnEnc, h, q, k, v, n, n, w, c.NHead, c.HeadDim, scale, qseq, qpos)
		// 3. x += o(attn).
		t := e.t[:n*d]
		if err := e.mm(t, b.wo, h, n); err != nil {
			return err
		}
		e.axpyRows(x, t)
		// 4. x += down(gelu(gate) * up)(LN(x)).
		e.rowsLN(h, x, b.ffnNorm, nil, n)
		g, u := e.gg[:n*c.NFFN], e.ff[:n*c.NFFN]
		if err := e.mm(g, b.gate, h, n); err != nil {
			return err
		}
		if err := e.mm(u, b.up, h, n); err != nil {
			return err
		}
		if e.violation == "up-gated" {
			g, u = u, g
		}
		e.actMul(g, u, e.gelu())
		if err := e.mm(t, b.down, g, n); err != nil {
			return err
		}
		e.axpyRows(x, t)
		if e.afterBlock != nil {
			e.afterBlock(li, x)
		}
	}
	e.rowsLN(x, x, mb.finalNorm, nil, n)
	return nil
}

// attend writes softmax(q k^T * scale) v into out for n query rows of nh
// heads of hd, on set's kernels: row i reads keys [i-w, i+w] cut to [0,
// keys) -- a window, or every key where w is keys. k and v are rows of the
// residual's width.
func (e *Embedder) attend(set *nn.AttnSet, out, q, k, v []float32, n, keys, w, nh, hd int, scale float32) {
	e.attendSeqs(set, out, q, k, v, n, keys, w, nh, hd, scale, nil, nil)
}

// attendSeqs is attend over packed sequences: query i is row qpos[i] of
// sequence qseq[i] and reads that sequence's keys inside its window. nil
// qseq is one sequence, query i at row i.
func (e *Embedder) attendSeqs(set *nn.AttnSet, out, q, k, v []float32, n, keys, w, nh, hd int, scale float32,
	qseq, qpos []int32) {
	j := &e.rg.attn
	if j.fn == nil {
		j.fn = e.attnHeads
	}
	j.set, j.out, j.q, j.k, j.v, j.n, j.keys, j.w, j.hd, j.scale = set, out, q, k, v, n, keys, w, hd, scale
	j.qseq, j.qpos = qseq, qpos
	e.jit.Parallel(nh, 1, j.fn)
	j.set, j.out, j.q, j.k, j.v, j.qseq, j.qpos = nil, nil, nil, nil, nil, nil, nil
}

// attnHeads is one chunk of attend: heads h0..h1-1.
func (e *Embedder) attnHeads(h0, h1 int) {
	j, d := &e.rg.attn, e.m.Cfg.NEmbd
	hd := j.hd
	for h := h0; h < h1; h++ {
		off := h * hd
		att := e.att[h*e.attStride:]
		acc := e.acc[h*hd : (h+1)*hd]
		for i := 0; i < j.n; i++ {
			lo, hi := max(0, i-j.w), min(j.keys, i+j.w+1)
			if j.qseq != nil && e.violation != "sequences-attend-across" {
				s, r := e.seqs[j.qseq[i]], int(j.qpos[i])
				lo, hi = max(s.Lo, r-j.w), min(s.Hi, r+j.w+1)
			}
			cnt := hi - lo
			j.set.AttnScores(att[:cnt], j.k[lo*d+off:], j.q[i*d+off:], cnt)
			nn.Scale32JIT(att[:cnt], j.scale)
			nn.Softmax32JIT(att, cnt)
			j.set.AttnAcc(acc, j.v[lo*d+off:], att[:cnt], cnt)
			copy(j.out[i*d+off:i*d+off+hd], acc)
		}
	}
}

// layaScores runs the decision head over e.x (n rows, the encoder's output)
// for a question of type t and writes the scorer's value at each marker row
// into out.
func (e *Embedder) layaScores(out []float32, n int, qt jlm.QuestionType, markers []int) error {
	m, c, mb := e.m, e.m.Cfg, e.m.enc.mb
	d := c.NEmbd
	x := e.x[:n*d]
	typeRow := func(qt jlm.QuestionType) []float32 {
		if e.violation == "type-row-0" {
			return mb.types[:d]
		}
		return mb.types[int(qt)*d : (int(qt)+1)*d]
	}
	qseq, qpos := e.packRows(n)
	if qseq == nil {
		e.addBiasRows(x, typeRow(qt), n)
	} else {
		for si, s := range e.seqs {
			e.addBiasRows(x[s.Lo*d:s.Hi*d], typeRow(e.seqTypes[si]), s.Hi-s.Lo)
		}
	}
	heads := mb.headHeads
	if e.violation == "encoder-heads" {
		heads = c.NHead
	}
	hd := d / heads
	scale := float32(1 / math.Sqrt(float64(hd)))
	nm := len(markers)
	// The marker rows the scorer reads are x's first nm once the last block
	// has run: the host's last block runs them alone, and a device's runs
	// every row, so they are gathered after it.
	gather := false
	for bi := 0; bi < len(mb.head); bi++ {
		if li := mb.trunk + bi; e.onDevice(li) {
			hi, err := e.devRun(li, mb.trunk+len(mb.head), n, x, nil, nil)
			if err != nil {
				return err
			}
			bi = hi - mb.trunk - 1
			gather = bi == len(mb.head)-1
			continue
		}
		if err := m.encBlockPageIn(mb.trunk + bi); err != nil {
			return err
		}
		b := &mb.head[bi]
		last := bi == len(mb.head)-1
		h := e.xb[:n*d]
		e.rowsLN(h, x, b.ln1W, b.ln1B, n)
		q, k, v := e.q[:n*d], e.k[:n*d], e.v[:n*d]
		for _, p := range []struct {
			w   tensor
			b   []float32
			dst []float32
		}{{b.wq, b.bq, q}, {b.wk, b.bk, k}, {b.wv, b.bv, v}} {
			if err := e.mm(p.dst, p.w, h, n); err != nil {
				return err
			}
			e.addBiasRows(p.dst, p.b, n)
		}
		// The last block's output is read only at the markers: every row's k
		// and v, and the rest of the block for the marker rows alone.
		rows := n
		aseq, apos := qseq, qpos
		if last {
			for j, r := range markers {
				copy(q[j*d:(j+1)*d], q[r*d:(r+1)*d])
				copy(x[j*d:(j+1)*d], x[r*d:(r+1)*d])
			}
			rows = nm
			// Each marker query reads its own sequence's keys.
			if qseq != nil {
				aseq, apos = e.qseq[:nm], e.qpos[:nm]
				for j, r := range markers {
					aseq[j], apos[j] = e.rowSeq[r], int32(r)
				}
			}
		}
		set := e.attnHead
		if heads != mb.headHeads {
			set = e.attnEnc
		}
		e.attendSeqs(set, h, q, k, v, rows, n, n, heads, hd, scale, aseq, apos)
		t := e.t[:rows*d]
		if err := e.mm(t, b.wo, h, rows); err != nil {
			return err
		}
		e.addBiasRows(t, b.bo, rows)
		xr := x[:rows*d]
		e.axpyRows(xr, t)
		e.rowsLN(h[:rows*d], xr, b.ln2W, b.ln2B, rows)
		ff := e.ff[:rows*mb.headFFN]
		if err := e.mm(ff, b.up, h, rows); err != nil {
			return err
		}
		e.addBiasRows(ff, b.bUp, rows)
		e.actRows(ff, nn.ActReLU)
		if err := e.mm(t, b.down, ff, rows); err != nil {
			return err
		}
		e.addBiasRows(t, b.bDown, rows)
		e.axpyRows(xr, t)
	}
	if gather {
		for j, r := range markers {
			copy(x[j*d:(j+1)*d], x[r*d:(r+1)*d])
		}
	}
	// The scorer on the marker rows: LN, Linear, GELU, Linear to one value.
	h := e.xb[:nm*d]
	e.rowsLN(h, x[:nm*d], mb.scNormW, mb.scNormB, nm)
	t := e.t[:nm*d]
	if err := e.mm(t, mb.sc, h, nm); err != nil {
		return err
	}
	e.addBiasRows(t, mb.scB, nm)
	e.actRows(t, e.gelu())
	if err := e.mm(out[:nm], mb.scOut, t, nm); err != nil {
		return err
	}
	e.addBiasRows(out[:nm], mb.scOutB, nm)
	return nil
}

// packRows is, where sequences are packed (Embedder.seqs), each of the n
// rows' sequence and the attention's query rows, the identity; nil, nil for
// one sequence. The query slices are the Embedder's scratch, rewritten by the
// last head block for its marker queries, so they are filled anew each call.
func (e *Embedder) packRows(n int) (qseq, qpos []int32) {
	if len(e.seqs) == 0 {
		return nil, nil
	}
	if cap(e.rowSeq) < n {
		e.rowSeq, e.qseq, e.qpos = make([]int32, n), make([]int32, n), make([]int32, n)
	}
	e.rowSeq, e.qseq, e.qpos = e.rowSeq[:n], e.qseq[:n], e.qpos[:n]
	for si, s := range e.seqs {
		for r := s.Lo; r < s.Hi; r++ {
			e.rowSeq[r] = int32(si)
		}
	}
	for r := range n {
		e.qseq[r], e.qpos[r] = e.rowSeq[r], int32(r)
	}
	return e.qseq, e.qpos
}

// encMB is the model's ModernBERT graph, or nil.
func (m *Model) encMB() *modernBERT {
	if m.enc == nil {
		return nil
	}
	return m.enc.mb
}

// gelu is ModernBERT's and Laya's scorer's GELU: the erf form, as
// transformers' GELUActivation computes it (and llama.cpp's GeGLU does not).
func (e *Embedder) gelu() nn.ActKind {
	if e.violation == "tanh-gelu" {
		return nn.ActGELU
	}
	return nn.ActGELUErf
}
