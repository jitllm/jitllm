package model

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// LFM2 made an encoder (jlm.ArchLFM2Enc): LiquidAI's LFM2.5-Encoder, d1-omni's
// trunk (its encoder.py, the reference this follows). LFM2's blocks with
//
//	no causal mask   every text row attends to every row; a media prefix's
//	                 rows (pictures, a voice clip, in front of the text) to
//	                 the prefix's alone
//	a centred conv   y_t = w0 bx_{t-1} + w1 bx_t + w2 bx_{t+1} over B*x,
//	                 the sequence padded with zeros, and the prefix's last
//	                 row not reading the text's first (keep_right)
//	a final norm     embedding_norm over every row
//
// and LFM2's per-head q/k RMSNorm, NEOX rotary over the whole head, GQA and
// SwiGLU MLP. Every product is the generated matmul, every elementwise step a
// generated kernel, on the Embedder's JIT and pool as the BERT family's
// blocks are.

// lfBlock is one trunk block: a short convolution or an attention, then the
// MLP.
type lfBlock struct {
	attn           bool
	norm, ffnNorm  []float32
	b, c, x, out   tensor    // the convolution's B, C and x projections and out_proj
	taps           []float32 // the convolution's three taps per channel, [tap][channel]
	wq, wk, wv, wo tensor
	qNorm, kNorm   []float32
	gate, up, down tensor
}

func (b *lfBlock) weights() []*tensor {
	if b.attn {
		return []*tensor{&b.wq, &b.wk, &b.wv, &b.wo, &b.gate, &b.up, &b.down}
	}
	return []*tensor{&b.b, &b.c, &b.x, &b.out, &b.gate, &b.up, &b.down}
}

// lfm2Enc is the part of an encoder only LFM2's has.
type lfm2Enc struct {
	blocks  []lfBlock
	outNorm []float32
	rope    nn.Rope
	// tapRows is each tap's channel weights repeated over rows (lfTaps), so
	// a tap over a whole sequence is one generated multiply; grown to the
	// longest sequence and kept.
	tapRows [3][]float32
}

// buildLFM2Enc loads the trunk, and its decision head (Laya's, on the
// modernBERT struct's head fields with no blocks of its own).
func buildLFM2Enc(c *jlm.File, m *Model, enc *encoder) error {
	cfg := m.Cfg
	d, hd := cfg.NEmbd, cfg.HeadDim
	kvw := cfg.NKVHead * hd
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
	if cfg.SSM.ConvKernel != 3 {
		return fmt.Errorf("model: an LFM2 encoder's centred convolution has 3 taps, this one %d", cfg.SSM.ConvKernel)
	}
	if cfg.NRot != hd || cfg.NHead*hd != d {
		return fmt.Errorf("model: %d heads of %d (rotary %d) for a %d-wide residual", cfg.NHead, hd, cfg.NRot, d)
	}
	lf := &lfm2Enc{rope: nn.Rope{NRot: cfg.NRot, Base: cfg.RopeBase, Neox: true, Scale: 1}}
	var err error
	if lf.outNorm, err = vec(jlm.RoleOutputNorm, jlm.DenseBlock, d); err != nil {
		return err
	}
	lf.blocks = make([]lfBlock, cfg.NLayer)
	for i := range lf.blocks {
		b, bi := &lf.blocks[i], int32(i)
		b.attn = cfg.LayerKind(i) == jlm.LayerFullAttn
		if b.norm, err = vec(jlm.RoleAttnNorm, bi, d); err != nil {
			return err
		}
		if b.ffnNorm, err = vec(jlm.RoleFFNNorm, bi, d); err != nil {
			return err
		}
		type mat struct {
			role    jlm.Role
			dst     *tensor
			rows, k int
		}
		ms := []mat{{jlm.RoleFFNGate, &b.gate, cfg.NFFN, d}, {jlm.RoleFFNUp, &b.up, cfg.NFFN, d},
			{jlm.RoleFFNDown, &b.down, d, cfg.NFFN}}
		if b.attn {
			ms = append(ms, mat{jlm.RoleAttnQ, &b.wq, d, d}, mat{jlm.RoleAttnK, &b.wk, kvw, d},
				mat{jlm.RoleAttnV, &b.wv, kvw, d}, mat{jlm.RoleAttnOut, &b.wo, d, d})
			if b.qNorm, err = vec(jlm.RoleAttnQNorm, bi, hd); err != nil {
				return err
			}
			if b.kNorm, err = vec(jlm.RoleAttnKNorm, bi, hd); err != nil {
				return err
			}
		} else {
			ms = append(ms, mat{jlm.RoleSCB, &b.b, d, d}, mat{jlm.RoleSCC, &b.c, d, d},
				mat{jlm.RoleSCX, &b.x, d, d}, mat{jlm.RoleSSMOut, &b.out, d, d})
			taps, err := vec(jlm.RoleSSMConv1d, bi, 3*d)
			if err != nil {
				return err
			}
			// [channel][tap] in the file (conv.weight, {3, d}); [tap][channel]
			// here, each tap a row the generated multiply reads whole.
			b.taps = make([]float32, 3*d)
			for ch := 0; ch < d; ch++ {
				for t := 0; t < 3; t++ {
					b.taps[t*d+ch] = taps[ch*3+t]
				}
			}
		}
		for _, w := range ms {
			if *w.dst, err = get(w.role, bi, w.rows, w.k); err != nil {
				return err
			}
		}
	}
	mb := &modernBERT{trunk: len(lf.blocks)}
	if err := buildLayaHead(c, m, mb, get, vec); err != nil {
		return err
	}
	enc.lf, enc.mb = lf, mb
	return nil
}

// lfTaps is the convolution's tap t of block b repeated over n rows.
func (e *Embedder) lfTaps(b *lfBlock, t, n int) []float32 {
	lf, d := e.m.enc.lf, e.m.Cfg.NEmbd
	r := &lf.tapRows[t]
	if len(*r) < n*d {
		*r = make([]float32, n*d)
	}
	rows := (*r)[:n*d]
	nn.Copy32JIT(rows, d, 1, b.taps[t*d:(t+1)*d], 0, 1, n, d)
	return rows
}

// lfm2Encode runs the trunk over a media prefix's rows (pre, rows of NEmbd;
// none for text alone) and ids after them, and leaves the final norm's
// output of the text rows in e.x from row 0: the head reads the text alone.
func (e *Embedder) lfm2Encode(ids []int32, pre []float32) error {
	m, c, lf := e.m, e.m.Cfg, e.m.enc.lf
	d := c.NEmbd
	p := len(pre) / d
	n := p + len(ids)
	hd, kvw := c.HeadDim, c.NKVHead*c.HeadDim
	e.grow(n)
	x := e.x[:n*d]
	copy(x, pre)
	for i, id := range ids {
		if err := m.embedRow(x[(p+i)*d:(p+i+1)*d], int(id)); err != nil {
			return err
		}
	}
	// Packed sequences (Embedder.seqs, a batch's prompts in one pass): each
	// row's positions count from its sequence's start, its keys and its
	// convolution's taps stay inside it.
	qseq, _ := e.packRows(n)
	for i := 0; i < n; i++ {
		pos := i
		if qseq != nil {
			pos = i - e.seqs[qseq[i]].Lo
		}
		e.jit.RopeTable(lf.rope, e.cs[i*c.NRot:(i+1)*c.NRot], pos)
	}
	scale := float32(1 / math.Sqrt(float64(hd)))
	h, t := e.xb[:n*d], e.t[:n*d]
	if err := e.devPrepare(n, -1, p); err != nil {
		return err
	}
	for li := 0; li < len(lf.blocks); li++ {
		// A run of placed blocks is one device call over the sequence.
		if e.onDevice(li) {
			hi, err := e.devRun(li, len(lf.blocks), n, x, e.cs[:n*c.NRot], nil)
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
		b := &lf.blocks[li]
		e.rowsRMS(h, x, b.norm, n, d)
		if b.attn {
			q, k, v := e.q[:n*d], e.k[:n*kvw], e.v[:n*kvw]
			for _, pr := range []struct {
				w   tensor
				dst []float32
			}{{b.wq, q}, {b.wk, k}, {b.wv, v}} {
				if err := e.mm(pr.dst, pr.w, h, n); err != nil {
					return err
				}
			}
			e.rowsRMS(q, q, b.qNorm, n*c.NHead, hd)
			e.rowsRMS(k, k, b.kNorm, n*c.NKVHead, hd)
			for i := 0; i < n; i++ {
				cs := e.cs[i*c.NRot : (i+1)*c.NRot]
				nn.RoPE32JIT(q[i*d:(i+1)*d], hd, cs, true)
				nn.RoPE32JIT(k[i*kvw:(i+1)*kvw], hd, cs, true)
			}
			e.lfAttend(h, q, k, v, n, p, scale)
			if err := e.mm(t, b.wo, h, n); err != nil {
				return err
			}
		} else {
			bb, cc, xx := e.q[:n*d], e.k[:n*d], e.v[:n*d]
			for _, pr := range []struct {
				w   tensor
				dst []float32
			}{{b.b, bb}, {b.c, cc}, {b.x, xx}} {
				if err := e.mm(pr.dst, pr.w, h, n); err != nil {
					return err
				}
			}
			e.actMul(bb, xx, nn.ActIdentity) // bx = B * x
			e.lfConv(b, h, bb, n, p)         // h = conv(bx)
			e.actMul(h, cc, nn.ActIdentity)  // C * conv(bx)
			e.jit.NewInput()
			if err := e.mm(t, b.out, h, n); err != nil {
				return err
			}
		}
		e.axpyRows(x, t)
		e.rowsRMS(h, x, b.ffnNorm, n, d)
		g, u := e.gg[:n*c.NFFN], e.ff[:n*c.NFFN]
		if err := e.mm(g, b.gate, h, n); err != nil {
			return err
		}
		if err := e.mm(u, b.up, h, n); err != nil {
			return err
		}
		e.actMul(g, u, nn.ActSiLU)
		if err := e.mm(t, b.down, g, n); err != nil {
			return err
		}
		e.axpyRows(x, t)
		if e.afterBlock != nil {
			e.afterBlock(li, x)
		}
	}
	e.rowsRMS(x, x, lf.outNorm, n, d)
	if p > 0 {
		copy(x, x[p*d:])
	}
	return nil
}

// lfConv writes the centred convolution of bx into y: the middle tap on
// every row, the left tap read one row back, the right tap one row ahead --
// except on the prefix's last row, which never reads the text (keep_right),
// and across a packed sequence's edge. bx is overwritten.
func (e *Embedder) lfConv(b *lfBlock, y, bx []float32, n, p int) {
	d := e.m.Cfg.NEmbd
	copy(y[:n*d], bx[:n*d])
	e.actMul(y[:n*d], e.lfTaps(b, 1, n), nn.ActIdentity)
	if n < 2 {
		return
	}
	// The left tap: y[t] += w0 * bx[t-1], t >= 1.
	l := e.ff[:n*d]
	copy(l, bx[:n*d])
	e.actMul(l, e.lfTaps(b, 0, n), nn.ActIdentity)
	// A sequence's first row reads nothing before it: the row l would
	// carry into it is cleared (it carries into no other).
	for _, sq := range e.seqs {
		if sq.Lo > 0 {
			nn.Scale32JIT(l[(sq.Lo-1)*d:sq.Lo*d], 0)
		}
	}
	e.axpyRows(y[d:n*d], l[:(n-1)*d])
	// The right tap: y[t] += w2 * bx[t+1], t <= n-2, not at t = p-1.
	if e.violation == "causal-conv" {
		return
	}
	e.actMul(bx[:n*d], e.lfTaps(b, 2, n), nn.ActIdentity)
	if p > 0 && p < n && e.violation != "prefix-reads-text" {
		nn.Scale32JIT(bx[p*d:(p+1)*d], 0)
	}
	for _, sq := range e.seqs {
		if sq.Hi < n {
			nn.Scale32JIT(bx[sq.Hi*d:(sq.Hi+1)*d], 0)
		}
	}
	e.axpyRows(y[:(n-1)*d], bx[d:n*d])
}

// lfAttend is the trunk's attention: nh query heads over the GQA key heads,
// every text row over every row, a prefix row over the prefix alone (p rows;
// zero is a sequence with none). k and v are rows of NKVHead heads.
func (e *Embedder) lfAttend(out, q, k, v []float32, n, p int, scale float32) {
	j := &e.rg.lfAttn
	if j.fn == nil {
		j.fn = e.lfAttnHeads
	}
	j.out, j.q, j.k, j.v, j.n, j.p, j.scale = out, q, k, v, n, p, scale
	e.jit.Parallel(e.m.Cfg.NHead, 1, j.fn)
	j.out, j.q, j.k, j.v = nil, nil, nil, nil
}

// lfAttnHeads is one chunk of lfAttend: heads h0..h1-1.
func (e *Embedder) lfAttnHeads(h0, h1 int) {
	j, c := &e.rg.lfAttn, e.m.Cfg
	d, hd := c.NEmbd, c.HeadDim
	group := c.NHead / c.NKVHead
	for h := h0; h < h1; h++ {
		off, koff := h*hd, (h/group)*hd
		att := e.att[h*e.attStride:]
		acc := e.acc[h*hd : (h+1)*hd]
		kvw := c.NKVHead * hd
		for i := 0; i < j.n; i++ {
			lo, hi := 0, j.n
			if len(e.seqs) > 0 {
				sq := e.seqs[e.rowSeq[i]]
				lo, hi = sq.Lo, sq.Hi
			}
			if i < j.p && e.violation != "prefix-reads-text" {
				hi = j.p
			}
			if e.violation == "causal-attn" {
				hi = i + 1
			}
			keys := hi - lo
			e.attnLF.AttnScores(att[:keys], j.k[lo*kvw+koff:], j.q[i*d+off:], keys)
			nn.Scale32JIT(att[:keys], j.scale)
			nn.Softmax32JIT(att, keys)
			e.attnLF.AttnAcc(acc, j.v[lo*kvw+koff:], att[:keys], keys)
			copy(j.out[i*d+off:i*d+off+hd], acc)
		}
	}
}

// lfAttnRegion is lfAttend's pool region.
type lfAttnRegion struct {
	out, q, k, v []float32
	n, p         int
	scale        float32
	fn           func(lo, hi int)
}

// rowsRMS is y[r] = RMSNorm(x[r]; w) over n rows of width wd.
func (e *Embedder) rowsRMS(y, x, w []float32, n, wd int) {
	j := &e.rg.rows
	j.op, j.y, j.x, j.w, j.b, j.d, j.eps = encRowRMS, y, x, w, nil, wd, e.m.Cfg.RMSEps
	e.rowRun(n)
}
