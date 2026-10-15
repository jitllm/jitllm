package model

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/cpu"
)

// Qwen3.8-Flash-Next (transformers' Qwen4ExpForCausalLM, llama.cpp's
// qwen4exp.cpp; jlm.ArchQwen4Exp).
//
// The residual is Config.HCMult streams, stream-major as Gemma 3n's and
// DeepSeek V4's are. Each sublayer reads a mix of them and writes into all of
// them:
//
//	xn     = each stream RMSNormed against its own row of the weight
//	gate   = sigmoid(up(silu(down(xn) / H)))          rank HCLowRank
//	in     = mean over the streams of gate * xn
//	w      = 2 * sigmoid(inject(xn) / H)              one weight a stream
//	x_k   += w_k * sublayer(in)
//
// and the head reads the mix alone, with no output norm. The sublayers are
// qwen3next's (the delta rule with a sigmoid output gate, the gated
// attention, the mixture with a shared expert); the attention blocks select
// their keys by QSA (qsa.go) and one linear block adds the n-gram embedding
// before its mix (ngram.go).
//
// The graph, transcribed: docs/engineering-history/model-correctness.md,
// "qwen4exp".

// q4eFault breaks one piece of the graph, set only by a gate
// (Config.q4eFault) to show the fixture can see it. It is the device's type
// (nn.Q4EPlan.Fault), so one violation reaches both tiers.
type q4eFault = nn.Q4EFault

const (
	q4eFaultNone = nn.Q4EFaultNone
	// q4eFaultNoGate mixes the streams by their plain mean.
	q4eFaultNoGate = nn.Q4EFaultNoGate
	// q4eFaultPlainPlace adds the sublayer output into every stream at
	// weight one.
	q4eFaultPlainPlace = nn.Q4EFaultPlainPlace
	// q4eFaultNoNGram drops the n-gram embedding.
	q4eFaultNoNGram = nn.Q4EFaultNoNGram
	// q4eFaultNoNGramConv drops its convolution.
	q4eFaultNoNGramConv = nn.Q4EFaultNoNGramConv
	// q4eFaultNoEOSReset hashes across an end-of-sequence token.
	q4eFaultNoEOSReset = nn.Q4EFaultNoEOSReset
	// q4eFaultDenseQSA attends every key on the QSA blocks.
	q4eFaultDenseQSA = nn.Q4EFaultDenseQSA
	// q4eFaultQSAFirstKey scores a QSA block by its first key, unpooled.
	q4eFaultQSAFirstKey = nn.Q4EFaultQSAFirstKey
	// q4eFaultNoTail drops the incomplete tail from a QSA selection.
	q4eFaultNoTail = nn.Q4EFaultNoTail
	// q4eFaultSiLUGate gates the delta rule's output by silu, qwen3next's.
	q4eFaultSiLUGate = nn.Q4EFaultSiLUGate
	// q4eFaultUnscaledGate drops the 1/H before the gate's silu.
	q4eFaultUnscaledGate = nn.Q4EFaultUnscaledGate
	// q4eFaultHalfPlace places at sigmoid(inject/H), without the factor of
	// two that centres a weight on one.
	q4eFaultHalfPlace = nn.Q4EFaultHalfPlace
)

// Q4E reports that the model is Qwen3.8-Flash-Next's graph: low-rank gated
// hyper-connections.
func (c *Config) Q4E() bool { return c.HCLowRank != 0 }

// QSAAt reports that block li selects its keys by QSA: every attention
// block, the prediction blocks after the trunk too (the release's is a QSA
// block, as its compress ratio states).
func (c *Config) QSAAt(li int) bool {
	return c.QSARatio != 0 && li >= 0 && li < c.NLayer+c.NMTP && !c.LayerKind(li).Recurrent()
}

// NGramAt reports that block li adds the n-gram embedding.
func (c *Config) NGramAt(li int) bool { return c.NGramBlock != 0 && li == c.NGramBlock-1 }

// ngramHeads is the n-gram embedding's head count.
func (c *Config) ngramHeads() int { return (c.NGram - 1) * c.NGramHeadsPer }

// ngramHist is how many earlier positions the n-gram convolution reads.
func (c *Config) ngramHist() int { return (c.NGramConv - 1) * c.NGram }

// ngramStateLen is one sequence's n-gram state: the NGram-1 tokens before the
// next position (stored as id+1, zero for none, which reads as the
// end-of-sequence id), then the convolution's history, ngramHist positions of
// every stream.
func (c *Config) ngramStateLen() int {
	return c.NGram - 1 + c.ngramHist()*c.HCMult*c.NEmbd
}

// q4eConfig copies and checks a Qwen3.8-Flash-Next container's own fields.
func q4eConfig(out *Config, c *jlm.Config) error {
	if c.Arch != jlm.ArchQwen4Exp {
		if c.HCLowRank != 0 || c.QSARatio != 0 || c.NGramBlock != 0 {
			return fmt.Errorf("model: %s carries Qwen3.8-Flash-Next's hyper-connections", out.Arch)
		}
		return nil
	}
	out.HCLowRank, out.QSARatio = int(c.HCLowRank), int(c.QSARatio)
	out.NGramBlock, out.NGram, out.NGramHeadsPer = int(c.NGramBlock), int(c.NGram), int(c.NGramHeadsPer)
	out.NGramRowDim, out.NGramConv = int(c.NGramRowDim), int(c.NGramConv)
	out.NGramEOS, out.NGramImage = int32(c.NGramEOS), int32(c.NGramImage)
	out.NGramMul, out.NGramOff, out.NGramVoc = c.NGramMul, c.NGramOff, c.NGramVoc
	out.DeltaSigmoidGate = c.Flags2.Has(jlm.Flag2DeltaSigmoidGate)
	switch {
	case out.HCMult < 2 || out.HCLowRank <= 0:
		return fmt.Errorf("model: %s: %d streams at rank %d", out.Arch, out.HCMult, out.HCLowRank)
	case !out.Hybrid() || out.ChanDecay() || out.SSD() || out.MLA():
		return fmt.Errorf("model: %s: hyper-connections around anything but qwen3next's hybrid", out.Arch)
	case out.QSARatio != 0 && (out.IdxHeads <= 0 || out.IdxHeadDim <= 0 || out.IdxHeadDim > out.HeadDim ||
		out.IdxTopK <= 0 || out.IdxTopK%out.QSARatio != 0 || out.NRot > out.IdxHeadDim || out.GeomSplit()):
		return fmt.Errorf("model: %s: QSA over %d heads of %d, a budget of %d at a ratio of %d, "+
			"on heads of %d rotating %d", out.Arch, out.IdxHeads, out.IdxHeadDim, out.IdxTopK, out.QSARatio,
			out.HeadDim, out.NRot)
	}
	if out.NGramBlock != 0 {
		if !out.LayerKind(out.NGramBlock - 1).Recurrent() {
			return fmt.Errorf("model: %s: the n-gram block %d is not a linear block", out.Arch, out.NGramBlock-1)
		}
		if out.NGramRowDim*out.ngramHeads() != out.NGramWidth() {
			return fmt.Errorf("model: %s: %d n-gram heads of %d", out.Arch, out.ngramHeads(), out.NGramRowDim)
		}
	}
	return nil
}

// NGramWidth is the gathered n-gram embedding's width: every head's row.
func (c *Config) NGramWidth() int { return c.ngramHeads() * c.NGramRowDim }

// q4eHC is one hyper-connection's weights: the grouped norm (HCMult rows of
// n_embd), the gate's two projections and the placement's.
type q4eHC struct {
	norm           []float32
	down, up, inj  tensor
	injected, head bool
}

// q4eLayer is a block's two hyper-connections and, on the n-gram block, the
// n-gram embedding's weights.
type q4eLayer struct {
	attn, ffn q4eHC
	// head is a prediction block's own head mix, which its draft's logits
	// and nothing else read (q4eCollapse); empty on a trunk block.
	head q4eHC
	// The n-gram block: the key and value projections of the gathered rows,
	// the three grouped norms and the convolution (HCMult*n_embd channels of
	// NGramConv taps, tap-major once loaded: tap t's channels at t*hcd).
	ngKey, ngVal                    tensor
	ngNormK, ngNormQ, ngNormC, ngCv []float32
}

func (h *q4eHC) tensors() [3]*tensor { return [3]*tensor{&h.down, &h.up, &h.inj} }

// tensors is every matrix of the block's, an array so binding them on the
// decode path allocates nothing.
func (d *q4eLayer) tensors() [10]*tensor {
	return [10]*tensor{&d.attn.down, &d.attn.up, &d.attn.inj, &d.ffn.down, &d.ffn.up, &d.ffn.inj,
		&d.ngKey, &d.ngVal, &d.head.down, &d.head.up}
}

// loadQ4EHC reads one hyper-connection's weights and checks their shapes.
func loadQ4EHC(h *q4eHC, bi int32, cfg *Config, norm, down, up, inj jlm.Role,
	get func(jlm.Role, int32, int32) (tensor, error), vec func(jlm.Role, int32) ([]float32, error)) error {
	var err error
	if h.norm, err = vec(norm, bi); err != nil {
		return err
	}
	if h.down, err = get(down, bi, -1); err != nil {
		return err
	}
	if h.up, err = get(up, bi, -1); err != nil {
		return err
	}
	hd, r := cfg.HCMult*cfg.NEmbd, cfg.HCLowRank
	if inj != 0 {
		if h.inj, err = get(inj, bi, -1); err != nil {
			return err
		}
		h.injected = true
		if h.inj.k != hd || h.inj.rows != cfg.HCMult {
			return fmt.Errorf("model: block %d's hyper-connection placement is %dx%d, want %dx%d",
				bi, h.inj.k, h.inj.rows, hd, cfg.HCMult)
		}
	}
	if len(h.norm) != hd || h.down.k != hd || h.down.rows != r || h.up.k != r || h.up.rows != hd {
		return fmt.Errorf("model: block %d's hyper-connection is a norm of %d, down %dx%d, up %dx%d; "+
			"want %d streams of %d at rank %d", bi, len(h.norm), h.down.k, h.down.rows, h.up.k, h.up.rows,
			cfg.HCMult, cfg.NEmbd, r)
	}
	return nil
}

// loadQ4E reads block bi's hyper-connections and, on the n-gram block, the
// n-gram embedding's weights.
func loadQ4E(l *layer, bi int32, cfg *Config,
	get func(jlm.Role, int32, int32) (tensor, error), vec func(jlm.Role, int32) ([]float32, error)) error {
	if !cfg.Q4E() {
		return nil
	}
	d := &q4eLayer{}
	l.q4e = d
	if err := loadQ4EHC(&d.attn, bi, cfg, jlm.RoleHCAttnNorm, jlm.RoleHCAttnDown, jlm.RoleHCAttnUp,
		jlm.RoleHCAttnInject, get, vec); err != nil {
		return err
	}
	if err := loadQ4EHC(&d.ffn, bi, cfg, jlm.RoleHCFFNNorm, jlm.RoleHCFFNDown, jlm.RoleHCFFNUp,
		jlm.RoleHCFFNInject, get, vec); err != nil {
		return err
	}
	if int(bi) >= cfg.NLayer {
		// A prediction block: its own head mix, the trunk's roles at its
		// index (llama.cpp's nextn.hc_head_*).
		if err := loadQ4EHC(&d.head, bi, cfg, jlm.RoleHCHeadNorm, jlm.RoleHCHeadDown, jlm.RoleHCHeadUp,
			0, get, vec); err != nil {
			return err
		}
		d.head.head = true
	}
	if !cfg.NGramAt(int(bi)) {
		return nil
	}
	var err error
	if d.ngKey, err = get(jlm.RoleNGramKey, bi, -1); err != nil {
		return err
	}
	if d.ngVal, err = get(jlm.RoleNGramValue, bi, -1); err != nil {
		return err
	}
	for _, v := range []struct {
		r   jlm.Role
		dst *[]float32
	}{{jlm.RoleNGramNormKey, &d.ngNormK}, {jlm.RoleNGramNormQuery, &d.ngNormQ},
		{jlm.RoleNGramNormConv, &d.ngNormC}, {jlm.RoleNGramConv, &d.ngCv}} {
		if *v.dst, err = vec(v.r, bi); err != nil {
			return err
		}
	}
	hd, w, kt := cfg.HCMult*cfg.NEmbd, cfg.NGramWidth(), cfg.NGramConv
	if d.ngKey.k != w || d.ngKey.rows != hd || d.ngVal.k != w || d.ngVal.rows != cfg.NEmbd ||
		len(d.ngNormK) != hd || len(d.ngNormQ) != hd || len(d.ngNormC) != hd || len(d.ngCv) != hd*kt {
		return fmt.Errorf("model: block %d's n-gram embedding is key %dx%d, value %dx%d, norms %d/%d/%d, "+
			"convolution %d; want %d wide in, %d streams of %d, %d taps", bi, d.ngKey.k, d.ngKey.rows,
			d.ngVal.k, d.ngVal.rows, len(d.ngNormK), len(d.ngNormQ), len(d.ngNormC), len(d.ngCv), w,
			cfg.HCMult, cfg.NEmbd, kt)
	}
	// The source's [hcd, taps] (channel-major) is re-laid tap-major, so each
	// tap is one vector of channels.
	tm := make([]float32, len(d.ngCv))
	for ch := 0; ch < hd; ch++ {
		for t := 0; t < kt; t++ {
			tm[t*hd+ch] = d.ngCv[ch*kt+t]
		}
	}
	d.ngCv = tm
	return nil
}

// q4eHead is the head's mix.
type q4eHead struct{ hc q4eHC }

// loadQ4EHead reads the head's mix and the n-gram table.
func loadQ4EHead(m *Model, cfg *Config, f *jlm.File,
	get func(jlm.Role, int32, int32) (tensor, error), vec func(jlm.Role, int32) ([]float32, error)) error {
	if !cfg.Q4E() {
		return nil
	}
	h := &q4eHead{}
	if err := loadQ4EHC(&h.hc, jlm.DenseBlock, cfg, jlm.RoleHCHeadNorm, jlm.RoleHCHeadDown,
		jlm.RoleHCHeadUp, 0, get, vec); err != nil {
		return err
	}
	h.hc.head = true
	m.q4eHead = h
	if cfg.NGramBlock == 0 {
		return nil
	}
	e, ok := f.Find(jlm.RoleNGramTable, jlm.DenseBlock, -1)
	if !ok || !jlm.RowTable(e) || e.NDim != 2 || int(e.Dims[0]) != cfg.NGramRowDim {
		return fmt.Errorf("model: %s has an n-gram block and no n-gram table %d wide", cfg.Arch, cfg.NGramRowDim)
	}
	t, ok := jlm.SourceType(e.Type)
	if !ok {
		return fmt.Errorf("model: the n-gram table is %v", e.Type)
	}
	switch t {
	case quant.F32, quant.F16, quant.BF16, quant.IQ4_NL:
	default:
		return fmt.Errorf("model: an n-gram table of %v has no row reader", t)
	}
	rb, err := jlm.RowBytes(e)
	if err != nil || rb > jlm.Align {
		return fmt.Errorf("model: n-gram rows of %d bytes (%v)", rb, err)
	}
	for h := range cfg.NGramVoc {
		if cfg.NGramOff[h]+cfg.NGramVoc[h] > e.Dims[1] {
			return fmt.Errorf("model: n-gram head %d reads past the table's %d rows", h, e.Dims[1])
		}
	}
	m.ngram = &ngramTable{f: f, e: e, typ: t, rowBytes: int(rb)}
	return nil
}

// q4eScratch is a State's Qwen3.8-Flash-Next buffers, grown to the widest
// call so a warm decode token and a warm step allocate nothing.
type q4eScratch struct {
	rows int
	// per row: every stream normed (xn), the gate's low rank (lo) and the
	// gate itself (gate), the mix (mix), the placement weights (w), and a
	// sublayer's output where it must not land in the residual (out).
	xn, lo, gate, mix, w, out []float32
	// twos is the placement's factor of two, a vector so the sigmoid's
	// generated multiply applies it.
	twos []float32
	ng   *ngramScratch
	qsa  *qsaScratch
	// rid is each row's token id for the n-gram block, -1 for a row with
	// none (q4eRowIDs).
	rid []int32
	// marks is the call's rotary marks as a device takes them
	// (qsaMarksToDevice), reused so a warm call allocates nothing.
	marks []int32
	// callState, callSlot and callIDs are a device call's rows (setCall).
	callState *State
	callSlot  int
	callIDs   []int32
	callSeq   []int
	callPos   []int
	callSlots []int
}

// q4eRowIDs records each of n rows' token ids for the n-gram block.
func (s *State) q4eRowIDs(n int, id func(i int) int32) {
	s.growQ4E(n)
	for i := 0; i < n; i++ {
		s.q4.rid[i] = id(i)
	}
}

// allocQ4E sizes a State's buffers for one row, and its streams.
func (s *State) allocQ4E() {
	c := s.m.Cfg
	if !c.Q4E() {
		return
	}
	s.xa = make([]float32, c.ResidW())
	s.x = s.xa[:c.NEmbd]
	s.q4 = &q4eScratch{}
	s.q4.twos = make([]float32, c.HCMult*(nn.ElemLanes+1))
	for i := range s.q4.twos {
		s.q4.twos[i] = 2
	}
	s.allocNGram()
	s.allocQSA()
	s.growQ4E(1)
}

// growQ4E sizes the per-row buffers for n rows.
func (s *State) growQ4E(n int) {
	c, g := s.m.Cfg, s.q4
	if g == nil || g.rows >= n {
		return
	}
	g.rows = n
	h, d := c.HCMult, c.NEmbd
	g.xn, g.gate = make([]float32, n*h*d), make([]float32, n*h*d)
	g.lo = make([]float32, n*c.HCLowRank)
	g.mix, g.out = make([]float32, n*d), make([]float32, n*d)
	g.w = make([]float32, n*h)
	g.rid = make([]int32, n)
	if len(g.twos) < n*h {
		g.twos = make([]float32, n*h)
		for i := range g.twos {
			g.twos[i] = 2
		}
	}
	s.growNGram(n)
	s.growQSARows(n)
}

// q4eExpand copies each row's embedding (stream 0) into the other streams:
// the residual starts as HCMult copies of it.
func (s *State) q4eExpand(x []float32, rows int) { s.q4eExpandAt(x, rows, 0) }

// q4eExpandAt is q4eExpand for a chunk whose first row is row base of the
// call. A prediction block's draft does not start from copies: its rows
// arrive with every stream (s.q4In, row-major, eh_proj's per-stream output),
// and those are laid into the chunk instead.
func (s *State) q4eExpandAt(x []float32, rows, base int) {
	c := s.m.Cfg
	if s.q4In != nil {
		w, d := c.ResidW(), c.NEmbd
		for r := 0; r < rows; r++ {
			in := s.q4In[(base+r)*w : (base+r+1)*w]
			for k := 0; k < c.HCMult; k++ {
				copy(c.stream(x, k, rows, r), in[k*d:(k+1)*d])
			}
		}
		return
	}
	for r := 0; r < rows; r++ {
		x0 := c.stream(x, 0, rows, r)
		for k := 1; k < c.HCMult; k++ {
			copy(c.stream(x, k, rows, r), x0)
		}
	}
}

// hcNorm writes each of n rows' streams into dst, row-major (row r's stream k
// at (r*H + k)*d), each RMSNormed against its own row of w.
func (s *State) hcNorm(dst, x []float32, n int, w []float32) {
	c := s.m.Cfg
	h, d := c.HCMult, c.NEmbd
	for r := 0; r < n; r++ {
		for k := 0; k < h; k++ {
			o := (r*h + k) * d
			s.rmsnorm(dst[o:o+d], c.stream(x, k, n, r), w[k*d:(k+1)*d], c.RMSEps)
		}
	}
}

// hcMix is a hyper-connection's mix for n rows of x: each row's sublayer
// input into g.mix (n_embd a row), and with a placement its weights into
// g.w (HCMult a row).
func (s *State) hcMix(x []float32, n int, hc *q4eHC) error {
	c, g := s.m.Cfg, s.q4
	h, d := c.HCMult, c.NEmbd
	hd := h * d
	xn, gate := g.xn[:n*hd], g.gate[:n*hd]
	s.hcNorm(xn, x, n, hc.norm)
	s.jit.NewInput()
	if c.q4eFault == q4eFaultNoGate {
		copy(gate, xn)
	} else {
		lo := g.lo[:n*c.HCLowRank]
		if err := s.mm(lo, hc.down, xn, n); err != nil {
			return err
		}
		if c.q4eFault != q4eFaultUnscaledGate {
			s.scale(lo, 1/float32(h))
		}
		nn.Act32JIT(lo, nn.ActSiLU)
		s.jit.NewInput()
		if err := s.mm(gate, hc.up, lo, n); err != nil {
			return err
		}
		nn.SigmoidMul32JIT(gate, xn)
		s.jit.NewInput()
	}
	for r := 0; r < n; r++ {
		m := g.mix[r*d : (r+1)*d]
		gr := gate[r*hd : (r+1)*hd]
		copy(m, gr[:d])
		for k := 1; k < h; k++ {
			s.axpy(m, gr[k*d:(k+1)*d], 1)
		}
		s.scale(m, 1/float32(h))
	}
	s.jit.NewInput()
	if !hc.injected {
		return nil
	}
	w := g.w[:n*h]
	if err := s.mm(w, hc.inj, xn, n); err != nil {
		return err
	}
	s.scale(w, 1/float32(h))
	nn.SigmoidMul32JIT(w, g.twos[:n*h])
	if c.q4eFault == q4eFaultHalfPlace {
		s.scale(w, 0.5)
	}
	s.jit.NewInput()
	return nil
}

// hcPlace adds each of n rows' sublayer output y into every stream of x at
// the weights hcMix left in g.w.
func (s *State) hcPlace(x []float32, n int, y []float32) {
	c, g := s.m.Cfg, s.q4
	h, d := c.HCMult, c.NEmbd
	for r := 0; r < n; r++ {
		yr := y[r*d : (r+1)*d]
		for k := 0; k < h; k++ {
			w := g.w[r*h+k]
			if c.q4eFault == q4eFaultPlainPlace {
				w = 1
			}
			s.axpy(c.stream(x, k, n, r), yr, w)
		}
	}
	s.jit.NewInput()
}

// q4eCollapse is the head's mix for n rows of x, written into stream 0 of
// each row, which the projection reads with no norm.
//
// The rows' residual, every stream, is a speculation step's hidden state on
// this model (rowsOut.tapStreams): what a prediction block reads, and what a
// draft chains to its next step. A prediction block's draft mixes through the
// block's own head mix.
func (s *State) q4eCollapse(x []float32, n int) error {
	// A chunk whose every block ran on a device grew nothing on the host; a
	// trunk's n-gram rows did, a prediction block's draft has none.
	s.growQ4E(n)
	c, g := s.m.Cfg, s.q4
	s.rows.tapStreams(s, x, n)
	hc := &s.m.q4eHead.hc
	if s.lo >= c.NLayer {
		hc = &s.m.layers[s.lo].q4e.head
	}
	if err := s.hcMix(x, n, hc); err != nil {
		return err
	}
	for r := 0; r < n; r++ {
		copy(c.stream(x, 0, n, r), g.mix[r*c.NEmbd:(r+1)*c.NEmbd])
	}
	return nil
}

// q4eSigScale is the delta rule's output gate: sigmoid on this model, silu
// under the gate's violation.
func (c *Config) q4eSigmoidGate() bool {
	return c.DeltaSigmoidGate && c.q4eFault != q4eFaultSiLUGate
}

// f32Ones is n ones.
func f32Ones(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = 1
	}
	return v
}

// invSqrt is 1/sqrt(n) as a float32.
func invSqrt(n int) float32 { return float32(1 / math.Sqrt(float64(n))) }

// attnSetF32 is a kernel set scoring rows of width w laid back to back.
func (s *State) attnSetF32(w int) *nn.AttnSet { return s.jit.AttnSetFor(w, w, w, cpu.KVF32) }
