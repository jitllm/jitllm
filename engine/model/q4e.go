package model

import (
	"fmt"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
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
// (Config.q4eFault) to show the fixture can see it: the front end states the
// broken graph, which every device composes. The hyper-connection's and
// QSA's pooling are taken out as rewrites of the composed statement instead
// (q4eComposedViolations).
type q4eFault uint8

const (
	q4eFaultNone q4eFault = iota
	// q4eFaultNoNGram drops the n-gram embedding.
	q4eFaultNoNGram
	// q4eFaultNoNGramConv drops its convolution.
	q4eFaultNoNGramConv
	// q4eFaultNoEOSReset hashes across an end-of-sequence token.
	q4eFaultNoEOSReset
	// q4eFaultSiLUGate gates the delta rule's output by silu, qwen3next's.
	q4eFaultSiLUGate
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
	if c.HCLowRank == 0 && c.QSARatio == 0 && c.NGramBlock == 0 {
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
	// The n-gram block: the key and value projections of the gathered rows
	// and the convolution (HCMult*n_embd channels of NGramConv taps, as the
	// file stores them: a channel's taps contiguous); its norms are in the
	// layer's vecs.
	ngKey, ngVal tensor
	ngCv         []float32
	// ngCvTaps is ngCv as a tensor of a row a channel, which the composer's
	// OpStateConv binds.
	ngCvTaps tensor
}

func (h *q4eHC) tensors() [3]*tensor { return [3]*tensor{&h.down, &h.up, &h.inj} }

// loadQ4EHC reads one hyper-connection's weights and checks their shapes.
func loadQ4EHC(h *q4eHC, bi int32, cfg *Config, norm, down, up, inj jlm.Role,
	getTo func(*tensor, jlm.Role, int32, int32) error, vec func(jlm.Role, int32) ([]float32, error)) error {
	var err error
	if h.norm, err = vec(norm, bi); err != nil {
		return err
	}
	if err = getTo(&h.down, down, bi, -1); err != nil {
		return err
	}
	if err = getTo(&h.up, up, bi, -1); err != nil {
		return err
	}
	hd, r := cfg.HCMult*cfg.NEmbd, cfg.HCLowRank
	if inj != 0 {
		if err = getTo(&h.inj, inj, bi, -1); err != nil {
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
	getTo func(*tensor, jlm.Role, int32, int32) error, vec func(jlm.Role, int32) ([]float32, error)) error {
	if !cfg.Q4E() {
		return nil
	}
	d := &q4eLayer{}
	l.q4e = d
	if err := loadQ4EHC(&d.attn, bi, cfg, jlm.RoleHCAttnNorm, jlm.RoleHCAttnDown, jlm.RoleHCAttnUp,
		jlm.RoleHCAttnInject, getTo, vec); err != nil {
		return err
	}
	if err := loadQ4EHC(&d.ffn, bi, cfg, jlm.RoleHCFFNNorm, jlm.RoleHCFFNDown, jlm.RoleHCFFNUp,
		jlm.RoleHCFFNInject, getTo, vec); err != nil {
		return err
	}
	if int(bi) >= cfg.NLayer {
		// A prediction block: its own head mix, the trunk's roles at its
		// index (llama.cpp's nextn.hc_head_*).
		if err := loadQ4EHC(&d.head, bi, cfg, jlm.RoleHCHeadNorm, jlm.RoleHCHeadDown, jlm.RoleHCHeadUp,
			0, getTo, vec); err != nil {
			return err
		}
		d.head.head = true
	}
	if !cfg.NGramAt(int(bi)) {
		return nil
	}
	var err error
	if err = getTo(&d.ngKey, jlm.RoleNGramKey, bi, -1); err != nil {
		return err
	}
	if err = getTo(&d.ngVal, jlm.RoleNGramValue, bi, -1); err != nil {
		return err
	}
	if d.ngCv, err = vec(jlm.RoleNGramConv, bi); err != nil {
		return err
	}
	hd, w, kt := cfg.HCMult*cfg.NEmbd, cfg.NGramWidth(), cfg.NGramConv
	if d.ngKey.k != w || d.ngKey.rows != hd || d.ngVal.k != w || d.ngVal.rows != cfg.NEmbd || len(d.ngCv) != hd*kt {
		return fmt.Errorf("model: block %d's n-gram embedding is key %dx%d, value %dx%d, convolution %d; "+
			"want %d wide in, %d streams of %d, %d taps", bi, d.ngKey.k, d.ngKey.rows, d.ngVal.k, d.ngVal.rows,
			len(d.ngCv), w, cfg.HCMult, cfg.NEmbd, kt)
	}
	d.ngCvTaps = f32Tensor(d.ngCv, hd, kt)
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
	// The head's mix is the model's, in no block's table.
	bind := func(dst *tensor, r jlm.Role, block, index int32) error {
		t, err := get(r, block, index)
		*dst = t
		return err
	}
	if err := loadQ4EHC(&h.hc, jlm.DenseBlock, cfg, jlm.RoleHCHeadNorm, jlm.RoleHCHeadDown,
		jlm.RoleHCHeadUp, 0, bind, vec); err != nil {
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

// q4eScratch is a instance's Qwen3.8-Flash-Next buffers, grown to the widest
// call so a warm decode token and a warm step allocate nothing.
type q4eScratch struct {
	rows int
	qsa  *qsaScratch
}

// allocQ4E sizes a instance's buffers for one row, and its streams.
func (s *Instance) allocQ4E() {
	c := s.m.Cfg
	if !c.Q4E() {
		return
	}
	s.q4 = &q4eScratch{}
	s.allocQSA()
	s.growQ4E(1)
}

// growQ4E sizes the per-row buffers for n rows.
func (s *Instance) growQ4E(n int) {
	g := s.q4
	if g == nil || g.rows >= n {
		return
	}
	g.rows = n
	s.growQSARows(n)
}

// q4eDraftStreams lays a prediction block's draft rows, which arrive with
// every stream (s.q4In, row-major, eh_proj's per-stream output), into a chunk
// of rows whose first is row base of the call.
func (s *Instance) q4eDraftStreams(x []float32, rows, base int) {
	c := s.m.Cfg
	w, d := c.ResidW(), c.NEmbd
	for r := 0; r < rows; r++ {
		in := s.q4In[(base+r)*w : (base+r+1)*w]
		for k := 0; k < c.HCMult; k++ {
			copy(c.stream(x, k, rows, r), in[k*d:(k+1)*d])
		}
	}
}

// q4eCollapse is the head's mix for n rows of x, written into stream 0 of
// each row, which the projection reads with no norm: the composed collapse,
// a prediction block's draft through the block's own head mix.
//
// The rows' residual, every stream, is a speculation step's hidden state on
// this model (rowsOut.tapStreams): what a prediction block reads, and what a
// draft chains to its next step.
func (s *Instance) q4eCollapse(x []float32, n int) error {
	s.growQ4E(n)
	s.rows.tapStreams(s, x, n)
	if why := s.cpuCollapse(x, n); why != composer.DeclineServes {
		return s.streamsDeclined("collapse", why)
	}
	return nil
}

// q4eSigScale is the delta rule's output gate: sigmoid on this model, silu
// under the gate's violation.
func (c *Config) q4eSigmoidGate() bool {
	return c.DeltaSigmoidGate && c.q4eFault != q4eFaultSiLUGate
}
