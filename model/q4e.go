package model

import (
	"fmt"

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
type Q4eHC struct {
	Norm           []float32
	Down, Up, Inj  Tensor
	injected, head bool
}

// q4eLayer is a block's two hyper-connections and, on the n-gram block, the
// n-gram embedding's weights.
type q4eLayer struct {
	Attn, Ffn Q4eHC
	// head is a prediction block's own head mix, which its draft's logits
	// and nothing else read (q4eCollapse); empty on a trunk block.
	head Q4eHC
	// The n-gram block: the key and value projections of the gathered rows
	// and the convolution (HCMult*n_embd channels of NGramConv taps, as the
	// file stores them: a channel's taps contiguous); its norms are in the
	// layer's vecs.
	NgKey, NgVal Tensor
	NgCv         []float32
	// ngCvTaps is ngCv as a tensor of a row a channel, which the composer's
	// OpStateConv binds.
	NgCvTaps Tensor
}

func (h *Q4eHC) tensors() [3]*Tensor { return [3]*Tensor{&h.Down, &h.Up, &h.Inj} }

// loadQ4EHC reads one hyper-connection's weights and checks their shapes.
func loadQ4EHC(h *Q4eHC, bi int32, cfg *Config, norm, down, up, inj jlm.Role,
	getTo func(*Tensor, jlm.Role, int32, int32) error, vec func(jlm.Role, int32) ([]float32, error)) error {
	var err error
	if h.Norm, err = vec(norm, bi); err != nil {
		return err
	}
	if err = getTo(&h.Down, down, bi, -1); err != nil {
		return err
	}
	if err = getTo(&h.Up, up, bi, -1); err != nil {
		return err
	}
	hd, r := cfg.HCMult*cfg.NEmbd, cfg.HCLowRank
	if inj != 0 {
		if err = getTo(&h.Inj, inj, bi, -1); err != nil {
			return err
		}
		h.injected = true
		if h.Inj.K != hd || h.Inj.Rows != cfg.HCMult {
			return fmt.Errorf("model: block %d's hyper-connection placement is %dx%d, want %dx%d",
				bi, h.Inj.K, h.Inj.Rows, hd, cfg.HCMult)
		}
	}
	if len(h.Norm) != hd || h.Down.K != hd || h.Down.Rows != r || h.Up.K != r || h.Up.Rows != hd {
		return fmt.Errorf("model: block %d's hyper-connection is a norm of %d, down %dx%d, up %dx%d; "+
			"want %d streams of %d at rank %d", bi, len(h.Norm), h.Down.K, h.Down.Rows, h.Up.K, h.Up.Rows,
			cfg.HCMult, cfg.NEmbd, r)
	}
	return nil
}

// loadQ4E reads block bi's hyper-connections and, on the n-gram block, the
// n-gram embedding's weights.
func loadQ4E(l *Layer, bi int32, cfg *Config,
	getTo func(*Tensor, jlm.Role, int32, int32) error, vec func(jlm.Role, int32) ([]float32, error)) error {
	if !cfg.Q4E() {
		return nil
	}
	d := &q4eLayer{}
	l.Q4e = d
	if err := loadQ4EHC(&d.Attn, bi, cfg, jlm.RoleHCAttnNorm, jlm.RoleHCAttnDown, jlm.RoleHCAttnUp,
		jlm.RoleHCAttnInject, getTo, vec); err != nil {
		return err
	}
	if err := loadQ4EHC(&d.Ffn, bi, cfg, jlm.RoleHCFFNNorm, jlm.RoleHCFFNDown, jlm.RoleHCFFNUp,
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
	if err = getTo(&d.NgKey, jlm.RoleNGramKey, bi, -1); err != nil {
		return err
	}
	if err = getTo(&d.NgVal, jlm.RoleNGramValue, bi, -1); err != nil {
		return err
	}
	if d.NgCv, err = vec(jlm.RoleNGramConv, bi); err != nil {
		return err
	}
	hd, w, kt := cfg.HCMult*cfg.NEmbd, cfg.NGramWidth(), cfg.NGramConv
	if d.NgKey.K != w || d.NgKey.Rows != hd || d.NgVal.K != w || d.NgVal.Rows != cfg.NEmbd || len(d.NgCv) != hd*kt {
		return fmt.Errorf("model: block %d's n-gram embedding is key %dx%d, value %dx%d, convolution %d; "+
			"want %d wide in, %d streams of %d, %d taps", bi, d.NgKey.K, d.NgKey.Rows, d.NgVal.K, d.NgVal.Rows,
			len(d.NgCv), w, cfg.HCMult, cfg.NEmbd, kt)
	}
	d.NgCvTaps = F32Tensor(d.NgCv, hd, kt)
	return nil
}

// q4eHead is the head's mix.
type q4eHead struct{ Hc Q4eHC }

// loadQ4EHead reads the head's mix and the n-gram table.
func loadQ4EHead(m *Model, cfg *Config, f *jlm.File,
	get func(jlm.Role, int32, int32) (Tensor, error), vec func(jlm.Role, int32) ([]float32, error)) error {
	if !cfg.Q4E() {
		return nil
	}
	h := &q4eHead{}
	// The head's mix is the model's, in no block's table.
	bind := func(dst *Tensor, r jlm.Role, block, index int32) error {
		t, err := get(r, block, index)
		*dst = t
		return err
	}
	if err := loadQ4EHC(&h.Hc, jlm.DenseBlock, cfg, jlm.RoleHCHeadNorm, jlm.RoleHCHeadDown,
		jlm.RoleHCHeadUp, 0, bind, vec); err != nil {
		return err
	}
	h.Hc.head = true
	m.Q4eHead = h
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
	m.Ngram = &ngramTable{F: f, E: e, typ: t, RowBytes: int(rb)}
	return nil
}

// q4eScratch is a instance's Qwen3.8-Flash-Next buffers, grown to the widest
// call so a warm decode token and a warm step allocate nothing.
type Q4eScratch struct {
	Rows int
}
